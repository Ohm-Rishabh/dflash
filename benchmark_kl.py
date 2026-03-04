import argparse
import time
import json
import random
from pathlib import Path
from itertools import chain
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def _kl_divergence_topk(draft_logits: "torch.Tensor", target_logits: "torch.Tensor", top_k: int = 50) -> "torch.Tensor":
    """
    Compute forward KL divergence D_KL(P_draft || Q_target) using a top-k approximation.
    Only sums over the top-k tokens from the draft distribution for efficiency.
    Lower KL = better alignment = draft more likely to be accepted.
    target_logits must match the spatial dimensions of draft_logits.
    Returns: (seq,) tensor of KL divergence per token position.
    """
    # shape: (1, seq, vocab), we just take batch 0 -> (seq, vocab)
    draft_log_p = F.log_softmax(draft_logits[0], dim=-1)
    target_log_p = F.log_softmax(target_logits[0], dim=-1)
    
    # Get top-k indices from draft (P_draft)
    k = min(top_k, draft_log_p.shape[-1])
    draft_topk_log_p, topk_indices = torch.topk(draft_log_p, k, dim=-1) # (seq, k)
    
    # Gather corresponding target log probs
    target_topk_log_p = torch.gather(target_log_p, dim=-1, index=topk_indices) # (seq, k)
    
    # KL = sum_v P(v) * (log P(v) - log Q(v))
    draft_topk_p = torch.exp(draft_topk_log_p)
    kl_div = torch.sum(draft_topk_p * (draft_topk_log_p - target_topk_log_p), dim=-1) # (seq,)
    
    return kl_div


def _block_size_from_kl_divergence(kl_pos_0_prev: float | None, kl_w10: float | None, batch_sizes: list[int]) -> int:
    """
    Choose block size using the previous step's kl_pos_0 (strongest signal, r=-0.385)
    blended with trailing w=10 (r=-0.241) for smoothing.
    
    Thresholds calibrated from 30-sample AIME25 profiling run:
    - ~80% of kl_pos_0 values are < 1.0 (these have acceptance_length ~8-16)
    - Values 1-5 show moderate acceptance (~4-10)
    - Values > 5 show low acceptance (~1-4)
    """
    if kl_pos_0_prev is None:
        return batch_sizes[-1]  # default max block size on first step
    
    # Primary signal: kl_pos_0 from previous step
    # Secondary signal: trailing w=10 mean (smoother, less noisy)
    # Blend: 70% kl_pos_0, 30% kl_w10 (when available)
    if kl_w10 is not None:
        kl_signal = 0.7 * kl_pos_0_prev + 0.3 * kl_w10
    else:
        kl_signal = kl_pos_0_prev
    
    # Calibrated thresholds from profiling data
    if kl_signal < 0.5:
        return 16 if 16 in batch_sizes else batch_sizes[-1]
    if kl_signal < 1.5:
        return 12 if 12 in batch_sizes else batch_sizes[-1]
    if kl_signal < 3.0:
        return 8 if 8 in batch_sizes else batch_sizes[0]
    return 4 if 4 in batch_sizes else batch_sizes[0]

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    mode: str = "profiling",
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    kl_divergence_log = []
    
    # KL tracking state
    kl_divergence_list = []   # Per-token KL values for trailing windows
    last_draft_logits = None
    prev_kl_pos_0 = None      # kl_pos_0 from the previous step (primary adaptive signal)
    
    draft_prefill = True

    while start < max_length:
        step_start_kl_overall = None
        
        if block_size > 1:
            # Compute all window metrics for logging
            kl_w10, kl_w5, kl_w3, kl_w1 = None, None, None, None
            if kl_divergence_list:
                kl_w10 = float(np.mean(kl_divergence_list[-10:])) if len(kl_divergence_list) >= 10 else float(np.mean(kl_divergence_list))
                kl_w5 = float(np.mean(kl_divergence_list[-5:])) if len(kl_divergence_list) >= 5 else float(np.mean(kl_divergence_list))
                kl_w3 = float(np.mean(kl_divergence_list[-3:])) if len(kl_divergence_list) >= 3 else float(np.mean(kl_divergence_list))
                kl_w1 = float(kl_divergence_list[-1])
            
            step_start_kl_overall = kl_w5

            if mode == "adaptive-kl":
                # Use previous step's kl_pos_0 (strongest signal, r=-0.385) blended with w=10
                block_size = _block_size_from_kl_divergence(prev_kl_pos_0, kl_w10, [4, 8, 12, 16])
            elif mode == "profiling":
                block_size = 16

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_out = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :]
            draft_logits = target.lm_head(draft_out)
            last_draft_logits = draft_logits.clone()
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)

        if block_size > 1:
            kl_div = _kl_divergence_topk(last_draft_logits, output.logits[:, :-1, :])  # (bs-1,)
            accepted_kl = kl_div[:max(1, acceptance_length)].cpu().tolist()
            kl_divergence_list.extend(accepted_kl)
            
            # The KL of the *first* drafted token in this block (position 0)
            # This is the most direct signal for the *current* step, rather than trailing window
            kl_pos_0 = kl_div[0].item() if len(kl_div) > 0 else None
            
            kl_divergence_log.append({
                "block_size": block_size,
                "kl_divergence": step_start_kl_overall,
                "kl_w10": kl_w10,
                "kl_w5": kl_w5,
                "kl_w3": kl_w3,
                "kl_w1": kl_w1,
                "kl_pos_0": kl_pos_0,
                "prev_kl_pos_0": prev_kl_pos_0,                  # The signal used for adaptive decisions
                "kl_divergence_per_pos": kl_div.cpu().tolist(),
                "acceptance_length": acceptance_length + 1,
            })
            
            # Save kl_pos_0 for the NEXT step's adaptive decision
            prev_kl_pos_0 = kl_pos_0

        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        kl_divergence_log=kl_divergence_log,
    )


def _plot_one_kl_acceptance(
    x: "np.ndarray",
    y: "np.ndarray",
    out_path: Path,
    title: str,
    use_hexbin: bool = True,
) -> None:
    """Single plot (hexbin or scatter) + marginals + regression for one (x, y) dataset."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    if len(x) < 2:
        return

    r = np.corrcoef(x, y)[0, 1]
    coefs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(coefs, x_line)

    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.02, wspace=0.02)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    if use_hexbin:
        hb = ax_main.hexbin(x, y, gridsize=24, cmap="Oranges", mincnt=1, edgecolors="none", alpha=0.85)
        fig.colorbar(hb, ax=ax_main, label="count")
    else:
        ax_main.scatter(x, y, alpha=0.5, s=14, c="darkorange", edgecolors="none")
    ax_main.plot(x_line, y_line, color="teal", linewidth=2, label=f"fit (r = {r:.3f})")
    ax_main.set_ylabel("Acceptance length")
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    ax_top.hist(x, bins=40, color="darkorange", alpha=0.7, edgecolor="none")
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)

    y_bins = np.arange(0, y.max() + 2, 1)
    ax_right.hist(y, bins=y_bins, orientation="horizontal", color="darkorange", alpha=0.7, edgecolor="none")
    ax_right.set_xlabel("Count")

    ax_main.set_xlabel("KL Divergence D(P_draft || Q_target)")
    fig.suptitle(title, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_kl_correlation_matrix(
    responses: list, base_out_path: Path, block_size: int
) -> None:
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        print(f"Skipping correlation plot (install pandas and matplotlib): {e}")
        return

    all_rows = []
    
    for r in responses:
        spec_resp = r.get(block_size)
        if not spec_resp or not hasattr(spec_resp, "kl_divergence_log"):
            continue
            
        for entry in spec_resp.kl_divergence_log:
            all_rows.append({
                "kl_w10": float(entry["kl_w10"]) if entry.get("kl_w10") is not None else None,
                "kl_w5": float(entry["kl_w5"]) if entry.get("kl_w5") is not None else None,
                "kl_w3": float(entry["kl_w3"]) if entry.get("kl_w3") is not None else None,
                "kl_w1": float(entry["kl_w1"]) if entry.get("kl_w1") is not None else None,
                "kl_pos_0": float(entry["kl_pos_0"]) if entry.get("kl_pos_0") is not None else None,
                "acceptance_length": entry["acceptance_length"],
            })

    if not all_rows:
        print("No KL divergence data available for plotting.")
        return

    df = pd.DataFrame(all_rows)
    y = df["acceptance_length"].values
    
    metrics_to_plot = {
        "kl_w10": "Trailing Window (w=10)",
        "kl_w5": "Trailing Window (w=5)",
        "kl_w3": "Trailing Window (w=3)",
        "kl_w1": "Previous Token Only (w=1)",
        "kl_pos_0": "Current Block Token 0",
    }
    
    for metric, name in metrics_to_plot.items():
        if df[metric].notna().any():
            valid_idx = df[metric].notna()
            x_valid = df[metric][valid_idx].values
            y_valid = y[valid_idx]
            
            out_path = base_out_path.with_name(f"{base_out_path.stem}_{metric}{base_out_path.suffix}")
            _plot_one_kl_acceptance(
                x_valid, y_valid, out_path, f"KL Divergence [{name}] vs acceptance length (block_size={block_size})", use_hexbin=True
            )
            print(f"[{metric}] KL correlation plot saved to {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="profiling", choices=["profiling", "adaptive-kl"], help="Run in standard profiling mode (fixed bs) or adaptive block size based on KL")
    parser.add_argument("--save-results", action="store_true", help="Save the execution results to a JSON file and plot correlation matrices")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    mode=args.mode,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    
    if args.save_results and dist.is_main():
        res_json = []
        for r in responses:
            spec_resp = r[block_size]
            res_json.append({"kl_divergence_log": spec_resp.kl_divergence_log})
        
        with open("benchmark_kl_results.json", "w") as f:
            json.dump(res_json, f, indent=2)
            
        _plot_kl_correlation_matrix(responses, Path("local_kl_acceptance_correlation.png"), block_size)

if __name__ == "__main__":
    main()