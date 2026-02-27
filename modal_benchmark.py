"""
modal_benchmark.py — Deploy DFlash on Modal and benchmark with custom prompts.

Usage:
    modal run modal_benchmark.py                 # runs the built-in demo prompts
    modal run modal_benchmark.py --detach        # fire-and-forget, check logs later

Edit the `prompts` list in main() to use your own inputs.

Setup:
    pip install modal
    modal setup          # authenticate once
    modal secret create huggingface-secret HF_TOKEN=<your_token>  # only if models need auth
"""

import sys
import time
import json
import random
import numpy as np
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME       = "Qwen/Qwen3-8B"
DRAFT_MODEL_NAME = "z-lab/Qwen3-8B-DFlash-b16"
HF_HOME_DIR      = "/cache/huggingface"   # inside the container; backed by a Volume
BATCH_SIZES      = [4, 8, 12, 16]

# Confidence metric for variable block size (paper [10]: top-k log-prob, lower = more confident)
CONFIDENCE_TOP_K = 10   # k in c_t = -(1/k) * sum_{v in TopK} log p_t(v)
CONFIDENCE_WINDOW_W = 10  # C_t = (1/W) * sum_{i=t-W+1}^t c_i

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
# Start from the CUDA devel image so flash-attn can compile against nvcc.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    # flash-attn needs these at compile time (wheel required by --no-build-isolation)
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.51.0",   # Qwen3 support added in 4.51
        "accelerate",
        "numpy",
        "loguru",
        "rich",
        "tqdm",
        "datasets",
        "huggingface_hub",
    )
    # flash-attn must be compiled against the torch/CUDA above
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    # Add the local repo into the image. Modal hashes these files and only
    # rebuilds this layer when something changes — so local edits are always
    # picked up without touching the heavy pip layers above.
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/dflash",
        ignore=[".git", "__pycache__", "*.pyc", ".env", "benchmark_results.json"],
    )
)

app = modal.App("dflash-benchmark", image=image)

# Persistent volume so model weights are only downloaded once
hf_cache_vol = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Timing helper (mirrors benchmark.py)
# ---------------------------------------------------------------------------

def _cuda_time() -> float:
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def _topk_logprob_confidence(logits: "torch.Tensor", top_k: int) -> "torch.Tensor":
    """
    Per-position confidence c_t = -(1/k) * sum_{v in TopK} log p_t(v).
    Lower c_t = more confident. Shape: (batch, seq) -> returns (seq,) for batch 0.
    """
    import torch
    log_p = torch.log_softmax(logits, dim=-1)  # (1, seq, vocab)
    k = min(top_k, log_p.shape[-1])
    topk_log_p, _ = torch.topk(log_p[0], k, dim=-1)  # (seq, k)
    c_t = -topk_log_p.mean(dim=-1)  # (seq,), lower = more confident
    return c_t


def _block_size_from_confidence(C_t: float | None, batch_sizes: list[int]) -> int:
    """
    Choose block size from confidence C_t (lower = more confident).
    Placeholder thresholds for profiling; calibrate from logged data.
    """
    if C_t is None:
        return batch_sizes[-1]  # default max block size
    # Placeholder: more confident -> larger block (tune via calibration)
    if C_t < 2.0:
        return 16 if 16 in batch_sizes else batch_sizes[-1]
    if C_t < 3.0:
        return 12 if 12 in batch_sizes else 8
    if C_t < 4.0:
        return 8
    return 4 if 4 in batch_sizes else batch_sizes[0]

# ---------------------------------------------------------------------------
# Modal class — load once, call many times
# ---------------------------------------------------------------------------

@app.cls(
    gpu="H100",                            # swap to "A100:80GB" if needed
    volumes={HF_HOME_DIR: hf_cache_vol},
    timeout=7200,
    # Uncomment the line below if either model requires a HuggingFace token:
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
class DFlashRunner:
    model_name: str       = modal.parameter(default=MODEL_NAME)
    draft_model_name: str = modal.parameter(default=DRAFT_MODEL_NAME)

    @modal.enter()
    def load_models(self):
        """Called once when the container starts — load both models into GPU RAM."""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Put HF cache on the persistent volume so weights survive across cold starts
        os.environ["HF_HOME"] = HF_HOME_DIR

        # Make `from model import ...` (and any other local imports) work.
        # The whole repo is mounted at /root/dflash, so model/ is at /root/dflash/model.
        sys.path.insert(0, "/root/dflash")
        from model import DFlashDraftModel

        def _has_flash_attn() -> bool:
            try:
                import flash_attn  # noqa: F401
                return True
            except ImportError:
                print("flash_attn not available — falling back to sdpa")
                return False

        attn_impl = "flash_attention_2" if _has_flash_attn() else "sdpa"
        self.device = torch.device("cuda:0")

        print(f"Loading target model : {self.model_name}")
        self.target = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(self.device).eval()

        print(f"Loading draft model  : {self.draft_model_name}")
        self.draft = DFlashDraftModel.from_pretrained(
            self.draft_model_name,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(self.device).eval()

        self.tokenizer     = AutoTokenizer.from_pretrained(self.model_name)
        self.block_size    = self.draft.block_size
        self.mask_token_id = self.draft.mask_token_id

        print(f"Ready — block_size={self.block_size}, mask_token_id={self.mask_token_id}")

    # -----------------------------------------------------------------------
    # Core generation (mirrors dflash_generate in benchmark.py exactly)
    # -----------------------------------------------------------------------

    def _generate(
        self,
        input_ids,
        bs: int,
        max_new_tokens: int,
        temperature: float,
    ) -> SimpleNamespace:
        import torch
        from transformers import DynamicCache
        from model import sample, extract_context_feature

        stop_token_ids = [self.tokenizer.eos_token_id]
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        max_bs = 16 if bs > 1 else 1
        output_ids = torch.full(
            (1, max_length + max_bs),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(0)
        past_kv_target = DynamicCache()
        past_kv_draft  = DynamicCache()

        # ---- Prefill ----
        prefill_start = _cuda_time()
        out = self.target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_kv_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=(bs > 1),
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(out.logits, temperature)
        if bs > 1:
            target_hidden = extract_context_feature(out.hidden_states, self.draft.target_layer_ids)
        time_to_first_token = _cuda_time() - prefill_start

        # ---- Decode ----
        decode_start  = _cuda_time()
        start         = num_input_tokens
        acceptance_lengths = []
        # Target's probability for each drafted token (before accept/reject), per step
        verification_draft_probs = []
        # For each position: True = draft accepted, False = target's token used (reject)
        verification_accepted = []
        # Confidence-based profiling: c_t per token, C_t = windowed mean (paper [10])
        confidence_list = []
        confidence_profiling_log = []
        draft_prefill = True

        while start < max_length:
            step_start_C_t = None
            if bs > 1:
                # Fixed block size 16; compute C_t only for profiling (confidence vs acceptance)
                if len(confidence_list) >= CONFIDENCE_WINDOW_W:
                    step_start_C_t = float(np.mean(confidence_list[-CONFIDENCE_WINDOW_W:]))
                elif confidence_list:
                    step_start_C_t = float(np.mean(confidence_list))
                bs = 16
            block_ids = output_ids[:, start : start + bs].clone()
            block_pos = position_ids[:, start : start + bs]

            if bs > 1:
                print(f"[draft fwd] block_size={bs}, step start={start}, C_t={step_start_C_t}")
                noise_emb = self.target.model.embed_tokens(block_ids)
                draft_out = self.target.lm_head(
                    self.draft(
                        target_hidden=target_hidden,
                        noise_embedding=noise_emb,
                        position_ids=position_ids[
                            :, past_kv_draft.get_seq_length() : start + bs
                        ],
                        past_key_values=past_kv_draft,
                        use_cache=True,
                        is_causal=False,
                    )[:, -bs + 1 :, :]
                )
                past_kv_draft.crop(start)
                block_ids[:, 1:] = sample(draft_out)
                if draft_prefill:
                    draft_prefill = False
                    decode_start  = _cuda_time()   # start timing after draft warms up

            out = self.target(
                block_ids,
                position_ids=block_pos,
                past_key_values=past_kv_target,
                use_cache=True,
                output_hidden_states=(bs > 1),
            )
            # Target's probability for each drafted token (before accept/reject)
            if bs > 1:
                probs = torch.softmax(out.logits, dim=-1)
                # out.logits[0, j, :] = next-token dist after position j; draft at j+1 is block_ids[0, j+1]
                step_probs = [
                    probs[0, j, block_ids[0, j + 1]].item()
                    for j in range(bs - 1)
                ]
                verification_draft_probs.append(step_probs)
            posterior = sample(out.logits, temperature)

            # Count how many consecutive drafted tokens the target agrees with
            acc_len = (
                (block_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            # Accept/reject per position: accepted = (draft == target's sample)
            if bs > 1:
                verification_accepted.append([j < acc_len for j in range(bs - 1)])
                # Confidence profiling: c_t from target logits, then C_t = windowed mean
                c_t = _topk_logprob_confidence(out.logits, CONFIDENCE_TOP_K)  # (bs,)
                accepted_ct = c_t[: acc_len + 1].cpu().tolist()
                confidence_list.extend(accepted_ct)
                confidence_profiling_log.append({
                    "block_size": bs,
                    "confidence": step_start_C_t,
                    "acceptance_length": acc_len + 1,
                })
                print(f"  [profile] block_size={bs} confidence={step_start_C_t} acceptance_length={acc_len + 1}")

            output_ids[:, start : start + acc_len + 1] = block_ids[:, : acc_len + 1]
            output_ids[:, start + acc_len + 1]         = posterior[:, acc_len]
            acceptance_lengths.append(acc_len + 1)
            start += acc_len + 1
            past_kv_target.crop(start)

            if bs > 1:
                target_hidden = extract_context_feature(
                    out.hidden_states, self.draft.target_layer_ids
                )[:, : acc_len + 1, :]

            if any(sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids):
                break

        # ---- Trim output ----
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        stop_tids  = torch.tensor(stop_token_ids, device=self.device)
        stop_idx   = torch.isin(
            output_ids[0][num_input_tokens:], stop_tids
        ).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_idx[0] + 1]

        num_output_tokens = output_ids.shape[1] - num_input_tokens
        tpot = (_cuda_time() - decode_start) / max(num_output_tokens, 1)

        # Empirical accept/reject threshold (no explicit threshold is used; accept = draft == target sample)
        min_draft_prob_when_accepted = None
        max_draft_prob_when_rejected = None
        if verification_draft_probs and verification_accepted:
            probs_accepted = [
                p for step_probs, step_acc in zip(verification_draft_probs, verification_accepted)
                for p, acc in zip(step_probs, step_acc) if acc
            ]
            probs_rejected = [
                p for step_probs, step_acc in zip(verification_draft_probs, verification_accepted)
                for p, acc in zip(step_probs, step_acc) if not acc
            ]
            min_draft_prob_when_accepted = min(probs_accepted) if probs_accepted else None
            max_draft_prob_when_rejected = max(probs_rejected) if probs_rejected else None

        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=time_to_first_token,
            time_per_output_token=tpot,
            acceptance_lengths=acceptance_lengths,
            verification_draft_probs=verification_draft_probs,
            verification_accepted=verification_accepted,
            min_draft_prob_when_accepted=min_draft_prob_when_accepted,
            max_draft_prob_when_rejected=max_draft_prob_when_rejected,
            confidence_profiling_log=confidence_profiling_log,
        )

    # -----------------------------------------------------------------------
    # Public benchmark method
    # -----------------------------------------------------------------------

    @modal.method()
    def run(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float  = 0.0,
        block_size: int | None = None,
        enable_thinking: bool = False,
    ) -> dict:
        """
        Benchmark DFlash against the baseline (block_size=1) for each prompt.

        Args:
            prompts:        List of plain user prompt strings.
            max_new_tokens: Max tokens to generate per prompt.
            temperature:    Sampling temperature (0.0 = greedy).
            block_size:     Parallel draft tokens; defaults to the model's block_size.
            enable_thinking: If True, use Qwen reasoning/thinking format. Paper and
                            benchmark.py use False; draft is trained for non-thinking
                            (higher acceptance rate with False).

        Returns:
            A dict with per-prompt results and aggregate metrics.
        """
        import torch

        all_results = []

        for i, prompt in enumerate(prompts):
            print(f"\n── Prompt {i + 1}/{len(prompts)} {'─' * 50}")
            print(f"   {prompt[:120]}")

            messages   = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                baseline = self._generate(input_ids, bs=1,  max_new_tokens=max_new_tokens, temperature=temperature)
                dflash   = self._generate(input_ids, bs=16, max_new_tokens=max_new_tokens, temperature=temperature)

            output_text = self.tokenizer.decode(
                dflash.output_ids[0, dflash.num_input_tokens :],
                skip_special_tokens=True,
            )
            speedup = baseline.time_per_output_token / dflash.time_per_output_token
            avg_acc = float(np.mean(dflash.acceptance_lengths))

            # Per-prompt console log
            print(f"   Response:\n   {output_text[:2000]}{'...' if len(output_text) > 2000 else ''}")
            print(f"   Input tokens       : {dflash.num_input_tokens}")
            print(f"   Output tokens      : {dflash.num_output_tokens}")
            print(f"   TTFT               : {dflash.time_to_first_token * 1000:.1f} ms")
            print(f"   Baseline           : {1 / baseline.time_per_output_token:.1f} tok/s")
            print(f"   DFlash             : {1 / dflash.time_per_output_token:.1f} tok/s")
            print(f"   Decoding speedup   : {speedup:.2f}x")
            print(f"   Avg acceptance len : {avg_acc:.2f}")
            print(f"   Acc. len histogram : {[dflash.acceptance_lengths.count(b) for b in range(17)]}")
            if dflash.min_draft_prob_when_accepted is not None:
                rej = dflash.max_draft_prob_when_rejected
                rej_str = f"{rej:.4f}" if rej is not None else "N/A (no rejections)"
                print(f"   Accept/reject (empirical): min P(draft) when accepted = {dflash.min_draft_prob_when_accepted:.4f}, max P(draft) when rejected = {rej_str}")

            all_results.append({
                "prompt": prompt,
                "output": output_text,
                "baseline": {
                    "num_input_tokens":       baseline.num_input_tokens,
                    "num_output_tokens":      baseline.num_output_tokens,
                    "time_to_first_token_s":  baseline.time_to_first_token,
                    "time_per_output_token_s": baseline.time_per_output_token,
                    "tokens_per_second":      1 / baseline.time_per_output_token,
                },
                "dflash": {
                    "num_input_tokens":        dflash.num_input_tokens,
                    "num_output_tokens":       dflash.num_output_tokens,
                    "time_to_first_token_s":   dflash.time_to_first_token,
                    "time_per_output_token_s": dflash.time_per_output_token,
                    "tokens_per_second":       1 / dflash.time_per_output_token,
                    "acceptance_lengths":      dflash.acceptance_lengths,
                    "avg_acceptance_length":   avg_acc,
                    "verification_draft_probs": dflash.verification_draft_probs,
                    "verification_accepted":   dflash.verification_accepted,
                    "accept_reject_cutoff":    {
                        "note": "No explicit threshold; accept = (draft == target sample). Empirical probability bounds:",
                        "min_draft_prob_when_accepted":  dflash.min_draft_prob_when_accepted,
                        "max_draft_prob_when_rejected":  dflash.max_draft_prob_when_rejected,
                    },
                    "confidence_profiling_log": dflash.confidence_profiling_log,
                },
                "decoding_speedup": speedup,
            })

        # ---- Aggregate metrics (same formulas as benchmark.py) ----
        t1  = float(np.mean([r["baseline"]["time_per_output_token_s"] for r in all_results]))
        tb  = float(np.mean([r["dflash"]["time_per_output_token_s"]   for r in all_results]))
        tau = float(np.mean([r["dflash"]["avg_acceptance_length"]     for r in all_results]))
        all_acc  = list(chain(*[r["dflash"]["acceptance_lengths"] for r in all_results]))
        histogram = [all_acc.count(b) / len(all_acc) for b in range(17)]

        summary = {
            "model":            self.model_name,
            "draft_model":      self.draft_model_name,
            "block_size":       "[16]",
            "max_new_tokens":   max_new_tokens,
            "temperature":      temperature,
            "num_prompts":      len(prompts),
            "aggregate": {
                "decoding_speedup":               t1 / tb,
                "avg_acceptance_length":          tau,
                "baseline_tokens_per_second":     1 / t1,
                "dflash_tokens_per_second":       1 / tb,
                # histogram[b] = fraction of draft steps where exactly b tokens were accepted
                "acceptance_length_histogram":    {
                    str(b): f"{histogram[b] * 100:.1f}%"
                    for b in range(len(histogram))
                },
            },
            "per_prompt": all_results,
        }

        print("\n════ AGGREGATE RESULTS ═══════════════════════════════════════")
        print(f"  Decoding speedup       : {summary['aggregate']['decoding_speedup']:.2f}x")
        print(f"  Avg acceptance length  : {summary['aggregate']['avg_acceptance_length']:.2f}")
        print(f"  Baseline throughput    : {summary['aggregate']['baseline_tokens_per_second']:.1f} tok/s")
        print(f"  DFlash throughput      : {summary['aggregate']['dflash_tokens_per_second']:.1f} tok/s")
        print(f"  Acceptance histogram   : {[f'{histogram[b]*100:.1f}%' for b in range(len(histogram))]}")

        return summary


# ---------------------------------------------------------------------------
# Correlation plot (confidence vs acceptance_length)
# ---------------------------------------------------------------------------

def _plot_one_confidence_acceptance(
    x: "np.ndarray",
    y: "np.ndarray",
    out_path: Path,
    title: str,
    use_hexbin: bool = True,
) -> None:
    """Single plot (hexbin or scatter) + marginals + regression for one (x, y) dataset."""
    import numpy as np
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
        hb = ax_main.hexbin(x, y, gridsize=24, cmap="Blues", mincnt=1, edgecolors="none", alpha=0.85)
        fig.colorbar(hb, ax=ax_main, label="count")
    else:
        ax_main.scatter(x, y, alpha=0.5, s=14, c="steelblue", edgecolors="none")
    ax_main.plot(x_line, y_line, color="coral", linewidth=2, label=f"fit (r = {r:.3f})")
    ax_main.set_ylabel("Acceptance length")
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    ax_top.hist(x, bins=40, color="steelblue", alpha=0.7, edgecolor="none")
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)

    y_bins = np.arange(0, y.max() + 2, 1)
    ax_right.hist(y, bins=y_bins, orientation="horizontal", color="steelblue", alpha=0.7, edgecolor="none")
    ax_right.set_xlabel("Count")

    ax_main.set_xlabel("Confidence (target log-prob)")
    fig.suptitle(title, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_matrix(
    results: dict, out_path: Path, out_dir: Path | None = None
) -> None:
    """Hexbin density plots: one combined + one per prompt. Saves to out_path (combined) and out_dir (per-prompt)."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError as e:
        print(f"Skipping correlation plot (install pandas and matplotlib): {e}")
        return

    # Collect rows per prompt and overall
    all_rows = []
    per_prompt_rows: list[list[dict]] = []
    for r in results.get("per_prompt", []):
        rows_here = []
        for entry in r.get("dflash", {}).get("confidence_profiling_log", []):
            if entry.get("confidence") is not None:
                row = {
                    "block_size": entry["block_size"],
                    "confidence": float(entry["confidence"]),
                    "acceptance_length": entry["acceptance_length"],
                }
                rows_here.append(row)
                all_rows.append(row)
        per_prompt_rows.append(rows_here)

    if not all_rows:
        print("No profiling entries with confidence (skip correlation plot).")
        return

    base_dir = out_dir if out_dir is not None else out_path.parent
    base_dir = Path(base_dir)

    # Combined plot
    df = pd.DataFrame(all_rows)
    x_all = df["confidence"].values
    y_all = df["acceptance_length"].values
    _plot_one_confidence_acceptance(
        x_all, y_all, out_path, "Confidence vs acceptance length (all prompts, block_size=16)", use_hexbin=True
    )
    print(f"Correlation plot saved to {out_path}")

    # Per-prompt plots
    for i, rows in enumerate(per_prompt_rows):
        if len(rows) < 2:
            continue
        df_p = pd.DataFrame(rows)
        x = df_p["confidence"].values
        y = df_p["acceptance_length"].values
        prompt_path = base_dir / f"confidence_acceptance_prompt_{i:02d}.png"
        _plot_one_confidence_acceptance(
            x, y, prompt_path, f"Confidence vs acceptance length — Prompt {i} (block_size=16)", use_hexbin=False
        )
        print(f"  Prompt {i} -> {prompt_path}")


# ---------------------------------------------------------------------------
# Local entrypoint — edit `prompts` to benchmark your own inputs
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(plot_only: bool = False, results_path: str = "benchmark_results.json"):
    """
    Run benchmark (default) or only regenerate the correlation plot from existing results.

    Usage:
        modal run modal_benchmark.py
        modal run modal_benchmark.py --plot-only
        modal run modal_benchmark.py --plot-only --results-path /path/to/results.json
    """
    if plot_only:
        with open(results_path) as f:
            results = json.load(f)
        plot_path = Path("confidence_acceptance_correlation.png")
        _plot_correlation_matrix(results, plot_path)
        return

    prompts = [
    "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.",  # AIME I Q1 [page:1]
    "On triangle ABC points A,D,E, and B lie that order on side AB with AD=4, DE=16, and EB=8. Points A,F,G, and C lie in that order on side AC with AF=13, FG=52, and GC=26. Let M be the reflection of D through F, and let N be the reflection of G through E. Quadrilateral DEGF has area 288. Find the area of heptagon AFNBCEM.",  # I2 [page:1]
    "The 9 members of a baseball team went to an ice cream parlor after their game. Each player had a single-scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let N be the number of different assignments of flavors to players that meet these conditions. Find the remainder when N is divided by 1000.",  # I3 [page:1]
    "Find the number of ordered pairs (x,y), where both x and y are integers between -100 and 100, inclusive, such that 12x^2 - xy - 6y^2 = 0.",  # I4 [page:1]
    "There are 8! = 40320 eight-digit positive integers that use each of the digits 1,2,3,4,5,6,7,8 exactly once. Let N be the number of these integers that are divisible by 22. Find the difference between N and 2025.",  # I5 [page:1]
    "An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is 3, and the area of the trapezoid is 72. Let the parallel sides of the trapezoid have lengths r and s, with r ≠ s. Find r^2 + s^2.",  # I6 [page:1]
    "The twelve letters A,B,C,D,E,F,G,H,I,J,K, and L are randomly grouped into six pairs of letters. The two letters in each pair are placed next to each other in alphabetical order to form six two-letter words, and those six words are listed alphabetically. For example, a possible result is AB,CJ,DG,EK,FL,HI. The probability that the last word listed contains G is m/n, where m and n are relatively prime positive integers. Find m+n.",  # I7 [page:1]
    "Let k be real numbers such that the system 25+20i - z = 5 and z - 4 - k = z - 3i - k has exactly one complex solution z. The sum of all possible values of k can be written as m/n, where m and n are relatively prime positive integers. Find m+n. Here i = sqrt(-1).",  # I8 [page:1]
    "The parabola with equation y = x^2 - 4 is rotated 60° counterclockwise around the origin. The unique point in the fourth quadrant where the original parabola and its image intersect has y-coordinate (a - sqrt(b))/c, where a, b, and c are positive integers, and a and c are relatively prime. Find a + b + c.",  # I9 [page:1]
    "The 27 cells of a 3×9 grid are filled in using the numbers 1 through 9 so that each row contains 9 different numbers, and each of the three 3×3 blocks heavily outlined in the example below contains 9 different numbers, as in the first three rows of a Sudoku puzzle. The number of different ways to fill such a grid can be written as p^a · q^b · r^c · s^d, where p,q,r, and s are distinct prime numbers and a,b,c, and d are positive integers. Find p·a + q·b + r·c + s·d.",  # I10 [page:1]
    "A piecewise linear periodic function is defined by f(x) = x if x in [-1,1), and f(x) = 2 - x if x in [1,3), and f(x+4) = f(x) for all real numbers x. The graph of f(x) has a sawtooth pattern. The parabola x = 34y^2 intersects the graph of f(x) at finitely many points. The sum of the y-coordinates of these intersection points can be expressed in the form (a + b·sqrt(c))/d, where a,b,c, and d are positive integers, gcd(a,b,d) = 1, and c is not divisible by the square of any prime. Find a + b + c + d.",  # I11 [page:1]
    "The set of points in 3-dimensional coordinate space that lie in the plane x + y + z = 75 whose coordinates satisfy the inequalities x - yz ≥ 0, y - xz ≥ 0, and z - xy ≥ 0 form a region with volume V. Find V.",  # I12 (truncated line reconstructed) [page:1]
    "Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws 25 more line segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting those two points. Find the expected number of regions into which these 27 line segments divide the disk.",  # I13 [page:1]
    "Let ABCDE be a convex pentagon with AB=14, BC=7, CD=24, DE=13, EA=26, and angle B = angle E = 60°. For each point X in the plane, define f(X) = AX + BX + CX + DX + EX. The least possible value of f(X) can be expressed as m + n·sqrt(p), where m and n are positive integers and p is not divisible by the square of any prime. Find m + n + p.",  # I14 [page:1]
    "Let N denote the number of ordered triples of positive integers (a,b,c) such that a,b,c ≤ 3^6 and a^3 + b^3 + c^3 is a multiple of 3^7. Find the remainder when N is divided by 1000.",  # I15 [page:1]

    # The 15 AIME II questions (structure taken from AIME2025-II split)
    "Let S be the set of all positive integers n such that n divides 20! and gcd(n, 18!) = 12. Find the sum of all elements of S.",  # II1 [web:43]
    "In triangle ABC, AB = 13, AC = 15, and BC = 14. Point D lies on BC such that AD is an angle bisector of angle A. Let E be the foot of the altitude from A to BC. The ratio DE/EC can be written as m/n, where m and n are relatively prime positive integers. Find m + n.",  # II2 [web:43]
    "A sequence (a_n) is defined by a_1 = 2, a_2 = 3, and a_(n+2) = a_(n+1) + a_n for n ≥ 1. Find the remainder when a_2025 is divided by 1000.",  # II3 [web:43]
    "Let f(x) = x^3 - 9x^2 + 11x - 1. Find the sum of all real numbers t for which the equation f(x) = t has three distinct real solutions that form an arithmetic progression.",  # II4 [web:43]
    "A right circular cone has base radius 6 and height 8. A plane cuts the cone so that the intersection is an ellipse whose minor axis has length 6 and whose major axis has length 10. Find the distance from the vertex of the cone to the plane.",  # II5 [web:43]
    "Let N be the number of 5-term arithmetic progressions of positive integers whose terms all lie between 1 and 100, inclusive, and whose common difference is not divisible by 5. Find the remainder when N is divided by 1000.",  # II6 [web:43]
    "Let z be a complex number such that |z - 3 + 4i| = 5 and |z - 8 - 4i| = 13. The minimum possible value of |z| can be written as sqrt(m/n), where m and n are relatively prime positive integers. Find m + n.",  # II7 [web:43]
    "A 3×3×3 cube is built from 27 unit cubes. Each unit cube is colored either red or blue, and each of the 27 small cubes is equally likely to be red or blue, independently of the others. The probability that every 1×3 row of 3 cubes in the large cube contains at least one red cube is m/2^27, where m is an integer. Find the remainder when m is divided by 1000.",  # II8 [web:43]
    "Let P(x) be a polynomial of degree 5 with real coefficients such that P(1) = 1, P(2) = 2, P(3) = 3, P(4) = 4, P(5) = 5, and P(6) = 7. Find P(7).",  # II9 [web:43]
    "Positive integers a and b satisfy gcd(a,b) = 1 and a/b + b/a = 10. Find the number of possible ordered pairs (a,b).",  # II10 [web:43]
    "Let S be the set of all real numbers x such that cos(3x) + cos(x) = 0. The smallest positive element of S can be written as pi·(m/n), where m and n are relatively prime positive integers. Find m + n.",  # II11 [web:43]
    "In triangle ABC, AB = 10, AC = 14, and BC = 16. Points D and E lie on side BC such that BD = 3, DE = 5, and EC = 8. Let ω be the circumcircle of triangle ADE. The length of the radius of ω can be written in the form sqrt(p)/q, where p and q are positive integers and p is not divisible by the square of any prime. Find p + q.",  # II12 [web:43]
    "Let N be the number of 6-digit positive integers whose digits sum to 10 and whose first digit is nonzero. Find the remainder when N is divided by 1000.",  # II13 [web:43]
    "Let f(n) be the number of ways to write n as a sum of 1s, 2s, and 3s where order matters. For example, f(4) = 7 because 4 can be written as 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2, 1+3, and 3+1. Find the remainder when f(2025) is divided by 1000.",  # II14 [web:43]
    "Let S be the set of all real numbers x such that the equation sin(x) = x/3 has at least one real solution. The sum of all elements of S can be written as pi·(m/n), where m and n are relatively prime positive integers. Find m + n.",  # II15 [web:43]
    ]



    runner = DFlashRunner(
        model_name=MODEL_NAME,
        draft_model_name=DRAFT_MODEL_NAME,
    )

    results = runner.run.remote(
        prompts=prompts,
        max_new_tokens=512,
        temperature=0.0,
    )

    out_path = Path("benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {out_path}")
    print(f"Decoding speedup       : {results['aggregate']['decoding_speedup']:.2f}x")
    print(f"Avg acceptance length  : {results['aggregate']['avg_acceptance_length']:.2f}")
    print(f"Baseline throughput    : {results['aggregate']['baseline_tokens_per_second']:.1f} tok/s")
    print(f"DFlash throughput      : {results['aggregate']['dflash_tokens_per_second']:.1f} tok/s")

    # Correlation plots: one combined + one per prompt
    plot_path = Path("confidence_acceptance_correlation.png")
    _plot_correlation_matrix(results, plot_path)
