"""
Pipeline orchestrator for model understanding data generation.

Runs the full pipeline end-to-end with resume support:
- Stage 1: Generate completions (single + multi-turn, via vLLM server)
- Stage 2: Screen completions for interesting behaviors (Anthropic API)
- Stage 3: Deep investigation with counterfactual experiments (Anthropic API + vLLM)
- Stage 4: Verification judge scoring (Anthropic API)
- Stage 5: Synthetic data generation (Anthropic API)

Usage:
    source .env
    .venv/bin/python data_pipelines/model_understanding/run_pipeline.py \
        --config data_pipelines/model_understanding/runs/test_mixed/config.json
"""

import argparse
import asyncio
import functools
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

from data_pipelines.pipeline_utils import load_dotenv

load_dotenv()

import anthropic
from openai import AsyncOpenAI

# Import stage modules
import data_pipelines.model_understanding.screen_completions as screen_completions
import data_pipelines.model_understanding.investigate as investigate_mod
import data_pipelines.model_understanding.verify_investigations as verify_investigations
import data_pipelines.model_understanding.generate_synthetic_data as generate_synthetic_data

# ---------------------------------------------------------------------------
# Pipeline defaults — all hyperparameters in one place.
#
# Per-run config.json overrides any of these. Keys in config.json that match
# a key here take precedence; anything not in config.json uses the default.
# ---------------------------------------------------------------------------

DEFAULTS = {
    # --- Global ---
    "async_concurrency": 40,           # concurrent API calls for all async stages
    "batch_chunk_size": 10000,          # requests per batch chunk (batch mode)
    "vllm_max_tokens": 500,            # max response tokens per completion (stages 1 + 3)
    "vllm_n_completions": 10,          # completions per prompt (stages 1 + 3)
    "vllm_temperature": 1.0,           # sampling temperature (stages 1 + 3)

    # --- Stage 1: Completion generation ---
    # "slurm": submit GPU jobs (production). "vllm_server": use a running
    # vLLM HTTP server for fast dev iteration when one is already up.
    "stage1_mode": "slurm",
    "stage1_n_prompts": 100,           # prompts to generate completions for
    "stage1_n_gpus": 1,                # GPUs for Slurm generation (slurm mode)
    "stage1_min_chars": 500,           # min total chars (all turns combined)
    "stage1_max_chars": 5000,          # max total chars (all turns combined)
    "stage1_max_turns": 5,             # max conversation turns to include
    "stage1_max_scan": 50000,          # max WildChat rows to scan

    # --- Stages 2 + 4: Screening & verification (LLM judge w/ tool call) ---
    "judge_model": "claude-opus-4-6",  # model for screening + verification
    "judge_max_tokens": 8192,          # max response tokens
    "judge_thinking_budget": 4096,     # extended thinking budget
    "stage2_n": 100,                   # prompts to screen (None = all)
    "stage2_mode": "async",            # "async" or "batch"
    "stage4_mode": "async",            # "async" or "batch"
    "stage4_batch_chunk_size": 500,    # verification prompts are large (~330KB each)

    # --- Stage 3: Investigation (Opus agent with tool use) ---
    "stage3_min_score": 3,             # min screening score to investigate
    "stage3_model": "claude-opus-4-6", # investigation model
    "stage3_max_tokens": 16384,        # max response tokens for investigator
    "stage3_thinking_budget": 10000,   # extended thinking budget for investigator
    "stage3_max_agent_turns": 30,      # max agent turns per investigation

    # --- Stage 5: Synthetic data generation ---
    "stage5_mode": "async",                               # "async" or "batch"
    "stage5_model": "claude-sonnet-4-5-20250929",         # Sonnet for synthesis
    "stage5_min_interest_score": 3,
    "stage5_min_verification_score": 7,
    "stage5_n_counterfactual_experiments": 3,
    "stage5_seed": 42,
}


def load_config(config_path: Path) -> dict:
    """Load per-run config, merged with defaults."""
    user_config = json.loads(config_path.read_text())
    merged = dict(DEFAULTS)
    merged.update(user_config)
    return merged


def update_status(run_dir: Path, status: dict, stage: str, data: dict):
    """Update a stage's status and write to disk."""
    status[stage] = data
    status["last_updated"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(f"  Status: {stage} = {data.get('status', '?')}")


# ---------------------------------------------------------------------------
# Stage 1: Generate completions (single + multi-turn, via vLLM server)
# ---------------------------------------------------------------------------

async def run_stage1_vllm_server(config: dict, run_dir: Path, status: dict) -> Path:
    """Generate completions via a running vLLM HTTP server (fast dev mode)."""
    output_file = run_dir / "completions.json"

    if status.get("stage1", {}).get("status") == "done" and output_file.exists():
        print("Stage 1 already done, skipping")
        return output_file

    update_status(run_dir, status, "stage1", {"status": "running"})

    # Stream and collect prompts
    from data_pipelines.model_understanding.slurm_scripts.prepare_shards import load_exclude_hashes, stream_prompts

    # Load exclusion sets if configured
    exclude_conv_hashes = None
    exclude_content_hashes = None
    if config.get("exclude_files"):
        exclude_conv_hashes, exclude_content_hashes = load_exclude_hashes(
            config["exclude_files"],
        )

    print(f"  Streaming WildChat...")
    all_candidates = list(stream_prompts(
        min_chars=config["stage1_min_chars"],
        max_chars=config["stage1_max_chars"],
        max_turns=config["stage1_max_turns"],
        max_scan=config["stage1_max_scan"],
        exclude_conv_hashes=exclude_conv_hashes,
        exclude_content_hashes=exclude_content_hashes,
    ))
    print(f"  Found {len(all_candidates)} candidates")

    import random
    random.seed(42)
    n_sample = min(config["stage1_n_prompts"], len(all_candidates))
    selected = random.sample(all_candidates, n_sample)
    for idx, s in enumerate(selected):
        s["id"] = f"prompt_{idx:05d}"

    # Generate completions via HTTP API
    vllm_job_name = config["vllm_job_name"]
    vllm_url = ensure_vllm_server(job_name=vllm_job_name)
    vllm_client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="unused")
    models = await vllm_client.models.list()
    model_name = models.data[0].id
    print(f"  vLLM model: {model_name}")

    results = []
    progress = {"done": 0}

    async def generate_one(prompt):
        response = await vllm_client.chat.completions.create(
            model=model_name,
            messages=prompt["messages"],
            n=config["vllm_n_completions"],
            temperature=config["vllm_temperature"],
            max_tokens=config["vllm_max_tokens"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        completions = [c.message.content for c in response.choices
                       if c.message.content and c.message.content.strip()]
        if completions:
            results.append({
                "id": prompt["id"],
                "user_message": prompt["user_message"],
                "messages": prompt["messages"],
                "n_turns": prompt["n_turns"],
                "total_chars": prompt["total_chars"],
                "conversation_hash": prompt.get("conversation_hash", ""),
                "completions": completions,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "response_token_counts": [len(c.split()) for c in completions],
            })
        progress["done"] += 1
        if progress["done"] % 10 == 0 or progress["done"] == len(selected):
            print(f"  Stage 1 progress: {progress['done']}/{len(selected)}")

    await asyncio.gather(*[generate_one(p) for p in selected])

    # Save
    from collections import Counter
    turn_dist = Counter(r["n_turns"] for r in results)
    output = {
        "metadata": {
            "model": model_name,
            "dataset": "allenai/WildChat-1M",
            "n_prompts": len(results),
            "n_completions_per_prompt": config["vllm_n_completions"],
            "temperature": config["vllm_temperature"],
            "max_response_tokens": config["vllm_max_tokens"],
            "min_chars": config["stage1_min_chars"],
            "max_chars": config["stage1_max_chars"],
            "max_turns": config["stage1_max_turns"],
            "turn_distribution": dict(turn_dist),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "prompts": results,
    }
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Saved {len(results)} prompts to {output_file}")
    for k in sorted(turn_dist):
        print(f"    {k}-turn: {turn_dist[k]}")

    update_status(run_dir, status, "stage1", {
        "status": "done",
        "n_prompts": len(results),
        "turn_distribution": dict(turn_dist),
    })
    return output_file


def run_stage1(config: dict, run_dir: Path, status: dict) -> Path:
    """Generate completions via Slurm GPU jobs.

    Stage 0: Prepare shards (stream WildChat, filter, tokenize, split)
    Stage 1: Submit Slurm jobs (one per GPU), wait, combine results.

    Returns the path to the combined completions file.
    """
    output_file = run_dir / "completions.json"

    if status.get("stage1", {}).get("status") == "done" and output_file.exists():
        print("Stage 1 already done, skipping")
        return output_file

    n_gpus = config["stage1_n_gpus"]
    n_prompts = config["stage1_n_prompts"]
    model = config["model"]
    quantization = config.get("quantization")
    shards_dir = run_dir / "shards"

    # --- Stage 0: Prepare shards ---
    if not (shards_dir / "shard_0.json").exists():
        update_status(run_dir, status, "stage1", {"status": "preparing_shards"})
        print(f"  Preparing {n_gpus} shards ({n_prompts} prompts)...")

        prepare_cmd = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(SCRIPT_DIR / "slurm_scripts" / "prepare_shards.py"),
            "--model", model,
            "--n-prompts", str(n_prompts),
            "--n-shards", str(n_gpus),
            "--output-dir", str(shards_dir),
            "--min-chars", str(config["stage1_min_chars"]),
            "--max-chars", str(config["stage1_max_chars"]),
            "--max-turns", str(config["stage1_max_turns"]),
            "--max-scan", str(config["stage1_max_scan"]),
        ]
        if config.get("exclude_files"):
            prepare_cmd.extend(["--exclude"] + config["exclude_files"])
        result = subprocess.run(prepare_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Shard preparation failed")
    else:
        print(f"  Shards already prepared in {shards_dir}")

    # --- Stage 1: Submit Slurm jobs ---
    # Use enforce_eager for small runs (faster startup)
    enforce_eager = n_prompts <= 1000

    part_files = []
    job_ids = []

    # Check for already-completed parts
    for shard_idx in range(n_gpus):
        part_file = run_dir / f"completions_part{shard_idx}.json"
        part_files.append(part_file)

    remaining_shards = [
        i for i in range(n_gpus) if not part_files[i].exists()
    ]

    if remaining_shards:
        update_status(run_dir, status, "stage1", {
            "status": "generating",
            "n_gpus": n_gpus,
            "remaining_shards": remaining_shards,
        })

        for shard_idx in remaining_shards:
            shard_file = shards_dir / f"shard_{shard_idx}.json"
            part_file = part_files[shard_idx]

            wrap_cmd = (
                f"cd {REPO_ROOT} && "
                f".venv/bin/python data_pipelines/model_understanding/generate_completions.py "
                f"--model {model} "
                f"--n-completions {config['vllm_n_completions']} "
                f"--max-tokens {config['vllm_max_tokens']} "
                f"--input-shard {shard_file} "
                f"--output {part_file}"
                + (f" --quantization {quantization}" if quantization else "")
            )

            sbatch_args = [
                "sbatch",
                "--gres=gpu:1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=4:00:00",
                "--partition=general",
                "--qos=high",
                f"--job-name=pipeline_gen_{shard_idx}",
                f"--output={run_dir}/stage1_shard{shard_idx}.log",
                "--wrap", wrap_cmd,
            ]

            result = subprocess.run(
                sbatch_args, capture_output=True, text=True, check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"  Submitted shard {shard_idx}: job {job_id}")

        # Poll until all jobs finish
        print(f"  Waiting for {len(job_ids)} Slurm jobs...")
        while job_ids:
            time.sleep(30)
            still_running = []
            for jid in job_ids:
                sq = subprocess.run(
                    ["squeue", "-j", jid, "--noheader"],
                    capture_output=True, text=True,
                )
                if sq.stdout.strip():
                    still_running.append(jid)
            if still_running:
                print(f"  {len(still_running)}/{len(job_ids)} jobs still running...")
                job_ids = still_running
            else:
                break

        # Verify all parts exist
        for shard_idx in remaining_shards:
            assert part_files[shard_idx].exists(), (
                f"Part file missing: {part_files[shard_idx]}. "
                f"Check {run_dir}/stage1_shard{shard_idx}.log"
            )
    else:
        print(f"  All {n_gpus} part files already exist")

    # --- Combine ---
    if not output_file.exists():
        print(f"  Combining {n_gpus} part files...")
        combine_cmd = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(SCRIPT_DIR / "slurm_scripts" / "combine_completions.py"),
            "--inputs", *[str(f) for f in part_files],
            "--output", str(output_file),
        ]
        result = subprocess.run(combine_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Combine failed")

    update_status(run_dir, status, "stage1", {
        "status": "done",
        "n_gpus": n_gpus,
        "n_prompts": n_prompts,
    })

    return output_file


# ---------------------------------------------------------------------------
# Stage 2: Screen completions
# ---------------------------------------------------------------------------

async def run_stage2(
    config: dict, run_dir: Path, status: dict, completions_path: Path,
):
    """Screen completions for interesting behaviors."""
    if status.get("stage2", {}).get("status") == "done":
        print("Stage 2 already done, skipping")
        return

    update_status(run_dir, status, "stage2", {"status": "running"})

    prompts = screen_completions.load_prompts(
        completions_path, n=config["stage2_n"],
    )

    screening_path = run_dir / "screening.json"
    mode = config["stage2_mode"]

    judge_args = dict(
        model=config["judge_model"],
        max_tokens=config["judge_max_tokens"],
        thinking_budget=config["judge_thinking_budget"],
    )

    if mode == "batch":
        results = screen_completions.screen_batch(
            prompts, screening_path,
            batch_chunk_size=config["batch_chunk_size"],
            **judge_args,
        )
    elif mode == "async":
        results = await screen_completions.screen_async(
            prompts, screening_path,
            concurrency=config["async_concurrency"],
            **judge_args,
        )
    else:
        raise ValueError(f"Unknown stage2_mode: {mode!r}")

    screen_completions.save_results(
        results, screening_path, completions_path,
        model=config["judge_model"],
    )

    score_dist = {}
    for s in range(1, 6):
        count = sum(1 for r in results if r.interest_score == s)
        if count > 0:
            score_dist[str(s)] = count

    update_status(run_dir, status, "stage2", {
        "status": "done",
        "total": len(prompts),
        "completed": len(results),
        "score_distribution": score_dist,
    })


# ---------------------------------------------------------------------------
# Stage 3: Investigation
# ---------------------------------------------------------------------------

def ensure_vllm_server(job_name: str) -> str:
    """Find a running vLLM server by Slurm job name. Fails if not found."""
    url = investigate_mod.discover_vllm_url(job_name=job_name)
    return url


async def run_stage3(config: dict, run_dir: Path, status: dict):
    """Deep investigation of interesting behaviors."""
    if status.get("stage3", {}).get("status") == "done":
        print("Stage 3 already done, skipping")
        return

    # Load screening results
    screening_path = run_dir / "screening.json"
    screening_data = json.loads(screening_path.read_text())
    min_score = config["stage3_min_score"]
    candidates = [
        r for r in screening_data["results"]
        if r["interest_score"] >= min_score
    ]
    candidates.sort(key=lambda r: r["interest_score"], reverse=True)

    print(f"  {len(candidates)} prompts with score >= {min_score}")

    if not candidates:
        print("  No prompts to investigate!")
        update_status(run_dir, status, "stage3", {
            "status": "done", "total": 0, "completed": 0,
        })
        return

    # Find or start vLLM server
    vllm_job_name = config["vllm_job_name"]
    vllm_url = ensure_vllm_server(job_name=vllm_job_name)
    vllm_client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="unused")

    # Auto-detect model name
    models = await vllm_client.models.list()
    model_name = models.data[0].id
    print(f"  vLLM model: {model_name}")

    # Quick health check
    test = await investigate_mod.run_vllm_prompt(
        vllm_client, model_name, user_message="Say hello.", n=1, max_tokens=20,
    )
    print(f"  vLLM OK: {test[0][:80]}")

    # Set up Anthropic client
    anthropic_client = anthropic.AsyncAnthropic()

    # Resume from jsonl
    jsonl_path = run_dir / "investigations.jsonl"
    completed_ids = set()
    existing_results = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["prompt_id"])
                    existing_results.append(record)
        print(f"  Resuming: {len(completed_ids)} already investigated")

    remaining = [r for r in candidates if r["prompt_id"] not in completed_ids]
    all_results = list(existing_results)
    skipped = {"count": 0}

    if not remaining:
        print("  All eligible prompts already investigated")
    else:
        update_status(run_dir, status, "stage3", {
            "status": "running",
            "total": len(candidates),
            "completed": len(completed_ids),
            "vllm_url": vllm_url,
        })

        write_lock = asyncio.Lock()
        jsonl_file = open(jsonl_path, "a")
        semaphore = asyncio.Semaphore(config["async_concurrency"])
        investigate_mod.token_rate_tracker.start()
        progress = {"done": len(completed_ids)}

        async def investigate_with_semaphore(screening_result: dict):
            async with semaphore:
                pid = screening_result["prompt_id"]
                try:
                    investigation = await investigate_mod.investigate_one(
                        screening_result=screening_result,
                        anthropic_client=anthropic_client,
                        vllm_client=vllm_client,
                        model_name=model_name,
                        output_dir=run_dir,
                        investigation_model=config["stage3_model"],
                        max_tokens=config["stage3_max_tokens"],
                        thinking_budget=config["stage3_thinking_budget"],
                        max_agent_turns=config["stage3_max_agent_turns"],
                        vllm_n_completions=config["vllm_n_completions"],
                        vllm_temperature=config["vllm_temperature"],
                        vllm_max_tokens=config["vllm_max_tokens"],
                    )
                    record = asdict(investigation)

                    async with write_lock:
                        all_results.append(record)
                        jsonl_file.write(json.dumps(record) + "\n")
                        jsonl_file.flush()
                except Exception as e:
                    skipped["count"] += 1
                    print(f"\n  SKIPPED {pid}: {e}")

                progress["done"] += 1
                print(f"\n  Investigation progress: "
                      f"{progress['done']}/{len(candidates)} "
                      f"(skipped: {skipped['count']})")

        await asyncio.gather(*[
            investigate_with_semaphore(r) for r in remaining
        ])

        if remaining:
            skip_rate = skipped["count"] / len(candidates)
            assert skip_rate < 0.001, (
                f"Investigation skip rate {skip_rate:.1%} "
                f"({skipped['count']}/{len(candidates)}) exceeds 0.1% threshold"
            )

        investigate_mod.token_rate_tracker.stop()
        jsonl_file.close()

    # Save final JSON
    output_path = run_dir / "investigations.json"
    output = {
        "metadata": {
            "investigation_model": config["stage3_model"],
            "input_file": str(screening_path),
            "vllm_url": vllm_url,
            "vllm_model": model_name,
            "n_investigated": len(all_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved {len(all_results)} investigations to {output_path}")

    update_status(run_dir, status, "stage3", {
        "status": "done",
        "total": len(candidates),
        "completed": len(all_results),
        "skipped": skipped["count"],
    })


# ---------------------------------------------------------------------------
# Stage 4: Verification
# ---------------------------------------------------------------------------

async def run_stage4(config: dict, run_dir: Path, status: dict):
    """Verification judge scoring."""
    if status.get("stage4", {}).get("status") == "done":
        print("Stage 4 already done, skipping")
        return

    investigations_path = run_dir / "investigations.json"
    investigations = verify_investigations.load_investigations(
        investigations_path, n=None,
    )

    if not investigations:
        print("  No investigations to verify!")
        update_status(run_dir, status, "stage4", {
            "status": "done", "total": 0, "completed": 0,
        })
        return

    update_status(run_dir, status, "stage4", {
        "status": "running",
        "total": len(investigations),
    })

    verification_path = run_dir / "verification.json"
    mode = config["stage4_mode"]

    judge_args = dict(
        model=config["judge_model"],
        max_tokens=config["judge_max_tokens"],
    )

    if mode == "batch":
        results = verify_investigations.verify_batch(
            investigations, verification_path,
            batch_chunk_size=config["stage4_batch_chunk_size"],
            **judge_args,
        )
    elif mode == "async":
        results = await verify_investigations.verify_async(
            investigations, verification_path,
            thinking_budget=config["judge_thinking_budget"],
            concurrency=config["async_concurrency"],
            **judge_args,
        )
    else:
        raise ValueError(f"Unknown stage4_mode: {mode!r}")

    verify_investigations.save_results(
        results, verification_path, investigations_path,
        model=config["judge_model"],
    )

    score_dist = {}
    for s in range(1, 11):
        count = sum(1 for r in results if r.score == s)
        if count > 0:
            score_dist[str(s)] = count

    update_status(run_dir, status, "stage4", {
        "status": "done",
        "total": len(investigations),
        "completed": len(results),
        "score_distribution": score_dist,
        "mean_score": round(
            sum(r.score for r in results) / len(results), 2,
        ) if results else 0,
    })


# ---------------------------------------------------------------------------
# Stage 5: Synthetic data generation
# ---------------------------------------------------------------------------

async def run_stage5(config: dict, run_dir: Path, status: dict):
    """Generate synthetic training data from pipeline outputs."""
    if status.get("stage5", {}).get("status") == "done":
        print("Stage 5 already done, skipping")
        return

    screening_path = run_dir / "screening.json"
    investigations_path = run_dir / "investigations.json"
    verification_path = run_dir / "verification.json"
    output_path = run_dir / "synthetic_data.json"

    for p in [screening_path, investigations_path, verification_path]:
        assert p.exists(), f"Required file missing: {p}"

    update_status(run_dir, status, "stage5", {"status": "running"})

    model = config["stage5_model"]
    min_interest = config["stage5_min_interest_score"]
    min_verification = config["stage5_min_verification_score"]
    n_cf_experiments = config["stage5_n_counterfactual_experiments"]
    seed = config["stage5_seed"]

    examples = generate_synthetic_data.load_joined_data(
        screening_path, investigations_path, verification_path,
        min_interest_score=min_interest,
        min_verification_score=min_verification,
    )

    requests = generate_synthetic_data.prepare_requests(
        examples,
        n_counterfactual_experiments=n_cf_experiments,
        seed=seed,
        model=model,
    )

    mode = config["stage5_mode"]
    if mode == "async":
        results = await generate_synthetic_data.generate_async(
            requests, output_path,
            concurrency=config["async_concurrency"],
        )
    elif mode == "batch":
        results = generate_synthetic_data.generate_batch(
            requests, output_path,
            batch_chunk_size=config["batch_chunk_size"],
        )
    else:
        raise ValueError(f"Unknown stage5_mode: {mode!r}")

    # Build args namespace for save_results
    args = argparse.Namespace(
        model=model,
        investigations=investigations_path,
        screening=screening_path,
        verification=verification_path,
        min_interest_score=min_interest,
        min_verification_score=min_verification,
        n_counterfactual_experiments=n_cf_experiments,
        seed=seed,
    )
    generate_synthetic_data.save_results(results, output_path, args)

    bp_count = sum(1 for r in results
                   if r.example_type == "behavior_prediction")
    cf_count = sum(1 for r in results
                   if r.example_type == "counterfactual_prediction")

    update_status(run_dir, status, "stage5", {
        "status": "done",
        "n_examples": len(results),
        "n_behavior_predictions": bp_count,
        "n_counterfactual_predictions": cf_count,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Run model understanding pipeline (stages 1-5)",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to config JSON (run directory = config file's parent dir)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.config.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load or create status
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    print(f"Pipeline: {config['run_name']}")
    print(f"Run dir:  {run_dir}")
    print(f"Config:\n{json.dumps(config, indent=2)}")

    t0 = time.time()

    # Stage 1: Generate completions (or use pre-existing file)
    if "completions_file" in config:
        raw = config["completions_file"]
        completions_path = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
        assert completions_path.exists(), (
            f"Completions file not found: {completions_path}"
        )
        print(f"\nUsing existing completions: {completions_path}")
    else:
        print(f"\n{'='*60}")
        print("STAGE 1: Generate completions")
        print(f"{'='*60}")
        t1 = time.time()
        if config["stage1_mode"] == "vllm_server":
            completions_path = await run_stage1_vllm_server(config, run_dir, status)
        elif config["stage1_mode"] == "slurm":
            completions_path = run_stage1(config, run_dir, status)
        else:
            raise ValueError(f"Unknown stage1_mode: {config['stage1_mode']!r}")
        print(f"  Stage 1 took {time.time() - t1:.0f}s")

    # Stage 2: Screening
    print(f"\n{'='*60}")
    print("STAGE 2: Screening")
    print(f"{'='*60}")
    t2 = time.time()
    await run_stage2(config, run_dir, status, completions_path)
    print(f"  Stage 2 took {time.time() - t2:.0f}s")

    # Stage 3: Investigation
    print(f"\n{'='*60}")
    print("STAGE 3: Investigation")
    print(f"{'='*60}")
    t3 = time.time()
    await run_stage3(config, run_dir, status)
    print(f"  Stage 3 took {time.time() - t3:.0f}s")

    # Stage 4: Verification
    print(f"\n{'='*60}")
    print("STAGE 4: Verification")
    print(f"{'='*60}")
    t4 = time.time()
    await run_stage4(config, run_dir, status)
    print(f"  Stage 4 took {time.time() - t4:.0f}s")

    # Stage 5: Synthetic data generation
    print(f"\n{'='*60}")
    print("STAGE 5: Synthetic data generation")
    print(f"{'='*60}")
    t5 = time.time()
    await run_stage5(config, run_dir, status)
    print(f"  Stage 5 took {time.time() - t5:.0f}s")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
