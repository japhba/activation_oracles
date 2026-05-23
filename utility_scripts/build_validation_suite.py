from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExistingSyntheticQAJsonSpec:
    kind: str
    component_name: str
    source_json_path: str
    sample_num_examples: int
    sample_seed: int


@dataclass
class GeneratedSyntheticQASpec:
    kind: str
    component_name: str
    run_name: str
    n_prompts: int
    sample_num_examples: int
    sample_seed: int
    use_batch_api: bool


@dataclass
class HiddenBiasActiveSpec:
    kind: str
    component_name: str
    run_name: str
    num_examples: int
    sample_num_examples: int
    sample_seed: int


@dataclass
class HiddenBiasInactiveSpec:
    kind: str
    component_name: str
    run_name: str
    source_run_name: str
    sample_num_examples: int
    sample_seed: int


@dataclass
class PastLensSpec:
    kind: str
    component_name: str
    num_examples: int
    seed: int
    min_k_tokens: int
    max_k_tokens: int
    min_k_activations: int
    max_k_activations: int
    max_length: int
    directions: list[str]
    vllm_max_new_tokens: int
    max_vllm_context_tokens: int
    future_chat_system_prompt_prob: float
    system_prompt_path: str
    english_only_temp_filter: bool


ValidationComponentSpec = (
    ExistingSyntheticQAJsonSpec
    | GeneratedSyntheticQASpec
    | HiddenBiasActiveSpec
    | HiddenBiasInactiveSpec
    | PastLensSpec
)


@dataclass
class ValidationSuiteConfig:
    suite_name: str
    validation_steps: int
    validation_on_start: bool
    components: list[ValidationComponentSpec]


@dataclass
class BuiltValidationArtifact:
    kind: str
    component_name: str
    prebuilt_pt_path: str
    prebuilt_pt_relpath: str
    sampled_json_path: str | None
    sampled_json_relpath: str | None
    source_json_path: str | None
    source_run_name: str | None
    num_examples: int


REPO_ROOT = Path(__file__).resolve().parent.parent


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def run_python(python_bin: Path, args: list[str]) -> None:
    cmd = [str(python_bin)] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def sample_entries(entries: list[dict], sample_num_examples: int, sample_seed: int) -> list[dict]:
    assert len(entries) >= sample_num_examples, (
        f"Need {sample_num_examples} entries, found only {len(entries)}"
    )
    rng = random.Random(sample_seed)
    return rng.sample(entries, sample_num_examples)


def parse_component(raw: dict) -> ValidationComponentSpec:
    kind = raw["kind"]
    if kind == "existing_synthetic_qa_json":
        return ExistingSyntheticQAJsonSpec(
            kind=kind,
            component_name=raw["component_name"],
            source_json_path=raw["source_json_path"],
            sample_num_examples=raw["sample_num_examples"],
            sample_seed=raw["sample_seed"],
        )
    if kind == "generate_synthetic_qa":
        return GeneratedSyntheticQASpec(
            kind=kind,
            component_name=raw["component_name"],
            run_name=raw["run_name"],
            n_prompts=raw["n_prompts"],
            sample_num_examples=raw["sample_num_examples"],
            sample_seed=raw["sample_seed"],
            use_batch_api=raw["use_batch_api"],
        )
    if kind == "generate_hidden_bias_active":
        return HiddenBiasActiveSpec(
            kind=kind,
            component_name=raw["component_name"],
            run_name=raw["run_name"],
            num_examples=raw["num_examples"],
            sample_num_examples=raw["sample_num_examples"],
            sample_seed=raw["sample_seed"],
        )
    if kind == "generate_hidden_bias_inactive":
        return HiddenBiasInactiveSpec(
            kind=kind,
            component_name=raw["component_name"],
            run_name=raw["run_name"],
            source_run_name=raw["source_run_name"],
            sample_num_examples=raw["sample_num_examples"],
            sample_seed=raw["sample_seed"],
        )
    if kind == "past_lens":
        return PastLensSpec(
            kind=kind,
            component_name=raw["component_name"],
            num_examples=raw["num_examples"],
            seed=raw["seed"],
            min_k_tokens=raw["min_k_tokens"],
            max_k_tokens=raw["max_k_tokens"],
            min_k_activations=raw["min_k_activations"],
            max_k_activations=raw["max_k_activations"],
            max_length=raw["max_length"],
            directions=raw["directions"],
            vllm_max_new_tokens=raw["vllm_max_new_tokens"],
            max_vllm_context_tokens=raw["max_vllm_context_tokens"],
            future_chat_system_prompt_prob=raw["future_chat_system_prompt_prob"],
            system_prompt_path=raw["system_prompt_path"],
            english_only_temp_filter=raw["english_only_temp_filter"],
        )
    raise ValueError(f"Unknown validation component kind: {kind}")


def load_suite_config(path: Path) -> ValidationSuiteConfig:
    payload = json.loads(path.read_text())
    components = [parse_component(raw) for raw in payload["components"]]
    component_names = [component.component_name for component in components]
    assert len(component_names) == len(set(component_names)), "Validation component names must be unique"
    seen_hidden_bias_active_runs: set[str] = set()
    for component in components:
        if isinstance(component, HiddenBiasActiveSpec):
            seen_hidden_bias_active_runs.add(component.run_name)
        if isinstance(component, HiddenBiasInactiveSpec):
            source_stage3_path = (
                REPO_ROOT
                / "datasets"
                / "training_data"
                / "artifacts"
                / "system_prompt_qa"
                / component.source_run_name
                / "stage3_hidden_bias.json"
            )
            assert component.source_run_name in seen_hidden_bias_active_runs or source_stage3_path.exists(), (
                "Hidden-bias inactive components require a generated active source run. "
                "List the active builder earlier in the suite, or point source_run_name at an existing run."
            )
    return ValidationSuiteConfig(
        suite_name=payload["suite_name"],
        validation_steps=payload["validation_steps"],
        validation_on_start=payload["validation_on_start"],
        components=components,
    )


def ensure_synthetic_qa_dataset(
    python_bin: Path,
    model_name: str,
    spec: GeneratedSyntheticQASpec,
) -> Path:
    run_dir = REPO_ROOT / "datasets" / "training_data" / "artifacts" / spec.run_name
    dataset_path = run_dir / "training_data.json"
    manifest_path = run_dir / "manifest.json"
    merged_path = run_dir / "stage12" / "merged.json"

    if not manifest_path.exists():
        run_python(
            python_bin,
            [
                "data_pipelines/training_data/generate_training_data.py",
                "prep",
                "--run",
                spec.run_name,
                "--model",
                model_name,
                "--n-prompts",
                str(spec.n_prompts),
            ],
        )

    if not merged_path.exists():
        run_python(
            python_bin,
            [
                "data_pipelines/training_data/generate_training_data.py",
                "stage12",
                "--run",
                spec.run_name,
            ],
        )
        run_python(
            python_bin,
            [
                "data_pipelines/training_data/generate_training_data.py",
                "merge-stage12",
                "--run",
                spec.run_name,
                "--num-shards",
                "1",
            ],
        )

    if not dataset_path.exists():
        stage3_args = [
            "data_pipelines/training_data/generate_training_data.py",
            "stage3",
            "--run",
            spec.run_name,
        ]
        if spec.use_batch_api:
            stage3_args.append("--batch")
        run_python(python_bin, stage3_args)

    assert dataset_path.exists(), f"Missing generated synthetic QA dataset: {dataset_path}"
    return dataset_path


def ensure_hidden_bias_active_dataset(
    python_bin: Path,
    model_name: str,
    spec: HiddenBiasActiveSpec,
) -> Path:
    run_dir = REPO_ROOT / "data_pipelines" / "training_data" / "artifacts" / "system_prompt_qa" / spec.run_name
    dataset_path = run_dir / "training_data.json"
    stage3_path = run_dir / "stage3_hidden_bias.json"

    if dataset_path.exists():
        return dataset_path

    run_python(
        python_bin,
        [
            "data_pipelines/training_data/generate_hidden_bias_data.py",
            "stage1",
            "--run",
            spec.run_name,
            "--model",
            model_name,
            "--num-examples",
            str(spec.num_examples),
        ],
    )
    run_python(
        python_bin,
        [
            "data_pipelines/training_data/generate_hidden_bias_data.py",
            "stage2",
            "--run",
            spec.run_name,
            "--model",
            model_name,
        ],
    )
    if not stage3_path.exists():
        run_python(
            python_bin,
            [
                "data_pipelines/training_data/generate_hidden_bias_data.py",
                "stage3",
                "--run",
                spec.run_name,
                "--model",
                model_name,
            ],
        )
    if not dataset_path.exists():
        run_python(
            python_bin,
            [
                "data_pipelines/training_data/generate_hidden_bias_data.py",
                "stage4",
                "--run",
                spec.run_name,
                "--model",
                model_name,
            ],
        )

    assert dataset_path.exists(), f"Missing hidden bias dataset: {dataset_path}"
    return dataset_path


def ensure_hidden_bias_inactive_dataset(
    python_bin: Path,
    model_name: str,
    spec: HiddenBiasInactiveSpec,
) -> Path:
    run_dir = REPO_ROOT / "data_pipelines" / "training_data" / "artifacts" / "system_prompt_qa" / spec.run_name
    dataset_path = run_dir / "training_data.json"
    if dataset_path.exists():
        return dataset_path

    run_python(
        python_bin,
        [
            "data_pipelines/training_data/generate_hidden_bias_data.py",
            "cross-pair",
            "--run",
            spec.run_name,
            "--model",
            model_name,
            "--source-run",
            spec.source_run_name,
        ],
    )

    assert dataset_path.exists(), f"Missing hidden bias inactive dataset: {dataset_path}"
    return dataset_path


def build_sampled_synthetic_qa_artifact(
    *,
    training_cfg,
    component_name: str,
    source_json_path: Path,
    sample_num_examples: int,
    sample_seed: int,
    sampled_json_dir: Path,
    prebuilt_pt_dir: Path,
) -> BuiltValidationArtifact:
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
    from nl_probes.dataset_classes.synthetic_qa_dataset import SyntheticQADatasetConfig, SyntheticQADatasetLoader

    raw = json.loads(source_json_path.read_text())
    sampled_entries = sample_entries(raw["entries"], sample_num_examples, sample_seed)

    sampled_json_path = sampled_json_dir / f"{sanitize_filename(component_name)}.json"
    sampled_json_payload = {
        "metadata": {
            "source_json_path": repo_relative(source_json_path),
            "sample_num_examples": sample_num_examples,
            "sample_seed": sample_seed,
            "component_name": component_name,
        },
        "entries": sampled_entries,
    }
    sampled_json_path.write_text(json.dumps(sampled_json_payload, indent=2))

    loader_config = DatasetLoaderConfig(
        custom_dataset_params=SyntheticQADatasetConfig(data_path=repo_relative(sampled_json_path)),
        num_train=sample_num_examples,
        num_test=0,
        splits=["train"],
        model_name=training_cfg.model_name,
        layer_combinations=training_cfg.layer_combinations,
        save_acts=False,
        batch_size=min(training_cfg.train_batch_size, sample_num_examples),
        dataset_name="",
        dataset_folder=repo_relative(prebuilt_pt_dir),
        seed=sample_seed,
    )
    loader = SyntheticQADatasetLoader(loader_config)
    loader.ensure_dataset_exists("train")
    prebuilt_pt_path = prebuilt_pt_dir / loader.get_dataset_filename("train")

    return BuiltValidationArtifact(
        kind="synthetic_qa",
        component_name=component_name,
        prebuilt_pt_path=str(prebuilt_pt_path),
        prebuilt_pt_relpath=repo_relative(prebuilt_pt_path),
        sampled_json_path=str(sampled_json_path),
        sampled_json_relpath=repo_relative(sampled_json_path),
        source_json_path=str(source_json_path),
        source_run_name=None,
        num_examples=sample_num_examples,
    )


def build_past_lens_artifact(
    *,
    training_cfg,
    spec: PastLensSpec,
    prebuilt_pt_dir: Path,
) -> BuiltValidationArtifact:
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
    from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader

    loader_config = DatasetLoaderConfig(
        custom_dataset_params=PastLensDatasetConfig(
            min_k_tokens=spec.min_k_tokens,
            max_k_tokens=spec.max_k_tokens,
            min_k_activations=spec.min_k_activations,
            max_k_activations=spec.max_k_activations,
            max_length=spec.max_length,
            directions=spec.directions,
            vllm_max_new_tokens=spec.vllm_max_new_tokens,
            max_vllm_context_tokens=spec.max_vllm_context_tokens,
            future_chat_system_prompt_prob=spec.future_chat_system_prompt_prob,
            system_prompt_path=spec.system_prompt_path,
            english_only_temp_filter=spec.english_only_temp_filter,
        ),
        num_train=spec.num_examples,
        num_test=0,
        splits=["train"],
        model_name=training_cfg.model_name,
        layer_combinations=training_cfg.layer_combinations,
        save_acts=False,
        batch_size=min(training_cfg.train_batch_size, spec.num_examples),
        dataset_name="",
        dataset_folder=repo_relative(prebuilt_pt_dir),
        seed=spec.seed,
    )
    loader = PastLensDatasetLoader(loader_config)
    loader.ensure_dataset_exists("train")
    prebuilt_pt_path = prebuilt_pt_dir / loader.get_dataset_filename("train")

    return BuiltValidationArtifact(
        kind="past_lens",
        component_name=spec.component_name,
        prebuilt_pt_path=str(prebuilt_pt_path),
        prebuilt_pt_relpath=repo_relative(prebuilt_pt_path),
        sampled_json_path=None,
        sampled_json_relpath=None,
        source_json_path=None,
        source_run_name=None,
        num_examples=spec.num_examples,
    )


def build_prebuilt_validation_loader_config(training_cfg, artifact: BuiltValidationArtifact) -> dict:
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
    from nl_probes.dataset_classes.prebuilt_pt_dataset import PrebuiltPTDatasetConfig

    loader_config = DatasetLoaderConfig(
        custom_dataset_params=PrebuiltPTDatasetConfig(
            data_path=artifact.prebuilt_pt_relpath,
            component_name=artifact.component_name,
        ),
        num_train=artifact.num_examples,
        num_test=0,
        splits=["train"],
        model_name=training_cfg.model_name,
        layer_combinations=training_cfg.layer_combinations,
        save_acts=False,
        batch_size=min(training_cfg.train_batch_size, artifact.num_examples),
        dataset_name="",
        dataset_folder=training_cfg.dataset_folder,
        seed=training_cfg.seed,
    )
    return asdict(loader_config)


def build_output_paths(training_config_path: Path, suite_name: str) -> tuple[Path, Path]:
    suite_root = REPO_ROOT / "sft_validation_data" / f"{training_config_path.stem}__{suite_name}"
    derived_config_path = training_config_path.with_name(f"{training_config_path.stem}__{suite_name}.json")
    return suite_root, derived_config_path


def print_plan(
    *,
    training_config_path: Path,
    suite_config_path: Path,
    suite: ValidationSuiteConfig,
    suite_root: Path,
    derived_config_path: Path,
) -> None:
    print("Validation suite build plan")
    print(f"  training config: {training_config_path}")
    print(f"  suite config: {suite_config_path}")
    print(f"  suite name: {suite.suite_name}")
    print(f"  output root: {suite_root}")
    print(f"  derived config: {derived_config_path}")
    print("  components:")
    for component in suite.components:
        print(f"    - {component.component_name}: {component.kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen validation artifacts for SFT training runs.")
    parser.add_argument("--training-config", required=True, help="Base training config JSON.")
    parser.add_argument("--suite-config", required=True, help="Validation suite config JSON.")
    parser.add_argument("--derived-config-out", default=None, help="Optional explicit output path for derived config.")
    parser.add_argument("--python-bin", default=".venv/bin/python", help="Python interpreter for generator subprocesses.")
    parser.add_argument("--plan-only", action="store_true", help="Print the build plan without generating artifacts.")
    args = parser.parse_args()

    training_config_path = Path(args.training_config)
    suite_config_path = Path(args.suite_config)
    python_bin = REPO_ROOT / args.python_bin

    assert training_config_path.exists(), f"Missing training config: {training_config_path}"
    assert suite_config_path.exists(), f"Missing suite config: {suite_config_path}"
    assert python_bin.exists(), f"Missing python interpreter: {python_bin}"

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    suite = load_suite_config(suite_config_path)
    suite_root, default_derived_config_path = build_output_paths(training_config_path, suite.suite_name)
    derived_config_path = default_derived_config_path if args.derived_config_out is None else Path(args.derived_config_out)

    if args.plan_only:
        print_plan(
            training_config_path=training_config_path,
            suite_config_path=suite_config_path,
            suite=suite,
            suite_root=suite_root,
            derived_config_path=derived_config_path,
        )
        return

    from nl_probes.configs.sft_config import read_training_config

    training_cfg = read_training_config(str(training_config_path))
    assert not training_cfg.validation_dataset_configs, (
        "Base training config already defines validation_dataset_configs. Start from the original train config."
    )

    sampled_json_dir = suite_root / "sampled_json"
    prebuilt_pt_dir = suite_root / "prebuilt_pt"
    sampled_json_dir.mkdir(parents=True, exist_ok=True)
    prebuilt_pt_dir.mkdir(parents=True, exist_ok=True)

    built_artifacts: list[BuiltValidationArtifact] = []

    for component in suite.components:
        print(f"\n=== Building validation component: {component.component_name} ({component.kind}) ===")
        if isinstance(component, ExistingSyntheticQAJsonSpec):
            source_json_path = REPO_ROOT / component.source_json_path
            assert source_json_path.exists(), f"Missing source JSON: {source_json_path}"
            artifact = build_sampled_synthetic_qa_artifact(
                training_cfg=training_cfg,
                component_name=component.component_name,
                source_json_path=source_json_path,
                sample_num_examples=component.sample_num_examples,
                sample_seed=component.sample_seed,
                sampled_json_dir=sampled_json_dir,
                prebuilt_pt_dir=prebuilt_pt_dir,
            )
            built_artifacts.append(artifact)
            continue

        if isinstance(component, GeneratedSyntheticQASpec):
            source_json_path = ensure_synthetic_qa_dataset(
                python_bin=python_bin,
                model_name=training_cfg.model_name,
                spec=component,
            )
            artifact = build_sampled_synthetic_qa_artifact(
                training_cfg=training_cfg,
                component_name=component.component_name,
                source_json_path=source_json_path,
                sample_num_examples=component.sample_num_examples,
                sample_seed=component.sample_seed,
                sampled_json_dir=sampled_json_dir,
                prebuilt_pt_dir=prebuilt_pt_dir,
            )
            artifact.source_run_name = component.run_name
            built_artifacts.append(artifact)
            continue

        if isinstance(component, HiddenBiasActiveSpec):
            source_json_path = ensure_hidden_bias_active_dataset(
                python_bin=python_bin,
                model_name=training_cfg.model_name,
                spec=component,
            )
            artifact = build_sampled_synthetic_qa_artifact(
                training_cfg=training_cfg,
                component_name=component.component_name,
                source_json_path=source_json_path,
                sample_num_examples=component.sample_num_examples,
                sample_seed=component.sample_seed,
                sampled_json_dir=sampled_json_dir,
                prebuilt_pt_dir=prebuilt_pt_dir,
            )
            artifact.source_run_name = component.run_name
            built_artifacts.append(artifact)
            continue

        if isinstance(component, HiddenBiasInactiveSpec):
            source_json_path = ensure_hidden_bias_inactive_dataset(
                python_bin=python_bin,
                model_name=training_cfg.model_name,
                spec=component,
            )
            artifact = build_sampled_synthetic_qa_artifact(
                training_cfg=training_cfg,
                component_name=component.component_name,
                source_json_path=source_json_path,
                sample_num_examples=component.sample_num_examples,
                sample_seed=component.sample_seed,
                sampled_json_dir=sampled_json_dir,
                prebuilt_pt_dir=prebuilt_pt_dir,
            )
            artifact.source_run_name = component.run_name
            built_artifacts.append(artifact)
            continue

        if isinstance(component, PastLensSpec):
            artifact = build_past_lens_artifact(
                training_cfg=training_cfg,
                spec=component,
                prebuilt_pt_dir=prebuilt_pt_dir,
            )
            built_artifacts.append(artifact)
            continue

        raise TypeError(f"Unhandled validation component: {component}")

    training_cfg.validation_dataset_configs = [
        build_prebuilt_validation_loader_config(training_cfg, artifact) for artifact in built_artifacts
    ]
    training_cfg.validation_dataset_loader_names = [artifact.component_name for artifact in built_artifacts]
    training_cfg.validation_steps = suite.validation_steps
    training_cfg.validation_on_start = suite.validation_on_start
    training_cfg.monitor_num_eval_examples_per_component = None
    training_cfg.monitor_num_eval_examples_classification_total = None
    training_cfg.monitor_eval_steps = None
    training_cfg.monitor_eval_on_start = False

    manifest_path = suite_root / "build_manifest.json"
    manifest_payload = {
        "suite_name": suite.suite_name,
        "training_config_path": str(training_config_path),
        "suite_config_path": str(suite_config_path),
        "derived_config_path": str(derived_config_path),
        "artifacts": [asdict(artifact) for artifact in built_artifacts],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))
    derived_config_path.write_text(json.dumps(asdict(training_cfg), indent=2))

    print("\nValidation suite build complete")
    print(f"  manifest: {manifest_path}")
    print(f"  derived config: {derived_config_path}")


if __name__ == "__main__":
    main()
