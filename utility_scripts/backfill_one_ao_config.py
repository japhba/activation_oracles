from pathlib import Path
import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download, upload_file

# =========================
# Edit these
# =========================
REPO_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
LOCAL_CONFIG = Path(
    "training_config_drafts/adamkarvonen__checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B/ao_config.json"
)
REMOTE_FILENAME = "ao_config.json"

# =========================
# Toggle steps
# =========================
DO_SHOW_LOCAL = True
DO_UPLOAD = True
DO_SHOW_REMOTE_FILES = False
DO_DOWNLOAD_AND_VERIFY = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    api = HfApi()

    if DO_SHOW_LOCAL:
        assert LOCAL_CONFIG.exists(), LOCAL_CONFIG
        cfg = json.loads(LOCAL_CONFIG.read_text())
        print("LOCAL:", LOCAL_CONFIG)
        print("model_name:", cfg["model_name"])
        print("created_at_utc:", cfg["created_at_utc"])
        print("git_commit:", cfg["git_commit"])
        print("dataset_configs:", len(cfg["dataset_configs"]))
        print("dataset_loader_names:", len(cfg["dataset_loader_names"]))
        print("sha256:", sha256(LOCAL_CONFIG))
        print()

    if DO_UPLOAD:
        assert LOCAL_CONFIG.exists(), LOCAL_CONFIG
        upload_file(
            path_or_fileobj=str(LOCAL_CONFIG),
            path_in_repo=REMOTE_FILENAME,
            repo_id=REPO_ID,
            commit_message="Backfill ao_config.json",
        )
        print(f"UPLOADED: {REPO_ID}/{REMOTE_FILENAME}")
        print()

    if DO_SHOW_REMOTE_FILES:
        files = api.list_repo_files(repo_id=REPO_ID)
        print(f"REMOTE FILES ({REPO_ID}):")
        for file in files:
            print(" ", file)
        print()

    if DO_DOWNLOAD_AND_VERIFY:
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=REMOTE_FILENAME,
            )
        )
        local_hash = sha256(LOCAL_CONFIG)
        remote_hash = sha256(downloaded)

        print("DOWNLOADED:", downloaded)
        print("local sha256 :", local_hash)
        print("remote sha256:", remote_hash)
        assert local_hash == remote_hash, "Hash mismatch between local and remote config"

        cfg = json.loads(downloaded.read_text())
        print("remote model_name:", cfg["model_name"])
        print("remote created_at_utc:", cfg["created_at_utc"])
        print("remote git_commit:", cfg["git_commit"])
        print("VERIFY OK")


if __name__ == "__main__":
    main()
