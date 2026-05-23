from pathlib import Path
import hashlib

from huggingface_hub import get_collection, hf_hub_download, upload_file

from nl_probes.configs.sft_config import TRAINING_CONFIG_FILENAME

# =========================
# Edit these
# =========================
COLLECTION_ID = "adamkarvonen/activation-oracles"
DRAFTS_DIR = Path("training_config_drafts")
REMOTE_FILENAME = TRAINING_CONFIG_FILENAME

# Optional subset filter. Keep empty to process the full collection.
ONLY_REPO_IDS: list[str] = []

# =========================
# Toggle steps
# =========================
DO_LIST_PLAN = True
DO_UPLOAD = True
DO_VERIFY = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_repo_ids() -> list[str]:
    collection = get_collection(COLLECTION_ID)
    repo_ids = [item.item_id for item in collection.items if item.item_type == "model"]
    if ONLY_REPO_IDS:
        repo_ids = [repo_id for repo_id in repo_ids if repo_id in ONLY_REPO_IDS]
    assert repo_ids, "No model repos found for the selected filter"
    return repo_ids


def get_local_config_path(repo_id: str) -> Path:
    return DRAFTS_DIR / repo_id.replace("/", "__") / REMOTE_FILENAME


def main() -> None:
    repo_ids = get_repo_ids()
    pairs: list[tuple[str, Path]] = []

    for repo_id in repo_ids:
        local_path = get_local_config_path(repo_id)
        assert local_path.exists(), f"Missing draft config for {repo_id}: {local_path}"
        pairs.append((repo_id, local_path))

    if DO_LIST_PLAN:
        print(f"Collection: {COLLECTION_ID}")
        print(f"Configs to process: {len(pairs)}")
        for repo_id, local_path in pairs:
            print(f"- {repo_id}")
            print(f"  local: {local_path}")
            print(f"  sha256: {sha256(local_path)}")
        print()

    if DO_UPLOAD:
        for repo_id, local_path in pairs:
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=REMOTE_FILENAME,
                repo_id=repo_id,
                commit_message="Backfill ao_config.json",
            )
            print(f"UPLOADED: {repo_id}/{REMOTE_FILENAME}")
        print()

    if DO_VERIFY:
        for repo_id, local_path in pairs:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=REMOTE_FILENAME,
                )
            )
            local_hash = sha256(local_path)
            remote_hash = sha256(downloaded)
            print(f"VERIFY: {repo_id}")
            print("  local sha256 :", local_hash)
            print("  remote sha256:", remote_hash)
            assert local_hash == remote_hash, f"Hash mismatch for {repo_id}"
        print("VERIFY OK")


if __name__ == "__main__":
    main()
