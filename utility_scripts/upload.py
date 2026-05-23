# %%

from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()

username = "adamkarvonen"

# Manual mode: uncomment and edit this to upload specific checkpoints.
# HF's upload_folder is idempotent (skips files already present), so safe to re-run.
#
# repo_ids = [
#     "500k_pl_31k_spqav2_199k_sqav3_126k_cls_r256_2ep",
# ]
#
# for repo_id in repo_ids:
#     folder = f"checkpoints/{repo_id}/final"
#     hf_repo_id = f"checkpoints_{repo_id}"
#     api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
#     api.upload_folder(folder_path=folder, repo_id=f"{username}/{hf_repo_id}", repo_type="model")

# Auto mode: finds and uploads all checkpoints/*/final directories.
checkpoints_dir = Path("checkpoints")

repo_ids = sorted(
    p.name for p in checkpoints_dir.iterdir() if p.is_dir() and (p / "final").exists()
)

print(f"Found {len(repo_ids)} checkpoints to upload:")
for r in repo_ids:
    print(f"  {r}")

for repo_id in repo_ids:
    folder = checkpoints_dir / repo_id / "final"
    hf_repo_id = f"checkpoints_{repo_id}"

    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading {repo_id} -> {username}/{hf_repo_id}...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=f"{username}/{hf_repo_id}",
        repo_type="model",
    )
    print(f"  Done: {repo_id}")
