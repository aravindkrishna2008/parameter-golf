import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

REPO_ID = "willdepueoai/parameter-golf"
REMOTE_ROOT_PREFIX = "datasets"
TOKENIZERS_DIR = Path("data/tokenizers")
TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)

files = ["fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab"]

for filename in files:
    # Try getting it from remote paths
    possible_paths = [
        f"{REMOTE_ROOT_PREFIX}/tokenizers/{filename}",
        f"tokenizers/{filename}",
        filename,
    ]

    found = False
    for path in possible_paths:
        print(f"Trying to download {path}...")
        try:
            p = Path(path)
            subfolder = p.parent.as_posix() if p.parent != Path(".") else None
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=p.name,
                subfolder=subfolder,
                repo_type="dataset",
            )
            print(f"Downloaded to {cached}")
            target = TOKENIZERS_DIR / filename
            shutil.copy2(cached, target)
            print(f"Copied to {target}")
            found = True
            break
        except Exception as e:
            print(f"Failed: {e}")

    if not found:
        print(f"Could not download {filename}")
