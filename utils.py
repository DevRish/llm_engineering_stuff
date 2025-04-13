from rich import print as rich_print
from huggingface_hub import hf_hub_download
import os

def download_model_from_hf(model_id, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    local_file = os.path.join(save_dir, filename)

    if not os.path.exists(local_file):
        rich_print(f"Downloading {filename} from {model_id}...")
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
    else:
        rich_print(f"[bold magenta]Model file already exists at {local_file}[/bold magenta]")
        return local_file