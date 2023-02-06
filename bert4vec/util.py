import os
from pathlib import Path
from typing import Dict, Union
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

def snapshot_download(
    repo_id: str,
    revision: str = None,
    cache_dir: Union[str, Path, None] = None,
    library_name: str = None,
    library_version: str = None,
    user_agent: Union[Dict, str, None] = None,
) -> str:
    """
    Download pretrained model from https://huggingface.co/
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    _api = HfApi()
    model_info = _api.model_info(repo_id=repo_id, revision=revision)

    storage_folder = os.path.join(cache_dir, repo_id.replace("/", "_"))
    for model_file in model_info.siblings:
        filename = os.path.join(*model_file.rfilename.split("/"))
        if filename.endswith(".h5") or filename.endswith(".ot") or filename.endswith(".msgpack"):
            continue
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=storage_folder,
            force_filename=filename,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")
    return storage_folder

snapshot_download("WangZeJun/roformer-sim-small-chinese")
