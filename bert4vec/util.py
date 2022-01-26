import os
from pathlib import Path
from typing import Dict, Union
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub import HfApi, hf_hub_url, cached_download
from huggingface_hub.snapshot_download import REPO_ID_SEPARATOR

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

    storage_folder = os.path.join(
        cache_dir, repo_id.replace("/", REPO_ID_SEPARATOR) + "." + model_info.sha
    )

    for model_file in model_info.siblings:

        url = hf_hub_url(
            repo_id, filename=model_file.rfilename, revision=model_info.sha
        )
        relative_filepath = os.path.join(*model_file.rfilename.split("/"))

        # Create potential nested dir
        nested_dirname = os.path.dirname(
            os.path.join(storage_folder, relative_filepath)
        )
        os.makedirs(nested_dirname, exist_ok=True)

        path = cached_download(
            url,
            cache_dir=storage_folder,
            force_filename=relative_filepath,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder
