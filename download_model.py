"""
Download trained relation extraction model 
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from wasabi import msg


DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".multi2oie")

models_url = {
    "model-epoch1-step16000-score1.9768.bin": "https://sciencedata.dk//shared/81ee2688645634814152e3965e74b7f7?download",
}


def models() -> list:
    """
    Returns a list of valid DaCy models

    Returns:
        list: list of valid DaCy models
    """
    return list(models_url.keys())


def download_model(
    model="model-epoch1-step16000-score1.9768.bin",
    save_path: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
    open_unverified_connection: bool = True,
) -> str:
    """
    Downloads and install a specified DaCy pipeline.

    Args:
        model (str): string indicating DaCy model, use dacy.models() to get a list of models
        save_path (str, optional): The path you want to save your model to. Is only used for DaCy models of v0.0.0 as later models are installed as modules to allow for better versioning. Defaults to None denoting the default cache directory. This can be found using using dacy.where_is_my_dacy().
        force (bool, optional): Should it download the model regardless of it already being present? Defaults to False.
        verbose (bool): Toggles the verbosity of the function. Defaults to True.

    Returns:
        a string of the model location

    Example:
        >>> download_model(model="da_dacy_medium_tft-0.0.0")
    """

    if save_path is None:
        save_path = DEFAULT_CACHE_DIR

    if open_unverified_connection:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    url = models_url[model]
    path = os.path.join(save_path, model)
    if os.path.exists(path) and force is False:
        return path

    if verbose is True:
        msg.info(f"\nDownloading '{model}'")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    download_url(url, path)
    if verbose is True:
        msg.info(f"\Model successfully downloaded")
    return path


class DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize=None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str) -> None:
    import urllib.request

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
