"""
https://python-docs.synapse.org/build/html/index.html
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from loguru import logger
import synapseclient
import synapseutils

from nextgenlp.config import config
from nextgenlp import genie_constants


def _read_secrets(
    secrets_path: str = config["Paths"]["SECRETS_PATH"],
) -> Dict[str, str]:
    return json.load(open(secrets_path, "r"))


def get_client(silent=True) -> synapseclient.Synapse:
    secrets = _read_secrets()
    return synapseclient.login(authToken=secrets["SYNAPSE_TOKEN"], silent=silent)


def sync_datasets(
    dataset_synids: Optional[Iterable[str]] = None,
    sync_path: Union[str, Path] = config["Paths"]["SYNAPSE_PATH"],
) -> List[synapseclient.entity.File]:
    if dataset_synids is None:
        dataset_synids = genie_constants.DATASET_NAME_TO_SYNID.values()
    syn = get_client()
    files = []
    for dataset_synid in dataset_synids:
        files.extend(
            synapseutils.syncFromSynapse(
                syn,
                dataset_synid,
                path=Path(sync_path) / dataset_synid,
            )
        )
    return files


if __name__ == "__main__":
    logger.info("syncing all default synapse datasets")
    files = sync_datasets()
