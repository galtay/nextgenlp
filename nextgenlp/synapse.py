"""
https://python-docs.synapse.org/build/html/index.html
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from loguru import logger
import synapseclient
import synapseutils

from nextgenlp.config import config


GENIE_12 = "genie-12.0-public"
GENIE_13 = "genie-13.3-consortium"
VALID_GENIE_VERSIONS = [GENIE_12, GENIE_13]

DATASET_NAME_TO_SYNID = {
    GENIE_12: "syn32309524",
    GENIE_13: "syn36709873",
}

ASSAY_INFORMATION = "assay_information"
DATA_CLINICAL_PATIENT = "data_clinical_patient"
DATA_CLINICAL_SAMPLE = "data_clinical_sample"
DATA_FUSIONS = "data_fusions"
DATA_GENE_MATRIX = "data_gene_matrix"
DATA_MUTATIONS_EXTENDED = "data_mutations_extended"
DATA_CNA = "data_CNA"
DATA_CNA_HG19_SEG = "data_cna_hg19"
GENE_PANELS = "gene_panels"
GENOMIC_INFORMATION = "genomic_information"


SYNC_PATH = config["Paths"]["SYNAPSE_PATH"]
SECRETS_PATH = config["Paths"]["SECRETS_PATH"]


def get_file_name_to_path(
    sync_path: str = SYNC_PATH,
    genie_version: str = GENIE_12,
) -> Dict[str, Path]:

    """Return the paths to GENIE datasets."""

    genie_path = Path(sync_path) / DATASET_NAME_TO_SYNID[genie_version]
    logger.info(f"genie_path={genie_path}")

    if genie_version == GENIE_12:
        file_name_to_path = {
            ASSAY_INFORMATION: genie_path / f"{ASSAY_INFORMATION}.txt",
            DATA_CLINICAL_PATIENT: genie_path / f"{DATA_CLINICAL_PATIENT}.txt",
            DATA_CLINICAL_SAMPLE: genie_path / f"{DATA_CLINICAL_SAMPLE}.txt",
            DATA_FUSIONS: genie_path / f"{DATA_FUSIONS}.txt",
            DATA_GENE_MATRIX: genie_path / f"{DATA_GENE_MATRIX}.txt",
            DATA_MUTATIONS_EXTENDED: genie_path / f"{DATA_MUTATIONS_EXTENDED}.txt",
            DATA_CNA: genie_path / f"{DATA_CNA}.txt",
            DATA_CNA_HG19_SEG: genie_path / f"genie_{DATA_CNA_HG19_SEG}.seg",
            GENE_PANELS: genie_path / GENE_PANELS,
            GENOMIC_INFORMATION: genie_path / f"{GENOMIC_INFORMATION}.txt",
        }

    elif genie_version == GENIE_13:
        file_name_to_path = {
            ASSAY_INFORMATION: genie_path / f"{ASSAY_INFORMATION}_13.3-consortium.txt",
            DATA_CLINICAL_PATIENT: genie_path
            / f"{DATA_CLINICAL_PATIENT}_13.3-consortium.txt",
            DATA_CLINICAL_SAMPLE: genie_path
            / f"{DATA_CLINICAL_SAMPLE}_13.3-consortium.txt",
            DATA_FUSIONS: genie_path / f"{DATA_FUSIONS}_13.3-consortium.txt",
            DATA_GENE_MATRIX: genie_path / f"{DATA_GENE_MATRIX}_13.3-consortium.txt",
            DATA_MUTATIONS_EXTENDED: genie_path
            / f"{DATA_MUTATIONS_EXTENDED}_13.3-consortium.txt",
            DATA_CNA: genie_path / f"{DATA_CNA}_13.3-consortium.txt",
            DATA_CNA_HG19_SEG: genie_path
            / f"genie_private_{DATA_CNA_HG19_SEG}_13.3-consortium.seg",
            GENE_PANELS: genie_path / GENE_PANELS,
            GENOMIC_INFORMATION: genie_path
            / f"{GENOMIC_INFORMATION}_13.3-consortium.txt",
        }

    else:
        raise ValueError(f"genie version must be one of {VALID_GENIE_VERSIONS}")

    return file_name_to_path


def _read_secrets() -> Dict[str, str]:
    return json.load(open(SECRETS_PATH, "r"))


def get_client(silent=True) -> synapseclient.Synapse:
    secrets = _read_secrets()
    return synapseclient.login(authToken=secrets["SYNAPSE_TOKEN"], silent=silent)


def sync_datasets(
    dataset_synids: Optional[Iterable[str]] = None,
    sync_path: Union[str, Path]=SYNC_PATH,
) -> List[synapseclient.entity.File]:
    if dataset_synids is None:
        dataset_synids = DATASET_NAME_TO_SYNID.values()
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
