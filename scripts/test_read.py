"""
Simple example of creating a GenieData object.
Here the paths are set using the config and genie_constants modules.
"""
from pathlib import Path
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config


GENIE_VERSION = genie_constants.GENIE_13p1
USE_CNA = False

synapse_directory = (
    Path(config["Paths"]["synapse_path"])
    / genie_constants.DATASET_NAME_TO_SYNID[GENIE_VERSION]
)

keep_keys = [
    "gene_panels",
    "data_clinical_patient",
    "data_clinical_sample",
    "data_mutations_extended",
]
if USE_CNA:
    keep_keys.append("data_CNA")

gd = genie.GenieData.from_synapse_directory(synapse_directory, read_cna=USE_CNA)
