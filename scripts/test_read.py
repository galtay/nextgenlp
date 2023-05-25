"""
Simple example of creating a GenieData object.
Here the paths are set using the config and genie_constants modules.
"""
from pathlib import Path
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config


GENIE_VERSION = genie_constants.GENIE_13p1
READ_CNA = True

synapse_directory = (
    Path(config["Paths"]["synapse_path"])
    / genie_constants.DATASET_NAME_TO_SYNID[GENIE_VERSION]
)
gd_all = genie.GenieData.from_synapse_directory(synapse_directory, read_cna=READ_CNA)
gd = (
    gd_all
    .subset_from_seq_assay_ids(genie_constants.SEQ_ASSAY_ID_GROUPS["MSK"])
    .subset_to_cna()
    .subset_to_cna_altered()
)
