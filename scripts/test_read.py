"""
Simple example of creating a GenieData object.
"""
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config


GENIE_VERSION = genie_constants.GENIE_12


syn_file_paths = genie_constants.get_file_name_to_path(
    sync_path=config["Paths"]["synapse_path"],
    genie_version=GENIE_VERSION,
)

keep_keys = [
    "gene_panels",
    "data_clinical_patient",
    "data_clinical_sample",
    "data_mutations_extended",
    "data_CNA",
]
read_file_paths = {k:v for k,v in syn_file_paths.items() if k in keep_keys}
gd = genie.GenieData.from_file_paths(**read_file_paths)
