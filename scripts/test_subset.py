"""
Simple example of creating GenieData subsets.
"""

from nextgenlp import genie
from nextgenlp import genie_constants


GENIE_VERSION = genie_constants.GENIE_12
#GENIE_VERSION = genie_constants.GENIE_13

syn_file_paths = genie.get_file_name_to_path(genie_version=GENIE_VERSION)
keep_keys = [
    "gene_panels",
    "data_clinical_patient",
    "data_clinical_sample",
    "data_mutations_extended",
    "data_CNA",
]
read_file_paths = {k:v for k,v in syn_file_paths.items() if k in keep_keys}

gds = {}
gds["ALL"] = genie.GenieData.from_file_paths(**read_file_paths)

# create specific subset
seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK"]
gd = (
    gds["ALL"]
    .subset_to_variants()
    .subset_from_seq_assay_ids(seq_assay_ids)
    .subset_from_path_score("Polyphen")
    .subset_from_path_score("SIFT")
    .subset_to_cna()
)

gd.make_sentences()
