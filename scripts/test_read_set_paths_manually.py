"""
Simple example of creating a GenieData object.
Here the paths to the dataset files are set manually
"""
import os
from nextgenlp import genie


# you will have to update this
syn_base_path = "/home/galtay/data/hack4nf-2022/synapse/syn32309524"
USE_CNA = False

gene_panels = os.path.join(syn_base_path, "gene_panels")
data_clinical_patient = os.path.join(syn_base_path, "data_clinical_patient.txt")
data_clinical_sample = os.path.join(syn_base_path, "data_clinical_sample.txt")
data_mutations_extended = os.path.join(syn_base_path, "data_mutations_extended.txt")
data_CNA = os.path.join(syn_base_path, "data_CNA.txt") if USE_CNA else None
gd = genie.GenieData.from_file_paths(
    gene_panels,
    data_clinical_patient,
    data_clinical_sample,
    data_mutations_extended,
    data_CNA,
)
