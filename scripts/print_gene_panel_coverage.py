import json
import pandas as pd

from nextgenlp.config import config
from nextgenlp import genie
from nextgenlp import genie_constants


GENIE_VERSION = genie_constants.GENIE_13p1
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
gds = {}
gds["ALL"] = genie.GenieData.from_file_paths(**read_file_paths)


rows = []
for seq_assay_id_group, seq_assay_ids in genie_constants.SEQ_ASSAY_ID_GROUPS.items():
    if seq_assay_id_group == "ALL":
        continue
    gd = gds["ALL"].subset_from_seq_assay_ids(seq_assay_ids)
    gd.make_sentences()
    gds[seq_assay_id_group] = gd

    num_nst = gd.df_dcs[gd.df_dcs['CANCER_TYPE']=='Nerve Sheath Tumor'].shape[0]
    num_nf1 = gd.df_dcs['sent_gene'].apply(lambda x: "NF1" in x).astype(int).sum()
    num_nf2 = gd.df_dcs['sent_gene'].apply(lambda x: "NF2" in x).astype(int).sum()

    row = {
        "subset": seq_assay_id_group,
        "panels": len(gd.seq_assay_ids),
        "genes": len(gd.seq_assay_genes),
        "samples": gd.df_dcs.shape[0],
        "samples (NF1)": num_nf1,
        "samples (NF2)": num_nf2,
        "samples (Nerve Sheath Tumor)": num_nst,
        "variants": gd.df_psm.shape[0],
    }
    rows.append(row)


df_report = pd.DataFrame.from_records(rows)
df_report = df_report.sort_values('samples').reset_index(drop=True)


# get example sentence
gd_poly = gd.subset_from_path_score("Polyphen")
row = gd_poly.df_dcs[gd_poly.df_dcs["sent_gene"].apply(lambda x: "NF1" in x)].iloc[1]
