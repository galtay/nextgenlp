import json
import pandas as pd

from nextgenlp import synapse
from nextgenlp import genie
from nextgenlp import genie_constants


GENIE_VERSION = "genie-12.0-public"
#GENIE_VERSION = "genie-13.3-consortium"

syn_file_paths = genie.get_file_name_to_path(genie_version=GENIE_VERSION)

gds = {}
gds["ALL"] = genie.GenieData.from_file_paths(syn_file_paths)


rows = []
for seq_assay_id_group in genie_constants.SEQ_ASSAY_ID_GROUPS.keys():
    if seq_assay_id_group == "ALL":
        continue
    gd = gds["ALL"].subset_from_seq_assay_id_group(seq_assay_id_group)
    gds[seq_assay_id_group] = gd

    num_nst = gd.df_dcs[gd.df_dcs['CANCER_TYPE']=='Nerve Sheath Tumor'].shape[0]
    num_nf1 = gd.df_dcs['gene_sent'].apply(lambda x: "NF1" in x).astype(int).sum()
    num_nf2 = gd.df_dcs['gene_sent'].apply(lambda x: "NF2" in x).astype(int).sum()

    row = {
        "subset": gd.seq_assay_id_group,
        "panels": len(gd.seq_assay_ids),
        "genes": len(gd.genes),
        "samples": gd.df_dcs.shape[0],
        "samples (NF1)": num_nf1,
        "samples (NF2)": num_nf2,
        "samples (Nerve Sheath Tumor)": num_nst,
        "variants": gd.df_mut.shape[0],


    }
    rows.append(row)


df_report = pd.DataFrame.from_records(rows)
df_report = df_report.sort_values('samples').reset_index(drop=True)


# get example sentence
gd_poly = gd.subset_from_path_score("Polyphen")
row = gd_poly.df_dcs[gd_poly.df_dcs['gene_sent'].apply(lambda x: "NF1" in x)].iloc[1]
