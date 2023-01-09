import os
import pandas as pd
from nextgenlp.config import config


embd_types = [
    "sent_gene_cna_count-svd",
    "sent_gene_flat_count-svd",
]

embd_types = [
    "sent_gene_polyphen_count",
    "sent_gene_flat_count",
]

EMBEDDINGS_PATH = config['Paths']["embeddings_path"]


df_vecs = pd.DataFrame()
df_meta = pd.DataFrame()

sample_ids = set()

for ii, embd_type in enumerate(embd_types):

    print(embd_type)
    out_path = os.path.join(EMBEDDINGS_PATH, embd_type)

    df1_vecs = pd.read_csv(os.path.join(out_path, f"{embd_type}_vecs.tsv"), sep="\t", header=None)
    df1_meta = pd.read_csv(os.path.join(out_path, f"{embd_type}_meta.tsv"), sep="\t")

    df1_vecs["SAMPLE_ID"] = df1_meta["SAMPLE_ID"]
    df1_meta = df1_meta.set_index("SAMPLE_ID")
    df1_vecs = df1_vecs.set_index("SAMPLE_ID")

    if ii == 0:
        sample_ids = set(df1_meta.index)
        df_vecs = df1_vecs.copy()
        df_meta = df1_meta.copy()
    else:
        sample_ids = set.intersection(sample_ids, set(df1_meta.index))

        df1_vecs = df1_vecs[df1_vecs.index.isin(sample_ids)]
        df_vecs = df_vecs[df_vecs.index.isin(sample_ids)]

        df1_meta = df1_meta[df1_meta.index.isin(sample_ids)]
        df_meta = df_meta[df_meta.index.isin(sample_ids)]

        df_vecs += df1_vecs
        assert (df_meta.index == df1_meta.index).all()


df_vecs = df_vecs / (ii+1)
df_vecs.to_csv(f"vecs.tsv", sep="\t", header=None, index=False)
df_meta.to_csv(f"meta.tsv", sep="\t", index=False)
