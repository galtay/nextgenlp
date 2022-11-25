import pandas as pd

embd_types = [
    "sent_gene_cna_count-svd",
    "sent_gene_flat_count-svd",
]

df_vecs = pd.DataFrame()
df_meta = pd.DataFrame()
for ii, embd_type in enumerate(embd_types):

    print(embd_type)
    df1_vecs = pd.read_csv(f"{embd_type}_vecs.tsv", sep="\t", header=None)
    df1_meta = pd.read_csv(f"{embd_type}_meta.tsv", sep="\t")
    if ii == 0:
        df_vecs = df1_vecs.copy()
        df_meta = df1_meta.copy()
    else:
        df_vecs += df1_vecs
        assert((df_meta == df1_meta).all().all())


df_vecs.to_csv(f"vecs.tsv", sep="\t", header=None, index=False)
df_meta.to_csv(f"meta.tsv", sep="\t", index=False)
