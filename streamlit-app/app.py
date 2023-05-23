from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import streamlit as st
import umap

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config
from nextgenlp.count_vectorizer import NextgenlpCountVectorizer


st.set_page_config(layout="wide")


GENIE_VERSIONS = [
    genie_constants.GENIE_12p1,
    genie_constants.GENIE_13p1,
]
DEFAULT_READ_KEYS = [
    "gene_panels",
    "data_clinical_patient",
    "data_clinical_sample",
    "data_mutations_extended",
]


with st.sidebar.form(key="data_selection_form"):

    st.header("Dataset Selection")
    genie_version = st.selectbox("GENIE version", GENIE_VERSIONS, index=1)
    read_cna = st.checkbox("read CNA", False)

    st.header("Filters")
    remove_no_var = st.checkbox("remove samples with no variants")
    remove_no_cna = st.checkbox("remove samples with no CNA data")

    seq_assay_id_group = st.selectbox(
        "Sequence Assay ID Group",
        sorted(list(genie_constants.SEQ_ASSAY_ID_GROUPS.keys())),
        index=6,
    )
    seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS[seq_assay_id_group]


    keep_keys = DEFAULT_READ_KEYS
    if read_cna:
        keep_keys.append("data_CNA")

    syn_file_paths = genie_constants.get_file_name_to_path(
        sync_path=config["Paths"]["synapse_path"],
        genie_version=genie_version,
    )
    read_file_paths = {k: v for k, v in syn_file_paths.items() if k in keep_keys}

    submitted = st.form_submit_button('Load Data')


@st.cache()
def read_genie(read_file_paths):
    print("read_file_paths: ", read_file_paths)
    gd = genie.GenieData.from_file_paths(**read_file_paths)
    return gd



gd_all = read_genie(read_file_paths)
gd = deepcopy(gd_all)

if remove_no_var:
    gd = gd.subset_to_variants()
if seq_assay_id_group != "ALL":
    gd = gd.subset_from_seq_assay_ids(seq_assay_ids)
if remove_no_cna:
    gd = gd.subset_to_cna()


gd.make_sentences()

with st.sidebar.form(key="embedding_selection_form"):

    st.header("Embedding Selection")
    ignore_sent_keys = ["sent_gene", "sent_var", "sent_cna"]
    sent_keys = sorted([
        col for col in gd.df_dcs.columns
        if col.startswith("sent_") and col not in ignore_sent_keys
    ])
    sent_key = st.selectbox(
        "Sentence Key",
        sent_keys,
        index=0,
    )
    unigram_weighter_method = st.selectbox(
        "Unigram Weighter Method",
        ["one", "identity", "abs"],
        index=2,
    )
    min_df = st.number_input("min_df", value=1)
    max_df = st.number_input("max_df", value=1.0)
    submitted = st.form_submit_button("Make Embeddings")


with st.sidebar:

    st.header("Sequence Assay IDs")
    st.write(seq_assay_ids)


ngp_cv = NextgenlpCountVectorizer(
    min_df=min_df,
    max_df=max_df,
    unigram_weighter_method=unigram_weighter_method,
)

st.dataframe(gd.df_dcs[sent_key].head(10))
xcv = ngp_cv.fit_transform(gd.df_dcs[sent_key])
xcvn = normalize(xcv, norm="l2", axis=1)

svd = TruncatedSVD(n_components=2)
xsvd = svd.fit_transform(xcvn)

ured = umap.UMAP()
xumap = ured.fit_transform(xcvn)

df_plt = pd.DataFrame(gd.df_dcs).reset_index()
df_plt["x_svd"] = xsvd[:,0]
df_plt["y_svd"] = xsvd[:,1]
df_plt["x_umap"] = xumap[:, 0]
df_plt["y_umap"] = xumap[:, 1]
hover_data = [
    "SAMPLE_ID",
    "ONCOTREE_CODE",
    "SEQ_ASSAY_ID",
    "CANCER_TYPE",
    "CANCER_TYPE_DETAILED",
    "sent_gene",
]
if read_cna:
    hover_data.append("sent_gene_cna")

projection = st.selectbox("Projection", ["svd", "umap"], index=0)

fig = px.scatter(
    df_plt, # .head(1000),
    x=f"x_{projection}",
    y=f"y_{projection}",
    hover_data=hover_data,
    color="CANCER_TYPE",
)
st.plotly_chart(fig, use_container_width=True)



st.title("GENIE Data")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patients", gd.df_psm["PATIENT_ID"].nunique())
with col2:
    st.metric("Samples", gd.df_dcs.index.nunique())
with col3:
    st.metric("Samples w/ Variants", gd.df_psm["SAMPLE_ID"].nunique())
with col4:
    st.metric("Variants", gd.df_psm.shape[0])


df_plt = (
    gd.df_dcs.groupby(["SEQ_ASSAY_ID", "CENTER"])
    .size()
    .sort_values()
    .to_frame("count")
    .reset_index()
)
fig = px.bar(
    df_plt,
    x="SEQ_ASSAY_ID",
    y="count",
    color="CENTER",
)
st.plotly_chart(fig, use_container_width=True)

st.header("Data Clinical Samples")
st.dataframe(gd.df_dcs.head(10))

st.header("Data Patient-Sample-Mutation")
st.dataframe(gd.df_psm.head(10))
