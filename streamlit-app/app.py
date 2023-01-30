import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
import streamlit as st

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config
from nextgenlp.count_vectorizer import NextgenlpCountVectorizer


st.set_page_config(layout="wide")


RANDOM_STATE = 9237
GENIE_VERSIONS = [genie_constants.GENIE_12]
DEFAULT_READ_KEYS = [
    "gene_panels",
    "data_clinical_patient",
    "data_clinical_sample",
    "data_mutations_extended",
]




with st.sidebar.form(key="data_selection_form"):

    st.header("Dataset Selection")
    genie_version = st.selectbox("GENIE version", GENIE_VERSIONS, index=0)
    read_cna = st.checkbox("Read CNA", False)
    remove_no_var = st.checkbox("Remove Samples with No Variants")
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

    submitted = st.form_submit_button('Read Data')


@st.cache()
def read_genie(read_file_paths):
    gd = genie.GenieData.from_file_paths(**read_file_paths)
    return gd



gd = read_genie(read_file_paths)
if remove_no_var:
    gd = gd.subset_to_variants()
if seq_assay_id_group != "ALL":
    gd = gd.subset_from_seq_assay_ids(seq_assay_ids)



gd.make_sentences()

with st.sidebar.form(key="embedding_selection_form"):

    st.header("Embedding Selection")
    ignore_sent_keys = ["sent_gene", "sent_var"]
    sent_keys = sorted([
        col for col in gd.df_dcs.columns
        if col.startswith("sent_") and col not in ignore_sent_keys
    ])
    sent_key = st.selectbox(
        "Sentence Key",
        sent_keys,
        index=0,
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
    unigram_weighter_method="identity",
)

st.dataframe(gd.df_dcs[sent_key].head(10))
xcv = ngp_cv.fit_transform(gd.df_dcs[sent_key])
svd = TruncatedSVD(n_components=2)
xsvd = svd.fit_transform(xcv)

df_plt = pd.DataFrame(gd.df_dcs).reset_index()
df_plt["x"] = xsvd[:,0]
df_plt["y"] = xsvd[:,1]
fig = px.scatter(
    df_plt.head(1000),
    x="x",
    y="y",
    hover_data=[
        "SAMPLE_ID",
        "ONCOTREE_CODE",
        "SEQ_ASSAY_ID",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
        "sent_gene",
    ],
    color="CANCER_TYPE",
)
st.plotly_chart(fig)



st.title("GENIE Data")

st.metric("Patients", gd.df_psm["PATIENT_ID"].nunique())
st.metric("Samples", gd.df_dcs.index.nunique())
st.metric("Samples w/ Variants", gd.df_psm["SAMPLE_ID"].nunique())
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
st.plotly_chart(fig) # , use_container_width=True)

st.header("Data Clinical Samples")
st.dataframe(gd.df_dcs.head(10))

st.header("Data Patient-Sample-Mutation")
st.dataframe(gd.df_psm.head(10))
