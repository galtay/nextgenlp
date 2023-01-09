"""
Test NextgenlpPmiVectorizer.
"""

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config
from nextgenlp.pmi_vectorizer import NextgenlpPmiVectorizer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse


RANDOM_STATE = 9237
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
    #    "data_CNA",
]
read_file_paths = {k: v for k, v in syn_file_paths.items() if k in keep_keys}

gds = {}
gds["ALL"] = genie.GenieData.from_file_paths(**read_file_paths)

# create specific subset
Y_PREDICT = "CANCER_TYPE"
seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK-IMPACT468"]
gd = (
    gds["ALL"]
    .subset_to_variants()
    .subset_from_seq_assay_ids(seq_assay_ids)
    .subset_from_path_score("Polyphen")
    .subset_from_path_score("SIFT")
    .subset_from_y_col(Y_PREDICT, 50)
    #    .subset_to_cna()
)

gd.make_sentences()
df_train, df_test = train_test_split(
    gd.df_dcs, stratify=gd.df_dcs[Y_PREDICT], random_state=RANDOM_STATE
)

min_df = 5
max_df = 1.0

sent_key = "sent_gene_flat"
pmi = NextgenlpPmiVectorizer(
    min_df=min_df,
    max_df=max_df,
    unigram_weighter_method="identity",
    transform_combines="pmi",
)
pmi.fit(df_train[sent_key])
xpmi_train = pmi.transform(df_train[sent_key])

xpmi_test = pmi.transform(df_test[sent_key])
clf = LogisticRegression(n_jobs=-1, max_iter=2000)
clf.fit(xpmi_train, df_train[Y_PREDICT])
y_pred = clf.predict(xpmi_test)
