"""
Test that CountVectorizer and NextgenlpCountVectorizer produce the same results.
"""

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.config import config
from nextgenlp.count_vectorizer import NextgenlpCountVectorizer

import json
import funcy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse


RANDOM_STATE = 9237
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


def get_count_vectorizer(**cv_kwargs):
    """Get an sklearn count vectorizer appropriate for pre-tokenized text"""
    return CountVectorizer(
        tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, **cv_kwargs
    )


min_df = 50
max_df = 1.0


skl_sent_key = "sent_gene"
skl_cv = get_count_vectorizer(min_df=min_df, max_df=max_df)
skl_cv.fit(df_train[skl_sent_key])
skl_xcv_train = skl_cv.transform(df_train[skl_sent_key])

skl_xcv_test = skl_cv.transform(df_test[skl_sent_key])
skl_clf = LogisticRegression(n_jobs=-1, max_iter=2000)
skl_clf.fit(skl_xcv_train, df_train[Y_PREDICT])
skl_y_pred = skl_clf.predict(skl_xcv_test)


ngp_sent_key = "sent_gene_flat"
ngp_cv = NextgenlpCountVectorizer(
    min_df=min_df,
    max_df=max_df,
    #    unigram_weighter_method="identity",
    unigram_weighter_method="one",
)
ngp_cv.fit(df_train[ngp_sent_key])
ngp_xcv_train = ngp_cv.transform(df_train[ngp_sent_key])

ngp_xcv_test = ngp_cv.transform(df_test[ngp_sent_key])
ngp_clf = LogisticRegression(n_jobs=-1, max_iter=2000)
ngp_clf.fit(ngp_xcv_train, df_train[Y_PREDICT])
ngp_y_pred = ngp_clf.predict(ngp_xcv_test)


print(f"stop words: {skl_cv.stop_words_}")
assert skl_cv.stop_words_ == ngp_cv.banned_unigrams
skl_feature_names = skl_cv.get_feature_names_out()


for ii in range(df_train.shape[0]):

    # get original sentence
    full_sent = sorted(df_train.iloc[ii]["sent_gene"])

    # remove stop words
    sent = [x for x in full_sent if x not in ngp_cv.banned_unigrams]

    # get the rows from the sklearn and nextgenlp sparse arrays
    skl_row = np.array(skl_xcv_train.getrow(ii).todense()).squeeze()
    ngp_row = np.array(ngp_xcv_train.getrow(ii).todense()).squeeze()

    assert len(skl_row.shape) == len(ngp_row.shape) == 1
    assert skl_row.size == ngp_row.size

    # assert that we recover the sentence from sklearn
    skl_genes = []
    for icol in range(skl_row.size):
        if skl_row[icol] != 0:
            skl_genes.extend([skl_feature_names[icol]] * skl_row[icol])
    skl_genes = sorted(skl_genes)
    assert skl_genes == sent

    # assert that we recover the sentence from ngp
    ngp_genes = []
    for icol in range(ngp_row.size):
        if ngp_row[icol] != 0:
            ngp_genes.extend([ngp_cv.index_to_unigram[icol]] * int(ngp_row[icol]))
    ngp_genes = sorted(ngp_genes)
    assert ngp_genes == sent
