"""
Test that CountVectorizer and NextgenlpCountVectorizer produce the same results.
"""

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp import embedders

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse


RANDOM_STATE = 9237
GENIE_VERSION = genie_constants.GENIE_12
#GENIE_VERSION = genie_constants.GENIE_13


class NextgenlpCountVectorizer:

    def __init__(self, min_unigram_weight=0, unigram_weighter = embedders.unigram_weighter_one):
        self.min_unigram_weight = min_unigram_weight
        self.unigram_weighter = unigram_weighter

    def fit(self, sentences):
        unigram_weights = embedders.calculate_unigrams(
            sentences,
            self.min_unigram_weight,
            self.unigram_weighter,
        )
        index_to_unigram = dict(enumerate(unigram_weights.keys()))
        unigram_to_index = {unigram: ii for ii, unigram in index_to_unigram.items()}

        self.unigram_weights = unigram_weights
        self.index_to_unigram = index_to_unigram
        self.unigram_to_index = unigram_to_index

    def transform(self, sentences):
        row_indexs = []
        col_indexs = []
        dat_values = []
        for isamp, sent in enumerate(sentences):
            for unigram, weight in sent:
                row_indexs.append(isamp)
                col_indexs.append(self.unigram_to_index[unigram])
                dat_values.append(weight)
        nrows = len(sentences)
        ncols = len(self.unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )

syn_file_paths = genie.get_file_name_to_path(genie_version=GENIE_VERSION)
keep_keys = [
    "gene_panels",
    "data_clinical_patient",
    "data_clinical_sample",
    "data_mutations_extended",
#    "data_CNA",
]
read_file_paths = {k:v for k,v in syn_file_paths.items() if k in keep_keys}

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
df_train, df_test = train_test_split(gd.df_dcs, stratify=gd.df_dcs[Y_PREDICT], random_state=RANDOM_STATE)



def get_count_vectorizer(**cv_kwargs):
    """Get a count vectorizer appropriate for pre-tokenized text"""
    return CountVectorizer(
        tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, **cv_kwargs
    )

skl_sent_key = "sent_gene"
skl_cv = get_count_vectorizer()
skl_cv.fit(df_train[skl_sent_key])
skl_xcv_train = skl_cv.transform(df_train[skl_sent_key])

skl_xcv_test = skl_cv.transform(df_test[skl_sent_key])
skl_clf = LogisticRegression(n_jobs=-1, max_iter=2000)
skl_clf.fit(skl_xcv_train, df_train[Y_PREDICT])
skl_y_pred = skl_clf.predict(skl_xcv_test)



ngp_sent_key = "sent_gene_flat"
ngp_cv = NextgenlpCountVectorizer(unigram_weighter=embedders.unigram_weighter_identity)
ngp_cv.fit(df_train[ngp_sent_key])
ngp_xcv_train = ngp_cv.transform(df_train[ngp_sent_key])

ngp_xcv_test = ngp_cv.transform(df_test[ngp_sent_key])
ngp_clf = LogisticRegression(n_jobs=-1, max_iter=2000)
ngp_clf.fit(ngp_xcv_train, df_train[Y_PREDICT])
ngp_y_pred = ngp_clf.predict(ngp_xcv_test)


skl_feature_names = skl_cv.get_feature_names_out()
for ii in range(df_train.shape[0]):
    sent = sorted(df_train.iloc[ii]['sent_gene'])

    skl_row = np.array(skl_xcv_train.getrow(ii).todense()).squeeze()
    ngp_row = np.array(ngp_xcv_train.getrow(ii).todense()).squeeze()

    assert len(skl_row.shape) == len(ngp_row.shape) == 1
    assert skl_row.size == ngp_row.size

    skl_genes = []
    for icol in range(skl_row.size):
        if skl_row[icol] != 0:
            skl_genes.extend([skl_feature_names[icol]] * skl_row[icol])
    skl_genes = sorted(skl_genes)
    assert(skl_genes == sent)

    ngp_genes = []
    for icol in range(ngp_row.size):
        if ngp_row[icol] != 0:
            ngp_genes.extend([ngp_cv.index_to_unigram[icol]] * int(ngp_row[icol]))
    ngp_genes = sorted(ngp_genes)
    assert(ngp_genes == sent)
