"""
Simple example of creating GenieData subsets.
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
            gd.df_dcs[sent_key],
            self.min_unigram_weight,
            self.unigram_weighter,
        )
        unigram_to_index = {unigram: ii for ii, (unigram, _) in enumerate(unigram_weights.most_common())}
        index_to_unigram = np.array([unigram for (unigram, _) in unigram_weights.most_common()])

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
                dat_values.append(self.unigram_weighter(weight))
        nrows = len(sentences)
        ncols = len(self.unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )


def get_clf_feature_importance(clf, class_name, feature_names, nmax=10):
    class_index = np.where(clf.classes_ == class_name)[0][0]
    coefs = clf.coef_[class_index,:]
    sindxs = np.argsort(coefs)[::-1]
    results = []
    for ii in range(nmax):
        results.append((
            feature_names[sindxs[ii]],
            coefs[sindxs[ii]],
        ))
    return results


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
#seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK-NOHEME"]
gd = (
    gds["ALL"]
    .subset_to_variants()
    .subset_from_seq_assay_ids(seq_assay_ids)
#    .subset_from_path_score("Polyphen")
#    .subset_from_path_score("SIFT")
    .subset_from_y_col(Y_PREDICT, 50)
#    .subset_to_cna()
)



gd.make_sentences()
df_train, df_test = train_test_split(gd.df_dcs, stratify=gd.df_dcs[Y_PREDICT], random_state=RANDOM_STATE)

all_sent_keys = [
    "sent_gene_flat", "sent_gene_sift", "sent_gene_polyphen",
    "sent_var_flat", "sent_var_sift", "sent_var_polyphen",
    "sent_gene_cna",
]
sent_keys = [sent_key for sent_key in all_sent_keys if sent_key in gd.df_dcs.columns]

df_clf_reports = {}
clfs = {}
cvs = {}

for sent_key in sent_keys:

    if sent_key == "sent_gene_cna":
        unigram_weighter = embedders.UnigramWeighter("abs", 0.0)
    else:
        unigram_weighter = embedders.UnigramWeighter("identity", 1.0)

    cv = NextgenlpCountVectorizer(unigram_weighter=unigram_weighter)
    cv.fit(df_train[sent_key])
    cvs[sent_key] = cv
    xcv_train = cv.transform(df_train[sent_key])
    xcv_test = cv.transform(df_test[sent_key])
    clf = LogisticRegression(n_jobs=-1, max_iter=2000)
    clf.fit(xcv_train, df_train[Y_PREDICT])
    y_pred = clf.predict(xcv_test)
    cls_report_dict = classification_report(
        df_test[Y_PREDICT], y_pred, output_dict=True, zero_division=0,
    )
    df_clf_report = (
        pd.DataFrame(cls_report_dict)
        .drop(columns=["accuracy", "macro avg", "weighted avg"])
        .T
    ).sort_values('f1-score')
    df_clf_reports[sent_key] = df_clf_report
    clfs[sent_key] = clf
