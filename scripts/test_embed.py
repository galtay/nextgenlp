"""
Simple example of creating GenieData subsets.
"""

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp import embedders

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse


RANDOM_STATE = 9237
GENIE_VERSION = genie_constants.GENIE_12
#GENIE_VERSION = genie_constants.GENIE_13


def get_count_vectorizer(**cv_kwargs):
    """Get a count vectorizer appropriate for pre-tokenized text"""
    return CountVectorizer(
        tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, **cv_kwargs
    )


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

cv = get_count_vectorizer()
cv.fit(df_train["sent_gene"])
xcv_train = cv.transform(df_train["sent_gene"])
xcv_test = cv.transform(df_test["sent_gene"])
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
)



sent_key = "sent_gene_sift"
ncv = NextgenlpCountVectorizer()
ncv.fit(df_train[sent_key])
xncv_train = ncv.transform(df_train[sent_key])
xncv_test = ncv.transform(df_test[sent_key])
nclf = LogisticRegression(n_jobs=-1, max_iter=2000)
nclf.fit(xncv_train, df_train[Y_PREDICT])
ny_pred = clf.predict(xncv_test)
ncls_report_dict = classification_report(
    df_test[Y_PREDICT], y_pred, output_dict=True, zero_division=0,
)
ndf_clf_report = (
    pd.DataFrame(cls_report_dict)
    .drop(columns=["accuracy", "macro avg", "weighted avg"])
    .T
)
