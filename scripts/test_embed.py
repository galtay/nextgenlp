"""
Simple example of creating GenieData subsets.
"""

from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp import embedders

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse
from tqdm import tqdm


USE_VARIANTS = False
RANDOM_STATE = 9237
GENIE_VERSION = genie_constants.GENIE_12
# GENIE_VERSION = genie_constants.GENIE_13



def get_clf_feature_importance(clf, class_name, feature_names, nmax=10):
    class_index = np.where(clf.classes_ == class_name)[0][0]
    coefs = clf.coef_[class_index, :]
    sindxs = np.argsort(coefs)[::-1]
    results = []
    for ii in range(nmax):
        results.append(
            (
                feature_names[sindxs[ii]],
                coefs[sindxs[ii]],
            )
        )
    return results


syn_file_paths = genie.get_file_name_to_path(genie_version=GENIE_VERSION)
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
# seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK-IMPACT468"]
seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK-NOHEME"]
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
df_train, df_test = train_test_split(
    gd.df_dcs, stratify=gd.df_dcs[Y_PREDICT], random_state=RANDOM_STATE
)

all_sent_keys = [
    "sent_gene_flat",
    "sent_gene_sift",
    "sent_gene_polyphen",
    "sent_var_flat",
    "sent_var_sift",
    "sent_var_polyphen",
    "sent_gene_cna",
]
sent_keys = [sent_key for sent_key in all_sent_keys if sent_key in gd.df_dcs.columns]

df_clf_reports = {}
pipes = {}

for sent_key in sent_keys:

    if not USE_VARIANTS:
        if "var" in sent_key:
            continue

    if sent_key == "sent_gene_cna":
        unigram_weighter_method = "abs"
        unigram_weighter_pre_shift = 0.0
        unigram_weighter_post_shift = 0.0
        skipgram_weighter_method = "norm"
        skipgram_weighter_pre_shift = 0.0
        skipgram_weighter_post_shift = 0.0

    else:
        unigram_weighter_method = "identity"
        unigram_weighter_pre_shift = 0.0
        unigram_weighter_post_shift = 1.0
        skipgram_weighter_method = "one"
        skipgram_weighter_pre_shift = 0.0
        skipgram_weighter_post_shift = 1.0


    pipe = Pipeline(
        [
            (
                "pmi",
                embedders.NextgenlpPmiVectorizer(
                    unigram_weighter_method=unigram_weighter_method,
                    unigram_weighter_pre_shift=unigram_weighter_pre_shift,
                    unigram_weighter_post_shift=unigram_weighter_post_shift,
                    skipgram_weighter_method=skipgram_weighter_method,
                    skipgram_weighter_pre_shift=skipgram_weighter_pre_shift,
                    skipgram_weighter_post_shift=skipgram_weighter_post_shift,
                ),
            ),
            ("clf", LogisticRegression(n_jobs=-1, max_iter=2000)),
        ]
    )
    pipe.fit(df_train[sent_key], df_train[Y_PREDICT])
    y_pred = pipe.predict(df_test[sent_key])
    cls_report_dict = classification_report(
        df_test[Y_PREDICT],
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    df_clf_report = (
        pd.DataFrame(cls_report_dict)
        .drop(columns=["accuracy", "macro avg", "weighted avg"])
        .T
    ).sort_values("f1-score")
    df_clf_reports[(sent_key, "pmi")] = df_clf_report
    pipes[(sent_key, "pmi")] = pipe


    pipe = Pipeline(
        [
            (
                "cv",
                embedders.NextgenlpCountVectorizer(
                    min_unigram_weight=0,
                    unigram_weighter_method=unigram_weighter_method,
                    unigram_weighter_pre_shift=unigram_weighter_pre_shift,
                    unigram_weighter_post_shift=unigram_weighter_post_shift,
                ),
            ),
            ("clf", LogisticRegression(n_jobs=-1, max_iter=2000)),
        ]
    )
    pipe.fit(df_train[sent_key], df_train[Y_PREDICT])
    y_pred = pipe.predict(df_test[sent_key])
    pipes[sent_key] = pipe

    cls_report_dict = classification_report(
        df_test[Y_PREDICT],
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    df_clf_report = (
        pd.DataFrame(cls_report_dict)
        .drop(columns=["accuracy", "macro avg", "weighted avg"])
        .T
    ).sort_values("f1-score")
    df_clf_reports[sent_key] = df_clf_report
    pipes[(sent_key, "count")] = pipe



    sys.exit(1)


sent_key = "sent_gene_flat"
xcv = pipes[sent_key].named_steps["cv"].transform(gd.df_dcs[sent_key])
vecs = xcv.todense()
df_vecs = pd.DataFrame(vecs)
df_vecs.to_csv("vecs.tsv", sep="\t", header=None, index=False)

meta_cols = [
    "SAMPLE_ID",
    "SEQ_ASSAY_ID",
    "CANCER_TYPE",
    "CANCER_TYPE_DETAILED",
    "CENTER",
    "SAMPLE_TYPE",
    "SAMPLE_TYPE_DETAILED",
    "sent_var",
]
df_meta = gd.df_dcs.reset_index()[meta_cols]
df_meta.to_csv("meta.tsv", sep="\t", index=False)
