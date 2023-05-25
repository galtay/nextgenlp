"""
Run embedders and classifiers
"""
import os
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse
from tqdm import tqdm

from nextgenlp.config import config
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp.count_vectorizer import NextgenlpCountVectorizer
from nextgenlp.pmi_vectorizer import NextgenlpPmiVectorizer


USE_PATH_SCORES = True
USE_VARIANTS = True
USE_CNA = True
RANDOM_STATE = 9237
GENIE_VERSION = genie_constants.GENIE_13p1
EMBEDDINGS_PATH = config['Paths']["embeddings_path"]


synapse_directory = (
    Path(config["Paths"]["synapse_path"])
    / genie_constants.DATASET_NAME_TO_SYNID[GENIE_VERSION]
)

gds = {}
gds["ALL"] = genie.GenieData.from_synapse_directory(synapse_directory, read_cna=USE_CNA)

# create specific subset
Y_PREDICT = "CANCER_TYPE"
seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK-IMPACT468"]
# seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS["MSK-NOHEME"]

gds["BASE"] = (
    gds["ALL"]
    .subset_to_variants()
    .subset_from_seq_assay_ids(seq_assay_ids)
    .subset_from_y_col(Y_PREDICT, 50)
)


sent_keys = [
    "sent_gene_flat",
]
if USE_VARIANTS:
    sent_keys += ["sent_var_flat"]
if USE_PATH_SCORES:
    sent_keys += ["sent_gene_sift", "sent_gene_polyphen"]
if USE_PATH_SCORES and USE_VARIANTS:
    sent_keys += ["sent_var_sift", "sent_var_polyphen"]
if USE_CNA:
    sent_keys += ["sent_gene_cna"]


df_clf_reports = {}
pipes = {}

for sent_key in sent_keys:

    logger.info(f"using sent_key={sent_key}")
    gd = gds["BASE"]

    if "sift" in sent_key:
        gd = gd.subset_from_path_score("SIFT")

    if "polyphen" in sent_key:
        gd = gd.subset_from_path_score("Polyphen")

    if "cna" in sent_key:
        gd = gd.subset_to_cna().subset_to_cna_altered()

    gd.make_sentences()
    df_train, df_test = train_test_split(
        gd.df_dcs, stratify=gd.df_dcs[Y_PREDICT], random_state=RANDOM_STATE
    )


    # start with defaults
    min_df = 0.0
    unigram_weighter_method = "identity"
    unigram_weighter_pre_shift = 0.0
    unigram_weighter_post_shift = 0.0
    skipgram_weighter_method = "norm"
    skipgram_weighter_pre_shift = 0.0
    skipgram_weighter_post_shift = 0.0

    # update values as needed
#    if sent_key == "sent_gene_cna":
#        unigram_weighter_method = "abs"

    if "sift" in sent_key or "polyphen" in sent_key:
        unigram_weighter_post_shift = 1.0
        skipgram_weighter_pre_shift = 1.0

    if "var" in sent_key:
        # TODO: choose an informed threshold
        min_df = 2


    # count vectorizer + SVD then logistic regression
    #======================================================
    pipe = Pipeline([
        (
            "vec",
            Pipeline([
                (
                    "count",
                    NextgenlpCountVectorizer(
                        min_df = min_df,
                        unigram_weighter_method=unigram_weighter_method,
                        unigram_weighter_pre_shift=unigram_weighter_pre_shift,
                        unigram_weighter_post_shift=unigram_weighter_post_shift,
                    )
                ),
                (
                    "svd",
                    TruncatedSVD(n_components=200)
                ),
            ]),
        ),
        (
            "clf",
            LogisticRegression(n_jobs=-1, max_iter=2000),
        ),
    ])
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
    df_clf_reports[(sent_key, "count-svd")] = df_clf_report
    pipes[(sent_key, "count-svd")] = pipe


    # count vectorizer then logistic regression
    #======================================================
    pipe = Pipeline(
        [
            (
                "vec",
                NextgenlpCountVectorizer(
                    min_df = min_df,
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
    df_clf_reports[(sent_key, "count")] = df_clf_report
    pipes[(sent_key, "count")] = pipe


    # PPMI + SVD  then logistic regression
    #======================================================
    pipe = Pipeline(
        [
            (
                "vec",
                NextgenlpPmiVectorizer(
                    min_df = min_df,
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


    # write out vectors on full dataset
    #======================================================
    for vec_type in ["count", "count-svd", "pmi"]:

        tag = f"{sent_key}_{vec_type}"
        out_path = os.path.join(EMBEDDINGS_PATH, tag)
        os.makedirs(out_path, exist_ok=True)

        pipe = sklearn.base.clone(pipes[(sent_key, vec_type)])
        xcv = pipe.named_steps["vec"].fit_transform(gd.df_dcs[sent_key])
        if type(xcv) == np.matrix or type(xcv) == np.ndarray:
            vecs = xcv
        else:
            vecs = xcv.todense()
        df_vecs = pd.DataFrame(vecs)
        df_vecs.to_csv(
            os.path.join(out_path, f"{tag}_vecs.tsv"),
            sep="\t",
            header=None,
            index=False,
        )

        df_dcs = gd.df_dcs.reset_index().copy()
        meta_cols = [
            "SAMPLE_ID",
            "PATIENT_ID",
            "SEQ_ASSAY_ID",
            "ONCOTREE_CODE",
            "CANCER_TYPE",
            "CANCER_TYPE_DETAILED",
            "CENTER",
            "SAMPLE_TYPE",
            "SAMPLE_TYPE_DETAILED",
            "sent_var",
        ]
        if "sent_gene_cna" in df_dcs.columns:
            meta_cols += ["sent_gene_cna"]

        df_meta = df_dcs[meta_cols].copy()

        HUGO_CODES = ["NF1", "NF2", "SMARCB1", "LZTR1"]
        for hugo in HUGO_CODES:
            df_meta[f"flag_{hugo}"] = (
                df_dcs["sent_gene"].apply(lambda x: hugo in x).astype(int)
            )

        ONCOTREE_CODES = ["NST", "MPNST", "NFIB", "SCHW", "CSCHW", "MSCHW"]
        for oncotree in ONCOTREE_CODES:
            df_meta[f"flag_{oncotree}"] = (df_dcs["ONCOTREE_CODE"] == oncotree).astype(
                int
            )

        df_meta.to_csv(
            os.path.join(out_path, f"{tag}_meta.tsv"),
            sep="\t",
            index=False,
        )

        df_clf_reports[(sent_key, vec_type)].reset_index().rename(columns={"index": Y_PREDICT}).to_csv(
            os.path.join(out_path, f"{tag}_clf_report.csv"),
            index=False,
        )


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


clf = pipes[('sent_gene_flat', 'count')].named_steps['clf']
class_name = 'Gastrointestinal Stromal Tumor'
feature_names = pipes[('sent_gene_flat', 'count')].named_steps['vec'].index_to_unigram
fi = get_clf_feature_importance(clf, class_name, feature_names)
