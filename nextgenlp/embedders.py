from collections import Counter
import itertools
import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm

from nextgenlp import genie


SEED = 29712836
random.seed(SEED)
MAX_PPMI_NDIM_WRITE = 10_000


def nPr(n, r=2):
    """Number of permuations of size 2"""
    return int(math.factorial(n) / math.factorial(n - r))


MAX_LEN = 1000
NUM_PERMS = {ii: nPr(ii) for ii in range(2, MAX_LEN + 1)}
SAMP_MULT = 6
PERM_RATIO = {ii: (SAMP_MULT * ii) / NUM_PERMS[ii] for ii in range(2, MAX_LEN + 1)}


def unigram_weighter_one(weight: float) -> float:
    return 1.0


def unigram_weighter_identity(weight: float) -> float:
    return weight


def unigram_weighter_abs(weight: float) -> float:
    return abs(weight)


class UnigramWeighter:
    def __init__(self, method: str, pre_shift: float = 0.0, post_shift: float = 0.0):
        self.method = method
        self.pre_shift = pre_shift
        self.post_shift = post_shift

        if method == "one":
            self.call_me = unigram_weighter_one
        elif method == "identity":
            self.call_me = unigram_weighter_identity
        elif method == "abs":
            self.call_me = unigram_weighter_abs
        else:
            raise ValueError("method not recognized")

    def __call__(self, weight: float) -> float:
        return self.call_me(weight + self.pre_shift) + self.post_shift


class NextgenlpCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        min_unigram_weight=0,
        max_unigram_weight=None,
        unigram_weighter_method="identity",
        unigram_weighter_pre_shift=0.0,
        unigram_weighter_post_shift=0.0,
    ):
        """
        * min_unigram_weight: discard unigrams with total corpus weight below this
        * max_unigram_weight: discard unigrams with total corpus weight above this
        * unigram_weighter_method: method to transform sentence weight to counter weight
        * unigram_weighter_shift: global shift to add to unigram weighter
        """
        self.min_unigram_weight = min_unigram_weight
        self.max_unigram_weight = max_unigram_weight
        self.unigram_weighter_method = unigram_weighter_method
        self.unigram_weighter_pre_shift = unigram_weighter_pre_shift
        self.unigram_weighter_post_shift = unigram_weighter_post_shift
        self.unigram_weighter = UnigramWeighter(
            unigram_weighter_method,
            pre_shift=unigram_weighter_pre_shift,
            post_shift=unigram_weighter_post_shift,
        )

    def calc_stats(self, weights):
        return {
            "min": np.min(weights),
            "max": np.max(weights),
            "mean": np.mean(weights),
            "median": np.median(weights),
            "std": np.std(weights),
            "var": np.var(weights),
        }

    def fit(self, X, y=None):
        sentence_weights = np.array(
            [weight for sentence in X for (unigram, weight) in sentence]
        )
        x_stats = self.calc_stats(sentence_weights)
        logger.info(f"fitting on X with sentence weight stats {x_stats}")

        unigram_weights = self.calculate_unigram_weights(X)
        stop_words = self.calculate_stop_words(unigram_weights)

        u_most_common = unigram_weights.most_common()
        u_weights = np.array([weight for (unigram, weight) in u_most_common])
        u_stats = self.calc_stats(u_weights)
        logger.info(f"unigram_weights stats {u_stats}")

        unigram_to_index = {
            unigram: ii
            for ii, (unigram, _) in enumerate(u_most_common)
            if unigram not in stop_words
        }
        index_to_unigram = np.array(
            [unigram for (unigram, _) in u_most_common if unigram not in stop_words]
        )

        self.unigram_weights = unigram_weights
        self.stop_words = stop_words
        self.unigram_to_index = unigram_to_index
        self.index_to_unigram = index_to_unigram

    def transform(self, X, y=None):
        sentence_weights = np.array(
            [weight for sentence in X for (unigram, weight) in sentence]
        )
        x_stats = self.calc_stats(sentence_weights)
        logger.info(f"transforming on X with sentence weight stats {x_stats}")

        row_indexs = []
        col_indexs = []
        dat_values = []
        for isamp, sent in enumerate(X):
            for unigram, weight in sent:
                if unigram in self.stop_words:
                    continue
                row_indexs.append(isamp)
                col_indexs.append(self.unigram_to_index[unigram])
                dat_values.append(self.unigram_weighter(weight))
        nrows = len(X)
        ncols = len(self.unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def calculate_unigram_weights(self, X: pd.Series) -> Counter:
        """Caclulate unigram weights from sentences."""
        unigram_weights = Counter()
        for sentence in tqdm(X, desc="calculating unigrams"):
            for unigram, weight in sentence:
                unigram_weights[unigram] += self.unigram_weighter(weight)
        logger.info("found {} unique unigrams".format(len(unigram_weights)))
        return unigram_weights

    def calculate_stop_words(self, unigram_weights: Counter) -> set:
        """Filter a counter removing values below `min_weight` and values above `max_weight`"""
        stop_words = set()
        for unigram, weight in unigram_weights.items():
            drop_min = (
                self.min_unigram_weight is not None and weight < self.min_unigram_weight
            )
            drop_max = (
                self.max_unigram_weight is not None and weight > self.max_unigram_weight
            )
            if drop_min or drop_max:
                stop_words.add(unigram)

        logger.info(
            "dropped {} unigrams after filtering by min_unigram_weight={}, max_unigram_weight={}".format(
                len(stop_words),
                self.min_unigram_weight,
                self.max_unigram_weight,
            )
        )
        return stop_words


def skipgram_weighter_one(weight_a: float, weight_b: float) -> float:
    return 1.0


def skipgram_weighter_product(weight_a: float, weight_b: float) -> float:
    return weight_a * weight_b


def skipgram_weighter_norm(weight_a: float, weight_b: float) -> float:
    return math.sqrt(weight_a**2 + weight_b**2)


class SkipgramWeighter:
    def __init__(self, method: str, pre_shift: float = 0.0, post_shift: float = 0.0):
        self.method = method
        self.pre_shift = pre_shift
        self.post_shift = post_shift

        if method == "one":
            self.call_me = skipgram_weighter_one
        elif method == "product":
            self.call_me = skipgram_weighter_product
        elif method == "norm":
            self.call_me = skipgram_weighter_norm
        else:
            raise ValueError("method not recognized")

    def __call__(self, weight_a: float, weight_b: float) -> float:
        return self.call_me(weight_a + self.pre_shift, weight_b + self.pre_shift) + self.post_shift


def random_int_except(a, b, no):
    """Generate a random integer between a and b (inclusive) avoiding no"""
    x = random.randint(a, b)
    while x == no:
        x = random.randint(a, b)
    return x



class NextgenlpPmiVectorizer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        min_unigram_weight=0,
        max_unigram_weight=None,
        unigram_weighter_method="identity",
        unigram_weighter_pre_shift=0.0,
        unigram_weighter_post_shift=0.0,
        skipgram_weighter_method="norm",
        skipgram_weighter_pre_shift=0.0,
        skipgram_weighter_post_shift=0.0,
        embedding_size=100,
        pmi_alpha=1.0,
        svd_p=1.0,
    ):
        """
        * min_unigram_weight: discard unigrams with total corpus weight below this
        * max_unigram_weight: discard unigrams with total corpus weight above this
        * unigram_weighter_method: method to transform sentence weight to counter weight

        """
        self.min_unigram_weight = min_unigram_weight
        self.max_unigram_weight = max_unigram_weight

        self.unigram_weighter_method = unigram_weighter_method
        self.unigram_weighter_pre_shift = unigram_weighter_pre_shift
        self.unigram_weighter_post_shift = unigram_weighter_post_shift
        self.unigram_weighter = UnigramWeighter(
            unigram_weighter_method,
            pre_shift=unigram_weighter_pre_shift,
            post_shift=unigram_weighter_post_shift,
        )

        self.skipgram_weighter_method = skipgram_weighter_method
        self.skipgram_weighter_pre_shift = skipgram_weighter_pre_shift
        self.skipgram_weighter_post_shift = skipgram_weighter_post_shift
        self.skipgram_weighter = SkipgramWeighter(
            skipgram_weighter_method,
            pre_shift=skipgram_weighter_pre_shift,
            post_shift=skipgram_weighter_post_shift,
        )

        self.embedding_size = embedding_size
        self.pmi_alpha = pmi_alpha
        self.svd_p = svd_p

        self.cv = NextgenlpCountVectorizer(
            min_unigram_weight=min_unigram_weight,
            max_unigram_weight=max_unigram_weight,
            unigram_weighter_method=unigram_weighter_method,
            unigram_weighter_pre_shift=unigram_weighter_pre_shift,
            unigram_weighter_post_shift=unigram_weighter_post_shift,
        )

    def fit(self, X, y=None):
        self.cv.fit(X)
        self.skipgram_weights = self.calculate_skipgrams(X)
        self.skipgram_matrix = self.create_skipgram_matrix()
        self.ppmi_matrix = self.calculate_ppmi_matrix()
        self.svd_matrix = self.calculate_svd_matrix()

    def transform(self, X, y=None):

        num_samples = X.size
        sample_vecs = np.zeros((num_samples, self.embedding_size))
        for ii, (sample_id, full_sentence) in tqdm(
            enumerate(X.items()), desc="making sample vectors"
        ):
            sentence = [
                (unigram, weight)
                for (unigram, weight) in full_sentence
                if unigram not in self.cv.stop_words
            ]
            sample_vec = np.zeros(self.embedding_size)
            norm = len(sentence) if len(sentence) > 0 else 1
            for unigram, weight in sentence:
                unigram_index = self.cv.unigram_to_index[unigram]
                unigram_vec = self.svd_matrix[unigram_index]
                sample_vec += unigram_vec
            sample_vec = sample_vec / norm
            sample_vecs[ii, :] = sample_vec
        return sample_vecs

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def calculate_skipgrams(self, X: pd.Series) -> Counter:
        """Caclulate unigrams and skipgrams from sentences."""

        logger.info(f"calculating skipgrams")

        skipgram_weights = Counter()
        for full_sentence in tqdm(X, desc="calculating skipgrams"):
            # filter out unigrams
            sentence = [
                (unigram, weight)
                for (unigram, weight) in full_sentence
                if unigram not in self.cv.stop_words
            ]
            num_toks = len(sentence)

            if num_toks < 2:
                continue

            # if sentence length is <= MAX_LEN take all permutations and normalize
            # if sentence length > MAX_LEN take SAMP_MULT random samples for each entry

            if num_toks <= MAX_LEN:
                perms = list(itertools.permutations(sentence, 2))
                length_norm = PERM_RATIO[num_toks]
                for (unigram_a, weight_a), (unigram_b, weight_b) in perms:
                    skipgram = (unigram_a, unigram_b)
                    skipgram_weights[skipgram] += (
                        self.skipgram_weighter(weight_a, weight_b) * length_norm
                    )
            else:
                for ii in range(num_toks):
                    unigram_a, weight_a = sentence[ii]
                    for nn in range(SAMP_MULT):
                        jj = random_int_except(0, num_toks - 1, ii)
                        unigram_b, weight_b = sentence[jj]
                        skipgram = (unigram_a, unigram_b)
                        skipgram_weights[skipgram] += self.skipgram_weighter(weight_a, weight_b)

        logger.info("found {} unique skipgrams".format(len(skipgram_weights)))

        return skipgram_weights

    def create_skipgram_matrix(self) -> sparse.csr_matrix:
        """Create a sparse skipgram matrix from a skipgram_weights counter.

        Input:
          * skipgram_weights: co-occurrence wrights between unigrams (counter)
          * unigram_to_index: maps unigram names to matrix indices
        """
        row_indexs = []
        col_indexs = []
        dat_values = []
        for (unigram_a, unigram_b), weight in tqdm(
            self.skipgram_weights.items(), desc="calculating skipgrams matrix"
        ):
            row_indexs.append(self.cv.unigram_to_index[unigram_a])
            col_indexs.append(self.cv.unigram_to_index[unigram_b])
            dat_values.append(weight)
        nrows = ncols = len(self.cv.unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )


    def calculate_ppmi_matrix(self) -> sparse.csr_matrix:

        """Calculates positive pointwise mutual information from skipgrams.

        This function uses the notation from LGD15
        https://aclanthology.org/Q15-1016/

        """

        # for normalizing counts to probabilities
        DD = self.skipgram_matrix.sum()

        # #(w) is a sum over contexts and #(c) is a sum over words
        pound_w_arr = np.array(self.skipgram_matrix.sum(axis=1)).flatten()
        pound_c_arr = np.array(self.skipgram_matrix.sum(axis=0)).flatten()

        # for context distribution smoothing (cds)
        pound_c_alpha_arr = pound_c_arr**self.pmi_alpha
        pound_c_alpha_norm = np.sum(pound_c_alpha_arr)

        row_indxs = []
        col_indxs = []
        dat_values = []

        for skipgram in tqdm(
            self.skipgram_weights.items(),
            total=len(self.skipgram_weights),
            desc="calculating ppmi matrix",
        ):

            (word, context), pound_wc = skipgram
            word_indx = self.cv.unigram_to_index[word]
            context_indx = self.cv.unigram_to_index[context]

            pound_w = pound_w_arr[word_indx]
            pound_c = pound_c_arr[context_indx]
            pound_c_alpha = pound_c_alpha_arr[context_indx]

            # this is how you would write the probabilities
            # Pwc = pound_wc / DD
            # Pw = pound_w / DD
            # Pc = pound_c / DD
            # Pc_alpha = pound_c_alpha / pound_c_alpha_norm

            # its more computationally effecient to use the counts directly
            # its less computationally efficient to also calculate the unsmoothed pmi
            # but we don't want it to feel left out
            pmi = np.log2((pound_wc * DD) / (pound_w * pound_c))
            pmi_alpha = np.log2((pound_wc * pound_c_alpha_norm) / (pound_w * pound_c_alpha))

            # turn pointwise mutual information into positive pointwise mutual information
            ppmi = max(pmi, 0)
            ppmi_alpha = max(pmi_alpha, 0)

            row_indxs.append(word_indx)
            col_indxs.append(context_indx)
            dat_values.append(ppmi_alpha)

        nrows = ncols = len(self.cv.unigram_to_index)
        return sparse.csr_matrix((dat_values, (row_indxs, col_indxs)), shape=(nrows, ncols))


    def calculate_svd_matrix(self) -> np.ndarray:
        """Singular Value Decomposition with eigenvalue weighting.

        See 3.3 of LGD15 https://aclanthology.org/Q15-1016/
        """
        lo_dim = min(self.embedding_size, self.ppmi_matrix.shape[0] - 1)
        uu, ss, vv = linalg.svds(self.ppmi_matrix, k=self.embedding_size)
        svd_matrix = uu.dot(np.diag(ss**self.svd_p))
        return svd_matrix


def calculate_sample_vectors(
    sentences: pd.Series,
    unigram_vecs,
    unigram_to_index,
    sample_to_index,
):

    """
    Input:
      * sentences: a pd.Series with lists of (unigram, weight) tuples.
      * unigram_vecs: unigram vecs to combine into sample vecs
      * unigram_to_index: unigram string -> matrix index
      * sample_to_index: sample ID -> matrix index
    """

    num_samples = sentences.size
    embedding_size = unigram_vecs.shape[1]
    sample_vecs = np.zeros((num_samples, embedding_size))
    for sample_id, full_sentence in tqdm(
        sentences.items(), desc="making sample vectors"
    ):
        sentence = [
            (unigram, weight)
            for (unigram, weight) in full_sentence
            if unigram in unigram_to_index
        ]
        sample_vec = np.zeros(embedding_size)
        norm = len(sentence) if len(sentence) > 0 else 1
        for unigram, weight in sentence:
            unigram_index = unigram_to_index[unigram]
            unigram_vec = unigram_vecs[unigram_index]
            sample_vec += unigram_vec
        sample_vec = sample_vec / norm
        sample_index = sample_to_index[sample_id]
        sample_vecs[sample_index, :] = sample_vec
    return sample_vecs


class PpmiEmbeddings:
    def __init__(
        self,
        df_dcs,
        min_unigram_weight,
        unigram_weighter,
        skipgram_weighter,
        embedding_size=100,
        ppmi_alpha=0.75,
        svd_p=1.0,
    ):

        self.df_dcs = df_dcs
        self.min_unigram_weight = min_unigram_weight
        self.unigram_weighter = unigram_weighter
        self.skipgram_weighter = skipgram_weighter
        self.embedding_size = embedding_size
        self.ppmi_alpha = ppmi_alpha
        self.svd_p = svd_p

    def create_embeddings(self, sent_col):

        unigram_weights = calculate_unigrams(
            self.df_dcs[sent_col],
            self.min_unigram_weight,
            self.unigram_weighter,
        )

        skipgram_weights = calculate_skipgrams(
            self.df_dcs[sent_col],
            unigram_weights,
            self.skipgram_weighter,
        )

        index_to_unigram = dict(enumerate(unigram_weights.keys()))
        unigram_to_index = {unigram: ii for ii, unigram in index_to_unigram.items()}
        skipgram_matrix = create_skipgram_matrix(skipgram_weights, unigram_to_index)

        ppmi_matrix = calculate_ppmi_matrix(
            skipgram_matrix,
            skipgram_weights,
            unigram_to_index,
            self.ppmi_alpha,
        )

        lo_dim = min(self.embedding_size, ppmi_matrix.shape[0] - 1)
        svd_matrix = calculate_svd_matrix(
            ppmi_matrix,
            lo_dim,
            self.svd_p,
        )

        index_to_sample = dict(enumerate(self.df_dcs.index))
        sample_to_index = {sample_id: ii for ii, sample_id in index_to_sample.items()}

        sample_vecs = calculate_sample_vectors(
            self.df_dcs[sent_col],
            svd_matrix,
            unigram_to_index,
            sample_to_index,
        )

        self.unigram_weights = unigram_weights
        self.skipgram_weights = skipgram_weights
        self.index_to_unigram = index_to_unigram
        self.unigram_to_index = unigram_to_index
        self.skipgram_matrix = skipgram_matrix
        self.ppmi_matrix = ppmi_matrix
        self.svd_matrix = svd_matrix
        self.index_to_sample = index_to_sample
        self.sample_to_index = sample_to_index
        self.sample_vecs = sample_vecs

    def write_unigram_projector_files(self, path, tag, df_meta_extra=None):

        os.makedirs(path, exist_ok=True)
        files_written = []

        # write out gene level embeddings
        # ====================================================================
        if self.ppmi_matrix.shape[0] < MAX_PPMI_NDIM_WRITE:
            fpath = os.path.join(path, f"{tag}_unigram_ppmi_vecs.tsv")
            files_written.append(fpath)
            df_vecs = pd.DataFrame(self.ppmi_matrix.todense())
            df_vecs.to_csv(fpath, sep="\t", index=False, header=False)
        else:
            logger.info(
                f"skipping PPMI vector write for shape {self.ppmi_matrix.shape}"
            )

        fpath = os.path.join(path, f"{tag}_unigram_svd_{self.embedding_size}_vecs.tsv")
        files_written.append(fpath)
        df_vecs = pd.DataFrame(self.svd_matrix)
        df_vecs.to_csv(fpath, sep="\t", index=False, header=False)

        # write out gene level metadata
        # ====================================================================

        # record unigram names -> index
        df_meta = pd.DataFrame(
            [self.index_to_unigram[ii] for ii in range(len(self.index_to_unigram))],
            columns=["unigram"],
        )

        # record unigram weights
        df_uweight = pd.DataFrame(
            self.unigram_weights.items(), columns=["unigram", "unigram_weight"]
        )
        df_meta = pd.merge(df_meta, df_uweight, on="unigram")

        df_meta["unigram_weight_norm"] = (
            df_meta["unigram_weight"] / df_meta["unigram_weight"].sum()
        )

        if df_meta_extra is not None:
            # add in extra metadata
            df_meta = pd.merge(df_meta, df_meta_extra, on="unigram", how="left")

        fpath = os.path.join(path, f"{tag}_unigram_meta.tsv")
        files_written.append(fpath)
        df_meta.to_csv(fpath, sep="\t", index=False)

        return files_written

    def write_sample_projector_files(self, path, tag, df_dcs):

        files_written = []

        # write out sample level embeddings
        # ====================================================================
        fpath = os.path.join(path, f"{tag}_sample_{self.embedding_size}_vecs.tsv")
        files_written.append(fpath)
        df_vecs = pd.DataFrame(self.sample_vecs)
        df_vecs.to_csv(
            fpath,
            sep="\t",
            index=False,
            header=False,
        )

        # write out sample level metadata
        # ====================================================================

        df_meta = pd.DataFrame(
            [self.index_to_sample[ii] for ii in range(len(self.index_to_sample))],
            columns=["SAMPLE_ID"],
        )

        # reocrd sample metadata from data clinical sample
        df_meta = pd.merge(
            df_meta,
            df_dcs,
            on="SAMPLE_ID",
            how="left",
        )

        # record sample metadata from data mutations extended
        df_tmp = df_dcs["gene_sent"].apply(set).to_frame("Hugos").reset_index()

        df_meta = pd.merge(
            df_meta,
            df_tmp,
            on="SAMPLE_ID",
        )

        df_meta["CENTER"] = df_meta["SAMPLE_ID"].apply(lambda x: x.split("-")[1])
        CENTER_CODES = ["DFCI", "MSK", "UCSF"]
        for center in CENTER_CODES:
            df_meta[f"{center}_flag"] = (df_meta["CENTER"] == center).astype(int)

        HUGO_CODES = ["NF1", "NF2", "SMARCB1", "LZTR1"]
        for hugo in HUGO_CODES:
            df_meta[f"{hugo}_mut"] = (
                df_meta["Hugos"].apply(lambda x: hugo in x).astype(int)
            )

        EXTRA_HUGO_CODES = ["KIT"]
        for hugo in EXTRA_HUGO_CODES:
            df_meta[f"{hugo}_mut"] = (
                df_meta["Hugos"].apply(lambda x: hugo in x).astype(int)
            )

        ONCOTREE_CODES = ["NST", "MPNST", "NFIB", "SCHW", "CSCHW", "MSCHW"]
        for oncotree in ONCOTREE_CODES:
            df_meta[f"{oncotree}_flag"] = (df_meta["ONCOTREE_CODE"] == oncotree).astype(
                int
            )

        df_meta["NST_CANCER_TYPE_FLAG"] = (
            df_meta["ONCOTREE_CODE"].isin(ONCOTREE_CODES)
        ).astype(int)

        EXTRA_ONCOTREE_CODES = ["GIST"]
        for oncotree in EXTRA_ONCOTREE_CODES:
            df_meta[f"{oncotree}_flag"] = (df_meta["ONCOTREE_CODE"] == oncotree).astype(
                int
            )

        df_meta["log_num_var"] = np.log10(df_meta["var_sent"].apply(len) + 1)

        to_drop = [
            "Hugos",
            "gene_sent",
            "gene_sent_flat",
            "var_sent_flat",
            "score_sent",
            "gene_sent_score",
            "var_sent_score",
        ]
        for col in to_drop:
            if col in df_meta.columns:
                df_meta = df_meta.drop(col, axis=1)

        fpath = os.path.join(path, f"{tag}_sample_meta.tsv")
        files_written.append(fpath)
        df_meta.to_csv(fpath, sep="\t", index=False)

        return files_written
