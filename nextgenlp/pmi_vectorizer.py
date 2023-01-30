from collections import Counter
import itertools
import math
import random
from typing import Callable, Dict, Iterable

from loguru import logger
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from tqdm import tqdm

from nextgenlp.count_vectorizer import NextgenlpCountVectorizer
from nextgenlp.count_vectorizer import UnigramWeighter


# type alias for sentences.
# for pathogenicity scores this could be,
# [
#   [(HER2, 1.0), (MSK, 0.5), ...],
#   [(GATA2, 0.2), ... ]
# ]
#
# for copy number alterations this could be,
# [
#   [(HER2, -2), (MSK, 0), ...],
#   [(GATA2, 1), ... ]
# ]
Sentences = Iterable[Iterable[tuple[str, float]]]


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
        return (
            self.call_me(weight_a + self.pre_shift, weight_b + self.pre_shift)
            + self.post_shift
        )


def random_int_except(a, b, no):
    """Generate a random integer between a and b (inclusive) avoiding no"""
    x = random.randint(a, b)
    while x == no:
        x = random.randint(a, b)
    return x


class NextgenlpPmiVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        min_df=1,
        max_df=1.0,
        unigram_weighter_method="identity",
        unigram_weighter_pre_shift=0.0,
        unigram_weighter_post_shift=0.0,
        skipgram_weighter_method="norm",
        skipgram_weighter_pre_shift=0.0,
        skipgram_weighter_post_shift=0.0,
        pmi_alpha=1.0,
        svd_keep=128,
        svd_p=1.0,
        transform_combines="svd",
    ):
        """
        * min_df: unigrams with document frequency below this are dropped
        * max_df: unigrams with document frequency above this are dropped
        * unigram_weighter_method: method to transform sentence weight to counter weight

        """
        self.min_df = min_df
        self.max_df = max_df
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

        self.pmi_alpha = pmi_alpha
        self.svd_keep = svd_keep
        self.svd_p = svd_p
        self.transform_combines = transform_combines

        self.cv = NextgenlpCountVectorizer(
            min_df=min_df,
            max_df=max_df,
            unigram_weighter_method=unigram_weighter_method,
            unigram_weighter_pre_shift=unigram_weighter_pre_shift,
            unigram_weighter_post_shift=unigram_weighter_post_shift,
        )

    def fit(self, X: Sentences, y=None):
        self.cv.fit(X)
        self.skipgram_weights = self.calculate_skipgrams(X, self.cv.unigram_to_index)
        self.skipgram_matrix = self.create_skipgram_matrix(
            self.skipgram_weights, self.cv.unigram_to_index
        )
        self.ppmi_matrix = self.calculate_ppmi_matrix(
            self.skipgram_matrix,
            self.skipgram_weights,
            self.cv.unigram_to_index,
            self.pmi_alpha,
        )

        svd_keep = min(self.svd_keep, self.ppmi_matrix.shape[0] - 1)
        self.svd_matrix = self.calculate_svd_matrix(svd_keep)

    def transform(self, X: Sentences, y=None):
        num_samples = len(X)
        if self.transform_combines == "svd":
            embd_size = self.svd_matrix.shape[1]
        elif self.transform_combines == "pmi":
            embd_size = self.ppmi_matrix.shape[1]
        else:
            raise ValueError()

        sample_vecs = np.zeros((num_samples, embd_size))

        for ii, (sample_id, full_sentence) in tqdm(
            enumerate(X.items()), desc="making sample vectors"
        ):
            sentence = [
                (unigram, weight)
                for (unigram, weight) in full_sentence
                if unigram in self.cv.unigram_to_index
            ]
            sample_vec = np.zeros(embd_size)
            norm = len(sentence) if len(sentence) > 0 else 1
            for unigram, weight in sentence:
                unigram_index = self.cv.unigram_to_index[unigram]
                if self.transform_combines == "svd":
                    unigram_vec = self.svd_matrix[unigram_index]
                elif self.transform_combines == "pmi":
                    unigram_vec = self.ppmi_matrix[unigram_index]
                sample_vec += unigram_vec
            sample_vec = sample_vec / norm
            sample_vecs[ii, :] = sample_vec
        return sample_vecs

    def fit_transform(self, X: Sentences, y=None):
        self.fit(X)
        return self.transform(X)

    def calculate_skipgrams(
        self, X: Sentences, unigram_to_index: Dict[str, int]
    ) -> Counter:
        """Caclulate unigrams and skipgrams from sentences."""

        logger.info(f"calculating skipgrams")

        skipgram_weights = Counter()
        for full_sentence in tqdm(X, desc="calculating skipgrams"):
            # filter out unigrams that are not in vocabulary
            sentence = [
                (unigram, weight)
                for (unigram, weight) in full_sentence
                if unigram in unigram_to_index
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
                        skipgram_weights[skipgram] += self.skipgram_weighter(
                            weight_a, weight_b
                        )

        logger.info("found {} unique skipgrams".format(len(skipgram_weights)))

        return skipgram_weights

    def create_skipgram_matrix(
        self, skipgram_weights, unigram_to_index
    ) -> sparse.csr_matrix:
        """Create a sparse skipgram matrix from a skipgram_weights counter.

        Input:
          * skipgram_weights: co-occurrence wrights between unigrams (counter)
          * unigram_to_index: maps unigram names to matrix indices
        """
        row_indexs = []
        col_indexs = []
        dat_values = []
        for (unigram_a, unigram_b), weight in tqdm(
            skipgram_weights.items(), desc="calculating skipgrams matrix"
        ):
            row_indexs.append(unigram_to_index[unigram_a])
            col_indexs.append(unigram_to_index[unigram_b])
            dat_values.append(weight)
        nrows = ncols = len(unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )

    def calculate_ppmi_matrix(
        self, skipgram_matrix, skipgram_weights, unigram_to_index, pmi_alpha
    ) -> sparse.csr_matrix:

        """Calculates positive pointwise mutual information from skipgrams.

        This function uses the notation from LGD15
        https://aclanthology.org/Q15-1016/

        """

        # for normalizing counts to probabilities
        DD = skipgram_matrix.sum()

        # #(w) is a sum over contexts and #(c) is a sum over words
        pound_w_arr = np.array(skipgram_matrix.sum(axis=1)).flatten()
        pound_c_arr = np.array(skipgram_matrix.sum(axis=0)).flatten()

        # for context distribution smoothing (cds)
        pound_c_alpha_arr = pound_c_arr**pmi_alpha
        pound_c_alpha_norm = np.sum(pound_c_alpha_arr)

        row_indxs = []
        col_indxs = []
        dat_values = []

        for skipgram in tqdm(
            skipgram_weights.items(),
            total=len(skipgram_weights),
            desc="calculating ppmi matrix",
        ):

            (word, context), pound_wc = skipgram
            word_indx = unigram_to_index[word]
            context_indx = unigram_to_index[context]

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
            pmi_alpha = np.log2(
                (pound_wc * pound_c_alpha_norm) / (pound_w * pound_c_alpha)
            )

            # turn pointwise mutual information into positive pointwise mutual information
            ppmi = max(pmi, 0)
            ppmi_alpha = max(pmi_alpha, 0)

            row_indxs.append(word_indx)
            col_indxs.append(context_indx)
            dat_values.append(ppmi_alpha)

        nrows = ncols = len(unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indxs, col_indxs)), shape=(nrows, ncols)
        )

    def calculate_svd_matrix(self, svd_keep) -> np.ndarray:
        """Singular Value Decomposition with eigenvalue weighting.

        See 3.3 of LGD15 https://aclanthology.org/Q15-1016/
        """
        uu, ss, vv = sparse.linalg.svds(self.ppmi_matrix, k=svd_keep)
        svd_matrix = uu.dot(np.diag(ss**self.svd_p))
        return svd_matrix
