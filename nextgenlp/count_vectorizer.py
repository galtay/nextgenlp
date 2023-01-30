from collections import Counter
from typing import Callable, Dict, Iterable, Union

from loguru import logger
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from tqdm import tqdm


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
Sentences = Iterable[Iterable[tuple[str, Union[int, float]]]]


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
        min_df=1,
        max_df=1.0,
        unigram_weighter_method="identity",
        unigram_weighter_pre_shift=0.0,
        unigram_weighter_post_shift=0.0,
    ):
        """
        * min_df: unigrams with document frequency below this are dropped
        * max_df: unigrams with document frequency above this are dropped
        * unigram_weighter_method: method to transform sentence weight to counter weight
        * unigram_weighter_pre_shift: (see UnigramWeighter)
        * unigram_weighter_post_shift: (see UnigramWeighter)
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

    def calc_stats(self, arr):
        return {
            "min": np.min(arr),
            "max": np.max(arr),
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr),
            "var": np.var(arr),
        }

    def calculate_unigram_counts(self, X: Sentences) -> Counter:
        """Caclulate unigram counts from sentences."""
        unigram_counts = Counter()
        for sentence in tqdm(X, desc="calculating unigram counts"):
            for unigram, weight in sentence:
                unigram_counts[unigram] += 1
        return unigram_counts

    def build_count_matrix(
        self, X: Sentences, unigram_to_index: Dict[str, int]
    ) -> sparse.csr_matrix:
        row_indexs = []
        col_indexs = []
        dat_values = []
        for isamp, sent in enumerate(X):
            for unigram, weight in sent:
                row_indexs.append(isamp)
                col_indexs.append(unigram_to_index[unigram])
                dat_values.append(1)
        nrows = len(X)
        ncols = len(unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )

    def fit(self, X: Sentences, y=None):
        u_counts = self.calculate_unigram_counts(X)
        unigram_to_index = {u: i for i, (u, _) in enumerate(u_counts.most_common())}
        index_to_unigram = np.array([u for (u, _) in u_counts.most_common()])

        x_count = self.build_count_matrix(X, unigram_to_index)
        n_docs, n_unigrams = x_count.shape
        doc_count = np.array(x_count.sum(axis=0)).squeeze()
        doc_freq = doc_count / n_docs
        weights = np.array([w for sentence in X for (u, w) in sentence])
        logger.info(f"fix on X with unigram weights {self.calc_stats(weights)}")
        logger.info(
            f"fit on X with unigram document counts {self.calc_stats(doc_count)}"
        )
        logger.info(
            f"fit on X with unigram document frequencies {self.calc_stats(doc_freq)}"
        )

        rare_unigrams = set()
        if isinstance(self.min_df, int):
            rare_unigrams = set(index_to_unigram[np.where(doc_count < self.min_df)[0]])
        elif isinstance(self.min_df, float):
            rare_unigrams = set(index_to_unigram[np.where(doc_freq < self.min_df)[0]])
        logger.info(
            f"dropping {len(rare_unigrams)} unigrams using min_df={self.min_df}"
        )

        common_unigrams = set()
        if isinstance(self.max_df, int):
            common_unigrams = set(
                index_to_unigram[np.where(doc_count > self.max_df)[0]]
            )
        elif isinstance(self.max_df, float):
            common_unigrams = set(index_to_unigram[np.where(doc_freq > self.max_df)[0]])
        logger.info(
            f"dropping {len(common_unigrams)} unigrams using max_df={self.max_df}"
        )

        banned_unigrams = set.union(rare_unigrams, common_unigrams)

        # update vocab
        for unigram in banned_unigrams:
            del u_counts[unigram]

        unigram_to_index = {u: i for i, (u, _) in enumerate(u_counts.most_common())}
        index_to_unigram = np.array([u for u, _ in u_counts.most_common()])

        self.doc_freq = doc_freq
        self.doc_count = doc_count
        self.unigram_to_index = unigram_to_index
        self.index_to_unigram = index_to_unigram
        self.banned_unigrams = banned_unigrams

    def build_weights_matrix(
        self,
        X: Sentences,
        unigram_to_index: Dict[str, int],
        unigram_weighter: Callable[[float], float],
    ) -> sparse.csr_matrix:
        row_indexs = []
        col_indexs = []
        dat_values = []
        for isamp, sent in enumerate(X):
            for unigram, weight in sent:
                if unigram not in unigram_to_index:
                    continue
                row_indexs.append(isamp)
                col_indexs.append(unigram_to_index[unigram])
                dat_values.append(unigram_weighter(weight))
        nrows = len(X)
        ncols = len(unigram_to_index)
        return sparse.csr_matrix(
            (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
        )

    def transform(self, X: Sentences, y=None):
        weights = np.array([w for sentence in X for (u, w) in sentence])
        logger.info(f"transform on X with unigram weights {self.calc_stats(weights)}")
        return self.build_weights_matrix(
            X, self.unigram_to_index, self.unigram_weighter
        )

    def fit_transform(self, X: Sentences, y=None):
        self.fit(X)
        return self.transform(X)
