from copy import deepcopy
from glob import glob
import os
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
import pandas as pd


class GenieData:
    def __init__(
        self,
        df_gp_wide: pd.DataFrame,
        df_psm: pd.DataFrame,
        df_dcs: pd.DataFrame,
        filters: Dict[str, set],
        df_cna: Optional[pd.DataFrame] = None,
        seq_assay_id_to_cna_genes: Optional[Dict[str, set]] = None,
    ):

        """Class to handle GENIE datasets.

        Args:
          df_gp_wide: (panels x genes) 1 if panel includes gene, otherwise 0
          df_psm: merged patient, clinical, mutations data. one row per variant
          df_dcs: clinical sample data. one row per sample
          filters: dict indicating which filters have been applied
          df_cna: (samples x genes) discrete copy number alteration data
          seq_assay_id_to_cna_genes: maps seq_assay_ids to genes with CNA data

        Note:
          See `from_file_paths` to create an instance from GENIE files.
          panel == seq_assay_id
        """

        self.df_gp_wide = df_gp_wide
        self.df_psm = df_psm
        self.df_dcs = df_dcs
        self.filters = filters
        self.df_cna = df_cna
        self.seq_assay_id_to_cna_genes = seq_assay_id_to_cna_genes

        if df_cna is not None and seq_assay_id_to_cna_genes is None:
            self._calc_cna_metadata()

        logger.info(self.__repr__())

    def _calc_cna_metadata(self):
        logger.info("calculating CNA metadata")

        # get all genes with non-null values for each sample
        df_cna_melted = get_melted_cna(self.df_cna, drop_zero=False)
        df_cna_gene_sets = (
            df_cna_melted.groupby("SAMPLE_ID")["gene"]
            .apply(frozenset)
            .to_frame("gene_set")
        )

        # add the seq_assay_id used for each sample and the number of non-null genes
        df_cna_gene_sets["SEQ_ASSAY_ID"] = self.df_dcs["SEQ_ASSAY_ID"]
        df_cna_gene_sets["num_genes"] = df_cna_gene_sets["gene_set"].apply(len)

        # infer which genes have CNA measurements for each panel
        grpd_num = df_cna_gene_sets.groupby(["SEQ_ASSAY_ID", "num_genes"]).size()
        grpd_gene = df_cna_gene_sets.groupby(["SEQ_ASSAY_ID", "gene_set"]).size()

        seq_assay_ids_with_cna = (
            grpd_num.to_frame().reset_index()["SEQ_ASSAY_ID"].unique()
        )

        seq_assay_id_to_cna_genes = {}
        for seq_assay_id in self.df_dcs["SEQ_ASSAY_ID"].unique():
            if seq_assay_id in seq_assay_ids_with_cna:
                seq_assay_id_to_cna_genes[seq_assay_id] = set(self.df_cna.columns)
            else:
                seq_assay_id_to_cna_genes[seq_assay_id] = set()

        for (seq_assay_id, genes), count in grpd_gene.items():
            seq_assay_id_to_cna_genes[seq_assay_id] = (
                seq_assay_id_to_cna_genes[seq_assay_id] & genes
            )

        self.seq_assay_id_to_cna_genes = seq_assay_id_to_cna_genes

    @property
    def seq_assay_genes(self):
        return set(self.df_gp_wide.columns)

    @property
    def seq_assay_ids(self):
        return set(self.df_gp_wide.index)

    @property
    def psm_genes(self):
        return set(self.df_psm["Hugo_Symbol"].unique())

    @property
    def cna_genes(self):
        if self.df_cna is None:
            return set()
        else:
            return set(self.df_cna.columns)

    @property
    def sample_ids(self):
        return set(self.df_dcs.index)

    @property
    def sample_ids_with_variants(self):
        return set(self.df_psm["SAMPLE_ID"].unique())

    @classmethod
    def from_file_paths(
        cls,
        gene_panels: str,
        data_clinical_patient: str,
        data_clinical_sample: str,
        data_mutations_extended: str,
        data_CNA: Optional[str] = None,
    ):

        """Create GenieData instance from GENIE file paths.

        Args:
          gene_panels: path to directory containing all data_gene_panel_<SEQ_ASSAY_ID>.txt files
          data_clinical_patient: path to data_clinical_patient.txt file
          data_clinical_sample: path to data_clinical_sample.txt file
          data_mutations_extended: path to data_mutations_extended.txt file
          df_CNA: path to data_CNA.txt file

        """

        df_gp_wide = read_gene_panels(gene_panels, style="wide")
        df_dcp = read_clinical_patient(data_clinical_patient)
        df_dcs = read_clinical_sample(data_clinical_sample)
        df_dme = read_mutations_extended(data_mutations_extended)
        if data_CNA is None:
            df_cna = None
        else:
            df_cna = read_cna(data_CNA)

        logger.info("merging patient, sample, and mutations data")
        # reset index so that "SAMPLE_ID" will be included in final columns
        # drop "CENTER" b/c it's in df_dme
        df_psm = pd.merge(
            df_dme,
            df_dcs.reset_index().drop(columns=["CENTER"]),
            left_on="Tumor_Sample_Barcode",
            right_on="SAMPLE_ID",
        )
        df_psm = pd.merge(
            df_psm,
            df_dcp,
            on="PATIENT_ID",
        )

        filters = {
            "seq_assay_ids": set(),
            "path_score": set(),
            "y_col": set(),
            "extra": set(),
        }
        return cls(df_gp_wide, df_psm, df_dcs, filters, df_cna=df_cna)

    def subset_to_variants(self):
        """Return new GenieData with samples that dont have variants removed."""

        logger.info(f"creating subset for samples with variants")
        keep_sample_ids = self.df_psm["SAMPLE_ID"].unique()

        df_dcs = self.df_dcs.loc[keep_sample_ids]
        logger.info(
            "dropped {} samples with no variants".format(
                self.df_dcs.shape[0] - df_dcs.shape[0]
            )
        )

        if self.df_cna is None:
            df_cna = None
        else:
            df_cna = self.df_cna[self.df_cna.index.isin(keep_sample_ids)].copy()

        filters = deepcopy(self.filters)
        filters["extra"].add("has_variant")
        return GenieData(
            self.df_gp_wide,
            self.df_psm,
            df_dcs.copy(),
            filters,
            df_cna=df_cna,
            seq_assay_id_to_cna_genes=self.seq_assay_id_to_cna_genes,
        )

    def subset_from_seq_assay_ids(self, seq_assay_ids):
        """Filter out samples not tested with seq_assay_ids."""

        logger.info(f"creating subset from seq_assay_ids={seq_assay_ids}")

        # get samples that used these panels and gene set intersection
        (sample_ids, seq_assay_genes) = get_genes_and_samples_from_seq_assay_ids(
            self.df_gp_wide, self.df_dcs, seq_assay_ids
        )
        # filter out samples using other panels
        df_dcs = self.df_dcs.loc[list(sample_ids)]
        # filter out variants found with other panels
        df_psm = self.df_psm[self.df_psm["SAMPLE_ID"].isin(df_dcs.index)]
        # filter out variants outside of gene set intersection
        df_psm = df_psm[df_psm["Hugo_Symbol"].isin(seq_assay_genes)]
        # remove samples with no remaining variants
        df_dcs = df_dcs.loc[df_psm["SAMPLE_ID"].unique()]
        # update gene panel dataframe
        df_gp_wide = self.df_gp_wide.loc[seq_assay_ids, list(seq_assay_genes)]

        if self.df_cna is None:
            df_cna = None
        else:
            # intersection between samples from gene panels and CNA samples
            keep_sample_ids = set(self.df_cna.index) & set(df_dcs.index)

            # intersection of CNA genes from different panels
            keep_cna_genes = set.intersection(
                *[
                    self.seq_assay_id_to_cna_genes[seq_assay_id]
                    for seq_assay_id in seq_assay_ids
                ]
            )
            # intersection between above and current kept CNA genes
            keep_cna_genes = list(set(self.df_cna.columns) & keep_cna_genes)

            if len(keep_sample_ids) == 0:
                logger.warning(
                    f"no samples with CNA data for seq_assay_ids={seq_assay_ids}"
                )
                df_cna = self.df_cna.loc[[], []].copy()

            elif len(keep_cna_genes) == 0:
                logger.warning(
                    f"CNA gene intersection is 0 for seq_assay_ids={seq_assay_ids}"
                )
                df_cna = self.df_cna.loc[[], []].copy()

            else:
                df_cna = self.df_cna.loc[list(keep_sample_ids), list(keep_cna_genes)].copy()

        filters = deepcopy(self.filters)
        filters["seq_assay_ids"] = filters["seq_assay_ids"] | set(seq_assay_ids)

        return GenieData(
            df_gp_wide.copy(),
            df_psm.copy(),
            df_dcs.copy(),
            filters,
            df_cna=df_cna,
            seq_assay_id_to_cna_genes=self.seq_assay_id_to_cna_genes,
        )

    def subset_to_cna(self):
        """Return new GenieData with samples that dont have CNA data removed."""
        logger.info(f"creating subset for samples with discrete CNA data")
        if self.df_cna is None:
            raise ValueError("CNA data was not read so can not create subset")

        sample_ids = set(self.df_dcs.index) & set(self.df_cna.index)
        if len(sample_ids) == 0:
            logger.warning(
                "no sample ID overlap between dcs and discrete CNA data in this subset"
            )
            df_cna = self.df_cna.loc[[], []]
        df_cna = self.df_cna.loc[list(sample_ids)]

        # check genes that have full CNA coverage
        bmask = df_cna.isnull().sum(axis=0) == 0
        cna_genes = list(bmask.index[bmask])
        if len(cna_genes) == 0:
            logger.warning("no genes have full discrete CNA coverage in this subset")
            df_cna = self.df_cna.loc[[], []]
        df_cna = df_cna[cna_genes]

        # check samples that have full CNA coverage
        bmask = df_cna.isnull().sum(axis=1) == 0
        cna_samples = list(bmask.index[bmask])
        df_cna = df_cna.loc[cna_samples]

        df_dcs = self.df_dcs.loc[df_cna.index]
        df_psm = self.df_psm[self.df_psm["SAMPLE_ID"].isin(df_cna.index)]

        filters = deepcopy(self.filters)
        filters["extra"].add("has_cna")

        return GenieData(
            self.df_gp_wide,
            df_psm.copy(),
            df_dcs.copy(),
            filters,
            df_cna=df_cna.copy(),
            seq_assay_id_to_cna_genes=self.seq_assay_id_to_cna_genes,
        )

    def subset_from_path_score(self, path_score):

        logger.info(f"creating subset for path_score={path_score}")

        if path_score == "Polyphen":
            bmask = self.df_psm["Polyphen_Score"].isnull()
            df_psm = self.df_psm[~bmask]
            df_dcs = self.df_dcs.loc[df_psm["SAMPLE_ID"].unique()]

        elif path_score == "SIFT":
            bmask = self.df_psm["SIFT_Score"].isnull()
            df_psm = self.df_psm[~bmask]
            df_dcs = self.df_dcs.loc[df_psm["SAMPLE_ID"].unique()]

        else:
            raise ValueError(
                f"path_score must be Polyphen or SIFT, but got {path_score}"
            )

        if self.df_cna is None:
            df_cna = None
        else:
            df_cna = self.df_cna[self.df_cna.index.isin(df_dcs.index)].copy()

        filters = deepcopy(self.filters)
        filters["path_score"].add(path_score)

        return GenieData(
            self.df_gp_wide,
            df_psm.copy(),
            df_dcs.copy(),
            filters,
            df_cna=df_cna,
            seq_assay_id_to_cna_genes=self.seq_assay_id_to_cna_genes,
        )

    def subset_from_y_col(self, y_col, y_min_count):

        logger.info(f"creating subset for y_col={y_col}, y_min_count={y_min_count}")

        y_counts = self.df_dcs[y_col].value_counts()
        y_keep = y_counts[y_counts >= y_min_count].index
        df_dcs = self.df_dcs[self.df_dcs[y_col].isin(y_keep)]
        df_psm = self.df_psm[self.df_psm["SAMPLE_ID"].isin(df_dcs.index)]

        if self.df_cna is None:
            df_cna = None
        else:
            df_cna = self.df_cna[self.df_cna.index.isin(df_dcs.index)].copy()

        filters = deepcopy(self.filters)
        filters["y_col"].add((y_col, y_min_count))

        return GenieData(
            self.df_gp_wide,
            df_psm.copy(),
            df_dcs.copy(),
            filters,
            df_cna=df_cna,
            seq_assay_id_to_cna_genes=self.seq_assay_id_to_cna_genes,
        )

    def get_sent_norm(self, sent_key: str, k: int = 1, p: int = 2) -> pd.Series:
        """Vector Norms [sum(|c_i,j|^p)]**(k/p)

        k=1,p=2 is L2 norm

        this was the mapper lens used in
        https://www.pnas.org/doi/10.1073/pnas.1102826108

        """
        self.df_dcs[sent_key].apply(lambda x: sum([abs(w) ** p for t, w in x]) ** k / p)

    def make_sentences(self, reverse_sift=True):

        # create variant token
        self.df_psm["var_token"] = (
            self.df_psm["Hugo_Symbol"] + "<>" + self.df_psm["HGVSp_Short"].fillna("")
        )

        # make sentences
        # first we groupby then we replace NaN with empty list
        # the NaN will be there for samples with no variants
        # ==============================================================
        check_cols = ["sent_var", "sent_gene_flat", "sent_var_flat"]

        self.df_dcs["sent_gene"] = self.df_psm.groupby("SAMPLE_ID")[
            "Hugo_Symbol"
        ].apply(list)
        self.df_dcs["sent_gene"] = self.df_dcs["sent_gene"].fillna("").apply(list)

        self.df_dcs["sent_var"] = self.df_psm.groupby("SAMPLE_ID")["var_token"].apply(
            list
        )
        self.df_dcs["sent_var"] = self.df_dcs["sent_var"].fillna("").apply(list)

        self.df_dcs["sent_gene_flat"] = self.df_dcs["sent_gene"].apply(
            lambda x: [(el, 1.0) for el in x]
        )

        self.df_dcs["sent_var_flat"] = self.df_dcs["sent_var"].apply(
            lambda x: [(el, 1.0) for el in x]
        )

        if "Polyphen" in self.filters["path_score"]:
            self.df_dcs["sent_polyphen"] = self.df_psm.groupby("SAMPLE_ID")[
                "Polyphen_Score"
            ].apply(list)
            self.df_dcs["sent_gene_polyphen"] = self.df_dcs.apply(
                lambda x: list(zip(x["sent_gene"], x["sent_polyphen"])), axis=1
            )
            self.df_dcs["sent_var_polyphen"] = self.df_dcs.apply(
                lambda x: list(zip(x["sent_var"], x["sent_polyphen"])), axis=1
            )
            check_cols = check_cols + [
                "sent_polyphen",
                "sent_gene_polyphen",
                "sent_var_polyphen",
            ]

        if "SIFT" in self.filters["path_score"]:
            self.df_dcs["sent_sift"] = self.df_psm.groupby("SAMPLE_ID")[
                "SIFT_Score"
            ].apply(list)
            if reverse_sift:
                self.df_dcs["sent_sift"] = self.df_dcs["sent_sift"].apply(
                    lambda x: [1.0 - el for el in x]
                )
            self.df_dcs["sent_gene_sift"] = self.df_dcs.apply(
                lambda x: list(zip(x["sent_gene"], x["sent_sift"])), axis=1
            )
            self.df_dcs["sent_var_sift"] = self.df_dcs.apply(
                lambda x: list(zip(x["sent_var"], x["sent_sift"])), axis=1
            )
            check_cols = check_cols + [
                "sent_sift",
                "sent_gene_sift",
                "sent_var_sift",
            ]

        if "has_cna" in self.filters["extra"]:
            self.df_dcs["sent_cna"] = self.df_cna.apply(
                lambda x: [cna for gene, cna in x.items() if cna != 0], axis=1
            )
            self.df_dcs["sent_gene_cna"] = self.df_cna.apply(
                lambda x: [(gene, cna) for gene, cna in x.items() if cna != 0], axis=1
            )

        for col in check_cols:
            assert (
                self.df_dcs["sent_gene"].apply(len) == self.df_dcs[col].apply(len)
            ).all()

    def __str__(self):
        return (
            "GenieData("
            "num_seq_assay_ids={}, "
            "num_seq_assay_genes={}, "
            "num_psm_genes={}, "
            "sample_rows={}, "
            "variant_rows={}, "
            "filters={}"
            ")"
        ).format(
            len(self.seq_assay_ids),
            len(self.seq_assay_genes),
            len(self.psm_genes),
            self.df_dcs.shape[0],
            self.df_psm.shape[0],
            self.filters,
        )

    def __repr__(self):
        return self.__str__()


def read_gene_panel(gp_path: str) -> pd.DataFrame:
    """Read one data_gene_panel_<SEQ_ASSAY_ID>.txt file"""
    with open(gp_path, "r") as fp:
        lines = fp.readlines()
    gene_panel = lines[0].strip().split("stable_id:")[-1].strip()
    num_genes = int(lines[1].strip().split("Number of Genes - ")[-1])
    genes = lines[2].strip().split("\t")[1:]
    assert num_genes == len(genes)
    df = pd.DataFrame(genes, columns=["Hugo_Symbol"])
    df["SEQ_ASSAY_ID"] = gene_panel
    return df


def read_gene_panels(gp_path: str, style="wide") -> pd.DataFrame:
    """Read all data_gene_panel_<SEQ_ASSAY_ID>.txt files"""
    fpaths = glob(os.path.join(gp_path, "data_gene_panel*.txt"))
    logger.info(
        "reading gene panel data from {} files in {}".format(len(fpaths), gp_path)
    )
    dfs = [read_gene_panel(fpath) for fpath in fpaths]
    df = pd.concat(dfs).reset_index(drop=True)
    if style == "tall":
        return df
    elif style == "wide":
        df["value"] = 1
        df = (
            df.pivot(index="SEQ_ASSAY_ID", columns="Hugo_Symbol")["value"]
            .fillna(0)
            .astype(int)
        )
        return df
    else:
        raise ValueError(f"style must be 'wide' or 'tall', got {style}")


def read_data_gene_matrix(fpath: str) -> pd.DataFrame:
    """Read data_gene_matrix.txt file"""
    logger.info(f"reading data gene matrix from {fpath}")
    df = pd.read_csv(fpath, sep="\t")
    df = df.set_index("SAMPLE_ID", verify_integrity=True)
    return df


def read_genomic_information(fpath: str) -> pd.DataFrame:
    """Read genomic_information.txt file"""
    logger.info(f"reading genomic information from {fpath}")
    df = pd.read_csv(fpath, sep="\t")
    return df


def read_assay_information(fpath: str) -> pd.DataFrame:
    """Read assay_information.txt file"""
    logger.info(f"reading assay information from {fpath}")
    df = pd.read_csv(fpath, sep="\t")
    df = df.set_index("SEQ_ASSAY_ID", verify_integrity=True)
    return df


def read_data_fusions(fpath: str) -> pd.DataFrame:
    """Read data_fusions.txt file"""
    logger.info(f"reading data fusions from {fpath}")
    df = pd.read_csv(fpath, sep="\t")
    return df


def read_clinical_patient(fpath: str) -> pd.DataFrame:
    """Read data_clinical_patient.txt file"""
    logger.info(f"reading data clinical patient from {fpath}")
    df = pd.read_csv(fpath, sep="\t", comment="#")
    df = df.set_index("PATIENT_ID", verify_integrity=True)
    return df


def read_clinical_sample(fpath: str) -> pd.DataFrame:
    """Read data_clinical_sample.txt file"""
    logger.info(f"reading data clinical sample from {fpath}")
    df = pd.read_csv(fpath, sep="\t", comment="#")
    df["CENTER"] = df["SAMPLE_ID"].apply(lambda x: x.split("-")[1])
    df = df.set_index("SAMPLE_ID", verify_integrity=True)
    return df


def read_cna_seg(fpath: str) -> pd.DataFrame:
    """Read genie_data_cna_hf19.seg file"""
    logger.info(f"reading seg CNA data from {fpath}")
    return pd.read_csv(fpath, sep="\t")


def read_cna(fpath: str) -> pd.DataFrame:
    """Read data_CNA.txt file

    This is discrete copy number data
    This rearranges so that we have
    * N-sample rows
    * N-gene columns
    """
    logger.info(f"reading discrete CNA data from {fpath}")
    df = pd.read_csv(fpath, sep="\t").set_index("Hugo_Symbol").T.sort_index()
    df.index.name = "SAMPLE_ID"
    return df


def read_mutations_extended(fpath: str) -> pd.DataFrame:
    """Read a data_mutations_extended.txt file"""
    logger.info(f"reading data mutations extended from {fpath}")
    return pd.read_csv(
        fpath,
        dtype={
            "Entrez_Gene_Id": pd.Int64Dtype(),
            "Chromosome": str,
            "Reference_Allele": str,
            "Tumor_Seq_Allele1": str,
            "Tumor_Seq_Allele2": str,
            "Match_Norm_Seq_Allele1": str,
            "Match_Norm_Seq_Allele2": str,
            "Matched_Norm_Sample_Barcode": str,
            "FILTER": str,
        },
        sep="\t",
    )


def read_pat_sam_mut(
    patient_fpath: str, sample_fpath: str, mutations_fpath: str
) -> pd.DataFrame:
    """Read and join the,
    * data_clinical_patient
    * data_clinical_sample
    * data_mutations_extended
    """
    df_dcp = read_clinical_patient(patient_fpath)
    # reset index so that "SAMPLE_ID" will be included in final columns
    # drop "CENTER" b/c it's in df_dme
    df_dcs = read_clinical_sample(sample_fpath).reset_index().drop(columns=["CENTER"])
    df_dme = read_mutations_extended(mutations_fpath)

    logger.info("merging patient, sample, and mutations data")
    df_psm = pd.merge(
        df_dme,
        df_dcs,
        left_on="Tumor_Sample_Barcode",
        right_on="SAMPLE_ID",
    )
    df_psm = pd.merge(
        df_psm,
        df_dcp,
        on="PATIENT_ID",
    )

    return df_psm


def get_melted_cna(
    df_cna: pd.DataFrame, drop_nan: bool = True, drop_zero: bool = True
) -> pd.DataFrame:
    """Melt discrete copy number data

    This transforms an N-sample by N-gene CNA dataframe into a dataframe that has
    one row for each cell in the original matrix.
    """
    df = df_cna.copy()
    df["SAMPLE_ID"] = df.index
    df = df.reset_index(drop=True)
    df_melted = pd.melt(df, id_vars="SAMPLE_ID", var_name="gene", value_name="dcna")
    if drop_nan:
        df_melted = df_melted[~df_melted["dcna"].isnull()]
    if drop_zero:
        df_melted = df_melted[df_melted["dcna"] != 0]
    return df_melted


def dme_to_cravat(df: pd.DataFrame) -> pd.DataFrame:
    """Create Open Cravat dataframe from data_mutations_extended dataframe

    NOTE: some inspiration from civicpy
    https://github.com/griffithlab/civicpy/blob/master/examples/Project%20GENIE.ipynb
    """
    df_cravat = pd.DataFrame()
    df_cravat["CHROM"] = df["Chromosome"].apply(lambda x: f"chr{x}")
    df_cravat["POS"] = df["Start_Position"]
    df_cravat["STRAND"] = df["Strand"]
    df_cravat["REF"] = df["Reference_Allele"]
    df_cravat["ALT"] = df["Tumor_Seq_Allele2"]
    df_cravat["INDIVIDUAL"] = df["Tumor_Sample_Barcode"]

    # decide if we should use Allele1 or Allele2 for ALT
    # turns out bmask is never true in the data i have looked at
    bmask = (
        (df["Reference_Allele"] != df["Tumor_Seq_Allele1"])
        & (~df["Tumor_Seq_Allele1"].isnull())
        & (df["Tumor_Seq_Allele1"] != df["Tumor_Seq_Allele2"])
    )
    df_cravat["ALT"] = df["Tumor_Seq_Allele2"]
    df_cravat.loc[bmask, "ALT"] = df["Tumor_Seq_Allele1"]

    return df_cravat


def get_genes_and_samples_from_seq_assay_ids(df_gp_wide, df_dcs, seq_assay_ids):
    sample_ids = set()
    genes = set(df_gp_wide.columns)
    for seq_assay_id in seq_assay_ids:
        seq_assay_id_genes = set(
            [gene for (gene, flag) in df_gp_wide.loc[seq_assay_id].items() if flag == 1]
        )
        seq_assay_id_sample_ids = set(
            df_dcs[df_dcs["SEQ_ASSAY_ID"] == seq_assay_id].index
        )
        genes = genes & seq_assay_id_genes
        sample_ids.update(seq_assay_id_sample_ids)
    logger.info(
        f"filtering samples using gene set intersection from {seq_assay_ids}. "
        f"found {len(sample_ids)} samples and {len(genes)} genes."
    )
    return sample_ids, genes
