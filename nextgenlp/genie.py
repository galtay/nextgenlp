from glob import glob
import os
from pathlib import Path
from typing import Dict

from loguru import logger
import pandas as pd

from nextgenlp.config import config
from nextgenlp import genie_constants


def get_file_name_to_path(
    sync_path: str = config["Paths"]["SYNAPSE_PATH"],
    genie_version: str = genie_constants.GENIE_12,
) -> Dict[str, Path]:

    """Return the paths to files in a GENIE dataset."""

    genie_path = Path(sync_path) / genie_constants.DATASET_NAME_TO_SYNID[genie_version]
    logger.info(f"genie_path={genie_path}")

    if genie_version == genie_constants.GENIE_12:
        file_name_to_path = {
            genie_constants.ASSAY_INFORMATION: genie_path
            / f"{genie_constants.ASSAY_INFORMATION}.txt",
            genie_constants.DATA_CLINICAL_PATIENT: genie_path
            / f"{genie_constants.DATA_CLINICAL_PATIENT}.txt",
            genie_constants.DATA_CLINICAL_SAMPLE: genie_path
            / f"{genie_constants.DATA_CLINICAL_SAMPLE}.txt",
            genie_constants.DATA_FUSIONS: genie_path
            / f"{genie_constants.DATA_FUSIONS}.txt",
            genie_constants.DATA_GENE_MATRIX: genie_path
            / f"{genie_constants.DATA_GENE_MATRIX}.txt",
            genie_constants.DATA_MUTATIONS_EXTENDED: genie_path
            / f"{genie_constants.DATA_MUTATIONS_EXTENDED}.txt",
            genie_constants.DATA_CNA: genie_path / f"{genie_constants.DATA_CNA}.txt",
            genie_constants.DATA_CNA_HG19_SEG: genie_path
            / f"genie_{genie_constants.DATA_CNA_HG19_SEG}.seg",
            genie_constants.GENE_PANELS: genie_path / genie_constants.GENE_PANELS,
            genie_constants.GENOMIC_INFORMATION: genie_path
            / f"{genie_constants.GENOMIC_INFORMATION}.txt",
        }

    elif genie_version == genie_constants.GENIE_13:
        file_name_to_path = {
            genie_constants.ASSAY_INFORMATION: genie_path
            / f"{genie_constants.ASSAY_INFORMATION}_13.3-consortium.txt",
            genie_constants.DATA_CLINICAL_PATIENT: genie_path
            / f"{genie_constants.DATA_CLINICAL_PATIENT}_13.3-consortium.txt",
            genie_constants.DATA_CLINICAL_SAMPLE: genie_path
            / f"{genie_constants.DATA_CLINICAL_SAMPLE}_13.3-consortium.txt",
            genie_constants.DATA_FUSIONS: genie_path
            / f"{genie_constants.DATA_FUSIONS}_13.3-consortium.txt",
            genie_constants.DATA_GENE_MATRIX: genie_path
            / f"{genie_constants.DATA_GENE_MATRIX}_13.3-consortium.txt",
            genie_constants.DATA_MUTATIONS_EXTENDED: genie_path
            / f"{genie_constants.DATA_MUTATIONS_EXTENDED}_13.3-consortium.txt",
            genie_constants.DATA_CNA: genie_path
            / f"{genie_constants.DATA_CNA}_13.3-consortium.txt",
            genie_constants.DATA_CNA_HG19_SEG: genie_path
            / f"genie_private_{genie_constants.DATA_CNA_HG19_SEG}_13.3-consortium.seg",
            genie_constants.GENE_PANELS: genie_path / genie_constants.GENE_PANELS,
            genie_constants.GENOMIC_INFORMATION: genie_path
            / f"{genie_constants.GENOMIC_INFORMATION}_13.3-consortium.txt",
        }

    else:
        raise ValueError(
            f"genie version must be one of {genie_constants.VALID_GENIE_VERSIONS}"
        )

    return file_name_to_path


class GenieData:
    def __init__(
        self,
        df_gp_wide,
        df_psm,
        df_dcs,
        filters,
        df_cna=None,
    ):

        self.df_gp_wide = df_gp_wide
        self.df_psm = df_psm
        self.df_dcs = df_dcs
        self.filters = filters
        self.df_cna = df_cna

        logger.info(self.__repr__())


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
        return set(self.df_psm['SAMPLE_ID'].unique())

    @classmethod
    def from_file_paths(cls, file_paths, include_cna=False):
        df_gp_wide = read_gene_panels(file_paths["gene_panels"], style="wide")

        df_dcp = read_clinical_patient(file_paths["data_clinical_patient"])
        df_dcs = read_clinical_sample(file_paths["data_clinical_sample"])
        df_dme = read_mutations_extended(file_paths["data_mutations_extended"])

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

        if include_cna:
            df_cna = read_cna(file_paths["data_CNA"])
        else:
            df_cna = None

        filters = {
            "seq_assay_ids": set(),
            "path_score": set(),
            "y_col": set(),
            "extra": set(),
        }
        return cls(
            df_gp_wide,
            df_psm,
            df_dcs,
            filters,
            df_cna = df_cna
        )


    def subset_to_variants(self):
        logger.info(f"creating subset for samples with variants")
        df_dcs = self.df_dcs.loc[self.df_psm["SAMPLE_ID"].unique()]
        self.filters["extra"] = self.filters["extra"] | set(["has_variant"])
        return GenieData(
            self.df_gp_wide,
            self.df_psm,
            df_dcs,
            self.filters,
            df_cna=self.df_cna,
        )


    def subset_to_cna(self):
        logger.info(f"creating subset for samples with discrete CNA data")
        sample_ids = set(self.df_dcs.index) & set(self.df_cna.index)
        if len(sample_ids) == 0:
            logger.warning("no sample ID overlap between dcs and discrete CNA data in this subset")

        df_cna = self.df_cna.loc[list(sample_ids)]
        bmask = df_cna.isnull().sum() == 0
        cna_genes = list(bmask.index[bmask])
        if len(cna_genes) == 0:
            logger.warning("no genes have full discrete CNA coverage in this subset")

        df_cna = df_cna[cna_genes]
        df_dcs = self.df_dcs.loc[df_cna.index]
        self.filters["extra"] = self.filters["extra"] | set(["has_cna"])
        return GenieData(
            self.df_gp_wide,
            self.df_psm,
            df_dcs,
            self.filters,
            df_cna = self.df_cna,
        )


    def subset_from_seq_assay_id_group(self, seq_assay_id_group):

        logger.info(f"creating subset for seq_assay_id_group={seq_assay_id_group}")
        if seq_assay_id_group == "ALL":
            raise ValueError("use from_file_paths to load ALL")

        seq_assay_ids = genie_constants.SEQ_ASSAY_ID_GROUPS[seq_assay_id_group]
        (sample_ids, seq_assay_genes) = get_genes_and_samples_from_seq_assay_ids(
            self.df_gp_wide, self.df_dcs, seq_assay_ids
        )
        df_dcs = self.df_dcs.loc[list(sample_ids)].copy()
        df_psm = self.df_psm[self.df_psm["SAMPLE_ID"].isin(df_dcs.index)].copy()
        df_psm = df_psm[df_psm["Hugo_Symbol"].isin(seq_assay_genes)]
        df_dcs = df_dcs.loc[df_psm["SAMPLE_ID"].unique()]
        df_gp_wide = self.df_gp_wide.loc[seq_assay_ids, list(seq_assay_genes)].copy()

        self.filters["seq_assay_ids"] = self.filters["seq_assay_ids"] | set(seq_assay_ids)

        return GenieData(
            df_gp_wide,
            df_psm,
            df_dcs,
            self.filters,
            df_cna=self.df_cna,
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

        self.filters["path_score"].add(path_score)

        return GenieData(
            self.df_gp_wide,
            df_psm,
            df_dcs,
            self.filters,
            df_cna=self.df_cna,
        )

    def subset_from_y_col(self, y_col, y_min_count):

        logger.info(f"creating subset for y_col={y_col}, y_min_count={y_min_count}")

        y_counts = self.df_dcs[y_col].value_counts()
        y_keep = y_counts[y_counts >= y_min_count].index
        df_dcs = self.df_dcs[self.df_dcs[y_col].isin(y_keep)].copy()
        df_psm = self.df_psm[self.df_psm["SAMPLE_ID"].isin(df_dcs.index)].copy()

        return GenieData(
            self.df_gp_wide,
            df_psm,
            df_dcs,
            self.seq_assay_id_group,
            self.seq_assay_genes,
            self.path_score,
            df_cna=self.df_cna,
        )

    def make_sentences(self):

        # create variant token
        self.df_psm["var_token"] = self.df_psm["Hugo_Symbol"] + "<>" + self.df_psm["HGVSp_Short"].fillna("")

        # make sentences
        # first we groupby then we replace NaN with empty list
        # the NaN will be there for samples with no mutations
        # ==============================================================
        self.df_dcs["gene_sent"] = self.df_psm.groupby("SAMPLE_ID")[
            "Hugo_Symbol"
        ].apply(list)
        self.df_dcs["gene_sent"] = self.df_dcs["gene_sent"].fillna("").apply(list)

        self.df_dcs["var_sent"] = self.df_psm.groupby("SAMPLE_ID")["var_token"].apply(
            list
        )
        self.df_dcs["var_sent"] = self.df_dcs["var_sent"].fillna("").apply(list)

        self.df_dcs["gene_sent_flat"] = self.df_dcs["gene_sent"].apply(
            lambda x: [(el, 1.0) for el in x]
        )

        self.df_dcs["var_sent_flat"] = self.df_dcs["var_sent"].apply(
            lambda x: [(el, 1.0) for el in x]
        )

        if self.path_score is not None:
            if self.path_score == "SIFT":
                # lower is more pathogenic for SIFT
                self.df_dcs["score_sent"] = self.df_psm.groupby("SAMPLE_ID")[
                    f"{self.path_score}_Score"
                ].apply(lambda x: [1.0 - el for el in list(x)])
            elif self.path_score == "Polyphen":
                self.df_dcs["score_sent"] = self.df_psm.groupby("SAMPLE_ID")[
                    f"{self.path_score}_Score"
                ].apply(list)

            self.df_dcs["gene_sent_score"] = self.df_dcs.apply(
                lambda x: list(zip(x["gene_sent"], x["score_sent"])), axis=1
            )
            self.df_dcs["var_sent_score"] = self.df_dcs.apply(
                lambda x: list(zip(x["var_sent"], x["score_sent"])), axis=1
            )

        check_cols = ["var_sent", "gene_sent_flat", "var_sent_flat"]
        if self.path_score is not None:
            check_cols = check_cols + [
                "score_sent",
                "gene_sent_score",
                "var_sent_score",
            ]
        for col in check_cols:
            assert (
                self.df_dcs["gene_sent"].apply(len) == self.df_dcs[col].apply(len)
            ).all()

    def __str__(self):
        return (
            "GenieData("
            "num_panels={}, "
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


def get_cna_norms(df_cna: pd.DataFrame, axis: int, k: int = 1, p: int = 2) -> pd.Series:
    """Vector Norms [sum(|c_i,j|^p)]**(k/p)

    k=1,p=2 is L2 norm
    axis=0 will do gene vectors
    axis=1 will do sample vectors

    this was the mapper lens used in
    https://www.pnas.org/doi/10.1073/pnas.1102826108

    TODO: do better imputation than just setting to 0
    """
    ser = (df_cna.fillna(0).abs() ** p).sum(axis=axis) ** (k / p)
    return ser


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
    df_melted = pd.melt(df, id_vars="SAMPLE_ID", var_name="hugo", value_name="dcna")
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


def get_cna_sentences(df_cna: pd.DataFrame, drop_nan=True, drop_zero=True) -> pd.Series:
    df_cna_melted = get_melted_cna(df_cna, drop_nan=drop_nan, drop_zero=drop_zero)
    cna_sentences = df_cna_melted.groupby("SAMPLE_ID").apply(
        lambda x: list(zip(x["hugo"], x["dcna"]))
    )
    return cna_sentences


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
    return sample_ids, genes
