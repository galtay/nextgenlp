from pathlib import Path
from typing import Dict
from loguru import logger


GENIE_12p1 = "genie-12p1"
GENIE_13p1 = "genie-13p1"
VALID_GENIE_VERSIONS = [GENIE_12p1, GENIE_13p1]

DATASET_NAME_TO_SYNID = {
    GENIE_12p1: "syn32309524",
    GENIE_13p1: "syn51355584",
}

PATHOLOGY_SCORES = ["Polyphen", "SIFT"]

TOKEN_TYPES = ["gene", "var"]

SEQ_ASSAY_ID_GROUPS = {
    "DFCI-ONCOPANEL-3.1": ["DFCI-ONCOPANEL-3.1"],
    "MSK-IMPACT468": ["MSK-IMPACT468"],
    "UCSF-IDTV5-TO": ["UCSF-IDTV5-TO"],
    "DFCI": [
        "DFCI-ONCOPANEL-1",
        "DFCI-ONCOPANEL-2",
        "DFCI-ONCOPANEL-3",
        "DFCI-ONCOPANEL-3.1",
    ],
    "MSK": [
        "MSK-IMPACT341",
        "MSK-IMPACT410",
        "MSK-IMPACT468",
        "MSK-IMPACT505",
        "MSK-IMPACT-HEME-400",
    ],
    "MSK-NOHEME": ["MSK-IMPACT341", "MSK-IMPACT410", "MSK-IMPACT468", "MSK-IMPACT505"],
    "UCSF": ["UCSF-NIMV4-TO", "UCSF-NIMV4-TN", "UCSF-IDTV5-TO", "UCSF-IDTV5-TN"],
    "DFCI-MSK-UCSF": [
        "DFCI-ONCOPANEL-1",
        "DFCI-ONCOPANEL-2",
        "DFCI-ONCOPANEL-3",
        "DFCI-ONCOPANEL-3.1",
        "MSK-IMPACT341",
        "MSK-IMPACT410",
        "MSK-IMPACT468",
        "MSK-IMPACT505",
        "UCSF-NIMV4-TO",
        "UCSF-NIMV4-TN",
        "UCSF-IDTV5-TO",
        "UCSF-IDTV5-TN",
    ],
    "ALL": [
        "CHOP-COMPT",
        "CHOP-HEMEP",
        "CHOP-STNGS",
        "COLU-CCCP-V1",
        "COLU-CSTP-V1",
        "COLU-TSACP-V1",
        "CRUK-TS",
        "DFCI-ONCOPANEL-1",
        "DFCI-ONCOPANEL-2",
        "DFCI-ONCOPANEL-3",
        "DFCI-ONCOPANEL-3.1",
        "DUKE-F1-DX1",
        "DUKE-F1-T5A",
        "DUKE-F1-T7",
        "GRCC-CHP2",
        "GRCC-CP1",
        "GRCC-MOSC3",
        "GRCC-MOSC4",
        "JHU-500STP",
        "JHU-50GP",
        "MDA-409-V1",
        "MDA-46-V1",
        "MDA-50-V1",
        "MSK-IMPACT-HEME-400",
        "MSK-IMPACT341",
        "MSK-IMPACT410",
        "MSK-IMPACT468",
        "MSK-IMPACT505",
        "NKI-CHPV2-NGS",
        "NKI-CHPV2-SOCV2-NGS",
        "NKI-PATH-NGS",
        "NKI-TSACP-MISEQ-NGS",
        "PROV-FOCUS-V1",
        "PROV-TSO500HT-V2",
        "PROV-TST170-V1",
        "SCI-PMP68-V1",
        "UCHI-ONCOHEME55-V1",
        "UCHI-ONCOSCREEN50-V1",
        "UCSF-IDTV5-TN",
        "UCSF-IDTV5-TO",
        "UCSF-NIMV4-TN",
        "UCSF-NIMV4-TO",
        "UHN-48-V1",
        "UHN-50-V2",
        "UHN-54-V1",
        "UHN-555-BLADDER-V1",
        "UHN-555-BREAST-V1",
        "UHN-555-GLIOMA-V1",
        "UHN-555-GYNE-V1",
        "UHN-555-HEAD-NECK-V1",
        "UHN-555-LUNG-V1",
        "UHN-555-MELANOMA-V1",
        "UHN-555-PAN-GI-V1",
        "UHN-555-PROSTATE-V1",
        "UHN-555-RENAL-V1",
        "UHN-555-V1",
        "UHN-555-V2",
        "UHN-OCA-V3",
        "VHIO-BILIARY-V01",
        "VHIO-BRAIN-V01",
        "VHIO-BREAST-V01",
        "VHIO-BREAST-V02",
        "VHIO-COLORECTAL-V01",
        "VHIO-ENDOMETRIUM-V01",
        "VHIO-GASTRIC-V01",
        "VHIO-GENERAL-V01",
        "VHIO-HEAD-NECK-V01",
        "VHIO-KIDNEY-V01",
        "VHIO-LUNG-V01",
        "VHIO-OVARY-V01",
        "VHIO-PANCREAS-V01",
        "VHIO-PAROTIDE-V01",
        "VHIO-SKIN-V01",
        "VHIO-URINARY-BLADDER-V01",
        "VICC-01-D2",
        "VICC-01-DX1",
        "VICC-01-MYELOID",
        "VICC-01-SOLIDTUMOR",
        "VICC-01-T4B",
        "VICC-01-T5A",
        "VICC-01-T6B",
        "VICC-01-T7",
        "WAKE-CA-01",
        "WAKE-CA-NGSQ3",
        "WAKE-CLINICAL-AB1",
        "WAKE-CLINICAL-AB2",
        "WAKE-CLINICAL-AB3",
        "WAKE-CLINICAL-CF2",
        "WAKE-CLINICAL-CF3",
        "WAKE-CLINICAL-DX1",
        "WAKE-CLINICAL-R2D2",
        "WAKE-CLINICAL-T5A",
        "WAKE-CLINICAL-T7",
        "YALE-HSM-V1",
        "YALE-OCP-V2",
        "YALE-OCP-V3",
    ],
}


def get_file_name_to_path(
    sync_path: str,
    genie_version: str,
) -> Dict[str, Path]:

    """Return the paths to files in a GENIE dataset."""

    genie_path = Path(sync_path) / DATASET_NAME_TO_SYNID[genie_version]
    logger.info(f"genie_path={genie_path}")

    file_name_to_path = {
        "assay_information": genie_path / "assay_information.txt",
        "data_clinical_patient": genie_path / "data_clinical_patient.txt",
        "data_clinical_sample": genie_path / "data_clinical_sample.txt",
        "data_fusions": genie_path / "data_fusions.txt",
        "data_gene_matrix": genie_path / "data_gene_matrix.txt",
        "data_mutations_extended": genie_path / "data_mutations_extended.txt",
        "data_CNA": genie_path / "data_CNA.txt",
        "data_cna_hg19": genie_path / f"genie_data_cna_hg19.seg",
        "gene_panels": genie_path / "gene_panels",
        "genomic_information": genie_path / "genomic_information.txt",
    }

    return file_name_to_path
