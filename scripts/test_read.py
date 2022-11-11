from nextgenlp import genie_constants
from nextgenlp import genie

syn_file_paths = genie.get_file_name_to_path(genie_version=genie_constants.GENIE_12)

# read everything but CNA
gd = genie.GenieData.from_file_paths(syn_file_paths, include_cna=True)
#gd.make_sentences()

gd1 = gd.subset_from_seq_assay_id_group("MSK-NOHEME")
gd2 = gd1.subset_to_cna()
gd2.make_sentences()

sys.exit(1)

gds = gd.subset_to_variants()


gd2 = gd1.subset_from_path_score("Polyphen")
gd3 = gd2.subset_from_path_score("SIFT")
gd4 = gd1.subset_from_y_col("CANCER_TYPE", 15)
gd1.make_sentences()
