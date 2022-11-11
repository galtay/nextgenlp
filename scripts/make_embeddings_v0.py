import os

import numpy as np
import pandas as pd
from loguru import logger

from nextgenlp import synapse
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp import embedders
from nextgenlp.config import config


SYNC_PATH = config["Paths"]["SYNAPSE_PATH"]
EMBEDDINGS_PATH = config["Paths"]["EMBEDDINGS_PATH"]
EMBEDDING_SIZE = 100

# we shift the Polyphen / SIFT scores from (0-1) to (1-2) so that
# we still keep some weight from the variants that have 0 scores
VAR_SHIFT = 1.0
MIN_UNIGRAM_WEIGHTS = {
    ("gene", "flat"): 0.0,
    ("var", "flat"): 0.0,
    ("gene", "Polyphen"): VAR_SHIFT + 0.0,
    ("gene", "SIFT"): VAR_SHIFT + 0.0,
    ("var", "Polyphen"): VAR_SHIFT + 0.0,
    ("var", "SIFT"): VAR_SHIFT + 0.0,
}


for genie_version in genie_constants.VALID_GENIE_VERSIONS:

    syn_file_paths = genie.get_file_name_to_path(genie_version=genie_version)

    logger.info(f"genie_version={genie_version}")
    gds = {}
    gds["ALL"] = genie.GenieData.from_file_paths(syn_file_paths)

    for seq_assay_id_group in genie_constants.SEQ_ASSAY_ID_GROUPS.keys():
        if seq_assay_id_group == "ALL":
            continue

        gds[seq_assay_id_group] = gds["ALL"].subset_from_seq_assay_id_group(
            seq_assay_id_group
        )

        for token_type in genie_constants.TOKEN_TYPES:
            # do flat var and gene here
            weighting = "flat"
            genie_subset = seq_assay_id_group
            embedding_subset = f"{token_type}-{weighting}-{EMBEDDING_SIZE}"
            tag = f"{genie_subset}-{token_type}-{weighting}"
            sent_col = f"{token_type}_sent_{weighting}"
            out_path = os.path.join(
                EMBEDDINGS_PATH, genie_version, genie_subset, embedding_subset
            )
            logger.info(
                f"genie_subset={genie_subset}, "
                f"embedding subset={embedding_subset}, "
                f"sent_col={sent_col}, "
                f"out_path={out_path}"
            )

            embds = embedders.PpmiEmbeddings(
                gds[genie_subset].df_dcs,
                min_unigram_weight=MIN_UNIGRAM_WEIGHTS[(token_type, weighting)],
                unigram_weighter=embedders.unigram_weighter_one,
                skipgram_weighter=embedders.skipgram_weighter_one,
                embedding_size=EMBEDDING_SIZE,
            )
            embds.create_embeddings(sent_col)
            embds.write_unigram_projector_files(
                out_path, tag, #  df_meta_extra=gds[genie_subset].dfs_meta_extra[token_type]
            )
            embds.write_sample_projector_files(out_path, tag, gds[genie_subset].df_dcs)
            sys.exit(1)
            for weighting in genie_constants.PATHOLOGY_SCORES:
                genie_subset = f"{seq_assay_id_group}-{weighting}"
                embedding_subset = f"{token_type}-{weighting}-{EMBEDDING_SIZE}"
                tag = f"{genie_subset}-{token_type}-{weighting}"
                sent_col = f"{token_type}_sent_score"
                out_path = os.path.join(
                    EMBEDDINGS_PATH, genie_version, genie_subset, embedding_subset
                )
                gds[genie_subset] = gds[seq_assay_id_group].subset_from_path_score(
                    weighting
                )
                logger.info(
                    f"genie_subset={genie_subset}, "
                    f"embedding subset={embedding_subset}, "
                    f"sent_col={sent_col}, "
                    f"out_path={out_path}"
                )

                embds = embedders.PpmiEmbeddings(
                    gds[genie_subset].df_dcs,
                    min_unigram_weight=MIN_UNIGRAM_WEIGHTS[(token_type, weighting)],
                    unigram_weighter=embedders.UnigramWeighter("identity", shift=VAR_SHIFT),
                    skipgram_weighter=embedders.SkipgramWeighter("norm", shift=VAR_SHIFT),
                    embedding_size=EMBEDDING_SIZE,
                )
                embds.create_embeddings(sent_col)
                embds.write_unigram_projector_files(
                    out_path, tag, df_meta_extra=gds[genie_subset].dfs_meta_extra[token_type]
                )
                embds.write_sample_projector_files(out_path, tag, gds[genie_subset].df_dcs)
