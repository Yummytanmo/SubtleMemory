from __future__ import annotations

from pathlib import Path

from infra.path_registry import archives_root, config_root, source_data_root


CATEGORY_CHOICES = ("complementary", "contradictory", "nuanced")

# Active assets participate in the integrated workflow and are the only files
# that generation/filter/repair commands should consume directly.
ACTIVE_CONFIG_FILES = {
    "complementary_generation": "complementary_generation.json",
    "contradictory_generation": "contradictory_generation.json",
    "conversation_types": "conversation_types.json",
    "filter_selection": "filter_selection.json",
    "nuanced_context_generation": "nuanced_context_generation.json",
    "nuanced_temporal_generation": "nuanced_temporal_generation.json",
}

ACTIVE_SOURCE_FILES = {
    "fanoutqa_kgt1": "complementary/fanoutqa_complementary_k_of_n.json",
    "musique_kgt1": "complementary/musique_complementary_k_of_n.json",
    "qacc_any_one": "complementary/qacc_complementary_any_one_of_n.json",
    "contradictory_source": "contradictory/contradictory_source.json",
    "context_light": "nuanced/light/ambig_context.json",
    "temporal_light": "nuanced/light/hoh_temporal.json",
}

# Archive assets are retained for audit or one-off maintenance flows and must
# stay separate from active benchmark inputs.
ARCHIVE_FILES = {
    "ambig_temporal": "ambiguous_context/ambig_temporal.json",
    "ambiguous_context_failed": "ambiguous_context/outputs_selected/nuanced/failed/context_10_samples.json",
    "ambiguous_context_passed": "ambiguous_context/outputs_selected/nuanced/passed/context_10_samples.json",
    "ambiguous_context_to_filter": "ambiguous_context/outputs_to_filter/nuanced/context_10_samples.json",
    "qacc_k1_failed": "qacc_k1/outputs_selected/complementary/failed/qacc_k1_10_samples.json",
    "qacc_k1_passed": "qacc_k1/outputs_selected/complementary/passed/qacc_k1_10_samples.json",
    "qacc_k1_to_filter": "qacc_k1/outputs_to_filter/complementary/qacc_k1_10_samples.json",
    "qacc_k1_source": "qacc_k1/source_data/complementary/qacc_complementary_k_of_n_k1.json",
}

COMPLEMENTARY_BATCH_SPECS = [
    ("qacc_any_one", "qacc_any_one_10_samples.json"),
    ("fanoutqa_kgt1", "fanoutqa_kgt1_10_samples.json"),
    ("musique_kgt1", "musique_kgt1_10_samples.json"),
]

CONTRADICTORY_BATCH_SPECS = [
    ("a_user_vs_user", "a_user_vs_user_10_samples.json"),
    ("b_user_vs_non_user", "b_user_vs_non_user_10_samples.json"),
    ("c_non_user_vs_non_user", "c_non_user_vs_non_user_10_samples.json"),
]

COMPLEMENTARY_FILE_TO_SOURCE = {
    filename: source for source, filename in COMPLEMENTARY_BATCH_SPECS
}

CONTRADICTORY_FILE_TO_SUBTYPE = {
    filename: subtype for subtype, filename in CONTRADICTORY_BATCH_SPECS
}

RECLASSIFIED_OUTPUT_FILENAME = "ambig_context_kgt1_reclassified.json"


def config_path(name: str) -> Path:
    return config_root() / ACTIVE_CONFIG_FILES[name]


def active_source_path(name: str) -> Path:
    return source_data_root() / ACTIVE_SOURCE_FILES[name]


def archive_path(name: str) -> Path:
    return archives_root() / ARCHIVE_FILES[name]


def active_config_paths() -> dict[str, Path]:
    return {name: config_path(name) for name in ACTIVE_CONFIG_FILES}


def active_source_paths() -> dict[str, Path]:
    return {name: active_source_path(name) for name in ACTIVE_SOURCE_FILES}


def archive_paths() -> dict[str, Path]:
    return {name: archive_path(name) for name in ARCHIVE_FILES}
