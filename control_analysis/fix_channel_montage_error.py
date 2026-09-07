"""
Rename ANT Neuro channel labels to Waveguard cap labels and export to FIF.
"""
from pathlib import Path
import mne


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("L:/Lab_LucaC/A_QNC_ANT_Data/TMS_MDD_EEG_acceleratedprotocol_data/D_250A4")
CNT_FILE = DATA_DIR / "D_250A4_1_2026-08-03_09-58-38.cnt"

# MNE convention: output filename should end in "_raw.fif" (or "-raw.fif")
# to avoid a naming-convention warning on save.
FIF_FILE = DATA_DIR / "D_250A4_1_2026-08-03_09-58-38.fif"

# {old_name: new_name}
CHANNEL_MAPPING = {
    'Fp1': '1Z', 'Fpz': '2Z', 'Fp2': '3Z',
    'F7': '4Z', 'F3': '6Z', 'Fz': '7Z', 'F4': '8Z', 'F8': '9Z',
    'FC5': '1L', 'FC1': '2L', 'FC2': '3L', 'FC6': '4L',
    'M1': '5L', 'T7': '6L', 'C3': '7L', 'Cz': '8L', 'C4': '9L',
    'T8': '10L', 'M2': '11L',
    'CP5': '1R', 'CP1': '2R', 'CP2': '3R', 'CP6': '4R',
    'P7': '5R', 'P3': '6R', 'Pz': '7R', 'P4': '8R', 'P8': '9R',
    'POz': '10R', 'O1': '11R', 'O2': '1LA',
    'EOG': '2LA', 'AF7': '3LA',
    'AF3': '1LB', 'AF4': '2LB', 'AF8': '3LB', 'F5': '4LB', 'F1': '5LB',
    'F2': '1LC', 'F6': '2LC', 'FC3': '3LC', 'FCz': '4LC', 'FC4': '5LC',
    'C5': '1LD', 'C1': '2LD', 'C2': '3LD', 'C6': '4LD',
    'CP3': '1RA', 'CP4': '2RA', 'P5': '3RA',
    'P1': '1RB', 'P2': '2RB', 'P6': '3RB', 'PO5': '4RB', 'PO3': '5RB',
    'PO4': '1RC', 'PO6': '2RC', 'FT7': '3RC', 'FT8': '4RC',
    'TP7': '5RC', 'TP8': '1RD', 'PO7': '2RD', 'PO8': '3RD', 'Oz': '4RD',
}


def rename_ant_channels(raw: mne.io.BaseRaw, mapping: dict) -> mne.io.BaseRaw:
    """Rename channels present in `raw`, warning about any mismatches."""
    ch_set = set(raw.ch_names)
    valid_mapping = {old: new for old, new in mapping.items() if old in ch_set}
    missing = set(mapping) - ch_set
    unmapped = ch_set - set(mapping)
    raw.rename_channels(valid_mapping)

    print(missing)
    print(unmapped)
    return raw


def main() -> None:
    if not CNT_FILE.exists():
        raise FileNotFoundError(f"CNT file not found: {CNT_FILE}")

    raw = mne.io.read_raw_cnt(CNT_FILE, preload=True)  # preload so save() writes actual data
    rename_ant_channels(raw, CHANNEL_MAPPING)
    raw.save(FIF_FILE, overwrite=True)


if __name__ == "__main__":
    main()