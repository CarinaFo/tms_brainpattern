from pathlib import Path

# Example str
OLD_STR = "256"
NEW_STR = "254"


def rename_brainvision_trio(vhdr_path, new_name, dry_run=True):
    """
    Rename a BrainVision EEG trio (.vhdr, .vmrk, .eeg/.dat) safely,
    updating internal file references inside the text headers.

    vhdr_path : path to the .vhdr file
    new_name  : new base name (no extension)
    dry_run   : if True, just prints what would happen without touching files
    """
    vhdr_path = Path(vhdr_path)
    folder = vhdr_path.parent
    old_base = vhdr_path.stem

    vmrk_path = folder / f"{old_base}.vmrk"
    eeg_path = folder / f"{old_base}.eeg"
    if not eeg_path.exists():
        eeg_path = folder / f"{old_base}.dat"

    for p in (vhdr_path, vmrk_path, eeg_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    new_vhdr = folder / f"{new_name}.vhdr"
    new_vmrk = folder / f"{new_name}.vmrk"
    new_eeg = folder / f"{new_name}{eeg_path.suffix}"

    print(f"  {vhdr_path.name} -> {new_vhdr.name}")
    print(f"  {vmrk_path.name} -> {new_vmrk.name}")
    print(f"  {eeg_path.name} -> {new_eeg.name}")

    if dry_run:
        return

    # Patch internal references in the .vhdr header (points to vmrk + eeg)
    vhdr_text = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    vhdr_text = vhdr_text.replace(vmrk_path.name, new_vmrk.name)
    vhdr_text = vhdr_text.replace(eeg_path.name, new_eeg.name)

    # Patch internal reference in the .vmrk header (points to eeg)
    vmrk_text = vmrk_path.read_text(encoding="utf-8", errors="ignore")
    vmrk_text = vmrk_text.replace(eeg_path.name, new_eeg.name)

    # Write new header files, remove old ones
    new_vhdr.write_text(vhdr_text, encoding="utf-8")
    new_vmrk.write_text(vmrk_text, encoding="utf-8")
    vhdr_path.unlink()
    vmrk_path.unlink()

    # Rename the (large, binary) data file directly - no content rewrite needed
    eeg_path.rename(new_eeg)


def batch_rename_folder(folder, old_str=OLD_STR, new_str=NEW_STR, dry_run=True):
    folder = Path(folder)
    vhdr_files = sorted(folder.glob("*.vhdr"))

    if not vhdr_files:
        print(f"No .vhdr files found in {folder}")
        return

    print(f"Found {len(vhdr_files)} .vhdr file(s). {'DRY RUN - ' if dry_run else ''}Replacing '{old_str}' -> '{new_str}':\n")

    for vhdr_path in vhdr_files:
        old_base = vhdr_path.stem

        if old_base.startswith('1_'): # remove leading 1
            new_base = old_base[2:]
        if old_str not in old_base:
            print(f"Skipping (no '{old_str}' in name): {vhdr_path.name}")
            continue
        
        new_base = old_base.replace(old_base, new_base)
        rename_brainvision_trio(vhdr_path, new_base, dry_run=dry_run)
        print()


if __name__ == "__main__":

    folder = Path("L:/Lab_LucaC/A_QNC_ANT_Data/TMS_MDD_EEG_data/D_251")

    # Step 1: dry run to check the planned renames
    batch_rename_folder(folder, dry_run=True)

    # Step 2: once it looks correct, uncomment to actually rename
    #batch_rename_folder(folder, dry_run=False)