from pathlib import Path

root = Path(".")          # change to the folder you want
dry_run = False            # set to False to actually delete

for f in root.rglob("*.out"):   # use root.glob("*.out") to skip subfolders
    print(("Would delete: " if dry_run else "Deleting: ") + str(f))
    if not dry_run:
        f.unlink()