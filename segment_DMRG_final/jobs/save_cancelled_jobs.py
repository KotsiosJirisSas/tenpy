#!/usr/bin/env python3
"""
Save the results of DMRG jobs that are no longer running (cancelled, killed by time/memory, crashed)
in the same format stop_converged_jobs.py uses:  <name_save>.pkl  with the data of the LAST sweep.

A .out file is saved when
  * it has at least one completed sweep (statistics + Filling printed), and
  * its job id is no longer in squeue (pending or running jobs are left alone), and
  * <name_save>.pkl does not exist yet (a finished job's own result, or an earlier save, is never overwritten).
There is no |Delta E| condition: whatever the last sweep was, it is stored.

Default is a DRY RUN. Add --apply to really write the files.

    python3 save_cancelled_jobs.py                       # dry run
    python3 save_cancelled_jobs.py --apply               # write the .pkl files
    python3 save_cancelled_jobs.py --apply --include-running   # also snapshot jobs that are still running
    python3 save_cancelled_jobs.py --apply --out-dir DIR # write DIR/<basename of name_save>.pkl instead
"""
import argparse
import ast
import glob
import math
import os
import pickle
import re
import subprocess

import numpy as np

SWEEP_KEYS = ["sweep", "N_updates", "E", "Delta_E", "S", "Delta_S", "max_S", "time",
              "max_trunc_err", "max_E_trunc", "max_chi", "norm_err"]


def parse_out(path):
    """Everything we need from a .out file (identical to stop_converged_jobs.py)."""
    t = open(path, errors="ignore").read()
    info = {"path": path,
            "ended": "Job output ends" in t,
            "traceback": "Traceback" in t}

    stats = {k: re.findall(rf"^{k}: (\S+)\s*$", t, re.M) for k in SWEEP_KEYS}
    filling = re.findall(r"^Filling:\s*\[([^\]]*)\]", t, re.M)
    n = min([len(v) for v in stats.values()] + [len(filling)])
    info["n_sweeps"] = n
    if n:
        info["sweep_stats"] = {k: [float(x) for x in stats[k][:n]] for k in SWEEP_KEYS}
        info["density"] = np.array(filling[n - 1].split(), dtype=float)
        info["energy"] = info["sweep_stats"]["E"][-1]
        info["delta_E"] = info["sweep_stats"]["Delta_E"][-1]
        info["sweep"] = int(info["sweep_stats"]["sweep"][-1])

    m = re.search(r"^number of electrons:\s*(\S+)", t, re.M)
    info["N"] = float(info["density"].sum()) if n else (float(m.group(1)) if m else None)

    def header(pattern, cast=str):
        m = re.search(pattern, t, re.M)
        return cast(m.group(1)) if m else None
    info["name_save"] = header(r"^name_save\s*:\s*(\S+)")
    info["L"] = header(r"^L\s*:\s*(\d+)", int)
    info["U0"] = header(r"^U0\s*:\s*(\S+)", float)
    info["x0"] = header(r"^x0 \(potential\)\s*:\s*(\S+)", float)
    info["w"] = header(r"^w\s+\(potential\)\s*:\s*(\S+)", float)

    info["model_par"] = None
    for line in t.splitlines():
        if line.startswith("{'verbose'") and "boundary_conditions" in line:
            try:
                mp = ast.literal_eval(re.sub(r"array\((\[[^\]]*\])\)", r"\1", line))
                if "root_config" in mp:
                    mp["root_config"] = np.array(mp["root_config"])
                info["model_par"] = mp
            except Exception:
                pass
            break

    dmrg_params = {}
    for key, val in re.findall(r"reading '(\w+)'=(\S+)", t):
        try:
            dmrg_params[key] = ast.literal_eval(val)
        except Exception:
            dmrg_params[key] = val
    info["dmrg_params"] = dmrg_params
    return info


def make_result(info):
    """Same keys as the result .pkl of the run script (identical to stop_converged_jobs.py)."""
    L, x0, w, U0 = info["L"], info["x0"], info["w"], info["U0"]
    potential = None
    if None not in (L, x0, w, U0):
        potential = (1 - np.tanh((np.arange(L) - x0) / w)) * U0 / 2
    return {
        "dmrg_params": info["dmrg_params"],
        "energy": info["energy"],
        "model_par": info["model_par"],
        "density": info["density"],
        "entanglement_entropy": None,
        "entanglement_spectrum": None,
        "potential": potential,
        "sweep_stats": info["sweep_stats"],
        "partial_from_out": {"file": info["path"], "sweep": info["sweep"], "delta_E": info["delta_E"]},
    }


def get_jobid(path, regex):
    ids = re.findall(regex, os.path.basename(path))
    return ids[-1] if ids else None


def jobs_in_queue():
    """All job ids of this user still in squeue (any state: PENDING, RUNNING, ...)."""
    r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%A"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("squeue failed (%s) -> stopping, nothing saved. Use --no-squeue to skip the check."
                         % r.stderr.strip())
    return set(r.stdout.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default="/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/jobs")
    ap.add_argument("--jobid-regex", default=r"(\d{5,})")
    ap.add_argument("--apply", action="store_true", help="really write the .pkl files (default: dry run)")
    ap.add_argument("--include-running", action="store_true",
                    help="also save jobs that are still in squeue (snapshot of their last sweep)")
    ap.add_argument("--no-squeue", action="store_true", help="do not ask squeue at all (treat every job as dead)")
    ap.add_argument("--out-dir", default=None,
                    help="write the .pkl here (basename of name_save) instead of next to name_save")
    args = ap.parse_args()

    queue = set() if (args.no_squeue or args.include_running) else jobs_in_queue()
    if args.out_dir and args.apply:
        os.makedirs(args.out_dir, exist_ok=True)

    count = {"saved": 0, "exists": 0, "in_queue": 0, "no_sweep": 0, "no_name": 0, "error": 0}
    for path in sorted(glob.glob(os.path.join(args.jobs_dir, "*.out"))):
        name = os.path.basename(path)
        try:
            info = parse_out(path)
            jobid = get_jobid(path, args.jobid_regex)
            if jobid in queue:
                count["in_queue"] += 1
                continue
            if info["n_sweeps"] == 0:
                count["no_sweep"] += 1
                continue
            if not info["name_save"]:
                count["no_name"] += 1
                print(f"[{name}] no name_save in the file -> skipped")
                continue
            base = info["name_save"] + ".pkl"
            target = os.path.join(args.out_dir, os.path.basename(base)) if args.out_dir else base
            if os.path.exists(target):
                count["exists"] += 1
                continue
            how = ("ended" if info["ended"] else "cut off") + (", traceback" if info["traceback"] else "")
            dE = info["delta_E"]
            dE_s = "nan" if math.isnan(dE) else f"{abs(dE):.2e}"
            msg = f"[{name}] job {jobid} ({how}): sweep {info['sweep']}, |Delta E| = {dE_s}, N = {info['N']:.3f}"
            if args.apply:
                with open(target, "wb") as f:
                    pickle.dump(make_result(info), f)
                print(msg + f"\n   saved {target}")
            else:
                print(msg + f"\n   [dry run] would save {target}")
            count["saved"] += 1
        except Exception as e:
            count["error"] += 1
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")

    print("\n%s: %d | already had a .pkl: %d | still in squeue: %d | no completed sweep: %d | "
          "no name_save: %d | errors: %d"
          % ("saved" if args.apply else "would save", count["saved"], count["exists"], count["in_queue"],
             count["no_sweep"], count["no_name"], count["error"]))


if __name__ == "__main__":
    main()
