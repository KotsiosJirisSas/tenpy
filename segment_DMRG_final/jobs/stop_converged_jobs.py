#!/usr/bin/env python3
"""
Watch the .out files of running DMRG jobs.

For every .out file in --jobs-dir whose job is still RUNNING (not finished):
  * if the last sweep has |Delta E| < --delta-e (default 1e-5):
        save the data of the last sweep to  <name_save>.pkl  (same keys as the result file of the
        run script), then cancel the job (scancel);
  * if the number of particles N > --max-n (default 120): cancel the job, whatever Delta E is
    (the data is also saved if Delta E is small).

Default is a DRY RUN (prints what it would do). Add --apply to really save + scancel.

    python3 stop_converged_jobs.py                      # dry run, one pass
    python3 stop_converged_jobs.py --apply              # save + cancel
    python3 stop_converged_jobs.py --apply --watch 600  # repeat every 10 minutes

The job id is taken from the .out file name (last run of >= 5 digits, e.g. slurm-13037795.out);
change --jobid-regex if your files are named differently.
"""
import argparse
import ast
import glob
import math
import os
import pickle
import re
import subprocess
import time

import numpy as np

SWEEP_KEYS = ["sweep", "N_updates", "E", "Delta_E", "S", "Delta_S", "max_S", "time",
              "max_trunc_err", "max_E_trunc", "max_chi", "norm_err"]


def parse_out(path):
    """Everything we need from a .out file (None if it has no completed sweep yet)."""
    t = open(path, errors="ignore").read()
    info = {"path": path, "finished": "Job output ends" in t or "Traceback" in t}

    # --- per-sweep statistics (printed at the end of every sweep) ---
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

    # --- number of particles: last Filling if there is one, else the initial 'number of electrons' ---
    m = re.search(r"^number of electrons:\s*(\S+)", t, re.M)
    info["N"] = float(info["density"].sum()) if n else (float(m.group(1)) if m else None)

    # --- header of the run ---
    def header(pattern, cast=str):
        m = re.search(pattern, t, re.M)
        return cast(m.group(1)) if m else None
    info["name_save"] = header(r"^name_save\s*:\s*(\S+)")
    info["L"] = header(r"^L\s*:\s*(\d+)", int)
    info["U0"] = header(r"^U0\s*:\s*(\S+)", float)
    info["x0"] = header(r"^x0 \(potential\)\s*:\s*(\S+)", float)
    info["w"] = header(r"^w\s+\(potential\)\s*:\s*(\S+)", float)

    # --- model_par: the dict printed by the run script (arrays turned into lists) ---
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

    # --- dmrg parameters the engine logged ("reading 'max_sweeps'=50") ---
    dmrg_params = {}
    for key, val in re.findall(r"reading '(\w+)'=(\S+)", t):
        try:
            dmrg_params[key] = ast.literal_eval(val)
        except Exception:
            dmrg_params[key] = val
    info["dmrg_params"] = dmrg_params
    return info


def make_result(info):
    """Same keys as the result .pkl of the run script. Entanglement data is not in the .out -> None."""
    L, x0, w, U0 = info["L"], info["x0"], info["w"], info["U0"]
    potential = None
    if None not in (L, x0, w, U0):          # N_layer = 1 is hardcoded in the run script
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


def is_running(jobid):
    out = subprocess.run(["squeue", "-j", jobid, "-h", "-o", "%T"], capture_output=True, text=True).stdout
    return out.strip().split("\n")[0].strip() == "RUNNING"


def process(path, args):
    info = parse_out(path)
    if info["finished"]:
        return
    small = info["n_sweeps"] > 0 and not math.isnan(info["delta_E"]) and abs(info["delta_E"]) < args.delta_e
    big = info["N"] is not None and info["N"] > args.max_n
    if not (small or big):
        return

    name = os.path.basename(path)
    jobid = get_jobid(path, args.jobid_regex)
    why = []
    if small:
        why.append(f"|Delta E| = {abs(info['delta_E']):.2e} < {args.delta_e:g} (sweep {info['sweep']})")
    if big:
        why.append(f"N = {info['N']:.3f} > {args.max_n:g}")
    print(f"[{name}] job {jobid}: " + "; ".join(why))
    if jobid is None:
        print("   no job id in the file name -> nothing done (use --jobid-regex)")
        return
    if not args.no_squeue and not is_running(jobid):
        print("   job is not RUNNING in squeue -> nothing done")
        return

    if small:
        target = info["name_save"] + ".pkl" if info["name_save"] else None
        if target is None:
            print("   no name_save in the file -> nothing done")
            return
        if os.path.exists(target):
            print(f"   {os.path.basename(target)} already exists -> not overwritten, nothing done")
            return
        if args.apply:
            with open(target, "wb") as f:
                pickle.dump(make_result(info), f)
            print(f"   saved {target}")
        else:
            print(f"   [dry run] would save {target}")
    if args.apply:
        subprocess.run(["scancel", jobid], check=True)
        print(f"   scancel {jobid}")
    else:
        print(f"   [dry run] would run: scancel {jobid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default="/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/jobs")
    ap.add_argument("--delta-e", type=float, default=1e-5, help="stop when |Delta E| of the last sweep is below this")
    ap.add_argument("--max-n", type=float, default=120, help="cancel the job when N is above this")
    ap.add_argument("--jobid-regex", default=r"(\d{5,})", help="regex for the job id in the .out file name (last match used)")
    ap.add_argument("--apply", action="store_true", help="really save and scancel (default: dry run)")
    ap.add_argument("--watch", type=int, default=0, help="repeat every this many seconds (0 = one pass)")
    ap.add_argument("--no-squeue", action="store_true", help="do not check that the job is RUNNING (for testing)")
    args = ap.parse_args()
    while True:
        for path in sorted(glob.glob(os.path.join(args.jobs_dir, "*.out"))):
            try:
                process(path, args)
            except Exception as e:
                print(f"[{os.path.basename(path)}] ERROR: {type(e).__name__}: {e}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
