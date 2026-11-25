"""The only code that reads the test split (official ILSVRC-2012 val). PROTOCOL.md: test is
evaluated once per final run, after every choice that depends on validation has been made and
recorded. Each access is appended to TEST_ACCESS_LOG.jsonl (committed with the results), so
the paper can show exactly when the test set was looked at.

  python evaluate_test.py --data data/in100 --runs runs/anchor_b0_fixed_lr5e-4_ep100_s1 [more runs]

Refuses runs that are unfinished, smoke runs, tuning-seed runs (seed 0) and runs already
evaluated, unless --again "<reason>" is given (the reason goes into the log and the deviations
log in PROTOCOL.md).
"""
import argparse
import datetime
import json
import os
import subprocess

import torch

from train import Packed, evaluate, loader, make_model, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "TEST_ACCESS_LOG.jsonl")


def protocol_commit():
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", ":/PROTOCOL.md"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--again", default="")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    done = set()
    if os.path.exists(LOG):
        done = {json.loads(l)["run"] for l in open(LOG) if l.strip()}
    commit = protocol_commit()
    if commit is None:
        raise SystemExit("PROTOCOL.md is not committed; commit it before touching the test set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        result = json.load(open(os.path.join(run, "result.json")))
        cfg = json.load(open(os.path.join(run, "config.json")))["args"]
        name = result["name"]
        if result["status"] != "finished" or result["smoke"] or result["seed"] == 0:
            print(f"skip {name}: status {result['status']}, smoke {result['smoke']}, seed {result['seed']}")
            continue
        if name in done and not args.again:
            print(f"skip {name}: test already evaluated (see {LOG}); --again '<reason>' to repeat")
            continue
        _, test_tf = transforms(argparse.Namespace(**cfg))
        ds = Packed(args.data, "test", test_tf, cfg["classes"])
        model = make_model(cfg["model"], cfg["classes"] or len(json.load(open(os.path.join(args.data, "meta.json")))["classes"]),
                           cfg["drop_path"]).to(device)
        model.load_state_dict(torch.load(os.path.join(run, "final.pt"), map_location=device))
        metrics = evaluate(model, loader(ds, 256, args.workers, False), device)
        entry = {"run": name, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "protocol_commit": commit, "test": metrics, "val": result["final_val"], "again": args.again or None}
        with open(LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        with open(os.path.join(run, "test.json"), "w") as fh:
            json.dump(entry, fh, indent=1)
        print(f"{name}: test top-1 {metrics['top1']:.2f} (val {result['final_val']['top1']:.2f})")


if __name__ == "__main__":
    main()
