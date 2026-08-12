"""
Local verification for the Heart Disease logistic regression project.

Runs every check that does NOT require an AWS account, so you can confirm the
repository is correct before opening SageMaker.

Usage (from the repository root):

    python verify_local.py            # fast checks only
    python verify_local.py --full     # also re-executes the notebook (~1-2 min)

Requires: numpy, pandas, matplotlib. The --full mode also needs nbformat + nbclient.
"""

import argparse
import json
import os
import subprocess
import sys

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
results = []


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append(bool(condition))
    print(f"{tag} {name}" + (f"  -> {detail}" if detail else ""))
    return bool(condition)


def warn(name, detail=""):
    print(f"{WARN} {name}" + (f"  -> {detail}" if detail else ""))


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
def check_files():
    section("1. Repository structure")
    required = [
        "heart.csv",
        "README.md",
        "heart_disease_lr_analysis.ipynb",
        "data/heart_train.csv",
        "data/heart_test.csv",
        "data/preprocessing_config.json",
        "sagemaker/train_heart_lr.py",
        "sagemaker/sagemaker_heart_lr.ipynb",
        "sagemaker/data/heart_train.csv",
        "sagemaker/data/heart_test.csv",
        "sagemaker/data/preprocessing_config.json",
    ]
    for path in required:
        check(f"exists: {path}", os.path.exists(path))

    if not os.path.isdir("images"):
        warn("images/ folder missing", "create it and add the 3 SageMaker screenshots")
    else:
        pngs = [f for f in os.listdir("images") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if len(pngs) >= 3:
            check("images/: at least 3 screenshots", True, f"{len(pngs)} found")
        else:
            warn(f"images/ has {len(pngs)} screenshot(s)",
                 "3 are required - add them after running SageMaker")


# ---------------------------------------------------------------------------
def check_dataset():
    section("2. Dataset integrity")
    import pandas as pd

    df = pd.read_csv("heart.csv")
    check("heart.csv loads", True, f"shape={df.shape}")

    expected_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    check("all 14 expected columns present",
          list(df.columns) == expected_cols,
          "columns differ" if list(df.columns) != expected_cols else "")

    check("no missing values", int(df.isna().sum().sum()) == 0)

    n_unique = len(df.drop_duplicates())
    check("302 unique patients after de-duplication", n_unique == 302,
          f"found {n_unique}")

    dups = int(df.duplicated().sum())
    print(f"       rows={len(df)}  duplicates={dups}  unique={n_unique}")

    binary_target = set(df["target"].unique()) <= {0, 1}
    check("target is binary {0,1}", binary_target)
    return df


# ---------------------------------------------------------------------------
def check_splits():
    section("3. Preprocessed splits (train/test leakage check)")
    import numpy as np
    import pandas as pd

    tr = pd.read_csv("data/heart_train.csv")
    te = pd.read_csv("data/heart_test.csv")

    check("train + test = 302 patients", len(tr) + len(te) == 302,
          f"{len(tr)} + {len(te)} = {len(tr) + len(te)}")
    check("70/30 split ratio", abs(len(tr) / 302 - 0.70) < 0.02,
          f"train fraction = {len(tr)/302:.3f}")

    # Stratification
    p_tr, p_te = tr["target"].mean(), te["target"].mean()
    check("split is stratified", abs(p_tr - p_te) < 0.03,
          f"positives train={p_tr:.3f} test={p_te:.3f}")

    # NO overlap between train and test - this is the critical one
    feat = [c for c in tr.columns if c != "target"]
    set_tr = set(map(tuple, np.round(tr[feat].values, 9)))
    overlap = sum(tuple(r) in set_tr for r in np.round(te[feat].values, 9))
    check("ZERO train/test overlap (no data leakage)", overlap == 0,
          f"{overlap} leaked rows" if overlap else "clean")

    # Normalization was fitted on train only
    check("train features standardized (mean~0)",
          np.abs(tr[feat].values.mean(axis=0)).max() < 1e-6)
    check("train features standardized (std~1)",
          np.abs(tr[feat].values.std(axis=0) - 1).max() < 1e-6)
    print(f"       test mean range: [{te[feat].values.mean(axis=0).min():.3f}, "
          f"{te[feat].values.mean(axis=0).max():.3f}]  (should NOT be exactly 0)")


# ---------------------------------------------------------------------------
def check_math():
    section("4. Model implementation (gradient check)")
    import numpy as np

    sys.path.insert(0, "sagemaker")
    import train_heart_lr as m

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (rng.random(40) > 0.5).astype(float)
    w = rng.normal(size=5) * 0.3
    b = 0.2
    lam = 0.7

    # sigmoid sanity
    check("sigmoid(0) == 0.5", abs(m.sigmoid(np.array([0.0]))[0] - 0.5) < 1e-12)
    check("sigmoid is bounded in (0,1)",
          0 < m.sigmoid(np.array([-800.0]))[0] and m.sigmoid(np.array([800.0]))[0] < 1 + 1e-12)

    # cost at w=0,b=0 must be log(2)
    J0 = m.compute_cost(np.zeros(5), 0.0, X, y, 0.0)
    check("cost at w=0,b=0 equals log(2)", abs(J0 - np.log(2)) < 1e-9, f"{J0:.10f}")

    # analytic vs numeric gradient
    dw, db = m.compute_gradient(w, b, X, y, lam)
    eps = 1e-6
    dw_num = np.zeros_like(w)
    for j in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        dw_num[j] = (m.compute_cost(wp, b, X, y, lam) -
                     m.compute_cost(wm, b, X, y, lam)) / (2 * eps)
    db_num = (m.compute_cost(w, b + eps, X, y, lam) -
              m.compute_cost(w, b - eps, X, y, lam)) / (2 * eps)

    err_w = float(np.max(np.abs(dw - dw_num)))
    err_b = float(abs(db - db_num))
    check("analytic gradient dw matches finite differences", err_w < 1e-6, f"max err={err_w:.2e}")
    check("analytic gradient db matches finite differences", err_b < 1e-6, f"err={err_b:.2e}")

    # bias must NOT be regularized
    _, db0 = m.compute_gradient(w, b, X, y, 0.0)
    check("bias b is NOT regularized", abs(db - db0) < 1e-15)

    # cost must decrease monotonically
    _, _, J_hist = m.gradient_descent(X, y, np.zeros(5), 0.0,
                                      alpha=0.1, lam=0.0, num_iters=300, print_every=0)
    diffs = np.diff(J_hist)
    check("cost decreases monotonically", bool((diffs <= 1e-12).all()),
          f"{int((diffs > 1e-12).sum())} increases")

    # metrics on a known confusion matrix
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0.])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0.])   # TP=3 FN=1 FP=1 TN=3
    ev = m.evaluate(y_true, y_pred)
    ok = (abs(ev["accuracy"] - 0.75) < 1e-9 and abs(ev["precision"] - 0.75) < 1e-9
          and abs(ev["recall"] - 0.75) < 1e-9 and abs(ev["f1"] - 0.75) < 1e-9)
    check("accuracy/precision/recall/F1 correct on a known case", ok, str(ev))


# ---------------------------------------------------------------------------
def check_training_script():
    section("5. SageMaker training script, run locally")
    import numpy as np

    out_dir = "_verify_model"
    cfg = json.load(open("data/preprocessing_config.json"))

    cmd = [sys.executable, "sagemaker/train_heart_lr.py",
           "--model-dir", out_dir, "--train", "data", "--test", "data",
           "--alpha", str(cfg["alpha"]), "--num-iters", str(cfg["num_iters"]),
           "--lam", str(cfg["lambda"])]
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    check("script exits with code 0", proc.returncode == 0,
          proc.stderr.strip()[-300:] if proc.returncode else "")
    check("script prints the completion marker",
          "TRAINING COMPLETED SUCCESSFULLY" in proc.stdout)
    check("model.npz written", os.path.exists(f"{out_dir}/model.npz"))
    check("metrics.json written", os.path.exists(f"{out_dir}/metrics.json"))

    if os.path.exists(f"{out_dir}/metrics.json"):
        got = json.load(open(f"{out_dir}/metrics.json"))["test"]
        exp = cfg["local_test_metrics"]
        diffs = {k: abs(got[k] - exp[k]) for k in ["accuracy", "precision", "recall", "f1"]}
        worst = max(diffs.values())
        check("test metrics reproduce the notebook exactly", worst < 1e-6,
              f"max diff = {worst:.2e}")
        print("       " + json.dumps({k: round(got[k], 4)
                                      for k in ["accuracy", "precision", "recall", "f1"]}))

    # cleanup
    if os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)


# ---------------------------------------------------------------------------
def check_sagemaker_data_copy():
    section("6. sagemaker/data matches data/")
    import hashlib

    for name in ["heart_train.csv", "heart_test.csv", "preprocessing_config.json"]:
        a = f"data/{name}"
        b = f"sagemaker/data/{name}"
        if not (os.path.exists(a) and os.path.exists(b)):
            check(f"{name} present in both", False)
            continue
        ha = hashlib.md5(open(a, "rb").read()).hexdigest()
        hb = hashlib.md5(open(b, "rb").read()).hexdigest()
        check(f"{name} identical in data/ and sagemaker/data/", ha == hb)


# ---------------------------------------------------------------------------
def check_notebook(full=False):
    section("7. Main notebook")
    try:
        import nbformat
    except ImportError:
        warn("nbformat not installed", "pip install nbformat nbclient  (skipping)")
        return

    nb = nbformat.read("heart_disease_lr_analysis.ipynb", as_version=4)
    check("notebook is valid JSON / nbformat", True, f"{len(nb.cells)} cells")

    n_err = sum(1 for c in nb.cells for o in c.get("outputs", [])
                if o.get("output_type") == "error")
    check("no error outputs stored", n_err == 0, f"{n_err} errors")

    n_plots = sum(1 for c in nb.cells for o in c.get("outputs", [])
                  if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
    check("plots are embedded in the notebook", n_plots >= 10, f"{n_plots} images")

    n_exec = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    check("all code cells were executed", n_exec == n_code, f"{n_exec}/{n_code}")

    if full:
        print("\n  Re-executing the notebook from scratch (this takes 1-2 minutes)...")
        try:
            from nbclient import NotebookClient
            nb2 = nbformat.read("heart_disease_lr_analysis.ipynb", as_version=4)
            NotebookClient(nb2, timeout=900, kernel_name="python3",
                           resources={"metadata": {"path": "."}}).execute()
            check("notebook re-executes end-to-end without errors", True)
        except Exception as e:
            check("notebook re-executes end-to-end without errors", False, str(e)[:250])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also re-execute the notebook from scratch")
    args = ap.parse_args()

    print("Heart Disease LR - local verification")
    print("Working directory:", os.getcwd())

    check_files()
    check_dataset()
    check_splits()
    check_math()
    check_training_script()
    check_sagemaker_data_copy()
    check_notebook(full=args.full)

    section("SUMMARY")
    total, passed = len(results), sum(results)
    print(f"{passed}/{total} checks passed")
    if passed == total:
        print("\nEverything works locally. You can move on to SageMaker (Step 5).")
        print("Remaining manual task: run sagemaker/sagemaker_heart_lr.ipynb in AWS,")
        print("save 3 screenshots into images/, and fill in the comparison row in README.md.")
    else:
        print(f"\n{total - passed} check(s) failed - review the [FAIL] lines above.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
