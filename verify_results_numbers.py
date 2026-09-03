"""Cross-check the numbers written in RESULTS.md against the committed JSON
outputs they came from. Run after any re-run of the analysis scripts.

Covers:
  Phase 3 - ADNI external validation (outputs/adni_external/)
  Phase 4 - Grad-CAM               (figures_gradcam*/gradcam_summary.json)
  Phase 5 - Temperature scaling    (outputs/temperature/)

Usage:  python tools/verify_results_numbers.py
"""
import json
import os
import sys

FAILURES = []
CHECKED = 0


def check(name, claimed, actual, tol=5e-5):
    global CHECKED
    CHECKED += 1
    if actual is None:
        FAILURES.append(f"{name}: not found in JSON")
    elif abs(claimed - actual) > tol:
        FAILURES.append(f"{name}: RESULTS.md={claimed} JSON={actual:.4f}")


# ---------------------------------------------------------------- Phase 3
def verify_phase3():
    path = "outputs/adni_external/adni_external_results.json"
    if not os.path.exists(path):
        print(f"SKIP Phase 3: {path} not found")
        return
    d = json.load(open(path))
    s = {(r["mode"], r["criterion"]): r
         for r in d if r.get("fold") == "pooled_summary"}
    mm = {"clinical": "clinical_only", "imaging": "imaging_only", "fusion": "fusion"}
    claimed = {
        ("clinical", "auc"):       (0.7795, 0.7407, 0.0388),
        ("clinical", "neg_brier"): (0.7877, 0.7759, 0.0118),
        ("imaging", "auc"):        (0.7699, 0.6661, 0.1039),
        ("imaging", "neg_brier"):  (0.7711, 0.7535, 0.0176),
        ("fusion", "auc"):         (0.8062, 0.6330, 0.1732),
        ("fusion", "neg_brier"):   (0.8221, 0.8014, 0.0208),
    }
    for (m, c), (pf, pooled, gap) in claimed.items():
        r = s.get((mm[m], c))
        if r is None:
            FAILURES.append(f"Phase3 {m}/{c}: missing")
            continue
        check(f"Phase3 {m}/{c} per-fold", pf, r["per_fold_mean_auc"])
        check(f"Phase3 {m}/{c} pooled", pooled, r["pooled_style_auc"])
        check(f"Phase3 {m}/{c} gap", gap, r["gap"])
    print("Phase 3 checked")


# ---------------------------------------------------------------- Phase 4
def verify_phase4():
    runs = []
    for p in ["figures_gradcam/gradcam_summary.json",
              "figures_gradcam_fusion/gradcam_summary.json"]:
        if os.path.exists(p):
            runs += json.load(open(p))
    if not runs:
        print("SKIP Phase 4: no gradcam_summary.json found")
        return
    s = {(r["mode"], r["criterion"], r["block"]): r for r in runs}
    # (mode, criterion, block): (sel_AD, sel_CN, entropy_AD)
    claimed = {
        ("fusion", "auc", "block5"):            (0.493, 0.586, 0.964),
        ("fusion", "neg_brier", "block5"):      (0.781, 0.630, 0.959),
        ("imaging_only", "auc", "block5"):      (0.624, 0.508, 0.966),
        ("imaging_only", "neg_brier", "block5"): (0.624, 0.508, 0.966),
    }
    for key, (ad, cn, ent) in claimed.items():
        r = s.get(key)
        if r is None:
            FAILURES.append(f"Phase4 {key}: missing")
            continue
        check(f"Phase4 {key} sel_AD", ad, r["ad_brain_selectivity"], tol=5e-4)
        check(f"Phase4 {key} sel_CN", cn, r["cn_brain_selectivity"], tol=5e-4)
        check(f"Phase4 {key} entropy_AD", ent, r["ad_entropy"], tol=5e-4)
        # the null result depends on selectivity being below chance
        if r["ad_brain_selectivity"] >= 1.0 or r["cn_brain_selectivity"] >= 1.0:
            FAILURES.append(
                f"Phase4 {key}: selectivity no longer below chance - the "
                f"null-result claim in RESULTS.md would need revising")
    print("Phase 4 checked")


# ---------------------------------------------------------------- Phase 5
def verify_phase5():
    path = "outputs/temperature/temperature_results.json"
    if not os.path.exists(path):
        print(f"SKIP Phase 5: {path} not found")
        return
    d = json.load(open(path))
    s = {(r["mode"], r["criterion"]): r for r in d}
    # (mode, criterion, condition): (gap, bacc, ece, n_degenerate)
    claimed = {
        ("clinical_only", "auc", "raw"):          (0.0222, 0.6765, 0.1905, 1),
        ("clinical_only", "auc", "scaled"):       (0.0110, 0.6765, 0.2424, 1),
        ("clinical_only", "neg_brier", "raw"):    (-0.0053, 0.7354, 0.2602, 0),
        ("clinical_only", "neg_brier", "scaled"): (0.0165, 0.7354, 0.3353, 0),
        ("imaging_only", "auc", "raw"):           (0.1201, 0.6719, 0.3606, 1),
        ("imaging_only", "auc", "scaled"):        (0.0735, 0.6719, 0.2720, 1),
        ("imaging_only", "neg_brier", "raw"):     (0.0227, 0.7581, 0.3667, 0),
        ("imaging_only", "neg_brier", "scaled"):  (0.0159, 0.7581, 0.3564, 0),
        ("fusion", "auc", "raw"):                 (0.1989, 0.6231, 0.3905, 1),
        ("fusion", "auc", "scaled"):              (0.1426, 0.6231, 0.2219, 1),
        ("fusion", "neg_brier", "raw"):           (0.0408, 0.7506, 0.3514, 0),
        ("fusion", "neg_brier", "scaled"):        (0.0330, 0.7506, 0.3362, 0),
    }
    for (mode, crit, cond), (gap, bacc, ece, ndeg) in claimed.items():
        r = s.get((mode, crit))
        if r is None:
            FAILURES.append(f"Phase5 {mode}/{crit}: missing")
            continue
        c = r[cond]
        check(f"Phase5 {mode}/{crit}/{cond} gap", gap, c["gap"])
        check(f"Phase5 {mode}/{crit}/{cond} bacc", bacc, c["per_fold_mean_bacc"])
        check(f"Phase5 {mode}/{crit}/{cond} ECE", ece, c["per_fold_mean_ece"])
        if c["n_degenerate_folds"] != ndeg:
            FAILURES.append(f"Phase5 {mode}/{crit}/{cond} degenerate: "
                            f"RESULTS.md={ndeg} JSON={c['n_degenerate_folds']}")

    # the central claim: temperature must leave balanced accuracy unchanged
    for (mode, crit), r in s.items():
        if abs(r["raw"]["per_fold_mean_bacc"] - r["scaled"]["per_fold_mean_bacc"]) > 1e-9:
            FAILURES.append(
                f"Phase5 {mode}/{crit}: temperature CHANGED balanced accuracy "
                f"({r['raw']['per_fold_mean_bacc']:.6f} -> "
                f"{r['scaled']['per_fold_mean_bacc']:.6f}). Monotonicity "
                f"violated - the Phase 5 argument depends on this being exact.")
        if abs(r["raw"]["per_fold_mean_auc"] - r["scaled"]["per_fold_mean_auc"]) > 1e-9:
            FAILURES.append(
                f"Phase5 {mode}/{crit}: temperature CHANGED AUC - "
                f"implementation bug, scaling must be rank-preserving.")
    print("Phase 5 checked")


if __name__ == "__main__":
    verify_phase3()
    verify_phase4()
    verify_phase5()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} PROBLEM(S) FOUND:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"ALL {CHECKED} CHECKED VALUES VERIFIED AGAINST COMMITTED JSON")
