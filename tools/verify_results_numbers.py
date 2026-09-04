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

        # AUC must be rank-preserved. Tiny deviations are possible without a
        # bug: at extreme temperatures the softmax saturates to exactly 0.0/1.0
        # in float64, creating ties, and AUC scores tied pairs at 0.5. That is a
        # representation limit, not a monotonicity violation. A large deviation
        # is a real bug. The threshold separates the two.
        d_auc = abs(r["raw"]["per_fold_mean_auc"] - r["scaled"]["per_fold_mean_auc"])
        extreme_T = [t for t in r["temperatures"] if t < 0.05 or t > 1e3]
        if d_auc > 1e-2:
            FAILURES.append(
                f"Phase5 {mode}/{crit}: temperature changed AUC by {d_auc:.4f} - "
                f"too large for float saturation, this is an implementation bug.")
        elif d_auc > 1e-9 and not extreme_T:
            FAILURES.append(
                f"Phase5 {mode}/{crit}: AUC moved by {d_auc:.2e} with no extreme "
                f"temperature to explain it (T range "
                f"{min(r['temperatures']):.3g}-{max(r['temperatures']):.3g}) - "
                f"investigate before trusting this row.")
        elif d_auc > 1e-9:
            print(f"  NOTE {mode}/{crit}: AUC moved {d_auc:.2e} due to float "
                  f"saturation at extreme T={min(extreme_T, key=lambda t: abs(t-1)):.3g} "
                  f"- expected, not a bug (see Phase 5 limitations)")
    print("Phase 5 checked")


# ---------------------------------------------------------------- Phase 6
def verify_phase6():
    path = "outputs/sample_size/sample_size_results.json"
    if not os.path.exists(path):
        print(f"SKIP Phase 6: {path} not found")
        return
    d = json.load(open(path))
    s = {(r["mode"], r["criterion"], r["n"]): r for r in d}
    # (mode, criterion): {n: claimed gap}
    claimed = {
        ("clinical_only", "auc"):       {100: 0.0388, 200: 0.0406, 400: 0.0378,
                                          800: 0.0394, 1287: 0.0388},
        ("clinical_only", "neg_brier"): {100: 0.0123, 200: 0.0119, 400: 0.0113,
                                          800: 0.0120, 1287: 0.0118},
        ("imaging_only", "auc"):        {100: 0.1033, 200: 0.1044, 400: 0.1032,
                                          800: 0.1043, 1287: 0.1039},
        ("imaging_only", "neg_brier"):  {100: 0.0184, 200: 0.0168, 400: 0.0173,
                                          800: 0.0176, 1287: 0.0176},
        ("fusion", "auc"):              {100: 0.1763, 200: 0.1715, 400: 0.1730,
                                          800: 0.1737, 1287: 0.1732},
        ("fusion", "neg_brier"):        {100: 0.0242, 200: 0.0194, 400: 0.0213,
                                          800: 0.0210, 1287: 0.0208},
    }
    for (mode, crit), by_n in claimed.items():
        for n, gap in by_n.items():
            r = s.get((mode, crit, n))
            if r is None:
                FAILURES.append(f"Phase6 {mode}/{crit}/n={n}: missing")
                continue
            check(f"Phase6 {mode}/{crit}/n={n} gap", gap, r["gap_mean"])

    # the claim: the gap must stay flat across sample size
    for (mode, crit), by_n in claimed.items():
        gaps = [s[(mode, crit, n)]["gap_mean"] for n in sorted(by_n)
                if (mode, crit, n) in s]
        if len(gaps) < 2:
            continue
        drift = abs(gaps[-1] - gaps[0])
        if drift > 0.02:
            FAILURES.append(
                f"Phase6 {mode}/{crit}: gap drifted {drift:.4f} from smallest "
                f"to largest n. RESULTS.md claims the gap is flat with sample "
                f"size - that claim would need revising.")

    # precision must improve with n, even though the gap does not move
    for (mode, crit), by_n in claimed.items():
        sds = [(n, s[(mode, crit, n)]["gap_sd"]) for n in sorted(by_n)
               if (mode, crit, n) in s and s[(mode, crit, n)]["n_repeats"] > 1]
        if len(sds) >= 2 and sds[-1][1] > sds[0][1]:
            FAILURES.append(
                f"Phase6 {mode}/{crit}: SD did not shrink with n "
                f"({sds[0][1]:.4f} at n={sds[0][0]} -> {sds[-1][1]:.4f} at "
                f"n={sds[-1][0]}) - unexpected, investigate.")
    print("Phase 6 checked")


# ---------------------------------------------------------------- Phase 7
def verify_phase7():
    """Phase 7 reports FIVE ResNet runs and an explicitly withdrawn conclusion.

    The checks here guard the corrected claim (the gap does not reliably
    reproduce on ResNet-18) rather than the original one (pretraining removes
    it), which was retracted after three further seeds.
    """
    paths = {
        "pretrained": "outputs/resnet2d/resnet2d_folds.json",
        "scratch_s0": "outputs/resnet2d_scratch/resnet2d_folds.json",
        "scratch_s1": "outputs/resnet2d_scratch_s1/resnet2d_folds.json",
        "scratch_s2": "outputs/resnet2d_scratch_s2/resnet2d_folds.json",
        "scratch_s3": "outputs/resnet2d_scratch_s3/resnet2d_folds.json",
    }
    runs = {k: json.load(open(p)) for k, p in paths.items() if os.path.exists(p)}
    if not runs:
        print("SKIP Phase 7: no resnet2d_folds.json found")
        return

    # (run, criterion): gap as written in the RESULTS.md table
    claimed = {
        ("pretrained", "auc"): 0.0054, ("pretrained", "auc_minus_ece"): 0.0012,
        ("pretrained", "gated_bacc"): 0.0054, ("pretrained", "neg_brier"): 0.0077,
        ("scratch_s0", "auc"): 0.0804, ("scratch_s0", "auc_minus_ece"): 0.0100,
        ("scratch_s0", "gated_bacc"): 0.0364, ("scratch_s0", "neg_brier"): 0.0138,
        ("scratch_s1", "auc"): 0.0274, ("scratch_s1", "auc_minus_ece"): 0.0343,
        ("scratch_s1", "gated_bacc"): 0.0274, ("scratch_s1", "neg_brier"): 0.0052,
        ("scratch_s2", "auc"): 0.0146, ("scratch_s2", "auc_minus_ece"): 0.0123,
        ("scratch_s2", "gated_bacc"): 0.0146, ("scratch_s2", "neg_brier"): 0.0192,
        ("scratch_s3", "auc"): 0.0150, ("scratch_s3", "auc_minus_ece"): 0.0150,
        ("scratch_s3", "gated_bacc"): 0.0026, ("scratch_s3", "neg_brier"): 0.0086,
    }
    for (run, crit), gap in claimed.items():
        if run not in runs:
            continue
        s_ = runs[run]["summary"].get(crit)
        if s_ is None:
            FAILURES.append(f"Phase7 {run}/{crit}: missing")
            continue
        check(f"Phase7 {run}/{crit} gap", gap, s_["gap"])

    # locked experimental constants must not have drifted in ANY run
    for run, d in runs.items():
        if abs(d.get("slice_fraction", -1) - 0.55) > 1e-9:
            FAILURES.append(f"Phase7 {run}: slice_fraction "
                            f"{d.get('slice_fraction')} != locked 0.55")
        if d.get("fold_seed") != 42:
            FAILURES.append(f"Phase7 {run}: fold_seed {d.get('fold_seed')} != 42")
        if d.get("slice_offsets") != [-1, 0, 1]:
            FAILURES.append(f"Phase7 {run}: slice_offsets "
                            f"{d.get('slice_offsets')} != locked [-1, 0, 1]")

    # every run must use the SAME subject splits - only training varies
    ref = None
    for run, d in sorted(runs.items()):
        splits = {k: d["folds"][i]["subjects"][k]
                  for i in range(len(d["folds"])) for k in ("train", "val", "test")}
        if ref is None:
            ref, ref_name = splits, run
        elif splits != ref:
            FAILURES.append(f"Phase7 {run}: subject splits differ from "
                            f"{ref_name} - runs are not comparable")

    # the corrected headline: the 3D CNN gap must exceed every ResNet run
    CNN_GAP = 0.1200
    for run, d in runs.items():
        g = d["summary"]["auc"]["gap"]
        if g >= CNN_GAP:
            FAILURES.append(
                f"Phase7 {run}: auc gap {g:.4f} >= the 3D CNN's {CNN_GAP}. "
                f"RESULTS.md claims no ResNet run approaches it - revise.")

    # the withdrawn claim must STAY withdrawn: scratch seeds must not all
    # exceed the pretrained run, or the pretraining story would be back
    scratch = [runs[k]["summary"]["auc"]["gap"]
               for k in ("scratch_s0", "scratch_s1", "scratch_s2", "scratch_s3")
               if k in runs]
    if "pretrained" in runs and len(scratch) >= 3:
        pre = runs["pretrained"]["summary"]["auc"]["gap"]
        spread = max(scratch) - min(scratch)
        if spread < 0.01:
            FAILURES.append(
                f"Phase7: scratch gaps span only {spread:.4f} across seeds. "
                f"RESULTS.md argues the seed-0 result was an outlier and the "
                f"effect is unstable - that reasoning would need revising.")
        if min(scratch) > pre * 3:
            FAILURES.append(
                f"Phase7: every scratch seed exceeds 3x the pretrained gap "
                f"({pre:.4f}). The withdrawn pretraining conclusion may "
                f"actually hold - re-examine before publishing the retraction.")
    print("Phase 7 checked")


if __name__ == "__main__":
    verify_phase3()
    verify_phase4()
    verify_phase5()
    verify_phase6()
    verify_phase7()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} PROBLEM(S) FOUND:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"ALL {CHECKED} CHECKED VALUES VERIFIED AGAINST COMMITTED JSON")