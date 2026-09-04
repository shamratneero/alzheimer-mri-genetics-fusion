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
    paths = {"pretrained": "outputs/resnet2d/resnet2d_folds.json",
             "scratch": "outputs/resnet2d_scratch/resnet2d_folds.json"}
    runs = {}
    for cond, p in paths.items():
        if os.path.exists(p):
            runs[cond] = json.load(open(p))
    if not runs:
        print("SKIP Phase 7: no resnet2d_folds.json found")
        return

    # (condition, criterion): (per_fold, pooled, gap, n_degenerate)
    claimed = {
        ("pretrained", "auc"):           (0.7864, 0.7811, 0.0054, 0),
        ("pretrained", "auc_minus_ece"): (0.7727, 0.7715, 0.0012, 0),
        ("pretrained", "gated_bacc"):    (0.7864, 0.7811, 0.0054, 0),
        ("pretrained", "neg_brier"):     (0.7733, 0.7656, 0.0077, 0),
        ("scratch", "auc"):              (0.8017, 0.7213, 0.0804, 2),
        ("scratch", "auc_minus_ece"):    (0.8020, 0.7921, 0.0100, 0),
        ("scratch", "gated_bacc"):       (0.7963, 0.7599, 0.0364, 1),
        ("scratch", "neg_brier"):        (0.7949, 0.7812, 0.0138, 1),
    }
    for (cond, crit), (pf, pooled, gap, ndeg) in claimed.items():
        if cond not in runs:
            continue
        s = runs[cond]["summary"].get(crit)
        if s is None:
            FAILURES.append(f"Phase7 {cond}/{crit}: missing")
            continue
        check(f"Phase7 {cond}/{crit} per-fold", pf, s["per_fold_mean_auc"])
        check(f"Phase7 {cond}/{crit} pooled", pooled, s["pooled_oof_auc"])
        check(f"Phase7 {cond}/{crit} gap", gap, s["gap"])
        if s["n_degenerate_folds"] != ndeg:
            FAILURES.append(f"Phase7 {cond}/{crit} degenerate: "
                            f"RESULTS.md={ndeg} JSON={s['n_degenerate_folds']}")

    # locked experimental constants must not have drifted
    for cond, d in runs.items():
        if abs(d.get("slice_fraction", -1) - 0.55) > 1e-9:
            FAILURES.append(f"Phase7 {cond}: slice_fraction is "
                            f"{d.get('slice_fraction')}, not the locked 0.55")
        if d.get("fold_seed") != 42:
            FAILURES.append(f"Phase7 {cond}: fold_seed is {d.get('fold_seed')}, "
                            f"not the locked 42 - splits may not match the "
                            f"primary model")
        if d.get("slice_offsets") != [-1, 0, 1]:
            FAILURES.append(f"Phase7 {cond}: slice_offsets "
                            f"{d.get('slice_offsets')} != locked [-1, 0, 1]")

    # the two conditions must differ ONLY in initialisation
    if len(runs) == 2:
        a, b = runs["pretrained"], runs["scratch"]
        if a.get("pretrained") is not True or b.get("pretrained") is not False:
            FAILURES.append("Phase7: the two runs are not a pretrained/scratch "
                            "pair - the ablation claim would be invalid")
        for key in ["slice_fraction", "fold_seed", "n_folds", "val_frac",
                    "slice_offsets", "arch"]:
            if a.get(key) != b.get(key):
                FAILURES.append(
                    f"Phase7: '{key}' differs between the pretrained and "
                    f"scratch runs ({a.get(key)} vs {b.get(key)}). The "
                    f"ablation requires initialisation to be the ONLY "
                    f"difference.")
        # every fold must use identical subject splits across the two runs
        for i, (fa, fb) in enumerate(zip(a["folds"], b["folds"]), start=1):
            for split in ["train", "val", "test"]:
                if fa["subjects"][split] != fb["subjects"][split]:
                    FAILURES.append(
                        f"Phase7 fold {i}: {split} subjects differ between "
                        f"pretrained and scratch runs - not a controlled "
                        f"ablation")
        # the headline claim
        gap_pre = a["summary"]["auc"]["gap"]
        gap_scr = b["summary"]["auc"]["gap"]
        if not (gap_scr > gap_pre):
            FAILURES.append(
                f"Phase7: scratch gap ({gap_scr:.4f}) is not larger than "
                f"pretrained ({gap_pre:.4f}). RESULTS.md claims pretraining "
                f"removes the instability - that claim would need revising.")
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