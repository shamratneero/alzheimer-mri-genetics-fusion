"""Cross-check the Phase 3 ADNI numbers written in RESULTS.md against the
actual committed results JSON. Run after any re-run of inference_adni.py.

Usage:  python tools/verify_phase3_numbers.py
"""
import json, re, sys

# Phase 3 ADNI numbers as written in RESULTS.md
claimed = {
    ('clinical','auc'):      (0.7795, 0.7407, 0.0388),
    ('clinical','neg_brier'):(0.7877, 0.7759, 0.0118),
    ('imaging','auc'):       (0.7699, 0.6661, 0.1039),
    ('imaging','neg_brier'): (0.7711, 0.7535, 0.0176),
    ('fusion','auc'):        (0.8062, 0.6330, 0.1732),
    ('fusion','neg_brier'):  (0.8221, 0.8014, 0.0208),
}
mode_map = {'clinical':'clinical_only','imaging':'imaging_only','fusion':'fusion'}

try:
    d = json.load(open('outputs/adni_external/adni_external_results.json'))
except FileNotFoundError:
    print('SKIP: adni_external_results.json not present in repo (gitignored?)')
    sys.exit(0)

summaries = {(r['mode'], r['criterion']): r for r in d if r.get('fold') == 'pooled_summary'}
ok = True
for (m, c), (pf, pooled, gap) in claimed.items():
    key = (mode_map[m], c)
    if key not in summaries:
        print(f'MISSING in JSON: {key}'); ok = False; continue
    r = summaries[key]
    for name, want, got in [('per-fold', pf, r['per_fold_mean_auc']),
                            ('pooled', pooled, r['pooled_style_auc']),
                            ('gap', gap, r['gap'])]:
        if abs(want - got) > 5e-5:
            print(f'MISMATCH {m}/{c} {name}: RESULTS.md={want} JSON={got:.4f}'); ok = False
print('ALL PHASE 3 NUMBERS VERIFIED AGAINST JSON' if ok else 'VERIFICATION FAILED')
