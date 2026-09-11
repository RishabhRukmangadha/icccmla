"""
kappa_analysis.py
-----------------
Computes Cohen's Kappa inter-rater reliability scores for human evaluation
of GSS, EVS, and EQS metrics in the ICCCMLA 2026 paper.

Usage:
    python kappa_analysis.py

Requires:
    - data/human_evaluation/human_judge_T1.xlsx
    - data/human_evaluation/human_judge_T2.xlsx
    - data/human_evaluation/human_judge_T3.xlsx
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import os

# ── File paths ────────────────────────────────────────────────────
FILES = {
    'T1': 'data/human_evaluation/human_judge_T1.xlsx',
    'T2': 'data/human_evaluation/human_judge_T2.xlsx',
    'T3': 'data/human_evaluation/human_judge_T3.xlsx',
}

SHEET = 'human_evaluation_sheet1'

# ── Load scores ───────────────────────────────────────────────────
def load_scores(path):
    df = pd.read_excel(path, sheet_name=SHEET)
    gss_col = [c for c in df.columns if 'GSS' in c and 'Human' in c][0]
    evs_col = [c for c in df.columns if 'EVS' in c and 'Human' in c][0]
    eqs_col = [c for c in df.columns if 'EQS' in c and 'Human' in c][0]
    claude_gss = [c for c in df.columns if 'GSS' in c and 'Claude' in c][0]
    claude_evs = [c for c in df.columns if 'EVS' in c and 'Claude' in c][0]
    claude_eqs = [c for c in df.columns if 'EQS' in c and 'Claude' in c][0]
    return {
        'GSS': df[gss_col],
        'EVS': df[evs_col],
        'EQS': df[eqs_col],
        'Claude_GSS': df[claude_gss],
        'Claude_EVS': df[claude_evs],
        'Claude_EQS': df[claude_eqs],
    }

# ── Kappa calculation ─────────────────────────────────────────────
def safe_kappa(a, b):
    mask = pd.notna(a) & pd.notna(b)
    a_c = a[mask].astype(int)
    b_c = b[mask].astype(int)
    if len(a_c) < 2:
        return None, 0
    try:
        return cohen_kappa_score(a_c, b_c, weights='linear'), len(a_c)
    except:
        return None, len(a_c)

def interpret(k):
    if k is None: return 'N/A'
    if k < 0.20: return 'Slight'
    elif k < 0.40: return 'Fair'
    elif k < 0.60: return 'Moderate'
    elif k < 0.80: return 'Substantial'
    else: return 'Almost Perfect'

# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Loading human evaluation files...")
    scores = {name: load_scores(path) for name, path in FILES.items()}

    # Fix Claude EVS=0 cases stored as NaN
    for name in scores:
        scores[name]['Claude_EVS'] = scores[name]['Claude_EVS'].fillna(0)

    pairs = [('T1','T2'), ('T1','T3'), ('T2','T3')]
    pair_labels = ['T1 vs T2', 'T1 vs T3', 'T2 vs T3']

    print("\n" + "="*60)
    print("HUMAN vs HUMAN — Inter-rater Kappa")
    print("="*60)
    hh_summary = {}
    for metric in ['GSS','EVS','EQS']:
        print(f"\n{metric}:")
        kappas = []
        for (t1, t2), label in zip(pairs, pair_labels):
            k, n = safe_kappa(scores[t1][metric], scores[t2][metric])
            if k is not None:
                print(f"  {label}: κ = {k:.3f} (n={n}) [{interpret(k)}]")
                kappas.append(k)
        if kappas:
            avg = np.mean(kappas)
            hh_summary[metric] = avg
            print(f"  >>> Average κ = {avg:.3f} [{interpret(avg)}]")

    print("\n" + "="*60)
    print("HUMAN vs CLAUDE — Validation Kappa")
    print("="*60)
    hc_summary = {}
    for metric in ['GSS','EVS','EQS']:
        print(f"\n{metric}:")
        kappas = []
        for teacher in ['T1','T2','T3']:
            k, n = safe_kappa(scores[teacher][metric],
                              scores[teacher][f'Claude_{metric}'])
            if k is not None:
                print(f"  {teacher} vs Claude: κ = {k:.3f} (n={n}) [{interpret(k)}]")
                kappas.append(k)
        if kappas:
            avg = np.mean(kappas)
            hc_summary[metric] = avg
            print(f"  >>> Average κ = {avg:.3f} [{interpret(avg)}]")

    print("\n" + "="*60)
    print("SUMMARY — As reported in paper")
    print("="*60)
    print(f"Human-Human GSS: κ = {hh_summary.get('GSS',0):.3f}")
    print(f"Human-Human EVS: κ = {hh_summary.get('EVS',0):.3f}")
    print(f"Human-Human EQS: κ = {hh_summary.get('EQS',0):.3f}")
    print(f"Human-Claude GSS: κ = {hc_summary.get('GSS',0):.3f}")
    print(f"Human-Claude EVS: κ = {hc_summary.get('EVS',0):.3f}")
    print(f"Human-Claude EQS: κ = {hc_summary.get('EQS',0):.3f}")

if __name__ == '__main__':
    main()
