#!/usr/bin/env python3
"""
Layer 4 - Demo app: Temporal Hepatotoxic Load explorer.

A self-contained Streamlit app. It duplicates the Layer 2 temporal model
(code/temporal_load.py) in pure Python so it can be deployed on its own
(Hugging Face Spaces / Streamlit Cloud) with no project files at runtime.

Model (identical to Layer 2 v1):
  - Per drug, a one-compartment oral (Bateman) concentration curve,
    normalised to peak 1 at the drug's Tmax:
        f(tau) = [exp(-ke*tau) - exp(-ka*tau)]
                 / [exp(-ke*Tmax) - exp(-ka*Tmax)],  tau >= 0, else 0
        ke = ln(2) / half_life;  ka solved from Tmax = ln(ka/ke)/(ka-ke).
  - Hazard from DILIrank 2.0 (Fixed Truth file 01):
        H = (SeverityClass / 8) * concern_weight
        concern_weight: vMost 1.0, vLess 0.6, Ambiguous 0.45, vNo 0.25.
  - Combined load: L(t) = sum_i H_i * DoseFraction_i * f_i(t - dose_time_i),
    on a 0-24 h grid at 0.05 h steps. Danger threshold 0.75, as in Layer 2.

Drug values are the Layer 2 v1 fixed points from
data/fixed_db/07_PK_Table_v0.1.csv (ranges replaced by the same midpoints
Layer 2 used) and alert notes are taken from
data/fixed_db/08_Structure_Alert_Matches_v1.csv. Nothing is invented.

RESEARCH PROTOTYPE ONLY. Not for clinical decision-making. Predictions
are associative, not causal.
"""

import math

import pandas as pd
import streamlit as st

THRESHOLD = 0.75
T_END = 24.0
DT = 0.05

DRUGS = {
    "acetaminophen": {
        "display": "Acetaminophen (paracetamol), oral immediate release",
        "tmax": 1.0, "half_life": 2.0, "severity": 5,
        "concern": "vMost-DILI-concern", "concern_weight": 1.0,
        "alert_note": "1 alert, 2 points (file 08): Phenol (quinone-imine "
                      "precursor). Consistent with oxidation to the "
                      "quinone-imine NAPQI.",
    },
    "diclofenac_sodium": {
        "display": "Diclofenac sodium, oral delayed release",
        "tmax": 2.0, "half_life": 1.5, "severity": 8,
        "concern": "vMost-DILI-concern", "concern_weight": 1.0,
        "alert_note": "0 of 13 alerts matched, 0 points (file 08). Its "
                      "maximum DILIrank severity (8) is NOT captured by "
                      "the alert panel: alerts are notes, not verdicts.",
    },
    "diclofenac_potassium": {
        "display": "Diclofenac potassium, oral conventional",
        "tmax": 1.0, "half_life": 1.5, "severity": 8,
        "concern": "vMost-DILI-concern", "concern_weight": 1.0,
        "alert_note": "Not a separate DILIrank entry; hazard and alerts "
                      "taken from the diclofenac (sodium salt) entry, "
                      "which matched 0 of 13 alerts (file 08). Only the "
                      "formulation PK differs (Tmax 1 h vs 2 h).",
    },
    "aspirin": {
        "display": "Aspirin (modelled as salicylate)",
        "tmax": 1.74, "half_life": 2.0, "severity": 0,
        "concern": "vLess-DILI-concern", "concern_weight": 0.6,
        "alert_note": "1 alert, 2 points (file 08): Carboxylic acid (acyl "
                      "glucuronide precursor). SeverityClass is 0, so the "
                      "v1 hazard weight is 0 despite the alert.",
    },
    "rabeprazole": {
        "display": "Rabeprazole sodium, oral delayed release",
        "tmax": 3.5, "half_life": 1.0, "severity": 3,
        "concern": "vLess-DILI-concern", "concern_weight": 0.6,
        "alert_note": "0 of 13 alerts matched, 0 points (file 08, as "
                      "rabeprazole sodium).",
    },
    "cyanocobalamin": {
        "display": "Cyanocobalamin (vitamin B12), zero-hazard control",
        "tmax": None, "half_life": None, "severity": 0,
        "concern": "vNo-DILI-concern", "concern_weight": 0.25,
        "alert_note": "File 08 flags 1 alert, 2 points: Furan "
                      "substructure. SeverityClass 0 and vNo give hazard "
                      "H = 0, so its load contribution is zero at every "
                      "time point, by construction. It is the project's "
                      "control drug: adding or removing it must never "
                      "change the result.",
    },
}


def hazard(drug) -> float:
    return (drug["severity"] / 8.0) * drug["concern_weight"]


def solve_ka(ke: float, tmax: float) -> float:
    """Solve Tmax = ln(ka/ke)/(ka-ke) for ka (same bisection as Layer 2)."""
    def g(ka):
        if abs(ka - ke) < 1e-9 * ke:
            return 1.0 / ke
        return math.log(ka / ke) / (ka - ke)

    lo, hi = 1e-8, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if g(mid) > tmax:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def bateman(tmax: float, half_life: float, tau: float) -> float:
    """Peak-normalised Bateman curve value at tau hours after the dose."""
    if tau < 0:
        return 0.0
    ke = math.log(2.0) / half_life
    ka = solve_ka(ke, tmax)
    denom = math.exp(-ke * tmax) - math.exp(-ka * tmax)
    return (math.exp(-ke * tau) - math.exp(-ka * tau)) / denom


def compute_load(selection):
    """selection: list of (key, dose_time_h, dose_fraction).

    Returns (times, L, per-drug contribution dict of lists).
    """
    n = int(T_END / DT) + 1
    times = [round(i * DT, 10) for i in range(n)]
    contrib = {}
    for key, dose_time, dose_fraction in selection:
        drug = DRUGS[key]
        h = hazard(drug)
        if h == 0.0 or drug["tmax"] is None:
            contrib[key] = [0.0] * n
        else:
            contrib[key] = [
                h * dose_fraction * bateman(drug["tmax"], drug["half_life"],
                                            t - dose_time)
                for t in times
            ]
    L = [sum(contrib[k][i] for k in contrib) for i in range(n)]
    return times, L, contrib


def danger_windows(times, L):
    windows, start = [], None
    above = [v >= THRESHOLD for v in L]
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        if start is not None and (not a or i == len(above) - 1):
            end = i if not a else i + 1
            windows.append((times[start], times[end - 1]))
            start = None
    return windows


def main():
    st.set_page_config(page_title="Temporal Hepatotoxic Load - Demo",
                       layout="wide")
    st.title("Temporal Hepatotoxic Load Explorer")
    st.warning(
        "Research prototype only. Not for clinical decision-making. "
        "Outputs are model scores built from public data (DILIrank 2.0, "
        "FAERS, published pharmacokinetics); they are associative, "
        "not causal, and are not a patient's risk.")

    st.markdown(
        "Pick 2 to 5 drugs, set when each dose is taken, and see how the "
        "combined hepatotoxic load L(t) evolves over 24 hours. The idea: "
        "a drug's hazard only bites while it is present at high "
        "concentration, so peaks that overlap in time add up, and the "
        "same drugs staggered in time score lower.")

    picked = st.multiselect(
        "Drugs in the combination",
        options=list(DRUGS.keys()),
        default=["acetaminophen", "diclofenac_sodium", "aspirin",
                 "rabeprazole", "cyanocobalamin"],
        format_func=lambda k: DRUGS[k]["display"],
    )
    if len(picked) < 2:
        st.info("Pick at least 2 drugs to compute a combined load.")
        return
    if len(picked) > 5:
        st.error("This prototype covers combinations of 2 to 5 drugs. "
                 "Remove one drug to continue.")
        return

    selection = []
    cols = st.columns(len(picked))
    for col, key in zip(cols, picked):
        with col:
            st.markdown(f"**{DRUGS[key]['display']}**")
            dose_time = st.slider("Dose time (h)", 0.0, 12.0, 0.0, 0.5,
                                  key=f"t_{key}")
            dose_fraction = st.slider(
                "Dose fraction (of max recommended dose)", 0.25, 2.0, 1.0,
                0.25, key=f"df_{key}")
            selection.append((key, dose_time, dose_fraction))

    times, L, contrib = compute_load(selection)
    imax = max(range(len(L)), key=lambda i: L[i])
    lmax = L[imax]
    auc = sum((L[i] + L[i + 1]) / 2.0 * DT for i in range(len(L) - 1))
    windows = danger_windows(times, L)
    time_above = sum(1 for v in L if v >= THRESHOLD) * DT

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak load Lmax", f"{lmax:.3f}")
    m2.metric("Time of peak (h)", f"{times[imax]:.2f}")
    m3.metric("Toxic AUC (0-24 h)", f"{auc:.3f}")
    m4.metric(f"Hours above threshold {THRESHOLD}", f"{time_above:.2f}")

    if windows:
        win_txt = "; ".join(f"{a:.2f} to {b:.2f} h" for a, b in windows)
        st.error(f"Danger window(s), L(t) at or above {THRESHOLD}: "
                 f"{win_txt}")
    else:
        st.success(f"L(t) never reaches the danger threshold "
                   f"({THRESHOLD}) for this schedule.")

    chart_data = {"Combined load L(t)": L}
    for key, _t, _df in selection:
        chart_data[DRUGS[key]["display"]] = contrib[key]
    df = pd.DataFrame(chart_data, index=times)
    df.index.name = "Hours after first dose"
    st.line_chart(df)

    st.subheader("Per-drug detail")
    rows = []
    for key, dose_time, dose_fraction in selection:
        d = DRUGS[key]
        rows.append({
            "Drug": d["display"],
            "Dose time (h)": dose_time,
            "Dose fraction": dose_fraction,
            "Tmax (h)": d["tmax"] if d["tmax"] is not None else "n/a",
            "Half-life (h)":
                d["half_life"] if d["half_life"] is not None else "n/a",
            "SeverityClass": d["severity"],
            "DILIrank concern": d["concern"],
            "Hazard H": round(hazard(d), 4),
            "Structural alert note": d["alert_note"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption(
        "Model: Layer 2 v1 of the project 'AI Framework for Multi-Drug "
        "Combination Hepatotoxicity Risk Prediction'. Hazards from "
        "DILIrank 2.0 (FDA LTKB); pharmacokinetic fixed points from the "
        "project PK table v0.1 (07_PK_Table_v0.1.csv, verified label "
        "values, ranges replaced by midpoints); structural alert notes "
        "from 08_Structure_Alert_Matches_v1.csv. Cyanocobalamin is the "
        "zero-hazard control: its hazard is 0, so it never changes L(t). "
        "Research prototype only, not for clinical decision-making.")



if __name__ == "__main__":
    main()
