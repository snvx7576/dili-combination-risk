#!/usr/bin/env python3
"""
Layer 4 - Demo app v2: Temporal Hepatotoxic Load Explorer.

Self-contained Streamlit app (deployed on Streamlit Community Cloud from
the project GitHub repo). v2 replaces the slider UI with a dynamic
row/table drug-entry interface and upgrades the engine from single doses
to multi-dose regimens: each drug is entered as strength in mg plus a
dosing frequency (1-5 times per 24 h), and the load is the linear
superposition of every dose over a 24 h window.

Model (identical core to Layer 2, code/temporal_load.py):
  - Per drug and per dose, a one-compartment oral (Bateman) concentration
    curve, normalised to peak 1 at the drug's Tmax:
        f(tau) = [exp(-ke*tau) - exp(-ka*tau)]
                 / [exp(-ke*Tmax) - exp(-ka*Tmax)],  tau >= 0, else 0
        ke = ln(2) / half_life;  ka solved from Tmax = ln(ka/ke)/(ka-ke).
  - Hazard from DILIrank 2.0 (Fixed Truth file 01):
        H = (SeverityClass / 8) * concern_weight
        concern_weight: vMost 1.0, vLess 0.6, Ambiguous 0.45, vNo 0.25.
  - Dose fraction from strength: F = entered mg / reference max single
    dose (mg). Reference maxima below are standard labelled values used
    ONLY to normalise the dose fraction; they are flagged in the UI and
    must be re-verified against current labels before any other use.
  - Multi-dose superposition: for frequency N, doses at
        t_k = offset + k * (24 / N),  k = 0..N-1
        L_i(t) = sum_k F_i * H_i * f_i(t - t_k)   (terms with t < t_k are 0)
        L(t)   = sum_i L_i(t),  grid 0-24 h at 0.05 h steps.
  - Danger threshold 0.75, as in Layer 2.

Drug PK fixed points are the Layer 2 v1 values from
data/fixed_db/07_PK_Table_v0.1.csv (ranges replaced by the same midpoints
Layer 2 used); alert notes from data/fixed_db/08_Structure_Alert_Matches_v1.csv.
With every drug at F = 1, frequency 1 and offset 0, the engine reproduces
Layer 2 scenario A exactly (Lmax 1.7356, Toxic AUC 11.0852).

RESEARCH PROTOTYPE ONLY. Not for clinical decision-making. Predictions
are associative, not causal.
"""

import math
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

THRESHOLD = 0.75
T_END = 24.0
DT = 0.05
MAX_ROWS = 6

FREQ_LABELS = {
    1: "1 x daily (every 24 h)",
    2: "2 x daily (every 12 h)",
    3: "3 x daily (every 8 h)",
    4: "4 x daily (every 6 h)",
    5: "5 x daily (every 4.8 h)",
}

# Reference maximum doses (mg) are standard labelled values, used only to
# normalise the entered strength into a dose fraction. Cyanocobalamin is
# the zero-hazard control (H = 0), so its fraction never affects the load.
DRUGS = {
    "acetaminophen": {
        "display": "Acetaminophen (paracetamol), oral immediate release",
        "tmax": 1.0, "half_life": 2.0, "severity": 5,
        "concern": "vMost-DILI-concern", "concern_weight": 1.0,
        "ref_max_single_mg": 1000.0, "ref_max_daily_mg": 4000.0,
        "alert_note": "1 alert, 2 points (file 08): Phenol (quinone-imine "
                      "precursor). Consistent with oxidation to the "
                      "quinone-imine NAPQI.",
    },
    "diclofenac_sodium": {
        "display": "Diclofenac sodium, oral delayed release",
        "tmax": 2.0, "half_life": 1.5, "severity": 8,
        "concern": "vMost-DILI-concern", "concern_weight": 1.0,
        "ref_max_single_mg": 50.0, "ref_max_daily_mg": 150.0,
        "alert_note": "0 of 13 alerts matched, 0 points (file 08). Its "
                      "maximum DILIrank severity (8) is NOT captured by "
                      "the alert panel: alerts are notes, not verdicts.",
    },
    "diclofenac_potassium": {
        "display": "Diclofenac potassium, oral conventional",
        "tmax": 1.0, "half_life": 1.5, "severity": 8,
        "concern": "vMost-DILI-concern", "concern_weight": 1.0,
        "ref_max_single_mg": 50.0, "ref_max_daily_mg": 150.0,
        "alert_note": "Not a separate DILIrank entry; hazard and alerts "
                      "taken from the diclofenac (sodium salt) entry, "
                      "which matched 0 of 13 alerts (file 08). Only the "
                      "formulation PK differs (Tmax 1 h vs 2 h).",
    },
    "aspirin": {
        "display": "Aspirin (modelled as salicylate)",
        "tmax": 1.74, "half_life": 2.0, "severity": 0,
        "concern": "vLess-DILI-concern", "concern_weight": 0.6,
        "ref_max_single_mg": 1000.0, "ref_max_daily_mg": 4000.0,
        "alert_note": "1 alert, 2 points (file 08): Carboxylic acid (acyl "
                      "glucuronide precursor). SeverityClass is 0, so the "
                      "v1 hazard weight is 0 despite the alert.",
    },
    "rabeprazole": {
        "display": "Rabeprazole sodium, oral delayed release",
        "tmax": 3.5, "half_life": 1.0, "severity": 3,
        "concern": "vLess-DILI-concern", "concern_weight": 0.6,
        "ref_max_single_mg": 20.0, "ref_max_daily_mg": 20.0,
        "alert_note": "0 of 13 alerts matched, 0 points (file 08, as "
                      "rabeprazole sodium).",
    },
    "cyanocobalamin": {
        "display": "Cyanocobalamin (vitamin B12), zero-hazard control",
        "tmax": None, "half_life": None, "severity": 0,
        "concern": "vNo-DILI-concern", "concern_weight": 0.25,
        "ref_max_single_mg": None, "ref_max_daily_mg": None,
        "alert_note": "File 08 flags 1 alert, 2 points: Furan "
                      "substructure. SeverityClass 0 and vNo give hazard "
                      "H = 0, so its load contribution is zero at every "
                      "time point, by construction. It is the project's "
                      "control drug: adding or removing it must never "
                      "change the result.",
    },
}


# ---------------------------------------------------------------- model

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


def dose_times(freq: int, offset: float):
    """Dose timestamps for a frequency over 24 h, shifted by offset."""
    step = 24.0 / freq
    return [offset + k * step for k in range(freq)]


def dose_fraction(drug, strength_mg):
    """F = strength / reference max single dose.

    Returns (fraction, warning or None). Drugs with no reference value
    fall back to F = 1 with an honest warning.
    """
    ref = drug["ref_max_single_mg"]
    if ref is None:
        return 1.0, "No reference max single dose on record; dose " \
                    "fraction set to 1.0."
    return strength_mg / ref, None


def compute_load(regimen):
    """regimen: list of dicts with keys drug_key, strength_mg, freq, offset.

    Returns (times list, L list, per-drug contribution dict, dose marker
    list of (time, drug display name)).
    """
    n = int(T_END / DT) + 1
    times = [round(i * DT, 10) for i in range(n)]
    contrib = {}
    markers = []
    for entry in regimen:
        drug = DRUGS[entry["drug_key"]]
        h = hazard(drug)
        frac, _warn = dose_fraction(drug, entry["strength_mg"])
        ts = dose_times(entry["freq"], entry["offset"])
        markers.extend((tk, drug["display"]) for tk in ts if tk <= T_END)
        if h == 0.0 or drug["tmax"] is None:
            contrib[entry["drug_key"]] = [0.0] * n
            continue
        curve = []
        for t in times:
            total = 0.0
            for tk in ts:
                if t >= tk:
                    total += bateman(drug["tmax"], drug["half_life"],
                                     t - tk)
            curve.append(h * frac * total)
        contrib[entry["drug_key"]] = curve
    L = [sum(contrib[k][i] for k in contrib) for i in range(n)]
    return times, L, contrib, markers


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


def metrics(times, L):
    arr = np.asarray(L, dtype=float)
    imax = int(np.argmax(arr))
    auc = float(np.trapezoid(arr, dx=DT))
    time_above = float(np.sum(arr >= THRESHOLD) * DT)
    return float(arr[imax]), times[imax], auc, time_above


# ------------------------------------------------------- session state

def _new_row(preset=None):
    rid = uuid.uuid4().hex[:8]
    st.session_state[f"drug_{rid}"] = preset
    drug = DRUGS.get(preset) if preset else None
    st.session_state[f"mg_{rid}"] = (
        float(drug["ref_max_single_mg"])
        if drug and drug["ref_max_single_mg"] else 0.0)
    st.session_state[f"freq_{rid}"] = 1
    st.session_state[f"off_{rid}"] = 0.0
    return rid


def _drop_row(rid):
    for prefix in ("drug_", "mg_", "freq_", "off_"):
        st.session_state.pop(prefix + rid, None)
    st.session_state.row_ids.remove(rid)


def _init_state():
    if "row_ids" not in st.session_state:
        st.session_state.row_ids = []
        st.session_state.row_ids.append(_new_row("acetaminophen"))
        st.session_state.row_ids.append(_new_row("diclofenac_sodium"))


def _on_drug_change(rid):
    """Prefill strength with the reference max single dose on selection."""
    key = st.session_state.get(f"drug_{rid}")
    drug = DRUGS.get(key) if key else None
    if drug and drug["ref_max_single_mg"]:
        st.session_state[f"mg_{rid}"] = float(drug["ref_max_single_mg"])


def _reset():
    for rid in list(st.session_state.row_ids):
        for prefix in ("drug_", "mg_", "freq_", "off_"):
            st.session_state.pop(prefix + rid, None)
    st.session_state.row_ids = []
    st.session_state.row_ids.append(_new_row("acetaminophen"))
    st.session_state.row_ids.append(_new_row("diclofenac_sodium"))


# ------------------------------------------------------------------ UI

def _render_rows():
    hdr = st.columns([3.5, 2, 2.5, 0.8])
    hdr[0].markdown("**Drug**")
    hdr[1].markdown("**Strength (mg)**")
    hdr[2].markdown("**Frequency (per 24 h)**")
    hdr[3].markdown("**\u00a0**")
    for rid in list(st.session_state.row_ids):
        c1, c2, c3, c4 = st.columns([3.5, 2, 2.5, 0.8])
        with c1:
            st.selectbox(
                "Drug", options=[None] + list(DRUGS.keys()),
                format_func=lambda k: ("Search and select a drug..."
                                       if k is None
                                       else DRUGS[k]["display"]),
                key=f"drug_{rid}", label_visibility="collapsed",
                on_change=_on_drug_change, args=(rid,))
        key = st.session_state.get(f"drug_{rid}")
        drug = DRUGS.get(key) if key else None
        with c2:
            st.number_input(
                "Strength (mg)", min_value=0.0, step=25.0,
                format="%.1f", key=f"mg_{rid}",
                label_visibility="collapsed")
            if drug and drug["ref_max_single_mg"]:
                st.caption(
                    f"Ref max single dose: "
                    f"{drug['ref_max_single_mg']:.0f} mg"
                    + (f" | max daily: {drug['ref_max_daily_mg']:.0f} mg"
                       if drug["ref_max_daily_mg"] else ""))
            elif drug:
                st.caption("Control drug: no reference dose, hazard is 0.")
        with c3:
            st.selectbox(
                "Frequency", options=[1, 2, 3, 4, 5],
                format_func=lambda n: FREQ_LABELS[n],
                key=f"freq_{rid}", label_visibility="collapsed")
            with st.expander("Advanced: offset start time"):
                st.number_input(
                    "First dose at (h)", min_value=0.0, max_value=23.0,
                    step=0.5, format="%.1f", key=f"off_{rid}")
        with c4:
            st.button("✕", key=f"del_{rid}",
                      disabled=len(st.session_state.row_ids) <= 1,
                      on_click=_drop_row, args=(rid,))


def _collect_regimen():
    regimen, warnings, seen = [], [], {}
    for rid in st.session_state.row_ids:
        key = st.session_state.get(f"drug_{rid}")
        mg = float(st.session_state.get(f"mg_{rid}") or 0.0)
        if key is None or mg <= 0:
            continue
        drug = DRUGS[key]
        if key in seen:
            warnings.append(
                f"Duplicate entry: {drug['display']} appears more than "
                f"once; all its doses are summed, as entered.")
        seen[key] = True
        frac, warn = dose_fraction(drug, mg)
        if warn and drug["ref_max_single_mg"] is None \
                and hazard(drug) > 0:
            warnings.append(f"{drug['display']}: {warn}")
        daily = mg * int(st.session_state.get(f"freq_{rid}") or 1)
        if drug["ref_max_daily_mg"] and daily > drug["ref_max_daily_mg"]:
            warnings.append(
                f"{drug['display']}: entered regimen totals {daily:.0f} "
                f"mg/day, above the reference max daily dose of "
                f"{drug['ref_max_daily_mg']:.0f} mg. The curve shows "
                f"this supratherapeutic exposure as entered.")
        regimen.append({
            "drug_key": key, "strength_mg": mg,
            "freq": int(st.session_state.get(f"freq_{rid}") or 1),
            "offset": float(st.session_state.get(f"off_{rid}") or 0.0),
            "fraction": frac,
        })
    return regimen, warnings


def _chart(times, L, contrib, markers, windows):
    fig = go.Figure()
    for key, curve in contrib.items():
        fig.add_trace(go.Scatter(
            x=times, y=curve, mode="lines",
            name=DRUGS[key]["display"], opacity=0.55,
            fill="tozeroy", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(
        x=times, y=L, mode="lines", name="Combined load L(t)",
        line=dict(width=3.5, color="#111111")))
    for a, b in windows:
        fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.12,
                      line_width=0)
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="Hepatotoxic Threshold (0.75)",
                  annotation_position="top left")
    for tk in sorted({t for t, _name in markers}):
        fig.add_vline(x=tk, line_dash="dot", line_color="gray",
                      line_width=1)
    fig.update_layout(
        hovermode="x unified", height=460,
        xaxis_title="Hours after first dose",
        yaxis_title="Hepatotoxic load L(t)",
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        margin=dict(t=60, b=40, l=50, r=20))
    return fig


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
        "Build a regimen row by row: pick a drug, enter the single-dose "
        "strength in mg and how many times a day it is taken. The engine "
        "sums every dose over 24 hours, so peaks that overlap in time "
        "add up, and the same drugs staggered in time score lower.")

    _init_state()
    _render_rows()

    b1, b2, _sp = st.columns([1.2, 1, 6])
    with b1:
        if st.button("+ Add Drug",
                     disabled=len(st.session_state.row_ids) >= MAX_ROWS):
            st.session_state.row_ids.append(_new_row())
            st.rerun()
    with b2:
        if st.button("Reset"):
            _reset()
            st.rerun()

    regimen, warnings = _collect_regimen()
    for w in warnings:
        st.warning(w)
    if len(regimen) < 1:
        st.info("Select at least one drug with a strength above 0 mg "
                "to compute the combined load.")
        return

    times, L, contrib, markers = compute_load(regimen)
    lmax, t_peak, auc, time_above = metrics(times, L)
    windows = danger_windows(times, L)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak load Lmax", f"{lmax:.3f}")
    m2.metric("Time of peak (h)", f"{t_peak:.2f}")
    m3.metric("Toxic AUC (0-24 h)", f"{auc:.3f}")
    m4.metric(f"Hours above threshold {THRESHOLD}", f"{time_above:.2f}")

    if windows:
        win_txt = "; ".join(f"{a:.2f} to {b:.2f} h" for a, b in windows)
        st.error(f"Danger window(s), L(t) at or above {THRESHOLD}: "
                 f"{win_txt}")
    else:
        st.success(f"L(t) never reaches the danger threshold "
                   f"({THRESHOLD}) for this regimen.")

    st.plotly_chart(_chart(times, L, contrib, markers, windows),
                    use_container_width=True)
    dose_note = "; ".join(
        f"{DRUGS[e['drug_key']]['display'].split(',')[0]} at "
        + ", ".join(f"{t:g} h" for t in dose_times(e["freq"],
                                                   e["offset"]))
        for e in regimen)
    st.caption(f"Dose times: {dose_note}. Dotted lines on the chart "
               f"mark each administration.")

    st.subheader("Per-drug detail")
    rows = []
    for e in regimen:
        d = DRUGS[e["drug_key"]]
        rows.append({
            "Drug": d["display"],
            "Strength (mg)": e["strength_mg"],
            "Dose fraction F": round(e["fraction"], 3),
            "Frequency (/24 h)": e["freq"],
            "Offset (h)": e["offset"],
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
        "Model: Layer 2 v2 of the project 'AI Framework for Multi-Drug "
        "Combination Hepatotoxicity Risk Prediction'. Hazards from "
        "DILIrank 2.0 (FDA LTKB); pharmacokinetic fixed points from the "
        "project PK table v0.1 (07_PK_Table_v0.1.csv, verified label "
        "values, ranges replaced by midpoints); structural alert notes "
        "from 08_Structure_Alert_Matches_v1.csv. Dose fractions use "
        "standard labelled reference maxima for normalisation only and "
        "must be re-verified against current labels before any other "
        "use. Cyanocobalamin is the zero-hazard control: its hazard is "
        "0, so it never changes L(t). Research prototype only, not for "
        "clinical decision-making.")


if __name__ == "__main__":
    main()
