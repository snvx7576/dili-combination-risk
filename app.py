#!/usr/bin/env python3
"""
Layer 4 - Demo app v5: Temporal Hepatotoxic Load Explorer.

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
  - Model v2.1 metabolite adjustment (app v5): the hazard used by the
    engine is H_adj = H x 1.25 when the drug has a toxic metabolite
    that is more toxic than the parent, H x 1.10 when it has any other
    toxic metabolite, otherwise H; H_adj is capped at 1.0. hazard()
    returns H_adj; hazard_base() returns the unadjusted H, and
    compute_load(regimen, use_adjusted=False) runs the base model.

Drug PK fixed points are the Layer 2 v1 values from
data/fixed_db/07_PK_Table_v0.1.csv (ranges replaced by the same midpoints
Layer 2 used); alert notes from data/fixed_db/08_Structure_Alert_Matches_v1.csv.
With every drug at F = 1, frequency 1 and offset 0, the base model
(use_adjusted=False) reproduces Layer 2 scenario A exactly (Lmax
1.7356, Toxic AUC 11.0852); the metabolite-adjusted Model v2.1 gives
Lmax 1.8774, Toxic AUC 11.7225 for the same scenario.

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

# ------------------------------------------------------------ drug data
# v3: the drug library is data-driven. drugs_data.json sits next to this
# file in the repo and is built by code/build_app_data.py from the Fixed
# Database: 01 DILIrank labels, 08 alert matches, 10 PK table v1
# (label-sourced, returned research, Cooper-verified, midpoints of label
# ranges derived and flagged at build time), 11 metabolites v1.
# Project-verified v0.1 fixed points override returned values for the
# original demo drugs, keeping the published Layer 2 scenarios exact.
import json
import os


@st.cache_data
def _load_drugs():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "drugs_data.json")
    with open(path) as f:
        return json.load(f)["drugs"]


DRUGS = _load_drugs()
DRUG_KEYS = sorted(DRUGS.keys(), key=lambda k: DRUGS[k]["display"].lower())

# ---------------------------------------------------------------- model

def hazard_base(drug) -> float:
    """Base hazard (original Layer 2 rule, no metabolite adjustment)."""
    return (drug["severity"] / 8.0) * drug["concern_weight"]


def metabolite_factor(drug) -> float:
    """Model v2.1 uplift factor from the drug's toxic metabolite.

    x1.25 if a toxic metabolite exists and is more toxic than the
    parent; x1.10 if a toxic metabolite exists with any other
    more_toxic value; 1.0 if there is no toxic metabolite.
    """
    tm = drug.get("toxic_metabolite")
    if not tm:
        return 1.0
    return 1.25 if tm.get("more_toxic") == "YES" else 1.10


def hazard(drug) -> float:
    """Model v2.1 adjusted hazard: base hazard x metabolite factor,
    capped at 1.0."""
    return min(hazard_base(drug) * metabolite_factor(drug), 1.0)


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


def compute_load(regimen, use_adjusted=True):
    """regimen: list of dicts with keys drug_key, strength_mg, freq, offset.

    use_adjusted=True (default) applies the Model v2.1 metabolite
    adjustment via hazard(); use_adjusted=False runs the base model
    via hazard_base().

    Returns (times list, L list, per-drug contribution dict, dose marker
    list of (time, drug display name)).
    """
    n = int(T_END / DT) + 1
    times = [round(i * DT, 10) for i in range(n)]
    contrib = []
    markers = []
    for entry in regimen:
        drug = DRUGS[entry["drug_key"]]
        h = hazard(drug) if use_adjusted else hazard_base(drug)
        frac, _warn = dose_fraction(drug, entry["strength_mg"])
        ts = dose_times(entry["freq"], entry["offset"])
        markers.extend((tk, drug["display"]) for tk in ts if tk <= T_END)
        if h == 0.0 or drug["tmax"] is None:
            contrib.append((entry["drug_key"], [0.0] * n))
            continue
        curve = []
        for t in times:
            total = 0.0
            for tk in ts:
                if t >= tk:
                    total += bateman(drug["tmax"], drug["half_life"],
                                     t - tk)
            curve.append(h * frac * total)
        contrib.append((entry["drug_key"], curve))
    L = [sum(c[i] for _k, c in contrib) for i in range(n)]
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
        st.session_state.row_ids.append(_new_row("diclofenac sodium"))


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
    st.session_state.row_ids.append(_new_row("diclofenac sodium"))


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
                "Drug", options=[None] + DRUG_KEYS,
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
            st.number_input(
                "Starts later by (hours)", min_value=0.0, max_value=23.0,
                step=0.5, format="%.1f", key=f"off_{rid}",
                help="0 means this medicine is taken at the same time "
                     "as the first one. Increase it to space the "
                     "medicines out through the day.")
            st.caption("Start delay: 0 = taken with the first medicine.")
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
    for key, curve in contrib:
        fig.add_trace(go.Scatter(
            x=times, y=curve, mode="lines",
            name=DRUGS[key]["display"].split(",")[0] + " alone",
            opacity=0.55,
            fill="tozeroy", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(
        x=times, y=L, mode="lines", name="Total liver load (all together)",
        line=dict(width=3.5, color="#111111")))
    Ly = [float(v) for v in L]
    pi = int(max(range(len(Ly)), key=lambda i: Ly[i]))
    fig.add_trace(go.Scatter(
        x=[times[pi]], y=[Ly[pi]], mode="markers",
        marker=dict(size=11, color="#c62828", symbol="circle"),
        name="Peak: medicines overlap most here", showlegend=False))
    fig.add_annotation(
        x=times[pi], y=Ly[pi], ax=60, ay=-45, showarrow=True,
        arrowcolor="#c62828",
        text=f"Peak {Ly[pi]:.2f}: the medicines overlap most here")
    for a, b in windows:
        fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.12,
                      line_width=0)
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="Danger line (0.75)",
                  annotation_position="top left")
    for tk in sorted({t for t, _name in markers}):
        fig.add_vline(x=tk, line_dash="dot", line_color="gray",
                      line_width=1)
    fig.update_layout(
        title="How the liver load rises and falls over 24 hours",
        hovermode="x unified", height=460,
        xaxis_title="Hours after first dose",
        yaxis_title="Liver load",
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        margin=dict(t=60, b=40, l=50, r=20))
    return fig


# ------------------------------------------------- people-first layer

def _human_duration(hours):
    """4.7 -> '4 hours 42 minutes'; 0.3 -> 'about 18 minutes'."""
    if hours <= 0:
        return "no time"
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    parts = []
    if h:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    return " ".join(parts) if parts else "under a minute"


def _verdict(lmax):
    """(level, colour, headline) for a peak load."""
    if lmax >= THRESHOLD:
        return "HIGH", "#c62828", "HIGH liver stress"
    if lmax >= 0.4:
        return "MODERATE", "#ef6c00", "MODERATE liver stress"
    return "LOW", "#2e7d32", "LOW liver stress"


def _timeline_html(times, L):
    """24 h strip: green safe, amber elevated, red danger."""
    segs = []
    def cls(v):
        return 2 if v >= THRESHOLD else (1 if v >= 0.4 else 0)
    start, cur = 0, cls(L[0])
    for i in range(1, len(L)):
        if cls(L[i]) != cur:
            segs.append((times[start], times[i - 1], cur))
            start, cur = i, cls(L[i])
    segs.append((times[start], times[-1], cur))
    colours = {0: "#43a047", 1: "#fb8c00", 2: "#c62828"}
    parts = ['<div style="display:flex;height:30px;border-radius:8px;'
             'overflow:hidden;border:1px solid rgba(128,128,128,.4);">']
    for a, b, c in segs:
        w = max((b - a) / T_END * 100.0, 0.4)
        parts.append(f'<div style="width:{w:.2f}%;background:'
                     f'{colours[c]};"></div>')
    parts.append("</div>")
    parts.append(
        '<div style="display:flex;justify-content:space-between;'
        'font-size:11px;opacity:.7;margin-top:4px;"><span>First dose '
        'taken</span><span>6 h</span><span>12 h</span><span>18 h</span>'
        '<span>24 h</span></div>')
    parts.append(
        '<div style="font-size:12px;margin-top:8px;">'
        '<span style="color:#c62828;">&#9632;</span> Danger hours &nbsp;'
        '<span style="color:#fb8c00;">&#9632;</span> Raised stress &nbsp;'
        '<span style="color:#43a047;">&#9632;</span> Safe hours</div>')
    return "".join(parts)


def _stagger_advice(regimen, cur_lmax):
    """Search single-drug delays; return the best plain-language advice.

    Tries delaying each drug on its own by 0.5 to 12 h. Prefers the
    smallest delay that brings the peak below the danger line; if no
    delay gets there, returns the delay with the lowest peak.
    Returns (drug display, delay h, new Lmax, new hours above) or None.
    """
    if len(regimen) < 2 or cur_lmax < THRESHOLD:
        return None
    best_clear = None   # (delay, idx, lmax, above)
    best_any = None     # (lmax, delay, idx, above)
    for idx in range(len(regimen)):
        for step in range(1, 25):
            d = step * 0.5
            trial = [dict(e) for e in regimen]
            trial[idx]["offset"] = trial[idx]["offset"] + d
            _t, L2, _c, _m = compute_load(trial)
            lm2, _tp, _auc, ab2 = metrics(_t, L2)
            if lm2 < THRESHOLD and (best_clear is None
                                    or d < best_clear[0]):
                best_clear = (d, idx, lm2, ab2)
            if best_any is None or lm2 < best_any[0]:
                best_any = (lm2, d, idx, ab2)
    if best_clear is not None:
        d, idx, lm2, ab2 = best_clear
        return (DRUGS[regimen[idx]["drug_key"]]["display"], d, lm2, ab2)
    if best_any is not None and best_any[0] < cur_lmax - 1e-9:
        lm2, d, idx, ab2 = best_any
        return (DRUGS[regimen[idx]["drug_key"]]["display"], d, lm2, ab2)
    return None


def main():
    st.set_page_config(page_title="HepMAT - Multi-drug Hepatotoxicity Assessment Tool",
                       layout="wide")
    st.title("HepMAT: Multi-drug Hepatotoxicity Assessment Tool")
    st.caption(
        "Research prototype only. Not for clinical decision-making. "
        "Scores are model estimates from public data (DILIrank 2.0, "
        "FAERS, published pharmacokinetics), not a patient's risk.")

    st.markdown(
        "Add your medicines below, with the dose in mg and how many "
        "times a day each is taken. You get one clear answer: how much "
        "stress the combination puts on the liver, and when.")

    st.markdown(
        '<div style="display:flex;gap:10px;flex-wrap:wrap;'
        'margin:4px 0 16px;">'
        '<div style="flex:1;min-width:180px;border:1px solid '
        'rgba(128,128,128,.35);border-radius:12px;padding:10px 14px;">'
        '<b>1. Pick your medicines</b><br><span style="font-size:13px;'
        'opacity:.8;">Add what you took, the dose in mg, and how '
        'often.</span></div>'
        '<div style="flex:1;min-width:180px;border:1px solid '
        'rgba(128,128,128,.35);border-radius:12px;padding:10px 14px;">'
        '<b>2. See the danger time</b><br><span style="font-size:13px;'
        'opacity:.8;">Red time means the liver is under heavy load. '
        'Green time is calm.</span></div>'
        '<div style="flex:1;min-width:180px;border:1px solid '
        'rgba(128,128,128,.35);border-radius:12px;padding:10px 14px;">'
        '<b>3. Space them out</b><br><span style="font-size:13px;'
        'opacity:.8;">Below, we show one safer gap between doses, in '
        'plain words.</span></div>'
        '</div>',
        unsafe_allow_html=True)
    st.markdown("#### Your medicines")

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
    if len(regimen) < 1:
        st.info("Select at least one drug with a strength above 0 mg "
                "to see the result.")
        return

    times, L, contrib, markers = compute_load(regimen)
    lmax, t_peak, auc, time_above = metrics(times, L)
    _tb, L_base, _cb, _mb = compute_load(regimen, use_adjusted=False)
    base_lmax, _btp, _bauc, _bta = metrics(times, L_base)
    windows = danger_windows(times, L)
    level, colour, headline = _verdict(lmax)

    # ---- Design A: verdict card -------------------------------------
    if level == "HIGH":
        sentence = (
            "Taken together, these medicines push the liver into the "
            f"danger zone for about {_human_duration(time_above)}.")
    elif level == "MODERATE":
        sentence = (
            "Together these medicines cause noticeable liver stress. "
            "The stress stays below the danger line, peaking about "
            f"{_human_duration(t_peak)} after the first dose.")
    else:
        sentence = (
            "Together these medicines cause only low liver stress, "
            "well below the danger line at all times.")
    meter_pct = min(lmax / 2.0, 1.0) * 100.0
    st.markdown(
        f'<div style="background:{colour};color:#ffffff;border-radius:'
        f'14px 14px 0 0;padding:16px 20px;">'
        f'<div style="font-size:28px;font-weight:800;">{headline}</div>'
        f'<div style="font-size:15px;opacity:.95;">{sentence}</div>'
        f'</div>',
        unsafe_allow_html=True)
    card = ['<div style="border:1px solid rgba(128,128,128,.4);'
            'border-top:none;border-radius:0 0 14px 14px;'
            'padding:14px 20px 18px;">']
    card.append('<div style="font-size:12px;opacity:.7;">LIVER STRESS '
                'METER</div>')
    card.append(
        '<div style="position:relative;height:14px;border-radius:7px;'
        'margin-top:6px;background:linear-gradient(90deg,#43a047 0%,'
        '#fdd835 55%,#c62828 100%);">'
        f'<div style="position:absolute;left:{meter_pct:.0f}%;top:-5px;'
        'width:4px;height:24px;background:#888;border-radius:2px;">'
        '</div></div>')
    card.append('<div style="display:flex;justify-content:space-between;'
                'font-size:11px;opacity:.7;margin-top:4px;"><span>Low'
                '</span><span>Medium</span><span>High</span></div>')
    imax = int(np.argmax(np.asarray(L, dtype=float)))
    shares = []
    for e, (key, curve) in zip(regimen, contrib):
        v = curve[imax]
        shares.append((v, DRUGS[key]["display"]))
    shares.sort(reverse=True)
    card.append('<div style="font-size:12px;opacity:.7;margin-top:14px;">'
                'HOW MUCH EACH MEDICINE ADDS AT THE WORST MOMENT</div>')
    for v, name in shares:
        if lmax <= 0 or v <= 0:
            tag, wdt = "no effect", 0.0
        else:
            share = v / lmax
            wdt = share * 100.0
            tag = ("the main driver" if share >= 0.5 else
                   "adds a lot" if share >= 0.25 else "adds a little")
        short = name.split(",")[0]
        card.append(
            f'<div style="font-size:13px;margin-top:7px;">{short} '
            f'<span style="display:inline-block;vertical-align:middle;'
            f'width:{max(wdt, 1.5):.0f}%;max-width:55%;height:10px;'
            f'background:{colour};border-radius:5px;margin:0 8px;">'
            f'</span> {tag}</div>')
    card.append("</div>")
    st.markdown("".join(card), unsafe_allow_html=True)

    # ---- Design B: your day, plus what to do -------------------------
    st.subheader("Your day, hour by hour")
    st.markdown(_timeline_html(times, L), unsafe_allow_html=True)
    if windows:
        win_txt = "; ".join(
            f"from about {_human_duration(a)} to about "
            f"{_human_duration(b)} after the first dose"
            for a, b in windows)
        st.markdown(f"**Danger time:** {win_txt}.")
    else:
        st.markdown("**No danger time:** the stress never reaches the "
                    "danger line in 24 hours.")
    advice = _stagger_advice(regimen, lmax)
    if advice is not None:
        name, delay, new_lmax, new_above = advice
        short = name.split(",")[0]
        if new_above <= 0:
            effect = "the danger time disappears completely"
        else:
            effect = (f"the danger time shrinks to about "
                      f"{_human_duration(new_above)}")
        if new_above <= 0:
            intro = "One change that helps a lot"
            tail = ""
        else:
            intro = "One change that helps"
            tail = (" The combination still enters the danger zone, "
                    "because one of these medicines reaches the danger "
                    "line on its own at this dose.")
        st.success(
            f"{intro}: take **{short}** about "
            f"**{_human_duration(delay)} later** than the other "
            f"medicine(s). The peak stress falls from {lmax:.2f} to "
            f"{new_lmax:.2f}, and {effect}.{tail} Please speak to a "
            f"doctor or pharmacist before changing how you take any "
            f"medicine.")

    # ---- Notes, tucked aside -----------------------------------------
    notes = list(warnings)
    any_adjusted = False
    for e in regimen:
        d = DRUGS[e["drug_key"]]
        tm = d.get("toxic_metabolite")
        if tm:
            factor = metabolite_factor(d)
            if factor > 1.0:
                any_adjusted = True
            notes.append(
                f"Toxic metabolite: {d['display'].split(',')[0]} forms "
                f"{tm['name']}"
                + (f" via {tm['enzyme']}" if tm["enzyme"] else "")
                + (", more toxic than the parent drug."
                   if tm["more_toxic"] == "YES" else ".")
                + f" The model raises this medicine's stress weight "
                  f"x{factor:.2f} because of it.")
    if any_adjusted:
        notes.append(
            "Metabolite adjustment is ON: a medicine that turns into "
            "a more toxic metabolite inside the body gets a higher "
            "stress weight (x1.25), and one with a toxic metabolite "
            "gets x1.10. This is the project's metabolite "
            "adjustment rule.")
    if notes:
        st.subheader("Good to know about these medicines")
        with st.container():
            for n in notes:
                st.markdown(f"- {n}")

    # ---- The science, always open -------------------------------------
    st.divider()
    st.subheader("The full picture: graph, numbers and medicine details")
    st.caption("Everything below stays open on the page. Nothing here "
               "is hidden behind a dropdown.")
    with st.container():
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Peak load (metabolite-adjusted)", f"{lmax:.3f}")
        m2.metric("Peak load (base model)", f"{base_lmax:.3f}")
        m3.metric("Time of peak (h)", f"{t_peak:.2f}")
        m4.metric("Toxic AUC (0-24 h)", f"{auc:.3f}")
        m5.metric(f"Hours above threshold {THRESHOLD}",
                  f"{time_above:.2f}")
        st.plotly_chart(_chart(times, L, contrib, markers, windows),
                        use_container_width=True)
        st.markdown(
            "**How to read this graph, in plain words:** the thick "
            "black line is the total load on the liver. The coloured "
            "lines show what each medicine would do on its own. Where "
            "the black line rises above the red dashed danger line, "
            "the liver is in the danger zone, and the pink band marks "
            "those hours. The marked peak is where the medicines "
            "overlap the most.")
        dose_note = "; ".join(
            f"{DRUGS[e['drug_key']]['display'].split(',')[0]} at "
            + ", ".join(f"{t:g} h" for t in dose_times(e["freq"],
                                                       e["offset"]))
            for e in regimen)
        st.caption(f"Dose times: {dose_note}.")
        st.subheader("Per-drug detail")
        rows = []
        for e in regimen:
            d = DRUGS[e["drug_key"]]
            tm = d.get("toxic_metabolite")
            factor = metabolite_factor(d)
            if factor == 1.25:
                adj_note = (f"x1.25 ({tm['name']} more toxic than "
                            f"parent)")
            elif factor == 1.10:
                adj_note = "x1.10"
            else:
                adj_note = "none"
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
                "Hazard H (base)": round(hazard_base(d), 4),
                "Hazard H (adjusted)": round(hazard(d), 4),
                "Metabolite adjustment": adj_note,
                "Formulation": d.get("formulation", ""),
                "Main enzymes": d.get("enzymes", ""),
                "Toxic metabolite": (d["toxic_metabolite"]["name"]
                                     if d.get("toxic_metabolite") else ""),
                "Structural alert note": d["alert_note"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption(
        "About this tool: HepMAT comes from the research project "
        "'An Artificial Intelligence Framework for Multi-Drug "
        "Combination Hepatotoxicity Risk Prediction with Machine "
        "Learning, Structural Alerts and Temporal Pharmacokinetic "
        "Modelling'. How the score is built: each medicine's hazard "
        "comes from DILIrank 2.0 (FDA Liver Toxicity Knowledge "
        "Base). A medicine that turns into a more toxic breakdown "
        "product (metabolite) inside the body has its hazard raised "
        "by 1.25 times, and one with any other toxic metabolite by "
        "1.10 times; the raised hazard is capped at the maximum, so "
        "the adjustment does not change medicines already at maximum "
        "hazard. Pharmacokinetic and metabolite values come from "
        "the project's fixed database of label-sourced values (where "
        "a label gives a range, its midpoint is used), and the "
        "structural alert notes come from the project's alert "
        "matching. Dose fractions use labelled reference maxima for "
        "normalisation only. Cyanocobalamin (vitamin B12) is the "
        "zero-hazard control: its hazard is 0, so it never changes "
        "the result. Research prototype only, not for clinical "
        "decision-making.")


if __name__ == "__main__":
    main()
