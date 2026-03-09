"""
Molecular Intelligence Explorer  —  app.py
Run:  streamlit run app.py
"""

import json
import re
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils import (
    fetch_pubchem_data,
    get_pubchem_image_url,
    get_pubchem_sdf,
    compute_rdkit_descriptors,
    lipinski_analysis,
    predict_solubility,
    generate_ai_explanation,
    compare_molecules,
    tanimoto_similarity,
    build_export_dataframe,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Molecular Intelligence Explorer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #080d16 !important;
    color: #e2e8f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1280px !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #94a3b8 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: #0f1c2e !important;
    color: #64748b !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.3rem 0.5rem !important;
    box-shadow: none !important;
    transition: all 0.15s !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #162033 !important;
    border-color: #2dd4bf !important;
    color: #2dd4bf !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── main search button ── */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.3rem !important;
    box-shadow: 0 0 20px rgba(6,182,212,0.3) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 30px rgba(6,182,212,0.5) !important;
}

/* ── text input ── */
.stTextInput > div > div > input {
    background: #0f1c2e !important;
    border: 1.5px solid #1e2d45 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #334155 !important; }
.stTextInput label { color: #334155 !important; font-size: 0.01rem !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1a !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 3px !important;
    border: 1px solid #1e2d45 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.1rem !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(59,130,246,0.2)) !important;
    color: #06b6d4 !important;
}

/* ── expander ── */
.streamlit-expanderHeader {
    background: #0f1c2e !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    border: 1px solid #1e2d45 !important;
}
.streamlit-expanderContent {
    background: #0a1220 !important;
    border-radius: 0 0 10px 10px !important;
    border: 1px solid #1e2d45 !important;
    border-top: none !important;
}

/* ── download button ── */
.stDownloadButton > button {
    background: #0f1c2e !important;
    color: #64748b !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
    transform: none !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    border-color: #06b6d4 !important;
    color: #06b6d4 !important;
    background: #0f1c2e !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── misc ── */
.stToggle > label { color: #64748b !important; font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important; }
.stSpinner > div { border-top-color: #06b6d4 !important; }
p { color: #94a3b8 !important; }
h1,h2,h3 { color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080d16; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── Reusable components ───────────────────────────────────────────────────────

def card(inner, pad="1.4rem", accent=None):
    border = f"border-left:3px solid {accent};" if accent else ""
    return f"""<div style="background:#0f1c2e;border:1px solid #1e2d45;
        border-radius:14px;padding:{pad};margin-bottom:0.9rem;{border}">{inner}</div>"""

def glow_card(inner, pad="1.4rem", glow="#06b6d4"):
    return f"""<div style="background:#0f1c2e;border:1px solid #1e2d45;
        border-radius:14px;padding:{pad};margin-bottom:0.9rem;
        box-shadow:0 0 30px {glow}18;">{inner}</div>"""

def metric_box(value, label, color):
    return f"""<div style="background:#0f1c2e;border:1px solid #1e2d45;
        border-radius:12px;padding:0.9rem 0.6rem;text-align:center;
        box-shadow:0 0 20px {color}15;">
        <div style="font-size:1.35rem;font-weight:700;color:{color};letter-spacing:-0.02em;line-height:1">{value}</div>
        <div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.25rem">{label}</div>
    </div>"""

def badge(text, color):
    return f"""<span style="background:{color}1a;color:{color};border:1px solid {color}33;
        border-radius:6px;font-size:0.72rem;font-weight:500;padding:0.12rem 0.6rem;
        margin:0.1rem;display:inline-block">{text}</span>"""

def section_head(icon, text, color="#06b6d4"):
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.5rem;
        margin:1.4rem 0 0.75rem;border-left:3px solid {color};padding-left:0.7rem;">
        <span>{icon}</span>
        <span style="font-size:0.97rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em">{text}</span>
    </div>""", unsafe_allow_html=True)

def prop_row(label, value, unit=""):
    val_str = f"{value} {unit}".strip() if str(value) not in ("—", "", "None") else "—"
    return f"""<div style="display:flex;justify-content:space-between;align-items:center;
        padding:0.38rem 0;border-bottom:1px solid #0f1c2e">
        <span style="color:#475569;font-size:0.8rem">{label}</span>
        <span style="color:#cbd5e1;font-weight:600;font-size:0.82rem;
              font-family:'JetBrains Mono',monospace">{val_str}</span>
    </div>"""

def rule_row(name, passed, val, threshold):
    icon = "✓" if passed else "✗"
    bg   = "#052e16" if passed else "#2d0a0a"
    clr  = "#10b981" if passed else "#f87171"
    return f"""<div style="display:flex;align-items:center;gap:0.6rem;
        padding:0.5rem 0.75rem;border-radius:8px;background:{bg};
        border:1px solid {clr}22;margin-bottom:0.4rem">
        <span style="color:{clr};font-weight:700;font-size:0.85rem;width:1rem">{icon}</span>
        <span style="color:#cbd5e1;font-size:0.83rem;flex:1">{name}</span>
        <span style="color:#475569;font-size:0.78rem;font-family:'JetBrains Mono',monospace">{val} / {threshold}</span>
    </div>"""


# ── Plotly charts  (NOTE: no 'margin' in BASE — added per-chart to avoid conflict) ──

def chart_base():
    """Return base layout dict WITHOUT margin — add margin per chart."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#64748b"),
    )

def make_radar(d):
    def norm(v, div, offset=0, cap=10):
        return min(max((float(v) + offset) / div, 0), cap)
    cats = ["MW", "LogP", "TPSA", "HB Don.", "HB Acc.", "RotBonds", "Rings"]
    vals = [
        norm(d.get("MolWt",300),   60),
        norm(d.get("LogP",2),      1.6, offset=3),
        norm(d.get("TPSA",60),     16),
        norm(d.get("HBD",2)*2,     1),
        norm(d.get("HBA",4),       1),
        min(float(d.get("RotBonds",3)), 10),
        min(float(d.get("RingCount",1))*2, 10),
    ]
    cats2 = cats + [cats[0]]; vals2 = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals2, theta=cats2, fill="toself",
        fillcolor="rgba(6,182,212,0.1)",
        line=dict(color="#06b6d4", width=2),
        marker=dict(size=5, color="#06b6d4"),
    ))
    base = chart_base()
    fig.update_layout(**base, height=270, showlegend=False,
        margin=dict(l=10,r=10,t=20,b=10),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,10],
                tickfont=dict(size=7, color="#334155"),
                gridcolor="#0f1c2e", linecolor="#0f1c2e"),
            angularaxis=dict(
                tickfont=dict(size=10, color="#64748b"),
                gridcolor="#0f1c2e", linecolor="#0f1c2e"),
        ))
    return fig

def make_pie(ac):
    colors = ["#06b6d4","#3b82f6","#8b5cf6","#10b981","#f59e0b","#f43f5e","#f97316","#84cc16"]
    fig = go.Figure(go.Pie(
        labels=list(ac.keys()), values=list(ac.values()), hole=0.55,
        marker=dict(colors=colors[:len(ac)], line=dict(color="#080d16", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="%{label}: %{value} atoms<extra></extra>",
    ))
    base = chart_base()
    fig.update_layout(**base, height=265,
        margin=dict(l=5,r=5,t=10,b=30),
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    return fig

def make_bond_bar(bt):
    color_map = {"SINGLE":"#06b6d4","DOUBLE":"#3b82f6","TRIPLE":"#8b5cf6","AROMATIC":"#10b981"}
    labels = [k for k,v in bt.items() if v > 0]
    values = [bt[k] for k in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=[color_map.get(l,"#64748b") for l in labels],
                    cornerradius=6,
                    line=dict(width=0)),
        text=values, textposition="outside",
        textfont=dict(color="#94a3b8", size=13),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    base = chart_base()
    fig.update_layout(**base, height=205,
        margin=dict(l=5,r=20,t=15,b=5),
        xaxis=dict(tickfont=dict(size=11, color="#64748b"), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(showticklabels=False, gridcolor="#0f1c2e"),
    )
    return fig

def make_gauge(score, label, color):
    # NOTE: margin here, NOT in BASE — this fixes the "multiple values" crash
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%", font=dict(size=24, color="#e2e8f0", family="Inter")),
        gauge=dict(
            axis=dict(range=[0,100], tickfont=dict(size=7, color="#334155"),
                      tickcolor="#1e2d45", nticks=6),
            bar=dict(color=color, thickness=0.2),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,35],  color="#2d0a0a"),
                dict(range=[35,65], color="#1c1a08"),
                dict(range=[65,100],color="#052e16"),
            ],
        ),
        title=dict(text=label, font=dict(size=11, color="#64748b", family="Inter")),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=185,
        margin=dict(l=15, r=15, t=30, b=5),   # ← margin only here, not spread across dicts
    )
    return fig

def make_cmp_bar(comp, n1, n2):
    labels = list(comp.keys())
    v1 = [float(comp[k][0]) for k in labels]
    v2 = [float(comp[k][1]) for k in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=n1, x=labels, y=v1,
        marker=dict(color="#06b6d4", cornerradius=5, line=dict(width=0))))
    fig.add_trace(go.Bar(name=n2, x=labels, y=v2,
        marker=dict(color="#8b5cf6", cornerradius=5, line=dict(width=0))))
    base = chart_base()
    fig.update_layout(**base, barmode="group", height=290,
        margin=dict(l=5,r=5,t=20,b=5),
        xaxis=dict(tickfont=dict(size=9, color="#64748b"), gridcolor="#0f1c2e"),
        yaxis=dict(tickfont=dict(size=9, color="#64748b"), gridcolor="#0f1c2e"),
        legend=dict(font=dict(size=11, color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    return fig


# ── 3D viewer ────────────────────────────────────────────────────────────────
def viewer_html(sdf, h=390):
    safe = sdf.replace("\\","\\\\").replace("`","\\`").replace("$","\\$")
    return f"""<!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <style>
      body{{margin:0;background:#080d16;}}
      #v{{width:100%;height:{h}px;border-radius:12px;overflow:hidden;}}
    </style></head><body>
    <div id="v"></div>
    <script>
      let v = $3Dmol.createViewer("v", {{backgroundColor:"#080d16"}});
      v.addModel(`{safe}`, "sdf");
      v.setStyle({{}}, {{
        stick: {{radius:0.12, colorscheme:"Jmol"}},
        sphere: {{scale:0.24, colorscheme:"Jmol"}}
      }});
      v.zoomTo(); v.spin(true); v.render();
    </script></body></html>"""


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.3rem">
      <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.02em">
        ⚗️ MolExplorer
      </div>
      <div style="font-size:0.7rem;color:#334155;margin-top:0.1rem">Chemistry Intelligence</div>
    </div>
    <div style="height:1px;background:#1e2d45;margin:0.6rem 0"></div>
    <div style="font-size:0.65rem;color:#334155;text-transform:uppercase;
                letter-spacing:0.09em;margin-bottom:0.5rem">Quick Examples</div>
    """, unsafe_allow_html=True)

    examples = [
        ("💊","Aspirin"),   ("☕","Caffeine"),
        ("💧","Water"),     ("🌿","Benzene"),
        ("🧪","Glucose"),   ("💉","Penicillin G"),
        ("🟡","Cholesterol"),("💙","Ibuprofen"),
        ("🫁","Dopamine"),  ("🍺","Ethanol"),
        ("🌸","Morphine"),  ("🧬","Adenine"),
    ]
    cols = st.columns(2)
    for i, (icon, name) in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"{icon} {name}", key=f"sb_{name}", use_container_width=True):
                st.session_state["sq"] = name
                st.session_state["trigger"] = True

    st.markdown('<div style="height:1px;background:#1e2d45;margin:0.75rem 0"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.4rem">Options</div>', unsafe_allow_html=True)
    show_3d = st.toggle("🔭 3D Viewer",      value=True)
    show_ai = st.toggle("🤖 AI Explanation",  value=True)


# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#0a0f1a;border:1px solid #1e2d45;border-radius:18px;
            padding:1.8rem 2.2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-80px;right:-80px;width:260px;height:260px;
              background:radial-gradient(circle,rgba(6,182,212,0.12),transparent 65%);
              border-radius:50%;pointer-events:none"></div>
  <div style="position:absolute;bottom:-60px;left:40%;width:200px;height:200px;
              background:radial-gradient(circle,rgba(139,92,246,0.1),transparent 65%);
              border-radius:50%;pointer-events:none"></div>
  <div style="position:relative;z-index:1">
    <div style="display:inline-flex;align-items:center;gap:0.4rem;
                background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);
                border-radius:6px;font-size:0.65rem;font-weight:600;color:#06b6d4;
                padding:0.15rem 0.65rem;letter-spacing:0.09em;
                text-transform:uppercase;margin-bottom:0.65rem">
      🔬 CHEMISTRY INTELLIGENCE
    </div>
    <div style="font-size:1.85rem;font-weight:700;color:#f1f5f9;
                letter-spacing:-0.03em;line-height:1.15;margin-bottom:0.4rem">
      Molecular Intelligence
      <span style="background:linear-gradient(90deg,#06b6d4,#8b5cf6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        Explorer
      </span>
    </div>
    <div style="color:#475569;font-size:0.85rem;max-width:480px;line-height:1.6">
      Search any element, molecule, or compound — instant structure, properties,
      drug-likeness, 3D viewer, and plain-English AI explanations.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  Explore", "⚖️  Compare", "📖  About"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORE
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    sc1, sc2 = st.columns([5, 1])
    with sc1:
        query = st.text_input(
            "q", label_visibility="collapsed",
            value=st.session_state.get("sq", ""),
            placeholder="🔎  Search — Aspirin · C6H6 · CCO · Caffeine · Dopamine",
            key="main_q",
        )
    with sc2:
        go_btn = st.button("Search →", use_container_width=True)

    run = go_btn or st.session_state.pop("trigger", False)

    # ── empty state ──────────────────────────────────────────────────────────
    if not query and not run:
        st.markdown(card("""
        <div style="text-align:center;padding:3rem 1rem">
          <div style="font-size:3rem;margin-bottom:0.75rem">⚗️</div>
          <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:0.35rem">
            Search any molecule to begin
          </div>
          <div style="color:#334155;font-size:0.85rem">
            Try: Aspirin · Caffeine · Benzene · Water · Dopamine
          </div>
        </div>"""), unsafe_allow_html=True)

    elif run and query:
        st.session_state["sq"] = query

        with st.spinner(f"Fetching **{query}** from PubChem…"):
            pub = fetch_pubchem_data(query)

        if not pub:
            st.error(f"❌  **{query}** not found. Try a different name, formula, or SMILES.")
            st.stop()

        smiles  = pub.get("CanonicalSMILES", "")
        cid     = pub.get("CID")
        rd      = compute_rdkit_descriptors(smiles) if smiles else {}
        lip     = lipinski_analysis({**pub, **rd})
        logp_v  = float(rd.get("LogP",  pub.get("XLogP", 2.5)) or 2.5)
        mw_v    = float(rd.get("MolWt", pub.get("MolecularWeight", 300)) or 300)
        sol     = predict_solubility(logp_v, mw_v)

        # ── identity card ─────────────────────────────────────────────────────
        iupac   = pub.get("IUPACName", query)
        formula = pub.get("MolecularFormula", "—")
        syns    = pub.get("Synonyms", [])[:5]
        badges  = " ".join(badge(s, "#06b6d4") for s in syns)
        badges += " " + badge(formula, "#8b5cf6")
        badges += " " + badge(f"CID {cid}", "#3b82f6")

        st.markdown(glow_card(f"""
        <div style="font-size:1.45rem;font-weight:700;color:#f1f5f9;
                    letter-spacing:-0.02em;margin-bottom:0.2rem">{iupac.title()}</div>
        <div style="color:#334155;font-size:0.82rem;margin-bottom:0.55rem">
            {formula} &nbsp;·&nbsp; PubChem CID {cid} &nbsp;·&nbsp;
            MW {pub.get("MolecularWeight","—")} g/mol
        </div>
        <div style="line-height:2">{badges}</div>
        """), unsafe_allow_html=True)

        # ── 6 metric cards ────────────────────────────────────────────────────
        m_cols = st.columns(6)
        metrics = [
            (rd.get("MolWt",    pub.get("MolecularWeight","—")), "Mol Weight",    "#06b6d4"),
            (rd.get("LogP",     pub.get("XLogP","—")),           "LogP",          "#3b82f6"),
            (rd.get("TPSA",     pub.get("TPSA","—")),            "TPSA Ų",        "#8b5cf6"),
            (rd.get("HBD",      pub.get("HBondDonorCount","—")), "HB Donors",     "#10b981"),
            (rd.get("HBA",      pub.get("HBondAcceptorCount","—")),"HB Accept.",  "#f59e0b"),
            (rd.get("QED","—"),                                    "QED Score",    "#f43f5e"),
        ]
        for col, (val, lbl, clr) in zip(m_cols, metrics):
            with col:
                st.markdown(metric_box(val, lbl, clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── two-column layout ────────────────────────────────────────────────
        left, right = st.columns([1, 1], gap="large")

        # LEFT ─ structures
        with left:
            section_head("🖼️", "2D Structure", "#06b6d4")
            if cid:
                img_url = get_pubchem_image_url(cid, 500)
                st.markdown(card(f"""
                <div style="text-align:center;background:#080d16;
                            border-radius:10px;padding:1.25rem">
                  <img src="{img_url}"
                       style="max-width:100%;border-radius:8px;
                              filter:invert(1) hue-rotate(180deg) brightness(0.85)"
                       alt="2D structure"/>
                </div>"""), unsafe_allow_html=True)

            if show_3d:
                section_head("🔭", "3D Interactive Viewer", "#3b82f6")
                with st.spinner("Building 3D model…"):
                    sdf = get_pubchem_sdf(cid) if cid else None
                if sdf:
                    st.components.v1.html(viewer_html(sdf, 390), height=390)
                    st.caption("Drag to rotate · Scroll to zoom · Auto-spinning")
                else:
                    st.markdown(card('<div style="color:#334155;text-align:center;padding:1rem">3D structure unavailable</div>'), unsafe_allow_html=True)

        # RIGHT ─ properties
        with right:
            section_head("🔬", "Molecular Properties", "#8b5cf6")

            all_props = [
                ("Molecular Weight",     rd.get("MolWt",    pub.get("MolecularWeight","—")),       "g/mol"),
                ("Exact Mass",           rd.get("ExactMolWt",pub.get("ExactMass","—")),             "g/mol"),
                ("LogP",                 rd.get("LogP",     pub.get("XLogP","—")),                  ""),
                ("TPSA",                 rd.get("TPSA",     pub.get("TPSA","—")),                    "Ų"),
                ("H-Bond Donors",        rd.get("HBD",      pub.get("HBondDonorCount","—")),         ""),
                ("H-Bond Acceptors",     rd.get("HBA",      pub.get("HBondAcceptorCount","—")),      ""),
                ("Rotatable Bonds",      rd.get("RotBonds","—"),                                     ""),
                ("Heavy Atoms",          rd.get("HeavyAtoms",pub.get("HeavyAtomCount","—")),         ""),
                ("Ring Count",           rd.get("RingCount","—"),                                    ""),
                ("Aromatic Rings",       rd.get("AromaticRings","—"),                                ""),
                ("sp³ Carbon Fraction",  rd.get("FractionCSP3","—"),                                 ""),
                ("Molar Refractivity",   rd.get("MolarRefractivity","—"),                            ""),
                ("Chiral Centers",       rd.get("NumChiralCenters","—"),                             ""),
                ("Heteroatoms",          rd.get("NumHeteroatoms","—"),                               ""),
                ("Total Bonds",          rd.get("NumBonds","—"),                                     ""),
                ("QED Score",            rd.get("QED","—"),                                          "/ 1.0"),
            ]
            rows_html = "".join(prop_row(p, v, u) for p, v, u in all_props)
            st.markdown(card(rows_html), unsafe_allow_html=True)

            section_head("🔗", "SMILES Strings", "#10b981")
            can = pub.get("CanonicalSMILES", "—")
            iso = pub.get("IsomericSMILES", "—")
            st.markdown(card(f"""
            <div style="margin-bottom:0.8rem">
              <div style="font-size:0.64rem;color:#334155;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:0.25rem">Canonical</div>
              <code style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;
                           color:#06b6d4;word-break:break-all;line-height:1.7">{can}</code>
            </div>
            <div>
              <div style="font-size:0.64rem;color:#334155;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:0.25rem">Isomeric</div>
              <code style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;
                           color:#8b5cf6;word-break:break-all;line-height:1.7">{iso}</code>
            </div>"""), unsafe_allow_html=True)

        # ── Lipinski + Solubility ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section_head("💊", "Drug-Likeness · Lipinski's Rule of Five", "#f59e0b")

        dl_col, sl_col = st.columns([1, 1], gap="large")

        with dl_col:
            rules_html = "".join(
                rule_row(name, ok, val, thresh)
                for name, (ok, val, thresh) in lip["rules"].items()
            )
            vc = lip["verdict_color"]
            vt = lip["verdict"]
            p  = lip["passed"]
            rules_html += f"""
            <div style="margin-top:0.8rem;padding:0.75rem;border-radius:10px;
                        background:{vc}12;border:1px solid {vc}30;text-align:center">
              <span style="font-weight:700;font-size:0.95rem;color:{vc}">{vt}</span>
              <span style="color:#334155;font-size:0.8rem"> &nbsp;·&nbsp; {p}/4 rules passed</span>
            </div>"""
            st.markdown(card(rules_html), unsafe_allow_html=True)

        with sl_col:
            section_head("💧", "Solubility Prediction", "#3b82f6")
            st.plotly_chart(
                make_gauge(sol["score"], sol["category"], sol["color"]),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.markdown(card(f"""
            <div style="text-align:center">
              <div style="font-weight:700;color:{sol['color']};font-size:0.95rem">
                {sol['category']}
              </div>
              <div style="color:#475569;font-size:0.8rem;margin-top:0.2rem">{sol['detail']}</div>
              <div style="color:#334155;font-size:0.73rem;margin-top:0.2rem;
                          font-family:'JetBrains Mono',monospace">
                log<i>S</i> ≈ {sol['logS']}
              </div>
            </div>""", "1rem"), unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section_head("📊", "Property Visualisations", "#06b6d4")
        ch1, ch2, ch3 = st.columns([1.1, 1, 0.85], gap="large")

        with ch1:
            st.markdown('<div style="font-size:0.75rem;color:#334155;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.06em">Property Radar</div>', unsafe_allow_html=True)
            if rd:
                st.plotly_chart(make_radar(rd), use_container_width=True, config={"displayModeBar":False})

        with ch2:
            st.markdown('<div style="font-size:0.75rem;color:#334155;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.06em">Atom Composition</div>', unsafe_allow_html=True)
            ac = rd.get("AtomComposition", {})
            if ac:
                st.plotly_chart(make_pie(ac), use_container_width=True, config={"displayModeBar":False})
            else:
                st.markdown('<div style="color:#334155;font-size:0.82rem;padding:1rem">No data</div>', unsafe_allow_html=True)

        with ch3:
            st.markdown('<div style="font-size:0.75rem;color:#334155;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.06em">Bond Types</div>', unsafe_allow_html=True)
            bt = rd.get("BondTypes", {})
            if bt and any(bt.values()):
                st.plotly_chart(make_bond_bar(bt), use_container_width=True, config={"displayModeBar":False})

        # ── AI Explanation ────────────────────────────────────────────────────
        if show_ai:
            st.markdown("<br>", unsafe_allow_html=True)
            section_head("🤖", "AI Explanation  —  Student Friendly", "#10b981")
            exp_text = generate_ai_explanation(pub, rd, lip)
            lines = exp_text.split("\n\n")
            rendered = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    quote = line.lstrip(">").strip()
                    rendered += f"""<div style="border-left:3px solid #06b6d4;
                        padding:0.6rem 0.9rem;margin:0.5rem 0;color:#475569;
                        font-size:0.82rem;font-style:italic;line-height:1.65;
                        background:#06b6d41a;border-radius:0 8px 8px 0">{quote}</div>"""
                else:
                    line_html = re.sub(
                        r"\*\*(.+?)\*\*",
                        r'<strong style="color:#06b6d4">\1</strong>',
                        line,
                    )
                    rendered += f'<p style="margin:0.5rem 0;line-height:1.75;color:#64748b;font-size:0.87rem">{line_html}</p>'
            st.markdown(card(rendered, "1.3rem"), unsafe_allow_html=True)

        # ── Description ───────────────────────────────────────────────────────
        desc_text = pub.get("Description", "")
        if desc_text and len(desc_text) > 40:
            with st.expander("📄  Full PubChem Description"):
                st.markdown(
                    f'<div style="color:#64748b;line-height:1.7;font-size:0.85rem;padding:0.5rem">{desc_text}</div>',
                    unsafe_allow_html=True,
                )

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section_head("📥", "Export Report", "#3b82f6")
        ex1, ex2 = st.columns(2)
        df_exp = build_export_dataframe(pub, rd, lip, sol)
        with ex1:
            if df_exp is not None:
                st.download_button(
                    "⬇️  Download CSV",
                    df_exp.to_csv(index=False).encode(),
                    file_name=f"{query.replace(' ','_')}.csv",
                    mime="text/csv", use_container_width=True,
                )
        with ex2:
            jdata = json.dumps({
                "pubchem":   pub,
                "rdkit":     {k: v for k, v in rd.items() if not isinstance(v, dict)},
                "lipinski":  {"passed": lip["passed"], "verdict": lip["verdict"]},
                "solubility": sol,
            }, indent=2)
            st.download_button(
                "⬇️  Download JSON",
                jdata.encode(),
                file_name=f"{query.replace(' ','_')}.json",
                mime="application/json", use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    section_head("⚖️", "Side-by-Side Molecule Comparison", "#06b6d4")
    st.markdown('<div style="color:#334155;font-size:0.85rem;margin-bottom:1rem">Compare two molecules — properties, similarity score, and charts</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1: mq1 = st.text_input("Molecule A", placeholder="e.g. Aspirin",    key="cmp1")
    with c2: mq2 = st.text_input("Molecule B", placeholder="e.g. Ibuprofen",  key="cmp2")
    cmp_go = st.button("⚖️  Compare Now", key="cmp_btn")

    if cmp_go and mq1 and mq2:
        with st.spinner("Fetching both molecules…"):
            d1 = fetch_pubchem_data(mq1)
            d2 = fetch_pubchem_data(mq2)

        if not d1 or not d2:
            st.error("Could not find one or both molecules. Please check the names.")
        else:
            r1 = compute_rdkit_descriptors(d1.get("CanonicalSMILES",""))
            r2 = compute_rdkit_descriptors(d2.get("CanonicalSMILES",""))
            l1 = lipinski_analysis({**d1, **r1})
            l2 = lipinski_analysis({**d2, **r2})
            n1 = d1.get("IUPACName", mq1).title()
            n2 = d2.get("IUPACName", mq2).title()

            ic1, ic2 = st.columns(2, gap="large")
            for col, data, rdesc, lip_r, name, glow in [
                (ic1, d1, r1, l1, n1, "#06b6d4"),
                (ic2, d2, r2, l2, n2, "#8b5cf6"),
            ]:
                with col:
                    cid_i = data.get("CID")
                    img   = get_pubchem_image_url(cid_i, 260) if cid_i else ""
                    img_t = (f'<img src="{img}" style="max-width:140px;margin:0.6rem 0;'
                             f'border-radius:8px;filter:invert(1) hue-rotate(180deg) brightness(0.85)"/>') if img else ""
                    st.markdown(glow_card(f"""
                    <div style="text-align:center">
                      <div style="font-weight:700;font-size:0.97rem;color:#e2e8f0">{name}</div>
                      <div style="color:#475569;font-size:0.78rem">{data.get("MolecularFormula","—")}</div>
                      {img_t}
                      <div>
                        {badge("MW: "+str(rdesc.get("MolWt","—")), "#06b6d4")}
                        {badge("LogP: "+str(rdesc.get("LogP","—")), "#8b5cf6")}
                        {badge(lip_r["verdict"].split()[0], "#10b981")}
                      </div>
                    </div>""", glow=glow), unsafe_allow_html=True)

            # Tanimoto similarity
            sim = tanimoto_similarity(
                d1.get("CanonicalSMILES",""), d2.get("CanonicalSMILES","")
            )
            if sim is not None:
                pct   = int(sim * 100)
                sc    = "#10b981" if pct > 60 else ("#f59e0b" if pct > 30 else "#f43f5e")
                label = ("Highly Similar" if pct > 60
                         else "Moderately Similar" if pct > 30 else "Structurally Dissimilar")
                st.markdown(card(f"""
                <div style="text-align:center;padding:0.4rem">
                  <div style="font-size:0.65rem;color:#334155;text-transform:uppercase;
                              letter-spacing:0.09em;margin-bottom:0.3rem">
                    Tanimoto Similarity · Morgan Fingerprints
                  </div>
                  <div style="font-size:2.8rem;font-weight:700;color:{sc};
                              letter-spacing:-0.04em;line-height:1">{sim}</div>
                  <div style="color:#475569;font-size:0.83rem;margin-top:0.25rem">
                    {label} — {pct}% structural overlap
                  </div>
                </div>"""), unsafe_allow_html=True)

            # Chart + table
            comp = compare_molecules(r1, r2)
            if comp:
                section_head("📊", "Property Chart", "#06b6d4")
                st.plotly_chart(make_cmp_bar(comp, n1, n2),
                                use_container_width=True, config={"displayModeBar":False})

                section_head("📋", "Comparison Table", "#8b5cf6")
                rows = [
                    {"Property": prop, n1: v1, n2: v2,
                     "Winner": f"🔵 {n1}" if w == 1 else (f"🟣 {n2}" if w == 2 else "—")}
                    for prop, (v1, v2, w) in comp.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    section_head("🧬", "About Molecular Intelligence Explorer", "#06b6d4")

    st.markdown(card("""
    <p style="color:#64748b;line-height:1.75;font-size:0.88rem;margin:0">
      Built as a <strong style="color:#06b6d4">BTech CSE (AI/ML) Semester 2 Chemistry Project</strong>.
      Combines cheminformatics, data science, and a minimal dark-lab UI to make
      molecular science accessible to students and researchers.
    </p>
    """), unsafe_allow_html=True)

    section_head("🏗️", "Architecture", "#3b82f6")
    a1, a2 = st.columns(2, gap="large")
    arch = [
        ("Frontend",  "Streamlit + Custom CSS dark lab theme"),
        ("Backend",   "Python · PubChem REST API · RDKit"),
        ("3D Viewer", "py3Dmol — WebGL embedded iframe"),
        ("Charts",    "Plotly — Radar · Pie · Bar · Gauge"),
    ]
    for i, (k, v) in enumerate(arch):
        with (a1 if i % 2 == 0 else a2):
            st.markdown(card(f"""
            <div style="font-size:0.63rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:0.2rem">{k}</div>
            <div style="color:#cbd5e1;font-size:0.85rem">{v}</div>
            """, "0.85rem"), unsafe_allow_html=True)

    section_head("⚗️", "Key Algorithms", "#8b5cf6")
    algorithms = [
        ("Lipinski's Rule of Five",   "Oral bioavailability drug screening — MW, LogP, HBD, HBA thresholds"),
        ("Yalkowsky Solubility",      "log S ≈ 0.5 − log P − 0.01 × (MW − 100)"),
        ("Morgan Fingerprints + Tanimoto", "Circular fingerprints for structural similarity (0–1 score)"),
        ("QED Score",                 "Quantitative Estimate of Drug-likeness (Bickerton et al., 2012)"),
    ]
    for name, desc in algorithms:
        st.markdown(card(f"""
        <div style="font-weight:600;color:#e2e8f0;font-size:0.88rem;margin-bottom:0.2rem">{name}</div>
        <div style="color:#475569;font-size:0.82rem">{desc}</div>
        """, "0.85rem"), unsafe_allow_html=True)

    section_head("📦", "Tech Stack", "#10b981")
    stack = ["Python 3.10+", "Streamlit", "RDKit", "PubChem API", "py3Dmol", "Plotly", "Pandas"]
    badges_html = " ".join(badge(t, "#06b6d4") for t in stack)
    st.markdown(card(f'<div style="line-height:2.2">{badges_html}</div>', "1rem"), unsafe_allow_html=True)
