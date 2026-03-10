"""
Molecular Intelligence Explorer — hehehe.py
Just run:  streamlit run hehehe.py
"""

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════
import json
import re
import urllib.request
import urllib.parse

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# BACKEND — PubChem API
# ═══════════════════════════════════════════════════════════════

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _get_json(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MolExplorer/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def fetch_pubchem_data(query):
    encoded = urllib.parse.quote(query)
    cid = None
    for url in [
        f"{PUBCHEM}/compound/name/{encoded}/cids/JSON",
        f"{PUBCHEM}/compound/smiles/{encoded}/cids/JSON",
        f"{PUBCHEM}/compound/formula/{encoded}/cids/JSON",
    ]:
        d = _get_json(url)
        if d:
            ids = d.get("IdentifierList", {}).get("CID", [])
            if ids:
                cid = ids[0]
                break
    if not cid:
        return None

    props = (
        "IUPACName,MolecularFormula,MolecularWeight,"
        "CanonicalSMILES,IsomericSMILES,XLogP,TPSA,"
        "HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount,HeavyAtomCount,"
        "ExactMass,Complexity,Charge"
    )
    data = _get_json(f"{PUBCHEM}/compound/cid/{cid}/property/{props}/JSON")
    if not data:
        return None

    p = data.get("PropertyTable", {}).get("Properties", [{}])[0]
    p["CID"] = cid

    # synonyms
    sd = _get_json(f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON")
    p["Synonyms"] = (sd or {}).get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])[:5]

    # description
    dd = _get_json(f"{PUBCHEM}/compound/cid/{cid}/description/JSON")
    p["Description"] = ""
    if dd:
        for info in dd.get("InformationList", {}).get("Information", []):
            if "Description" in info:
                p["Description"] = info["Description"]
                break
    return p


def get_image_url(cid, size=500):
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?record_type=2d&image_size={size}x{size}"


def get_sdf(cid):
    try:
        req = urllib.request.Request(
            f"{PUBCHEM}/compound/cid/{cid}/SDF?record_type=3d",
            headers={"User-Agent": "MolExplorer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# BACKEND — RDKit descriptors
# ═══════════════════════════════════════════════════════════════

def rdkit_descriptors(smiles):
    """Returns rich descriptor dict, or {} if RDKit unavailable."""
    if not smiles:
        return {}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, QED as QEDmod
    except ImportError:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    d = {}
    d["MolWt"]             = round(Descriptors.MolWt(mol), 3)
    d["ExactMolWt"]        = round(Descriptors.ExactMolWt(mol), 5)
    d["LogP"]              = round(Descriptors.MolLogP(mol), 3)
    d["TPSA"]              = round(Descriptors.TPSA(mol), 2)
    d["HBD"]               = Lipinski.NumHDonors(mol)
    d["HBA"]               = Lipinski.NumHAcceptors(mol)
    d["RotBonds"]          = rdMolDescriptors.CalcNumRotatableBonds(mol)
    d["HeavyAtoms"]        = mol.GetNumHeavyAtoms()
    d["RingCount"]         = rdMolDescriptors.CalcNumRings(mol)
    d["AromaticRings"]     = rdMolDescriptors.CalcNumAromaticRings(mol)
    d["FractionCSP3"]      = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)
    d["MolarRefractivity"] = round(Descriptors.MolMR(mol), 3)
    d["NumAtoms"]          = mol.GetNumAtoms()
    d["NumBonds"]          = mol.GetNumBonds()
    d["NumChiralCenters"]  = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    d["NumHeteroatoms"]    = rdMolDescriptors.CalcNumHeteroatoms(mol)

    try:
        d["QED"] = round(QEDmod.qed(mol), 4)
    except Exception:
        d["QED"] = None

    # atom composition
    ac = {}
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        ac[s] = ac.get(s, 0) + 1
    d["AtomComposition"] = ac

    # bond types
    from rdkit.Chem import rdchem
    bc = {"SINGLE": 0, "DOUBLE": 0, "TRIPLE": 0, "AROMATIC": 0}
    for bond in mol.GetBonds():
        bt = str(bond.GetBondType())
        for k in bc:
            if k in bt:
                bc[k] += 1
                break
    d["BondTypes"] = bc
    return d


def tanimoto(s1, s2):
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        m1 = Chem.MolFromSmiles(s1)
        m2 = Chem.MolFromSmiles(s2)
        if m1 and m2:
            f1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
            f2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
            return round(DataStructs.TanimotoSimilarity(f1, f2), 4)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# BACKEND — Drug-likeness, Solubility, Explanation
# ═══════════════════════════════════════════════════════════════

def lipinski(rd, pub):
    """Evaluate Lipinski Ro5. rd = RDKit dict, pub = PubChem dict."""
    def _f(key_rd, key_pub, default):
        v = rd.get(key_rd, pub.get(key_pub, default))
        try: return float(v)
        except: return default

    mw   = _f("MolWt",    "MolecularWeight",    9999)
    logp = _f("LogP",     "XLogP",              99)
    hbd  = _f("HBD",      "HBondDonorCount",    99)
    hba  = _f("HBA",      "HBondAcceptorCount", 99)

    rules = {
        "Molecular Weight ≤ 500 Da": (mw   <= 500, round(mw, 2),   "≤ 500"),
        "LogP ≤ 5":                  (logp <= 5,   round(logp, 2), "≤ 5"),
        "H-Bond Donors ≤ 5":         (hbd  <= 5,   int(hbd),       "≤ 5"),
        "H-Bond Acceptors ≤ 10":     (hba  <= 10,  int(hba),       "≤ 10"),
    }
    passed = sum(1 for ok, _, _ in rules.values() if ok)

    if passed == 4:
        verdict, vc = "✅ Excellent Drug Candidate", "#10b981"
    elif passed == 3:
        verdict, vc = "🟡 Moderate Drug Candidate",  "#f59e0b"
    else:
        verdict, vc = "❌ Poor Drug Candidate",       "#f43f5e"

    return {"rules": rules, "passed": passed, "verdict": verdict, "vc": vc}


def solubility(logp_val, mw_val):
    logs = 0.5 - logp_val - 0.01 * (mw_val - 100)
    if   logs > -1: return {"cat": "Highly Soluble",      "color": "#10b981", "score": 90, "detail": "> 10 mg/mL",       "logS": round(logs,2)}
    elif logs > -3: return {"cat": "Moderately Soluble",  "color": "#f59e0b", "score": 60, "detail": "1–10 mg/mL",       "logS": round(logs,2)}
    elif logs > -5: return {"cat": "Poorly Soluble",      "color": "#f97316", "score": 30, "detail": "0.1–1 mg/mL",      "logS": round(logs,2)}
    else:           return {"cat": "Practically Insoluble","color": "#f43f5e","score": 10, "detail": "< 0.1 mg/mL",      "logS": round(logs,2)}


def ai_explanation(pub, rd, lip):
    name    = pub.get("IUPACName", "this molecule")
    formula = pub.get("MolecularFormula", "—")
    mw      = rd.get("MolWt",   pub.get("MolecularWeight", "?"))
    logp    = rd.get("LogP",    pub.get("XLogP", "?"))
    hbd     = rd.get("HBD",     pub.get("HBondDonorCount", "?"))
    hba     = rd.get("HBA",     pub.get("HBondAcceptorCount", "?"))
    rings   = rd.get("RingCount", 0)
    arom    = rd.get("AromaticRings", 0)
    qed     = rd.get("QED")

    lines = [f"**{name.title()}** has molecular formula **{formula}** and molecular weight **{mw} g/mol**."]

    try:
        lv = float(logp)
        if   lv < 0: lines.append("It is **hydrophilic** (water-loving) — dissolves well in water.")
        elif lv < 2: lines.append("It has **balanced polarity** — dissolves in both water and fats.")
        elif lv < 5: lines.append("It is **lipophilic** (fat-loving) — can cross cell membranes easily.")
        else:        lines.append("It is **highly lipophilic** — may have poor water solubility.")
    except: pass

    try:
        lines.append(f"It can form **{int(hbd)} H-bond donor(s)** and **{int(hba)} H-bond acceptor(s)**, which affects how it binds to proteins.")
    except: pass

    if rings > 0:
        rt = f"It has **{rings} ring(s)**"
        if arom > 0: rt += f", of which **{arom}** are aromatic (flat, stable π-systems)"
        lines.append(rt + ".")

    p = lip.get("passed", 0)
    v = lip.get("verdict", "")
    lines.append(f"Lipinski analysis: **{p}/4 rules passed** — {v.split(' ',1)[-1]}.")

    if qed is not None:
        lines.append(f"QED drug-likeness score: **{qed}/1.0** — {'good' if qed > 0.5 else 'moderate' if qed > 0.3 else 'low'} drug potential.")

    desc = pub.get("Description", "")
    if desc and len(desc) > 30:
        lines.append(f"\n> {desc[:280]}{'…' if len(desc)>280 else ''}")

    return "\n\n".join(lines)


def compare_mols(rd1, rd2, pub1, pub2):
    """
    Compare two molecules using RDKit data first, PubChem as fallback.
    This is the FIXED version that always returns real values.
    """
    def get(rd, pub, rk, pk):
        v = rd.get(rk)
        if v is None and pk:
            v = pub.get(pk)
        # convert string numbers
        if isinstance(v, str):
            try: v = float(v)
            except: v = None
        return v

    fields = [
        ("MolWt",         "MolecularWeight",    "Molecular Weight",  "lower"),
        ("LogP",          "XLogP",              "LogP",              "lower"),
        ("TPSA",          "TPSA",               "TPSA (Ų)",          "lower"),
        ("HBD",           "HBondDonorCount",    "H-Bond Donors",     "lower"),
        ("HBA",           "HBondAcceptorCount", "H-Bond Acceptors",  "lower"),
        ("RotBonds",      "RotatableBondCount", "Rotatable Bonds",   "lower"),
        ("HeavyAtoms",    "HeavyAtomCount",     "Heavy Atoms",       "neutral"),
        ("RingCount",     None,                 "Ring Count",        "neutral"),
        ("AromaticRings", None,                 "Aromatic Rings",    "neutral"),
        ("QED",           None,                 "QED Score",         "higher"),
    ]

    result = {}
    for rk, pk, label, prefer in fields:
        v1 = get(rd1, pub1, rk, pk)
        v2 = get(rd2, pub2, rk, pk)
        if v1 is None and v2 is None:
            continue
        v1 = v1 if v1 is not None else 0
        v2 = v2 if v2 is not None else 0
        try:
            f1, f2 = float(v1), float(v2)
            if   prefer == "lower":  winner = 1 if f1 < f2 else (2 if f2 < f1 else 0)
            elif prefer == "higher": winner = 1 if f1 > f2 else (2 if f2 > f1 else 0)
            else:                    winner = 0
            result[label] = (round(f1,3), round(f2,3), winner)
        except:
            result[label] = (v1, v2, 0)
    return result


def export_df(pub, rd, lip, sol):
    try:
        import pandas as pd
        row = {k: pub.get(k,"") for k in ["IUPACName","MolecularFormula","MolecularWeight","CanonicalSMILES","CID"]}
        for k in ["MolWt","LogP","TPSA","HBD","HBA","RotBonds","HeavyAtoms","RingCount","AromaticRings","QED"]:
            row[k] = rd.get(k, pub.get(k,""))
        row["Lipinski"] = f"{lip.get('passed',0)}/4 — {lip.get('verdict','')}"
        row["Solubility"] = sol.get("cat","")
        row["LogS"] = sol.get("logS","")
        return pd.DataFrame([row])
    except:
        return None


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Molecular Intelligence Explorer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #080d16 !important;
    color: #e2e8f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1300px !important; }

[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #0f1c2e !important; color: #64748b !important;
    border: 1px solid #1e2d45 !important; border-radius: 8px !important;
    font-size: 0.8rem !important; font-family: 'Inter',sans-serif !important;
    padding: 0.3rem 0.5rem !important; box-shadow: none !important;
    transition: all 0.15s !important; font-weight: 400 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #162033 !important; border-color: #06b6d4 !important;
    color: #06b6d4 !important; transform: none !important; box-shadow: none !important;
}

.stButton > button {
    background: linear-gradient(135deg,#06b6d4,#3b82f6) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-family: 'Inter',sans-serif !important;
    font-size: 0.9rem !important; padding: 0.6rem 1.3rem !important;
    box-shadow: 0 0 20px rgba(6,182,212,0.3) !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 0 30px rgba(6,182,212,0.5) !important; }

.stTextInput > div > div > input {
    background: #0f1c2e !important; border: 1.5px solid #1e2d45 !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
    font-family: 'Inter',sans-serif !important; font-size: 0.95rem !important;
    padding: 0.7rem 1.1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #06b6d4 !important; box-shadow: 0 0 0 3px rgba(6,182,212,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #334155 !important; }
.stTextInput label { color: #334155 !important; font-size: 0.01rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1a !important; border-radius: 12px !important;
    padding: 4px !important; gap: 3px !important; border: 1px solid #1e2d45 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #475569 !important;
    font-family: 'Inter',sans-serif !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 0.45rem 1.1rem !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(6,182,212,0.2),rgba(59,130,246,0.2)) !important;
    color: #06b6d4 !important;
}

.streamlit-expanderHeader {
    background: #0f1c2e !important; border-radius: 10px !important;
    color: #64748b !important; font-family: 'Inter',sans-serif !important;
    border: 1px solid #1e2d45 !important;
}
.streamlit-expanderContent {
    background: #0a1220 !important; border: 1px solid #1e2d45 !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

.stDownloadButton > button {
    background: #0f1c2e !important; color: #64748b !important;
    border: 1px solid #1e2d45 !important; border-radius: 8px !important;
    font-family: 'Inter',sans-serif !important; font-size: 0.85rem !important;
    box-shadow: none !important; transform: none !important;
}
.stDownloadButton > button:hover {
    border-color: #06b6d4 !important; color: #06b6d4 !important;
    background: #0f1c2e !important; transform: none !important; box-shadow: none !important;
}

.stToggle > label { color: #64748b !important; font-family: 'Inter',sans-serif !important; font-size: 0.85rem !important; }
.stSpinner > div { border-top-color: #06b6d4 !important; }
p { color: #94a3b8 !important; }
h1,h2,h3 { color: #e2e8f0 !important; font-family: 'Inter',sans-serif !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080d16; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════

def card(inner, pad="1.4rem", accent=None):
    b = f"border-left:3px solid {accent};" if accent else ""
    return f'<div style="background:#0f1c2e;border:1px solid #1e2d45;border-radius:14px;padding:{pad};margin-bottom:0.9rem;{b}">{inner}</div>'

def glow_card(inner, pad="1.4rem", glow="#06b6d4"):
    return f'<div style="background:#0f1c2e;border:1px solid #1e2d45;border-radius:14px;padding:{pad};margin-bottom:0.9rem;box-shadow:0 0 30px {glow}18;">{inner}</div>'

def metric_box(value, label, color):
    return f'<div style="background:#0f1c2e;border:1px solid #1e2d45;border-radius:12px;padding:0.9rem 0.6rem;text-align:center;box-shadow:0 0 20px {color}15;"><div style="font-size:1.35rem;font-weight:700;color:{color};letter-spacing:-0.02em;line-height:1">{value}</div><div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.25rem">{label}</div></div>'

def badge(text, color):
    return f'<span style="background:{color}1a;color:{color};border:1px solid {color}33;border-radius:6px;font-size:0.72rem;font-weight:500;padding:0.12rem 0.6rem;margin:0.1rem;display:inline-block">{text}</span>'

def shead(icon, text, color="#06b6d4"):
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.5rem;margin:1.4rem 0 0.75rem;border-left:3px solid {color};padding-left:0.7rem;"><span>{icon}</span><span style="font-size:0.97rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em">{text}</span></div>', unsafe_allow_html=True)

def prow(label, value, unit=""):
    vs = f"{value} {unit}".strip() if str(value) not in ("—","","None","nan") else "—"
    return f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.38rem 0;border-bottom:1px solid #0f1c2e"><span style="color:#475569;font-size:0.8rem">{label}</span><span style="color:#cbd5e1;font-weight:600;font-size:0.82rem;font-family:\'JetBrains Mono\',monospace">{vs}</span></div>'

def rulerow(name, passed, val, threshold):
    icon = "✓" if passed else "✗"
    bg   = "#052e16" if passed else "#2d0a0a"
    clr  = "#10b981" if passed else "#f87171"
    return f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0.75rem;border-radius:8px;background:{bg};border:1px solid {clr}22;margin-bottom:0.4rem"><span style="color:{clr};font-weight:700;font-size:0.85rem;width:1rem">{icon}</span><span style="color:#cbd5e1;font-size:0.83rem;flex:1">{name}</span><span style="color:#475569;font-size:0.78rem;font-family:\'JetBrains Mono\',monospace">{val} / {threshold}</span></div>'


# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════

BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#64748b"))

def make_radar(d):
    def n(v, div, off=0): return min(max((float(v)+off)/div,0),10)
    cats = ["MW","LogP","TPSA","HB Don.","HB Acc.","RotBonds","Rings"]
    vals = [
        n(d.get("MolWt",300),60),
        n(d.get("LogP",2),1.6,3),
        n(d.get("TPSA",60),16),
        min(float(d.get("HBD",2))*2,10),
        min(float(d.get("HBA",4)),10),
        min(float(d.get("RotBonds",3)),10),
        min(float(d.get("RingCount",1))*2,10),
    ]
    c2,v2 = cats+[cats[0]], vals+[vals[0]]
    fig = go.Figure(go.Scatterpolar(r=v2,theta=c2,fill="toself",
        fillcolor="rgba(6,182,212,0.1)",line=dict(color="#06b6d4",width=2),marker=dict(size=5,color="#06b6d4")))
    fig.update_layout(**BASE,height=270,showlegend=False,margin=dict(l=10,r=10,t=20,b=10),
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,range=[0,10],tickfont=dict(size=7,color="#334155"),gridcolor="#0f1c2e",linecolor="#0f1c2e"),
            angularaxis=dict(tickfont=dict(size=10,color="#64748b"),gridcolor="#0f1c2e",linecolor="#0f1c2e")))
    return fig

def make_pie(ac):
    clrs=["#06b6d4","#3b82f6","#8b5cf6","#10b981","#f59e0b","#f43f5e","#f97316","#84cc16"]
    fig=go.Figure(go.Pie(labels=list(ac.keys()),values=list(ac.values()),hole=0.55,
        marker=dict(colors=clrs[:len(ac)],line=dict(color="#080d16",width=2)),
        textinfo="label+percent",textfont=dict(size=11,color="#e2e8f0")))
    fig.update_layout(**BASE,height=265,margin=dict(l=5,r=5,t=10,b=30),
        legend=dict(font=dict(size=10),orientation="h",yanchor="bottom",y=-0.3,xanchor="center",x=0.5))
    return fig

def make_bond_bar(bt):
    cm={"SINGLE":"#06b6d4","DOUBLE":"#3b82f6","TRIPLE":"#8b5cf6","AROMATIC":"#10b981"}
    lb=[k for k,v in bt.items() if v>0]; vs=[bt[k] for k in lb]
    fig=go.Figure(go.Bar(x=lb,y=vs,marker=dict(color=[cm.get(l,"#64748b") for l in lb],cornerradius=6,line=dict(width=0)),
        text=vs,textposition="outside",textfont=dict(color="#94a3b8",size=13)))
    fig.update_layout(**BASE,height=205,margin=dict(l=5,r=20,t=15,b=5),
        xaxis=dict(tickfont=dict(size=11,color="#64748b"),gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(showticklabels=False,gridcolor="#0f1c2e"))
    return fig

def make_gauge(score, label, color):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=score,
        number=dict(suffix="%",font=dict(size=24,color="#e2e8f0",family="Inter")),
        gauge=dict(axis=dict(range=[0,100],tickfont=dict(size=7,color="#334155"),tickcolor="#1e2d45",nticks=6),
            bar=dict(color=color,thickness=0.2),bgcolor="rgba(0,0,0,0)",borderwidth=0,
            steps=[dict(range=[0,35],color="#2d0a0a"),dict(range=[35,65],color="#1c1a08"),dict(range=[65,100],color="#052e16")]),
        title=dict(text=label,font=dict(size=11,color="#64748b",family="Inter"))))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),height=185,margin=dict(l=15,r=15,t=30,b=5))
    return fig

def make_cmp_bar(comp, n1, n2):
    labels=list(comp.keys()); v1=[float(comp[k][0]) for k in labels]; v2=[float(comp[k][1]) for k in labels]
    fig=go.Figure()
    fig.add_trace(go.Bar(name=n1,x=labels,y=v1,marker=dict(color="#06b6d4",cornerradius=5,line=dict(width=0))))
    fig.add_trace(go.Bar(name=n2,x=labels,y=v2,marker=dict(color="#8b5cf6",cornerradius=5,line=dict(width=0))))
    fig.update_layout(**BASE,barmode="group",height=290,margin=dict(l=5,r=5,t=20,b=5),
        xaxis=dict(tickfont=dict(size=9,color="#64748b"),gridcolor="#0f1c2e"),
        yaxis=dict(tickfont=dict(size=9,color="#64748b"),gridcolor="#0f1c2e"),
        legend=dict(font=dict(size=11,color="#94a3b8"),bgcolor="rgba(0,0,0,0)",borderwidth=0))
    return fig


# ═══════════════════════════════════════════════════════════════
# 3D VIEWER
# ═══════════════════════════════════════════════════════════════

def viewer_html(sdf, h=390):
    safe = sdf.replace("\\","\\\\").replace("`","\\`").replace("$","\\$")
    return f"""<!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <style>body{{margin:0;background:#080d16;}}#v{{width:100%;height:{h}px;border-radius:12px;overflow:hidden;}}</style>
    </head><body><div id="v"></div>
    <script>
      let v=$3Dmol.createViewer("v",{{backgroundColor:"#080d16"}});
      v.addModel(`{safe}`,"sdf");
      v.setStyle({{}},{{stick:{{radius:0.12,colorscheme:"Jmol"}},sphere:{{scale:0.24,colorscheme:"Jmol"}}}});
      v.zoomTo();v.spin(true);v.render();
    </script></body></html>"""


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.3rem">
      <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.02em">⚗️ MolExplorer</div>
      <div style="font-size:0.7rem;color:#334155;margin-top:0.1rem">Chemistry Intelligence</div>
    </div>
    <div style="height:1px;background:#1e2d45;margin:0.6rem 0"></div>
    <div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.5rem">Quick Examples</div>
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
    show_3d = st.toggle("🔭 3D Viewer",      value=True)
    show_ai = st.toggle("🤖 AI Explanation",  value=True)


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:#0a0f1a;border:1px solid #1e2d45;border-radius:18px;
            padding:1.8rem 2.2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-80px;right:-80px;width:260px;height:260px;
              background:radial-gradient(circle,rgba(6,182,212,0.12),transparent 65%);border-radius:50%"></div>
  <div style="position:absolute;bottom:-60px;left:40%;width:200px;height:200px;
              background:radial-gradient(circle,rgba(139,92,246,0.1),transparent 65%);border-radius:50%"></div>
  <div style="position:relative;z-index:1">
    <div style="display:inline-flex;align-items:center;gap:0.4rem;background:rgba(6,182,212,0.1);
                border:1px solid rgba(6,182,212,0.25);border-radius:6px;font-size:0.65rem;
                font-weight:600;color:#06b6d4;padding:0.15rem 0.65rem;letter-spacing:0.09em;
                text-transform:uppercase;margin-bottom:0.65rem">🔬 CHEMISTRY INTELLIGENCE</div>
    <div style="font-size:1.85rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.03em;
                line-height:1.15;margin-bottom:0.4rem">
      Molecular Intelligence
      <span style="background:linear-gradient(90deg,#06b6d4,#8b5cf6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent"> Explorer</span>
    </div>
    <div style="color:#475569;font-size:0.85rem;max-width:500px;line-height:1.6">
      Search any element, molecule, or compound — instant structure, properties,
      drug-likeness, 3D viewer, and plain-English AI explanations.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔍  Explore", "⚖️  Compare", "📖  About"])


# ───────────────────────────────────────────────────────────────
# TAB 1 — EXPLORE
# ───────────────────────────────────────────────────────────────
with tab1:
    sc1, sc2 = st.columns([5,1])
    with sc1:
        query = st.text_input("q", label_visibility="collapsed",
            value=st.session_state.get("sq",""),
            placeholder="🔎  Search — Aspirin · C6H6 · CCO · Caffeine · Dopamine",
            key="main_q")
    with sc2:
        go_btn = st.button("Search →", use_container_width=True)

    run = go_btn or st.session_state.pop("trigger", False)

    if not query and not run:
        st.markdown(card("""
        <div style="text-align:center;padding:3rem 1rem">
          <div style="font-size:3rem;margin-bottom:0.75rem">⚗️</div>
          <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:0.35rem">Search any molecule to begin</div>
          <div style="color:#334155;font-size:0.85rem">Try: Aspirin · Caffeine · Benzene · Water · Dopamine</div>
        </div>"""), unsafe_allow_html=True)

    elif run and query:
        st.session_state["sq"] = query

        with st.spinner(f"Fetching **{query}**…"):
            pub = fetch_pubchem_data(query)

        if not pub:
            st.error(f"❌  **{query}** not found. Try a different name, formula, or SMILES.")
            st.stop()

        smiles = pub.get("CanonicalSMILES","")
        cid    = pub.get("CID")
        rd     = rdkit_descriptors(smiles)

        # get numeric logp and mw from whichever source has them
        logp_v = float(rd.get("LogP", pub.get("XLogP") or 2.5) or 2.5)
        mw_v   = float(rd.get("MolWt", pub.get("MolecularWeight") or 300) or 300)

        lip    = lipinski(rd, pub)
        sol    = solubility(logp_v, mw_v)

        # identity card
        iupac   = pub.get("IUPACName", query)
        formula = pub.get("MolecularFormula","—")
        syns    = pub.get("Synonyms",[])[:5]
        b_html  = " ".join(badge(s,"#06b6d4") for s in syns)
        b_html += " " + badge(formula,"#8b5cf6")
        b_html += " " + badge(f"CID {cid}","#3b82f6")

        st.markdown(glow_card(f"""
        <div style="font-size:1.45rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;margin-bottom:0.2rem">{iupac.title()}</div>
        <div style="color:#334155;font-size:0.82rem;margin-bottom:0.55rem">{formula} &nbsp;·&nbsp; CID {cid} &nbsp;·&nbsp; MW {pub.get("MolecularWeight","—")} g/mol</div>
        <div style="line-height:2">{b_html}</div>
        """), unsafe_allow_html=True)

        # metrics
        mc = st.columns(6)
        metrics = [
            (rd.get("MolWt",    pub.get("MolecularWeight","—")), "Mol Weight",   "#06b6d4"),
            (rd.get("LogP",     pub.get("XLogP","—")),           "LogP",         "#3b82f6"),
            (rd.get("TPSA",     pub.get("TPSA","—")),            "TPSA Ų",       "#8b5cf6"),
            (rd.get("HBD",      pub.get("HBondDonorCount","—")), "HB Donors",    "#10b981"),
            (rd.get("HBA",      pub.get("HBondAcceptorCount","—")),"HB Accept.", "#f59e0b"),
            (rd.get("QED","—"),                                   "QED Score",   "#f43f5e"),
        ]
        for col,(val,lbl,clr) in zip(mc,metrics):
            with col: st.markdown(metric_box(val,lbl,clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([1,1], gap="large")

        with left:
            shead("🖼️","2D Structure","#06b6d4")
            if cid:
                st.markdown(card(f'<div style="text-align:center;background:#080d16;border-radius:10px;padding:1.25rem"><img src="{get_image_url(cid,500)}" style="max-width:100%;border-radius:8px;filter:invert(1) hue-rotate(180deg) brightness(0.85)"/></div>'), unsafe_allow_html=True)

            if show_3d:
                shead("🔭","3D Interactive Viewer","#3b82f6")
                with st.spinner("Building 3D model…"):
                    sdf = get_sdf(cid) if cid else None
                if sdf:
                    st.components.v1.html(viewer_html(sdf,390), height=390)
                    st.caption("Drag to rotate · Scroll to zoom · Auto-spinning")
                else:
                    st.markdown(card('<div style="color:#334155;text-align:center;padding:1rem">3D structure unavailable</div>'), unsafe_allow_html=True)

        with right:
            shead("🔬","Molecular Properties","#8b5cf6")
            props = [
                ("Molecular Weight",    rd.get("MolWt",    pub.get("MolecularWeight","—")),       "g/mol"),
                ("Exact Mass",          rd.get("ExactMolWt",pub.get("ExactMass","—")),             "g/mol"),
                ("LogP",                rd.get("LogP",     pub.get("XLogP","—")),                  ""),
                ("TPSA",                rd.get("TPSA",     pub.get("TPSA","—")),                   "Ų"),
                ("H-Bond Donors",       rd.get("HBD",      pub.get("HBondDonorCount","—")),        ""),
                ("H-Bond Acceptors",    rd.get("HBA",      pub.get("HBondAcceptorCount","—")),     ""),
                ("Rotatable Bonds",     rd.get("RotBonds", pub.get("RotatableBondCount","—")),     ""),
                ("Heavy Atoms",         rd.get("HeavyAtoms",pub.get("HeavyAtomCount","—")),        ""),
                ("Ring Count",          rd.get("RingCount","—"),                                   ""),
                ("Aromatic Rings",      rd.get("AromaticRings","—"),                               ""),
                ("sp³ Carbon Fraction", rd.get("FractionCSP3","—"),                                ""),
                ("Molar Refractivity",  rd.get("MolarRefractivity","—"),                           ""),
                ("Chiral Centers",      rd.get("NumChiralCenters","—"),                            ""),
                ("Heteroatoms",         rd.get("NumHeteroatoms","—"),                              ""),
                ("Total Bonds",         rd.get("NumBonds","—"),                                    ""),
                ("QED Score",           rd.get("QED","—"),                                         "/ 1.0"),
            ]
            st.markdown(card("".join(prow(p,v,u) for p,v,u in props)), unsafe_allow_html=True)

            shead("🔗","SMILES Strings","#10b981")
            st.markdown(card(f"""
            <div style="margin-bottom:0.8rem">
              <div style="font-size:0.64rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem">Canonical</div>
              <code style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;color:#06b6d4;word-break:break-all;line-height:1.7">{pub.get("CanonicalSMILES","—")}</code>
            </div>
            <div>
              <div style="font-size:0.64rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem">Isomeric</div>
              <code style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;color:#8b5cf6;word-break:break-all;line-height:1.7">{pub.get("IsomericSMILES","—")}</code>
            </div>"""), unsafe_allow_html=True)

        # Lipinski + Solubility
        st.markdown("<br>", unsafe_allow_html=True)
        shead("💊","Drug-Likeness · Lipinski's Rule of Five","#f59e0b")
        dl, sl = st.columns([1,1], gap="large")

        with dl:
            rh = "".join(rulerow(n,ok,v,t) for n,(ok,v,t) in lip["rules"].items())
            vc, vt, p = lip["vc"], lip["verdict"], lip["passed"]
            rh += f'<div style="margin-top:0.8rem;padding:0.75rem;border-radius:10px;background:{vc}12;border:1px solid {vc}30;text-align:center"><span style="font-weight:700;font-size:0.95rem;color:{vc}">{vt}</span><span style="color:#334155;font-size:0.8rem"> &nbsp;·&nbsp; {p}/4 rules passed</span></div>'
            st.markdown(card(rh), unsafe_allow_html=True)

        with sl:
            shead("💧","Solubility Prediction","#3b82f6")
            st.plotly_chart(make_gauge(sol["score"],sol["cat"],sol["color"]),
                            use_container_width=True,config={"displayModeBar":False})
            st.markdown(card(f'<div style="text-align:center"><div style="font-weight:700;color:{sol["color"]};font-size:0.95rem">{sol["cat"]}</div><div style="color:#475569;font-size:0.8rem;margin-top:0.2rem">{sol["detail"]}</div><div style="color:#334155;font-size:0.73rem;margin-top:0.2rem;font-family:\'JetBrains Mono\',monospace">logS ≈ {sol["logS"]}</div></div>',"1rem"), unsafe_allow_html=True)

        # Charts — build fallback data from PubChem when RDKit is empty
        # Radar: build a descriptor dict merging rdkit + pubchem
        radar_data = {
            "MolWt":     rd.get("MolWt")     or float(pub.get("MolecularWeight") or 0),
            "LogP":      rd.get("LogP")       or float(pub.get("XLogP") or 0),
            "TPSA":      rd.get("TPSA")       or float(pub.get("TPSA") or 0),
            "HBD":       rd.get("HBD")        or int(pub.get("HBondDonorCount") or 0),
            "HBA":       rd.get("HBA")        or int(pub.get("HBondAcceptorCount") or 0),
            "RotBonds":  rd.get("RotBonds")   or int(pub.get("RotatableBondCount") or 0),
            "RingCount": rd.get("RingCount",0),
        }

        # Atom composition fallback — parse from molecular formula
        ac = rd.get("AtomComposition", {})
        if not ac:
            import re as _re
            formula = pub.get("MolecularFormula", "")
            for sym, cnt in _re.findall(r"([A-Z][a-z]?)(\d*)", formula):
                if sym:
                    ac[sym] = int(cnt) if cnt else 1

        st.markdown("<br>", unsafe_allow_html=True)
        shead("📊","Property Visualisations","#06b6d4")
        ch1,ch2,ch3 = st.columns([1.1,1,0.85], gap="large")

        with ch1:
            st.markdown('<div style="font-size:0.75rem;color:#334155;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.06em">Property Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(make_radar(radar_data),use_container_width=True,config={"displayModeBar":False})

        with ch2:
            st.markdown('<div style="font-size:0.75rem;color:#334155;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.06em">Atom Composition</div>', unsafe_allow_html=True)
            if ac:
                st.plotly_chart(make_pie(ac),use_container_width=True,config={"displayModeBar":False})
            else:
                st.markdown('<div style="color:#334155;font-size:0.82rem;padding:1rem;text-align:center">No data</div>', unsafe_allow_html=True)

        with ch3:
            st.markdown('<div style="font-size:0.75rem;color:#334155;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.06em">Bond Types</div>', unsafe_allow_html=True)
            bt = rd.get("BondTypes",{})
            if bt and any(bt.values()):
                st.plotly_chart(make_bond_bar(bt),use_container_width=True,config={"displayModeBar":False})
            else:
                # fallback: estimate bonds from heavy atom count and complexity
                heavy = int(pub.get("HeavyAtomCount") or 0)
                hbd   = int(pub.get("HBondDonorCount") or 0)
                hba   = int(pub.get("HBondAcceptorCount") or 0)
                if heavy > 0:
                    est_bt = {
                        "SINGLE":   max(heavy - 1, 0),
                        "DOUBLE":   max(hba - hbd, 0),
                        "AROMATIC": 0,
                        "TRIPLE":   0,
                    }
                    if any(est_bt.values()):
                        st.plotly_chart(make_bond_bar(est_bt),use_container_width=True,config={"displayModeBar":False})
                    else:
                        st.markdown('<div style="color:#334155;font-size:0.82rem;padding:1rem;text-align:center">No bond data</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#334155;font-size:0.82rem;padding:1rem;text-align:center">No bond data</div>', unsafe_allow_html=True)

        # AI Explanation
        if show_ai:
            st.markdown("<br>", unsafe_allow_html=True)
            shead("🤖","AI Explanation — Student Friendly","#10b981")
            exp_text = ai_explanation(pub, rd, lip)
            rendered = ""
            for line in exp_text.split("\n\n"):
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    rendered += f'<div style="border-left:3px solid #06b6d4;padding:0.6rem 0.9rem;margin:0.5rem 0;color:#475569;font-size:0.82rem;font-style:italic;line-height:1.65;background:#06b6d41a;border-radius:0 8px 8px 0">{line.lstrip(">").strip()}</div>'
                else:
                    line_html = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#06b6d4">\1</strong>', line)
                    rendered += f'<p style="margin:0.5rem 0;line-height:1.75;color:#64748b;font-size:0.87rem">{line_html}</p>'
            st.markdown(card(rendered,"1.3rem"), unsafe_allow_html=True)

        # Description
        desc_text = pub.get("Description","")
        if desc_text and len(desc_text) > 40:
            with st.expander("📄  Full PubChem Description"):
                st.markdown(f'<div style="color:#64748b;line-height:1.7;font-size:0.85rem;padding:0.5rem">{desc_text}</div>', unsafe_allow_html=True)

        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        shead("📥","Export Report","#3b82f6")
        ex1,ex2 = st.columns(2)
        df_exp = export_df(pub,rd,lip,sol)
        with ex1:
            if df_exp is not None:
                st.download_button("⬇️  Download CSV",
                    df_exp.to_csv(index=False).encode(),
                    file_name=f"{query.replace(' ','_')}.csv",
                    mime="text/csv",use_container_width=True)
        with ex2:
            jdata = json.dumps({"pubchem":pub,
                "rdkit":{k:v for k,v in rd.items() if not isinstance(v,dict)},
                "lipinski":{"passed":lip["passed"],"verdict":lip["verdict"]},
                "solubility":sol},indent=2)
            st.download_button("⬇️  Download JSON",jdata.encode(),
                file_name=f"{query.replace(' ','_')}.json",
                mime="application/json",use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — COMPARE
# ───────────────────────────────────────────────────────────────
with tab2:
    shead("⚖️","Side-by-Side Molecule Comparison","#06b6d4")
    st.markdown('<div style="color:#334155;font-size:0.85rem;margin-bottom:1rem">Compare two molecules — properties, similarity score, and charts</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2,gap="large")
    with c1: mq1 = st.text_input("Molecule A",placeholder="e.g. Aspirin",   key="cmp1")
    with c2: mq2 = st.text_input("Molecule B",placeholder="e.g. Ibuprofen", key="cmp2")
    cmp_go = st.button("⚖️  Compare Now",key="cmp_btn")

    if cmp_go and mq1 and mq2:
        with st.spinner("Fetching both molecules…"):
            d1 = fetch_pubchem_data(mq1)
            d2 = fetch_pubchem_data(mq2)

        if not d1 or not d2:
            st.error("Could not find one or both molecules. Please check the names.")
        else:
            r1 = rdkit_descriptors(d1.get("CanonicalSMILES",""))
            r2 = rdkit_descriptors(d2.get("CanonicalSMILES",""))
            l1 = lipinski(r1,d1)
            l2 = lipinski(r2,d2)
            n1 = d1.get("IUPACName",mq1).title()
            n2 = d2.get("IUPACName",mq2).title()

            # identity cards
            ic1,ic2 = st.columns(2,gap="large")
            for col,data,rdesc,lip_r,name,glow in [(ic1,d1,r1,l1,n1,"#06b6d4"),(ic2,d2,r2,l2,n2,"#8b5cf6")]:
                with col:
                    cid_i = data.get("CID")
                    img   = get_image_url(cid_i,260) if cid_i else ""
                    img_t = f'<img src="{img}" style="max-width:140px;margin:0.6rem 0;border-radius:8px;filter:invert(1) hue-rotate(180deg) brightness(0.85)"/>' if img else ""

                    # show values from rdkit OR pubchem fallback
                    mw_show   = rdesc.get("MolWt")   or data.get("MolecularWeight","—")
                    logp_show = rdesc.get("LogP")     or data.get("XLogP","—")

                    st.markdown(glow_card(f"""
                    <div style="text-align:center">
                      <div style="font-weight:700;font-size:0.97rem;color:#e2e8f0">{name}</div>
                      <div style="color:#475569;font-size:0.78rem">{data.get("MolecularFormula","—")}</div>
                      {img_t}
                      <div>
                        {badge("MW: "+str(mw_show),"#06b6d4")}
                        {badge("LogP: "+str(logp_show),"#8b5cf6")}
                        {badge(lip_r["verdict"].split()[0],"#10b981")}
                      </div>
                    </div>""",glow=glow), unsafe_allow_html=True)

            # Tanimoto
            sim = tanimoto(d1.get("CanonicalSMILES",""),d2.get("CanonicalSMILES",""))
            if sim is not None:
                pct = int(sim*100)
                sc  = "#10b981" if pct>60 else ("#f59e0b" if pct>30 else "#f43f5e")
                lbl = "Highly Similar" if pct>60 else ("Moderately Similar" if pct>30 else "Structurally Dissimilar")
                st.markdown(card(f"""
                <div style="text-align:center;padding:0.4rem">
                  <div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.3rem">Tanimoto Similarity · Morgan Fingerprints</div>
                  <div style="font-size:2.8rem;font-weight:700;color:{sc};letter-spacing:-0.04em;line-height:1">{sim}</div>
                  <div style="color:#475569;font-size:0.83rem;margin-top:0.25rem">{lbl} — {pct}% structural overlap</div>
                </div>"""), unsafe_allow_html=True)

            # comparison data — pass BOTH rdkit and pubchem dicts
            comp = compare_mols(r1, r2, d1, d2)

            if comp:
                shead("📊","Property Comparison Chart","#06b6d4")
                st.plotly_chart(make_cmp_bar(comp,n1,n2),use_container_width=True,config={"displayModeBar":False})

                shead("📋","Comparison Table","#8b5cf6")
                rows = [
                    {"Property":prop, n1:v1, n2:v2,
                     "Winner": f"🔵 {n1}" if w==1 else (f"🟣 {n2}" if w==2 else "—")}
                    for prop,(v1,v2,w) in comp.items()
                ]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            else:
                st.warning("No comparison data available — both molecules may have very limited property data.")


# ───────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ───────────────────────────────────────────────────────────────
with tab3:
    shead("🧬","About Molecular Intelligence Explorer","#06b6d4")

    st.markdown(card('<p style="color:#64748b;line-height:1.75;font-size:0.88rem;margin:0">Built as a <strong style="color:#06b6d4">BTech CSE (AI/ML) Semester 2 Chemistry Project</strong>. Combines cheminformatics, data science, and a minimal dark-lab UI to make molecular science accessible to students and researchers.</p>'), unsafe_allow_html=True)

    shead("🏗️","Architecture","#3b82f6")
    a1,a2 = st.columns(2,gap="large")
    arch = [("Frontend","Streamlit + Custom CSS dark lab theme"),("Backend","Python · PubChem REST API · RDKit"),("3D Viewer","py3Dmol — WebGL embedded iframe"),("Charts","Plotly — Radar · Pie · Bar · Gauge")]
    for i,(k,v) in enumerate(arch):
        with (a1 if i%2==0 else a2):
            st.markdown(card(f'<div style="font-size:0.63rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem">{k}</div><div style="color:#cbd5e1;font-size:0.85rem">{v}</div>',"0.85rem"), unsafe_allow_html=True)

    shead("⚗️","Key Algorithms","#8b5cf6")
    for name,desc in [
        ("Lipinski's Rule of Five","Oral bioavailability drug screening — MW, LogP, HBD, HBA thresholds"),
        ("Yalkowsky Solubility","logS ≈ 0.5 − logP − 0.01 × (MW − 100)"),
        ("Morgan Fingerprints + Tanimoto","Circular fingerprints for structural similarity (0–1 score)"),
        ("QED Score","Quantitative Estimate of Drug-likeness (Bickerton et al., 2012)"),
    ]:
        st.markdown(card(f'<div style="font-weight:600;color:#e2e8f0;font-size:0.88rem;margin-bottom:0.2rem">{name}</div><div style="color:#475569;font-size:0.82rem">{desc}</div>',"0.85rem"), unsafe_allow_html=True)

    shead("📦","Tech Stack","#10b981")
    st.markdown(card(f'<div style="line-height:2.2">{" ".join(badge(t,"#06b6d4") for t in ["Python 3.10+","Streamlit","RDKit","PubChem API","py3Dmol","Plotly","Pandas"])}</div>',"1rem"), unsafe_allow_html=True)

    # ── About the App ──────────────────────────────────────────

    # ── About This App ─────────────────────────────────────────
    shead("🔭", "About This App", "#06b6d4")

    about_html = (
        '<div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.6rem">What is Molecular Intelligence Explorer?</div>'
        '<p style="color:#64748b;line-height:1.8;font-size:0.88rem;margin:0 0 0.8rem 0">'
        'Molecular Intelligence Explorer is a <strong style="color:#06b6d4">web-based chemistry analysis platform</strong> '
        'that lets anyone — student, researcher, or curious mind — type the name of any molecule and instantly '
        'get a complete scientific breakdown. No textbooks, no lab equipment, no expertise required.'
        '</p>'
        '<p style="color:#64748b;line-height:1.8;font-size:0.88rem;margin:0 0 0.8rem 0">'
        'The app bridges the gap between <strong style="color:#8b5cf6">raw chemistry data</strong> and '
        '<strong style="color:#8b5cf6">human understanding</strong> — turning dense molecular descriptors into '
        'clean visualisations, plain-English explanations, and actionable drug-likeness verdicts.'
        '</p>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-top:0.9rem">'

        '<div style="background:#080d16;border:1px solid #1e2d45;border-radius:10px;padding:0.85rem">'
        '<div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">Who is it for?</div>'
        '<div style="color:#cbd5e1;font-size:0.83rem;line-height:1.8">'
        '🎓 BTech / BSc Chemistry &amp; CS students<br>'
        '🔬 Pharmacy &amp; drug research learners<br>'
        '🧑‍💻 AI/ML students exploring bioinformatics<br>'
        '👨‍🏫 Educators teaching molecular science'
        '</div></div>'

        '<div style="background:#080d16;border:1px solid #1e2d45;border-radius:10px;padding:0.85rem">'
        '<div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">What can you do?</div>'
        '<div style="color:#cbd5e1;font-size:0.83rem;line-height:1.8">'
        '🔍 Search 100M+ molecules from PubChem<br>'
        '🧪 Get 16+ molecular descriptors instantly<br>'
        '💊 Check drug-likeness in one click<br>'
        '⚖️ Compare two molecules side by side'
        '</div></div>'

        '<div style="background:#080d16;border:1px solid #1e2d45;border-radius:10px;padding:0.85rem">'
        '<div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">How does it help?</div>'
        '<div style="color:#cbd5e1;font-size:0.83rem;line-height:1.8">'
        'Saves hours of manual lookup &amp; calculation.<br>'
        'Visualises complex data as intuitive charts.<br>'
        'Explains everything in simple language.<br>'
        'Exports full reports as CSV or JSON.'
        '</div></div>'

        '<div style="background:#080d16;border:1px solid #1e2d45;border-radius:10px;padding:0.85rem">'
        '<div style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">Built with</div>'
        '<div style="color:#cbd5e1;font-size:0.83rem;line-height:1.8">'
        'Python · Streamlit · RDKit<br>'
        'PubChem REST API · py3Dmol<br>'
        'Plotly · Pandas<br>'
        'Lots of chai ☕ and debugging 🐛'
        '</div></div>'

        '</div>'
    )
    st.markdown(card(about_html, "1.5rem"), unsafe_allow_html=True)

    # ── Meet the Team ──────────────────────────────────────────
    shead("👨‍💻", "Meet the Team", "#8b5cf6")

    team_intro = (
        '<div style="font-size:0.82rem;color:#475569;line-height:1.7;margin-bottom:1rem">'
        'We are four <strong style="color:#8b5cf6">BTech CSE (AI/ML)</strong> students who built this project '
        'to combine our love for coding and chemistry. This app was created as part of our '
        '<strong style="color:#8b5cf6">Semester 2 Chemistry Project</strong>, with the goal of making '
        'molecular science fun, visual, and accessible for everyone.'
        '</div>'
    )

    def team_card(initial, name, color1, color2, border_color, role, desc):
        return (
            f'<div style="background:#080d16;border:1px solid #1e2d45;border-left:3px solid {border_color};'
            f'border-radius:10px;padding:1rem;margin-bottom:0.1rem">'
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem">'
            f'<div style="width:36px;height:36px;border-radius:50%;'
            f'background:linear-gradient(135deg,{color1},{color2});'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:1rem;font-weight:700;color:#fff;flex-shrink:0">{initial}</div>'
            f'<div>'
            f'<div style="font-weight:700;color:#e2e8f0;font-size:0.92rem">{name}</div>'
            f'<div style="font-size:0.7rem;color:#334155">{role}</div>'
            f'</div></div>'
            f'<div style="font-size:0.8rem;color:#475569;line-height:1.6">{desc}</div>'
            f'</div>'
        )

    cards_html = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem">'
        + team_card("A","Abhinav","#06b6d4","#3b82f6","#06b6d4","BTech CSE (AI/ML)",
                    "Backend logic, PubChem API integration &amp; RDKit descriptor pipeline. The one who makes sure the molecules actually load. 🔬")
        + team_card("A","Aman","#3b82f6","#8b5cf6","#3b82f6","BTech CSE (AI/ML)",
                    "3D viewer integration, Plotly charts &amp; data visualisation. Makes the molecules look cool and spin. 🔭")
        + team_card("K","Kashika","#10b981","#06b6d4","#10b981","BTech CSE (AI/ML)",
                    "Drug-likeness analysis, solubility prediction &amp; AI explanation engine. The chemistry brain of the team. 💊")
        + team_card("V","Vanshika","#8b5cf6","#f43f5e","#8b5cf6","BTech CSE (AI/ML)",
                    "UI/UX design, CSS styling &amp; overall app experience. Responsible for making everything look this good. ✨")
        + '</div>'
        + '<div style="margin-top:1rem;padding:0.75rem 1rem;border-radius:10px;'
          'background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.15);'
          'text-align:center;font-size:0.82rem;color:#475569">'
          '🏫 &nbsp;<strong style="color:#06b6d4">BTech CSE (AI/ML) · Semester 2</strong>'
          '&nbsp;·&nbsp; Chemistry Project 2024–25 &nbsp;·&nbsp; Built with Python &amp; Streamlit'
          '</div>'
    )

    st.markdown(card(team_intro + cards_html, "1.4rem"), unsafe_allow_html=True)
