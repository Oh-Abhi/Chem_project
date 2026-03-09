"""
utils.py — Backend Logic for Molecular Intelligence Explorer
============================================================
Handles all data fetching, RDKit calculations, drug-likeness,
solubility prediction, and molecule comparisons.
"""

import re
import json
import urllib.request
import urllib.parse
import urllib.error


# ─────────────────────────────────────────────
# 1. PubChem API
# ─────────────────────────────────────────────

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_pubchem_data(query: str) -> dict | None:
    """
    Fetch molecule data from PubChem by name, formula, or SMILES.
    Returns a dictionary of raw PubChem properties, or None on failure.
    """
    # Step 1 – resolve the CID from a name/formula/SMILES
    encoded = urllib.parse.quote(query)
    cid = _resolve_cid(query, encoded)
    if cid is None:
        return None

    # Step 2 – fetch full property sheet
    props = (
        "IUPACName,MolecularFormula,MolecularWeight,"
        "CanonicalSMILES,IsomericSMILES,XLogP,TPSA,"
        "HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount,HeavyAtomCount,"
        "Complexity,ExactMass,MonoisotopicMass,"
        "Charge,CovalentUnitCount"
    )
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON"
    data = _get_json(url)
    if not data:
        return None

    props_dict = data.get("PropertyTable", {}).get("Properties", [{}])[0]
    props_dict["CID"] = cid

    # Step 3 – synonyms
    syn_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    syn_data = _get_json(syn_url)
    synonyms = []
    if syn_data:
        synonyms = (
            syn_data.get("InformationList", {})
            .get("Information", [{}])[0]
            .get("Synonym", [])[:5]
        )
    props_dict["Synonyms"] = synonyms

    # Step 4 – description
    desc_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/description/JSON"
    desc_data = _get_json(desc_url)
    description = ""
    if desc_data:
        for info in desc_data.get("InformationList", {}).get("Information", []):
            if "Description" in info:
                description = info["Description"]
                break
    props_dict["Description"] = description

    return props_dict


def _resolve_cid(query: str, encoded: str) -> int | None:
    """Try multiple PubChem lookup strategies to get a CID."""
    strategies = [
        f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON",
        f"{PUBCHEM_BASE}/compound/smiles/{encoded}/cids/JSON",
        f"{PUBCHEM_BASE}/compound/formula/{encoded}/cids/JSON",
    ]
    for url in strategies:
        data = _get_json(url)
        if data:
            ids = data.get("IdentifierList", {}).get("CID", [])
            if ids:
                return ids[0]
    return None


def _get_json(url: str) -> dict | None:
    """Simple HTTP GET returning parsed JSON or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MolecularExplorer/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def get_pubchem_image_url(cid: int, size: int = 400) -> str:
    """Return the PubChem 2-D structure image URL for a CID."""
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?record_type=2d&image_size={size}x{size}"


def get_pubchem_sdf(cid: int) -> str | None:
    """Download the 3-D SDF for a CID (used by py3Dmol)."""
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type=3d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MolecularExplorer/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode()
    except Exception:
        return None


# ─────────────────────────────────────────────
# 2. RDKit Molecular Descriptors
# ─────────────────────────────────────────────

def compute_rdkit_descriptors(smiles: str) -> dict:
    """
    Compute a rich set of RDKit molecular descriptors from a SMILES string.
    Returns an empty dict if RDKit is unavailable or the SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, QED
    except ImportError:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    desc = {}

    # Core descriptors
    desc["MolWt"]               = round(Descriptors.MolWt(mol), 4)
    desc["ExactMolWt"]          = round(Descriptors.ExactMolWt(mol), 6)
    desc["LogP"]                = round(Descriptors.MolLogP(mol), 4)
    desc["TPSA"]                = round(Descriptors.TPSA(mol), 2)
    desc["HBD"]                 = Lipinski.NumHDonors(mol)
    desc["HBA"]                 = Lipinski.NumHAcceptors(mol)
    desc["RotBonds"]            = rdMolDescriptors.CalcNumRotatableBonds(mol)
    desc["HeavyAtoms"]          = mol.GetNumHeavyAtoms()
    desc["RingCount"]           = rdMolDescriptors.CalcNumRings(mol)
    desc["AromaticRings"]       = rdMolDescriptors.CalcNumAromaticRings(mol)
    desc["FractionCSP3"]        = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)
    desc["MolarRefractivity"]   = round(Descriptors.MolMR(mol), 4)
    desc["NumAtoms"]            = mol.GetNumAtoms()
    desc["NumBonds"]            = mol.GetNumBonds()
    desc["NumChiralCenters"]    = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    desc["NumHeteroatoms"]      = rdMolDescriptors.CalcNumHeteroatoms(mol)
    desc["NumStereocenters"]    = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

    # QED drug-likeness score (0–1)
    try:
        desc["QED"] = round(QED.qed(mol), 4)
    except Exception:
        desc["QED"] = None

    # Atom composition (symbol → count)
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
    desc["AtomComposition"] = atom_counts

    # Bond type counts
    from rdkit.Chem import rdchem
    bond_counts = {"SINGLE": 0, "DOUBLE": 0, "TRIPLE": 0, "AROMATIC": 0}
    for bond in mol.GetBonds():
        bt = str(bond.GetBondType())
        for key in bond_counts:
            if key in bt:
                bond_counts[key] += 1
                break
    desc["BondTypes"] = bond_counts

    return desc


# ─────────────────────────────────────────────
# 3. Lipinski Rule of Five
# ─────────────────────────────────────────────

def lipinski_analysis(desc: dict) -> dict:
    """
    Evaluate Lipinski's Rule of Five and return a detailed breakdown.
    `desc` should be the output of compute_rdkit_descriptors().
    """
    mw   = desc.get("MolWt",   desc.get("MolecularWeight", 9999))
    logp = desc.get("LogP",    desc.get("XLogP", 99))
    hbd  = desc.get("HBD",     desc.get("HBondDonorCount", 99))
    hba  = desc.get("HBA",     desc.get("HBondAcceptorCount", 99))

    rules = {
        "Molecular Weight ≤ 500 Da": (float(mw) <= 500,  round(float(mw), 2),  "≤ 500"),
        "LogP ≤ 5":                  (float(logp) <= 5,   round(float(logp), 2), "≤ 5"),
        "H-Bond Donors ≤ 5":         (int(hbd) <= 5,      int(hbd),              "≤ 5"),
        "H-Bond Acceptors ≤ 10":     (int(hba) <= 10,     int(hba),              "≤ 10"),
    }

    passed = sum(1 for ok, _, _ in rules.values() if ok)

    if passed == 4:
        verdict = "✅ Excellent Drug Candidate"
        verdict_color = "#22c55e"
    elif passed == 3:
        verdict = "🟡 Moderate Drug Candidate"
        verdict_color = "#f59e0b"
    else:
        verdict = "❌ Poor Drug Candidate"
        verdict_color = "#ef4444"

    return {
        "rules": rules,
        "passed": passed,
        "total": 4,
        "verdict": verdict,
        "verdict_color": verdict_color,
    }


# ─────────────────────────────────────────────
# 4. Solubility Prediction
# ─────────────────────────────────────────────

def predict_solubility(logp: float, mw: float) -> dict:
    """
    Rule-based aqueous solubility prediction using LogP and MW.
    Returns category, description, and a numeric score 0-100.
    """
    # Simple Yalkowsky-like estimate: logS ≈ 0.5 - logP - 0.01*(MW - 100)
    log_s = 0.5 - logp - 0.01 * (mw - 100)

    if log_s > -1:
        category = "Highly Soluble"
        color     = "#22c55e"
        score     = 90
        detail    = "Dissolves easily in water (> 10 mg/mL)"
    elif log_s > -3:
        category = "Moderately Soluble"
        color     = "#f59e0b"
        score     = 60
        detail    = "Moderate aqueous solubility (1–10 mg/mL)"
    elif log_s > -5:
        category = "Poorly Soluble"
        color     = "#f97316"
        score     = 30
        detail    = "Low aqueous solubility (0.1–1 mg/mL)"
    else:
        category = "Practically Insoluble"
        color     = "#ef4444"
        score     = 10
        detail    = "Very low solubility in water (< 0.1 mg/mL)"

    return {
        "category": category,
        "logS": round(log_s, 2),
        "color": color,
        "score": score,
        "detail": detail,
    }


# ─────────────────────────────────────────────
# 5. AI Plain-English Explanation
# ─────────────────────────────────────────────

def generate_ai_explanation(pubchem_data: dict, rdkit_desc: dict, lipinski: dict) -> str:
    """
    Build a readable, student-friendly explanation of the molecule
    without calling any external AI API — purely rule-based text generation.
    """
    name   = pubchem_data.get("IUPACName", "this molecule")
    formula = pubchem_data.get("MolecularFormula", "unknown")
    mw     = rdkit_desc.get("MolWt", pubchem_data.get("MolecularWeight", "?"))
    logp   = rdkit_desc.get("LogP", pubchem_data.get("XLogP", "?"))
    hbd    = rdkit_desc.get("HBD", pubchem_data.get("HBondDonorCount", "?"))
    hba    = rdkit_desc.get("HBA", pubchem_data.get("HBondAcceptorCount", "?"))
    rings  = rdkit_desc.get("RingCount", 0)
    arom   = rdkit_desc.get("AromaticRings", 0)
    qed    = rdkit_desc.get("QED")

    lines = []
    lines.append(f"**{name.title()}** has the molecular formula **{formula}** "
                 f"and a molecular weight of **{mw} g/mol**.")

    # Lipophilicity
    try:
        lv = float(logp)
        if lv < 0:
            lines.append("It is **hydrophilic** (water-loving), meaning it dissolves well in water.")
        elif lv < 2:
            lines.append("It has **balanced polarity** — dissolves reasonably in both water and fats.")
        elif lv < 5:
            lines.append("It is **lipophilic** (fat-loving), which helps it cross cell membranes.")
        else:
            lines.append("It is **highly lipophilic**, which may cause poor aqueous solubility.")
    except (TypeError, ValueError):
        pass

    # H-bonds
    try:
        donors    = int(hbd)
        acceptors = int(hba)
        lines.append(
            f"It can form **{donors} hydrogen bond donor(s)** and **{acceptors} hydrogen bond acceptor(s)**, "
            "influencing how it interacts with biological targets."
        )
    except (TypeError, ValueError):
        pass

    # Ring systems
    if rings > 0:
        ring_text = f"The molecule contains **{rings} ring(s)**"
        if arom > 0:
            ring_text += f", of which **{arom}** are aromatic (flat, highly stable π systems)"
        lines.append(ring_text + ".")

    # Drug-likeness
    verdict = lipinski.get("verdict", "")
    passed  = lipinski.get("passed", 0)
    lines.append(
        f"Lipinski's Rule of Five analysis shows **{passed}/4 rules passed** — {verdict.split(' ', 1)[-1]}."
    )

    # QED
    if qed is not None:
        lines.append(
            f"The **QED drug-likeness score** is **{qed}** (scale 0–1; higher is more drug-like)."
        )

    # Description from PubChem
    desc = pubchem_data.get("Description", "")
    if desc and len(desc) > 30:
        truncated = desc[:300] + ("…" if len(desc) > 300 else "")
        lines.append(f"\n> {truncated}")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# 6. Molecule Comparison
# ─────────────────────────────────────────────

def compare_molecules(desc1: dict, desc2: dict) -> dict:
    """
    Compare two molecules by their descriptor dicts.
    Returns a table-ready dict of {property: (val1, val2, winner)}.
    """
    fields = [
        ("MolWt",       "Molecular Weight (g/mol)", "lower"),
        ("LogP",        "LogP",                     "lower"),
        ("TPSA",        "TPSA (Ų)",                 "lower"),
        ("HBD",         "H-Bond Donors",             "lower"),
        ("HBA",         "H-Bond Acceptors",          "lower"),
        ("RotBonds",    "Rotatable Bonds",            "lower"),
        ("HeavyAtoms",  "Heavy Atoms",                "neutral"),
        ("RingCount",   "Ring Count",                 "neutral"),
        ("AromaticRings","Aromatic Rings",            "neutral"),
        ("QED",         "QED Score",                  "higher"),
    ]

    result = {}
    for key, label, prefer in fields:
        v1 = desc1.get(key)
        v2 = desc2.get(key)
        if v1 is None or v2 is None:
            continue
        try:
            f1, f2 = float(v1), float(v2)
            if prefer == "lower":
                winner = 1 if f1 < f2 else (2 if f2 < f1 else 0)
            elif prefer == "higher":
                winner = 1 if f1 > f2 else (2 if f2 > f1 else 0)
            else:
                winner = 0
        except (TypeError, ValueError):
            winner = 0
        result[label] = (v1, v2, winner)

    return result


def tanimoto_similarity(smiles1: str, smiles2: str) -> float | None:
    """
    Compute Tanimoto similarity between two molecules using RDKit Morgan fingerprints.
    Returns a float 0–1, or None if RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        return None

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)


# ─────────────────────────────────────────────
# 7. Export Helpers
# ─────────────────────────────────────────────

def build_export_dataframe(pubchem_data: dict, rdkit_desc: dict, lipinski: dict, solubility: dict):
    """
    Build a flat pandas DataFrame for CSV export.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    row = {}
    # PubChem fields
    for k in ["IUPACName", "MolecularFormula", "MolecularWeight",
              "CanonicalSMILES", "IsomericSMILES", "CID"]:
        row[k] = pubchem_data.get(k, "")

    # RDKit fields
    for k in ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
              "HeavyAtoms", "RingCount", "AromaticRings", "QED"]:
        row[k] = rdkit_desc.get(k, "")

    row["Lipinski_Passed"]  = f"{lipinski.get('passed',0)}/4"
    row["Lipinski_Verdict"] = lipinski.get("verdict", "")
    row["Solubility"]       = solubility.get("category", "")
    row["LogS"]             = solubility.get("logS", "")

    return pd.DataFrame([row])
