# 🧬 Molecular Intelligence Explorer

A premium chemistry web application built with Python & Streamlit.
Enter any element, molecule, or compound — get instant chemical insights,
3D visualisation, drug-likeness analysis, and student-friendly AI explanations.

---

## 🚀 Quick Start

### 1. Clone / Download
```bash
git clone https://github.com/your-username/molecular-explorer
cd molecular_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note for Windows users:** If `rdkit-pypi` fails, try:
> ```bash
> conda install -c conda-forge rdkit
> ```

### 3. Run the app
```bash
streamlit run app.py
```
The app opens automatically at **http://localhost:8501**

---

## 📂 Project Structure
```
molecular_app/
├── app.py           # Main Streamlit application (UI + orchestration)
├── utils.py         # Backend: PubChem API, RDKit, drug-likeness, export
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## 🧩 Features

| Feature | Description |
|---|---|
| 🔍 Smart Search | Name, formula, or SMILES input |
| 🖼️ 2D Structure | High-res PubChem structure image |
| 🔭 3D Viewer | Interactive py3Dmol with spin animation |
| 📊 Property Radar | Normalised RDKit descriptor radar chart |
| 🥧 Atom Composition | Interactive pie chart of element distribution |
| 📊 Bond Analysis | Bar chart of single/double/triple/aromatic bonds |
| 💊 Lipinski Ro5 | Full rule-by-rule drug-likeness analysis |
| 💧 Solubility | Rule-based aqueous solubility prediction |
| 🤖 AI Explanation | Plain-English student-friendly description |
| ⚖️ Comparison | Side-by-side molecular comparison + Tanimoto similarity |
| 📥 Export | CSV and JSON report download |
| 🌙 Dark Mode | Toggle in sidebar |

---

## 🔬 Data Sources
- **PubChem** (NIH) — molecular data, structures, synonyms, descriptions
- **RDKit** — cheminformatics descriptors, fingerprints, QED

## 📚 Algorithms
- Lipinski's Rule of Five
- Yalkowsky aqueous solubility estimation
- Morgan fingerprints + Tanimoto coefficient
- QED drug-likeness score (Bickerton et al., 2012)

---

## 👨‍💻 Built With
Python · Streamlit · RDKit · PubChem API · py3Dmol · Plotly · Pandas

---

*Built as a BTech CSE (AI/ML) Semester 2 Chemistry Project.*
