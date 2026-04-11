from __future__ import annotations

"""
M2 · Molecular Simulation Engine v4
──────────────────────────────────────────────────────────────────────────────
Features:
  · Intelligent input: Chinese/English name -> SMILES via Qwen3-32b / PubChem
  · Small molecule (SMILES / CIF / PDB upload)
  · Polymer Builder: homopolymer + copolymer (RDKit core)
    Optional: PySoftK (complex topologies), SMiPoly (BigSMILES notation)
  · UFF / MMFF geometry optimization
  · 3D viewer: drag-rotate, scroll-zoom-at-cursor, bond count overlay,
    atom-click panel (bond length / bond energy / functional group reactivity)
  · Molecular properties + intermolecular forces analysis
  · Download module:
      - Data   (selectable JSON/CSV)
      - Image  (live preview editor, data overlays, PNG capture)
      - Model  (SDF / PDB / XYZ)
  · SQLite + SDF persistence
──────────────────────────────────────────────────────────────────────────────
Tool integration:
  PySoftK  — complex polymer 3D topology builder (optional)
  SMiPoly  — BigSMILES polymer notation input  (optional)
  RadonPy  — polymer property estimation       (optional, heavy dep)
  MolPy    — not integrated (no stable public API found)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import re
import math
import sqlite3
import datetime
import tempfile
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

# ── numpy ────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

# ── RDKit ────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                             Draw, rdMolTransforms)
    RDKIT_OK = True
    try:
        from rdkit.Chem import rdDetermineBonds
        RDKIT_BONDS_OK = True
    except ImportError:
        RDKIT_BONDS_OK = False
except ImportError:
    RDKIT_OK = False
    RDKIT_BONDS_OK = False

# ── PubChemPy ─────────────────────────────────────────────────────────────────
try:
    import pubchempy as pcp
    PUBCHEM_OK = True
except ImportError:
    PUBCHEM_OK = False

# ── ASE (CIF parsing) ─────────────────────────────────────────────────────────
try:
    from ase.io import read as _ase_read
    ASE_OK = True
except ImportError:
    ASE_OK = False

# ── PySoftK (optional: complex polymer topologies) ───────────────────────────
try:
    from pysoftk.linear_polymer.super_monomer import Sm as _PySoftK_Sm
    from pysoftk.linear_polymer.linear_polymer import Lp as _PySoftK_Lp
    PYSOFTK_OK = True
except Exception:
    PYSOFTK_OK = False

# ── SMiPoly (optional: BigSMILES polymer notation) ───────────────────────────
try:
    from smipoly import SMiPoly as _SMiPoly
    SMIPOLY_OK = True
except Exception:
    SMIPOLY_OK = False

# ── Legacy utils hook ─────────────────────────────────────────────────────────
try:
    from utils.database import save_record as _legacy_save
    _UTILS_OK = True
except ImportError:
    _UTILS_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

_DB_PATH         = Path("lab_storage.db")
_MODELS_DIR      = Path("data") / "models"
_POLYMER_DB_PATH = Path("data") / "polymer_db.json"


# ══════════════════════════════════════════════════════════════════════════════
# BOND ENERGY TABLE  (kJ/mol, approximate average values)
# ══════════════════════════════════════════════════════════════════════════════

# Key: (sym1, sym2, bond_order) — sym1 <= sym2 alphabetically
_BOND_ENERGY: dict[tuple, float] = {
    ("C","C",1): 347.0,  ("C","C",2): 614.0,  ("C","C",3): 839.0,
    ("C","H",1): 413.0,
    ("C","N",1): 305.0,  ("C","N",2): 615.0,  ("C","N",3): 891.0,
    ("C","O",1): 358.0,  ("C","O",2): 799.0,
    ("C","F",1): 485.0,
    ("C","Cl",1):339.0,
    ("C","Br",1):285.0,
    ("C","I",1): 213.0,
    ("C","S",1): 272.0,  ("C","S",2): 536.0,
    ("C","P",1): 264.0,
    ("H","N",1): 391.0,
    ("H","O",1): 467.0,
    ("H","S",1): 339.0,
    ("N","N",1): 163.0,  ("N","N",2): 418.0,  ("N","N",3): 945.0,
    ("N","O",1): 201.0,  ("N","O",2): 607.0,
    ("O","O",1): 157.0,  ("O","O",2): 498.0,
    ("O","S",2): 522.0,
    ("S","S",1): 266.0,
    ("P","O",1): 335.0,  ("P","O",2): 544.0,
    ("Si","O",1):452.0,
    ("Si","C",1):318.0,
    # Aromatic (bond order 1.5)
    ("C","C",1.5): 507.0,
    ("C","N",1.5): 460.0,
}

_BOND_TYPE_LABEL = {1: "Single", 2: "Double", 3: "Triple", 1.5: "Aromatic"}


def _bond_energy_lookup(sym1: str, sym2: str, order: float) -> float | None:
    s1, s2 = (sym1, sym2) if sym1 <= sym2 else (sym2, sym1)
    # try exact order
    v = _BOND_ENERGY.get((s1, s2, order))
    if v is not None:
        return v
    # fallback to integer order
    v = _BOND_ENERGY.get((s1, s2, round(order)))
    return v


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL GROUP PATTERNS  (SMARTS, reactivity info)
# ══════════════════════════════════════════════════════════════════════════════

_FG_PATTERNS: list[dict] = [
    {
        "name": "Hydroxyl (-OH)",
        "smarts": "[OX2H]",
        "reactions": "Esterification, Oxidation to carbonyl, Dehydration, Etherification",
        "reagents": "Carboxylic acid / acid chloride (esterif.); oxidizing agent; conc. H2SO4",
        "min_equiv": 1,
    },
    {
        "name": "Carboxyl (-COOH)",
        "smarts": "[CX3](=O)[OX2H1]",
        "reactions": "Esterification, Amide formation, Decarboxylation, Salt formation",
        "reagents": "Alcohol + acid catalyst; amine; heat (decarboxylation); base",
        "min_equiv": 1,
    },
    {
        "name": "Ester (-COO-)",
        "smarts": "[#6][CX3](=O)[OX2][#6]",
        "reactions": "Hydrolysis, Transesterification, Aminolysis",
        "reagents": "H2O + acid/base; alcohol + catalyst; amine",
        "min_equiv": 1,
    },
    {
        "name": "Primary Amine (-NH2)",
        "smarts": "[NX3H2]",
        "reactions": "Amide formation, Alkylation, Schiff base formation, Diazotization",
        "reagents": "Acid chloride; alkyl halide; aldehyde/ketone; NaNO2 + HCl",
        "min_equiv": 1,
    },
    {
        "name": "Secondary Amine (-NH-)",
        "smarts": "[NX3H1]",
        "reactions": "Amide formation, Alkylation",
        "reagents": "Acid chloride; alkyl halide",
        "min_equiv": 1,
    },
    {
        "name": "Aldehyde (-CHO)",
        "smarts": "[CX3H1](=O)[#6,H]",
        "reactions": "Nucleophilic addition, Oxidation to carboxyl, Cannizzaro, Aldol",
        "reagents": "Nu: (Grignard, NaBH4, HCN); oxidizing agent; base (Cannizzaro/Aldol)",
        "min_equiv": 1,
    },
    {
        "name": "Ketone (C=O)",
        "smarts": "[#6][CX3](=O)[#6]",
        "reactions": "Nucleophilic addition, Enolization, Aldol condensation",
        "reagents": "Nu: (Grignard, NaBH4); base (aldol); LDA (enolate)",
        "min_equiv": 1,
    },
    {
        "name": "Alkene (C=C)",
        "smarts": "[CX3]=[CX3]",
        "reactions": "Addition (HX, X2, H2O, H2), Epoxidation, Ozonolysis, Polymerization",
        "reagents": "HX; X2; H2O + acid; H2 + catalyst; mCPBA; O3 then reductant",
        "min_equiv": 1,
    },
    {
        "name": "Alkyne (C≡C)",
        "smarts": "[CX2]#[CX2]",
        "reactions": "Addition, Hydration to ketone/aldehyde, Terminal alkyne deprotonation",
        "reagents": "HX / X2; H2O + HgSO4; NaNH2 (terminal)",
        "min_equiv": 1,
    },
    {
        "name": "Aromatic Ring",
        "smarts": "c1ccccc1",
        "reactions": "Electrophilic aromatic substitution (EAS), Birch reduction",
        "reagents": "E+ (NO2+, Br2/FeBr3, SO3/H2SO4, R+); Na/NH3(l) (Birch)",
        "min_equiv": 1,
    },
    {
        "name": "Alkyl Halide (C-X)",
        "smarts": "[CX4][F,Cl,Br,I]",
        "reactions": "SN1/SN2 Substitution, E1/E2 Elimination",
        "reagents": "Nu: (OH-, CN-, NR3); base (E2); heat (E1)",
        "min_equiv": 1,
    },
    {
        "name": "Nitrile (C≡N)",
        "smarts": "[CX2]#[NX1]",
        "reactions": "Hydrolysis to amide/acid, Reduction to amine",
        "reagents": "H2O + acid/base; LiAlH4 or H2/catalyst",
        "min_equiv": 1,
    },
    {
        "name": "Amide (-CONH-)",
        "smarts": "[NX3][CX3](=O)[#6]",
        "reactions": "Hydrolysis, Reduction to amine, Hofmann rearrangement",
        "reagents": "H2O + acid/base; LiAlH4; Br2 + NaOH (Hofmann)",
        "min_equiv": 1,
    },
    {
        "name": "Thiol (-SH)",
        "smarts": "[SX2H]",
        "reactions": "Oxidation to disulfide, Alkylation, Metal complexation",
        "reagents": "Oxidizing agent (I2, H2O2); alkyl halide; metal salts",
        "min_equiv": 2,
    },
    {
        "name": "Epoxide",
        "smarts": "[OX2r3]",
        "reactions": "Ring-opening (acid/base), Nucleophilic addition",
        "reagents": "H2O + acid; H2O + base; Nu: (Grignard, amine, thiol)",
        "min_equiv": 1,
    },
    {
        "name": "Phosphate / Phosphonate",
        "smarts": "[PX4](=O)([O,N])[O,N]",
        "reactions": "Hydrolysis, Transphosphorylation, Coordination chemistry",
        "reagents": "H2O + acid/base; nucleophile; metal ions",
        "min_equiv": 1,
    },
]


def _build_fg_lookup(molblock: str) -> dict[int, list[dict]]:
    """
    Map each atom index to a list of functional groups it belongs to.
    Returns {} if RDKit unavailable or parse fails.
    """
    if not RDKIT_OK:
        return {}
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            return {}
        result: dict[int, list] = {}
        for fg in _FG_PATTERNS:
            pat = Chem.MolFromSmarts(fg["smarts"])
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                for idx in match:
                    result.setdefault(idx, [])
                    if fg["name"] not in [x["name"] for x in result[idx]]:
                        result[idx].append({
                            "name":      fg["name"],
                            "reactions": fg["reactions"],
                            "reagents":  fg["reagents"],
                            "min_equiv": fg["min_equiv"],
                        })
        return result
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# BOND DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_bond_data(molblock: str) -> tuple[list[dict], dict[int, list]]:
    """
    Extract bond geometry + energy from a mol block.

    Returns:
      bonds_list  — list of bond dicts
      bonds_by_atom — {atom_idx: [bond_dict, ...]}
    """
    if not RDKIT_OK:
        return [], {}
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None or mol.GetNumConformers() == 0:
            return [], {}
        conf  = mol.GetConformer()
        bonds = []
        bonds_by_atom: dict[int, list] = {}
        for bond in mol.GetBonds():
            i    = bond.GetBeginAtomIdx()
            j    = bond.GetEndAtomIdx()
            sym1 = mol.GetAtomWithIdx(i).GetSymbol()
            sym2 = mol.GetAtomWithIdx(j).GetSymbol()
            p1   = conf.GetAtomPosition(i)
            p2   = conf.GetAtomPosition(j)
            length = math.sqrt(
                (p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2
            )
            order  = bond.GetBondTypeAsDouble()
            energy = _bond_energy_lookup(sym1, sym2, order)
            bd = {
                "idx":    bond.GetIdx(),
                "atom1":  i,
                "atom2":  j,
                "sym1":   sym1,
                "sym2":   sym2,
                "order":  order,
                "label":  _BOND_TYPE_LABEL.get(order, f"{order}"),
                "length": round(length, 3),
                "energy": energy,
            }
            bonds.append(bd)
            bonds_by_atom.setdefault(i, []).append(bd)
            bonds_by_atom.setdefault(j, []).append(bd)
        return bonds, bonds_by_atom
    except Exception:
        return [], {}


# ══════════════════════════════════════════════════════════════════════════════
# INTERMOLECULAR FORCES
# ══════════════════════════════════════════════════════════════════════════════

def calc_intermolecular_forces(smiles: str, molblock: str | None = None) -> dict:
    """
    Estimate types and relative strengths of intermolecular forces.
    Returns structured dict with qualitative + semi-quantitative data.
    """
    if not RDKIT_OK:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None and molblock:
            mol = Chem.RemoveHs(Chem.MolFromMolBlock(molblock, removeHs=False))
        if mol is None:
            return {}

        mw          = Descriptors.MolWt(mol)
        logp        = Descriptors.MolLogP(mol)
        hbd         = rdMolDescriptors.CalcNumHBD(mol)
        hba         = rdMolDescriptors.CalcNumHBA(mol)
        n_aromatic  = rdMolDescriptors.CalcNumAromaticRings(mol)
        tpsa        = Descriptors.TPSA(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        # London dispersion: scales with polarizability (proxy: MW, heavy atoms)
        london_score = min(100, mw / 5)
        london_strength = ("Weak" if london_score < 20
                           else "Moderate" if london_score < 50 else "Strong")
        london_est_kj = round(0.05 * mw, 1)

        # Check for polar bonds (dipole-dipole prerequisite)
        polar_atoms = sum(1 for a in mol.GetAtoms()
                          if a.GetSymbol() in ("N","O","F","Cl","S","P"))
        has_dipole = polar_atoms > 0

        # Dipole-dipole
        dipole_score = min(100, polar_atoms * 15 + tpsa * 0.3)
        dipole_strength = ("None" if not has_dipole
                           else "Weak" if dipole_score < 25
                           else "Moderate" if dipole_score < 60 else "Strong")
        dipole_est_kj = round(dipole_score * 0.3, 1) if has_dipole else 0

        # H-bonding
        hb_active = min(hbd, hba)
        hb_strength = ("None" if hb_active == 0
                       else "Weak" if hb_active == 1
                       else "Moderate" if hb_active <= 3 else "Strong")
        hb_est_kj = round(hb_active * 20, 1)   # ~20 kJ/mol per H-bond

        # Pi-pi stacking
        pi_strength = ("None" if n_aromatic == 0
                       else "Weak" if n_aromatic == 1
                       else "Moderate" if n_aromatic <= 3 else "Strong")
        pi_est_kj = round(n_aromatic * 8, 1)

        # Ion-dipole: check for ionic groups (charged atoms, formal charge)
        charged = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
        ion_dipole = "Present" if charged > 0 else "Absent"
        ion_est_kj = round(charged * 40, 1)

        # Dominant force
        forces_ranked = [
            ("H-bonding",      hb_est_kj),
            ("London dispersion", london_est_kj),
            ("Dipole-dipole",  dipole_est_kj),
            ("Pi-pi stacking", pi_est_kj),
            ("Ion-dipole",     ion_est_kj),
        ]
        dominant = max(forces_ranked, key=lambda x: x[1])[0]

        return {
            "London Dispersion": {
                "present": True,
                "strength": london_strength,
                "estimated_kJ_mol": london_est_kj,
                "note": f"Based on MW = {mw:.1f} g/mol; always present in all molecules",
            },
            "Dipole-Dipole": {
                "present": has_dipole,
                "strength": dipole_strength,
                "estimated_kJ_mol": dipole_est_kj,
                "note": (f"Polar atoms (N/O/F/Cl/S/P): {polar_atoms}; TPSA = {tpsa:.1f} A2"
                         if has_dipole else "No significant polar bonds detected"),
            },
            "Hydrogen Bonding": {
                "present": hb_active > 0,
                "strength": hb_strength,
                "estimated_kJ_mol": hb_est_kj,
                "note": f"H-bond donors: {hbd}, acceptors: {hba}; ~20 kJ/mol per H-bond",
            },
            "Pi-Pi Stacking": {
                "present": n_aromatic > 0,
                "strength": pi_strength,
                "estimated_kJ_mol": pi_est_kj,
                "note": (f"Aromatic rings: {n_aromatic}; ~8 kJ/mol per ring pair"
                         if n_aromatic > 0 else "No aromatic rings detected"),
            },
            "Ion-Dipole": {
                "present": charged > 0,
                "strength": ion_dipole,
                "estimated_kJ_mol": ion_est_kj,
                "note": (f"Formally charged atoms: {charged}"
                         if charged > 0 else "No formally charged atoms detected"),
            },
            "_dominant": dominant,
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# POLYMER LOCAL DATABASE
# ══════════════════════════════════════════════════════════════════════════════

_POLYMER_DB_DEFAULT: dict = {
    "PE":       {"zh": "聚乙烯",              "en": "Polyethylene",               "monomer_smiles": "C=C",              "monomer_en": "ethylene"},
    "PP":       {"zh": "聚丙烯",              "en": "Polypropylene",              "monomer_smiles": "C=CC",             "monomer_en": "propylene"},
    "PVC":      {"zh": "聚氯乙烯",            "en": "Polyvinyl chloride",         "monomer_smiles": "C=CCl",            "monomer_en": "vinyl chloride"},
    "PS":       {"zh": "聚苯乙烯",            "en": "Polystyrene",                "monomer_smiles": "C=Cc1ccccc1",      "monomer_en": "styrene"},
    "PMMA":     {"zh": "聚甲基丙烯酸甲酯",    "en": "Polymethyl methacrylate",    "monomer_smiles": "C=C(C)C(=O)OC",   "monomer_en": "methyl methacrylate"},
    "PAN":      {"zh": "聚丙烯腈",            "en": "Polyacrylonitrile",          "monomer_smiles": "C=CC#N",           "monomer_en": "acrylonitrile"},
    "PVDF":     {"zh": "聚偏氟乙烯",          "en": "Polyvinylidene fluoride",    "monomer_smiles": "FC(F)=C",          "monomer_en": "vinylidene fluoride"},
    "PTFE":     {"zh": "聚四氟乙烯",          "en": "Polytetrafluoroethylene",    "monomer_smiles": "FC(F)=C(F)F",      "monomer_en": "tetrafluoroethylene"},
    "PLA":      {"zh": "聚乳酸",              "en": "Polylactic acid",            "monomer_smiles": "C[C@@H](O)C(=O)O", "monomer_en": "lactic acid"},
    "PB":       {"zh": "聚丁二烯",            "en": "Polybutadiene",              "monomer_smiles": "C=CC=C",           "monomer_en": "butadiene"},
    "Nylon-6":  {"zh": "尼龙-6",              "en": "Nylon-6",                    "monomer_smiles": "C1CCCCC(=O)N1",   "monomer_en": "caprolactam"},
    "PET":      {"zh": "聚对苯二甲酸乙二酯",  "en": "Polyethylene terephthalate", "monomer_smiles": "OCC(=O)c1ccc(cc1)C(=O)O", "monomer_en": "ethylene + terephthalic acid"},
    "PC":       {"zh": "聚碳酸酯",            "en": "Polycarbonate",              "monomer_smiles": "OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)", "monomer_en": "bisphenol A carbonate"},
    "PEG":      {"zh": "聚乙二醇",            "en": "Polyethylene glycol",        "monomer_smiles": "OCCO",             "monomer_en": "ethylene glycol"},
    "PVA":      {"zh": "聚乙烯醇",            "en": "Polyvinyl alcohol",          "monomer_smiles": "CC(O)",            "monomer_en": "vinyl alcohol"},
    "PAA":      {"zh": "聚丙烯酸",            "en": "Polyacrylic acid",           "monomer_smiles": "C=CC(=O)O",        "monomer_en": "acrylic acid"},
    "PI":       {"zh": "聚酰亚胺",            "en": "Polyimide",                  "monomer_smiles": "O=C1NC(=O)c2ccccc21", "monomer_en": "phthalimide unit"},
    "PEEK":     {"zh": "聚醚醚酮",            "en": "Poly(ether ether ketone)",   "monomer_smiles": "O=C(c1ccc(Oc2ccccc2)cc1)c1ccccc1", "monomer_en": "PEEK repeat unit"},
}


def _init_polymer_db() -> dict:
    _POLYMER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _POLYMER_DB_PATH.exists():
        _POLYMER_DB_PATH.write_text(
            json.dumps(_POLYMER_DB_DEFAULT, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    try:
        return json.loads(_POLYMER_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _POLYMER_DB_DEFAULT.copy()


def _polymer_db_lookup(query: str, db: dict) -> dict | None:
    q = query.strip()
    for key, data in db.items():
        if (q.upper() == key.upper()
                or q == data.get("zh", "")
                or q.lower() == data.get("en", "").lower()):
            return {**data, "key": key}
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS molecules (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT,
                smiles     TEXT,
                energy     REAL,
                formula    TEXT,
                file_path  TEXT,
                created_at TEXT
            )
        """)


def _db_lookup(name: str):
    with sqlite3.connect(_DB_PATH) as conn:
        return conn.execute(
            "SELECT name, smiles, energy, formula, file_path "
            "FROM molecules WHERE LOWER(name)=LOWER(?) LIMIT 1",
            (name,),
        ).fetchone()


def _db_save(name: str, smiles: str, energy, formula: str, file_path: str) -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO molecules (name, smiles, energy, formula, file_path, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (name, smiles, energy, formula, file_path,
             datetime.datetime.now().isoformat()),
        )


def _save_sdf(name: str, molblock: str) -> Path:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = _MODELS_DIR / f"{safe}_{ts}.sdf"
    path.write_text(molblock, encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT INPUT LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _looks_like_smiles(text: str) -> bool:
    if _has_chinese(text) or " " in text.strip():
        return False
    smiles_chars = set("CNOSPFClBrI=#@+\\-/[]()%0123456789cnos.")
    return len(text) > 0 and all(c in smiles_chars for c in text)


def _qwen_lookup(text: str) -> tuple[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        try:
            api_key = (
                st.secrets.get("OPENROUTER_API_KEY")
                or st.secrets.get("DASHSCOPE_API_KEY")
                or ""
            )
        except Exception:
            api_key = ""
    if not api_key:
        return "", ""

    if os.getenv("DASHSCOPE_API_KEY"):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model    = "qwen3-32b"
    else:
        base_url = "https://openrouter.ai/api/v1"
        model    = "qwen/qwen3-32b"

    system_prompt = (
        "You are a chemistry database assistant. "
        "Given a chemical name (possibly in Chinese), respond ONLY with a valid JSON object "
        'containing exactly two fields: "english_name" and "smiles". '
        "english_name must be the standard IUPAC or widely-used English name. "
        "smiles must be a valid canonical SMILES string. "
        "If you cannot determine SMILES with high confidence, set smiles to null. "
        "Output raw JSON only — no markdown, no explanation, no preamble.\n\n"
        'Example: Input: 苯乙烯  Output: {"english_name":"styrene","smiles":"C=Cc1ccccc1"}'
    )
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model":      model,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": text.strip()},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        data = json.loads(raw)
        return data.get("english_name", ""), data.get("smiles") or ""
    except Exception:
        return "", ""


def _pubchem_lookup(query: str) -> tuple[str, str]:
    if not PUBCHEM_OK:
        return "", "pubchempy not installed"
    try:
        hits = pcp.get_compounds(query.strip(), "name")
        if hits:
            c = hits[0]
            return c.isomeric_smiles or "", c.molecular_formula or ""
        return "", f"PubChem: no results for '{query}'"
    except Exception as e:
        return "", str(e)


def get_structure_logic(user_input: str, polymer_db: dict) -> dict:
    text = user_input.strip()
    if not text:
        return {"english_name": "", "smiles": "", "formula": "",
                "source": "failed", "error": "Empty input"}
    if _looks_like_smiles(text):
        return {"english_name": text, "smiles": text, "formula": "",
                "source": "smiles_direct", "error": ""}
    hit = _polymer_db_lookup(text, polymer_db)
    if hit:
        return {"english_name": hit["en"], "smiles": hit["monomer_smiles"],
                "formula": "", "source": "polymer_db", "error": ""}
    row = _db_lookup(text)
    if row:
        _, _smiles, _, _formula, _ = row
        return {"english_name": text, "smiles": _smiles or "",
                "formula": _formula or "", "source": "db_cache", "error": ""}
    if _has_chinese(text):
        en_name, smiles = _qwen_lookup(text)
        if smiles:
            return {"english_name": en_name, "smiles": smiles,
                    "formula": "", "source": "qwen", "error": ""}
        if en_name:
            smiles, formula = _pubchem_lookup(en_name)
            if smiles:
                return {"english_name": en_name, "smiles": smiles,
                        "formula": formula, "source": "pubchem", "error": ""}
    smiles, formula = _pubchem_lookup(text)
    if smiles:
        return {"english_name": text, "smiles": smiles,
                "formula": formula, "source": "pubchem", "error": ""}
    return {
        "english_name": text, "smiles": "", "formula": "",
        "source": "failed",
        "error": (
            f"Cannot resolve '{text}'.\n"
            "Try: (1) direct SMILES input  "
            "(2) configure OPENROUTER_API_KEY / DASHSCOPE_API_KEY for LLM lookup  "
            "(3) install pubchempy: pip install pubchempy"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CIF / PDB FILE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_structure_file(file_bytes: bytes, ext: str) -> tuple[str | None, str | None]:
    if not RDKIT_OK:
        return None, "RDKit not installed"
    ext = ext.lower().lstrip(".")
    if ext == "pdb":
        try:
            content = file_bytes.decode("utf-8", errors="replace")
            mol = Chem.MolFromPDBBlock(content, removeHs=False, sanitize=False)
            if mol is None:
                return None, "PDB parse failed"
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            if mol.GetNumConformers() == 0:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            return Chem.MolToMolBlock(mol), None
        except Exception as exc:
            return None, str(exc)

    elif ext == "cif":
        if not ASE_OK:
            return None, "CIF parsing requires ASE — run: pip install ase"
        try:
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)
            atoms     = _ase_read(str(tmp_path))
            tmp_path.unlink(missing_ok=True)
            positions = atoms.get_positions()
            symbols   = atoms.get_chemical_symbols()
            rwmol = Chem.RWMol()
            conf  = Chem.Conformer(len(symbols))
            for i, (sym, pos) in enumerate(zip(symbols, positions)):
                try:
                    atom = Chem.Atom(sym)
                except Exception:
                    atom = Chem.Atom(6)
                rwmol.AddAtom(atom)
                conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
            rwmol.AddConformer(conf, assignId=True)
            if RDKIT_BONDS_OK:
                try:
                    rdDetermineBonds.DetermineConnectivity(rwmol)
                    rdDetermineBonds.DetermineBondOrders(rwmol, charge=0)
                except Exception:
                    pass
            try:
                Chem.SanitizeMol(rwmol, catchErrors=True)
            except Exception:
                pass
            return Chem.MolToMolBlock(rwmol.GetMol()), None
        except Exception as exc:
            return None, f"CIF parse error: {exc}"
    else:
        return None, f"Unsupported format: .{ext}  (accepts .pdb / .cif)"


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def smiles_to_3d(smiles: str) -> tuple[str | None, str | None]:
    if not RDKIT_OK:
        return None, "RDKit not installed"
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, f"Invalid SMILES: {smiles}"
        mol = Chem.AddHs(mol)
        ps  = AllChem.ETKDGv3()
        ps.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps) == -1:
            ps2 = AllChem.ETKDG()
            ps2.randomSeed = 42
            if AllChem.EmbedMolecule(mol, ps2) == -1:
                return None, "3D embedding failed"
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return Chem.MolToMolBlock(mol), None
    except Exception as exc:
        return None, str(exc)


def _build_single_polymer(monomer_smiles: str, n: int,
                          use_pysoftk: bool = False) -> tuple[str | None, str | None]:
    """
    Build a linear homopolymer of n repeat units.

    If PySoftK is installed and use_pysoftk=True, delegates to PySoftK for
    more accurate 3D topology. Falls back to RDKit chain-building otherwise.
    """
    if not RDKIT_OK:
        return None, "RDKit not installed"

    n = max(2, min(int(n), 30))

    # ── PySoftK path (optional) ───────────────────────────────────────────────
    if use_pysoftk and PYSOFTK_OK:
        try:
            sm  = _PySoftK_Sm(monomer_smiles, monomer_smiles, "Br")
            lp  = _PySoftK_Lp(sm.mon_to_poly(), n, shift=1.25, core="Au")
            mol = Chem.MolFromSmiles(lp.linear_polymer())
            if mol is not None:
                mol = Chem.AddHs(mol)
                ps  = AllChem.ETKDGv3()
                ps.randomSeed = 42
                AllChem.EmbedMolecule(mol, ps)
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                return Chem.MolToMolBlock(mol), None
        except Exception:
            pass   # fall through to RDKit path

    # ── RDKit chain-building path ─────────────────────────────────────────────
    try:
        mono = Chem.MolFromSmiles(monomer_smiles.strip())
        if mono is None:
            return None, f"Invalid monomer SMILES: {monomer_smiles}"
        N         = mono.GetNumAtoms()
        vinyl_pat = Chem.MolFromSmarts("[#6:1]=[#6:2]")
        match     = mono.GetSubstructMatch(vinyl_pat)
        has_vinyl = bool(match)
        skip_atoms = set(match) if has_vinyl else set()
        rwmol = Chem.RWMol()

        def _add_unit(rep: int) -> None:
            offset = rep * N
            for atom in mono.GetAtoms():
                new = Chem.Atom(atom.GetAtomicNum())
                new.SetFormalCharge(atom.GetFormalCharge())
                rwmol.AddAtom(new)
            for bond in mono.GetBonds():
                a1   = bond.GetBeginAtomIdx() + offset
                a2   = bond.GetEndAtomIdx()   + offset
                orig = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
                btype = (Chem.BondType.SINGLE
                         if skip_atoms and orig == skip_atoms
                         else bond.GetBondType())
                rwmol.AddBond(a1, a2, btype)

        if has_vinyl:
            idx_a, idx_b = match[0], match[1]
            for rep in range(n):
                _add_unit(rep)
                if rep > 0:
                    rwmol.AddBond(idx_b + (rep - 1) * N,
                                  idx_a +  rep      * N,
                                  Chem.BondType.SINGLE)
        else:
            heavy = [a.GetIdx() for a in mono.GetAtoms() if a.GetAtomicNum() > 1]
            terms = [a.GetIdx() for a in mono.GetAtoms()
                     if a.GetAtomicNum() > 1 and a.GetDegree() == 1]
            if len(terms) < 2:
                if len(heavy) >= 2:
                    terms = [heavy[0], heavy[-1]]
                else:
                    return None, "No linkable heavy atoms found"
            head, tail = terms[0], terms[-1]
            for rep in range(n):
                _add_unit(rep)
                if rep > 0:
                    rwmol.AddBond(tail + (rep - 1) * N,
                                  head +  rep      * N,
                                  Chem.BondType.SINGLE)

        mol = rwmol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return None, f"Sanitization error: {e}"
        mol = Chem.AddHs(mol)
        ps  = AllChem.ETKDGv3()
        ps.randomSeed = 42; ps.maxIterations = 2000
        if AllChem.EmbedMolecule(mol, ps) == -1:
            ps2 = AllChem.ETKDG()
            ps2.randomSeed = 42
            if AllChem.EmbedMolecule(mol, ps2) == -1:
                return None, (
                    f"3D embedding failed for n={n}. "
                    "Try a smaller n (<=10) or a different monomer."
                )
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return Chem.MolToMolBlock(mol), None
    except Exception as exc:
        return None, str(exc)


def build_copolymer(monomers: list[dict],
                    use_pysoftk: bool = False) -> tuple[str | None, str | None]:
    """
    Build a block copolymer: [A]_n1 — [B]_n2 — ...
    Segments joined tail-to-head with a single bond.
    """
    if not RDKIT_OK:
        return None, "RDKit not installed"
    valid = [m for m in monomers
             if m.get("smiles", "").strip() and int(m.get("n", 1)) >= 1]
    if not valid:
        return None, "Please provide at least one valid monomer"
    if len(valid) == 1:
        return _build_single_polymer(valid[0]["smiles"], int(valid[0]["n"]),
                                     use_pysoftk=use_pysoftk)

    segments: list = []
    for spec in valid:
        mb, err = _build_single_polymer(spec["smiles"], int(spec["n"]),
                                        use_pysoftk=use_pysoftk)
        if err:
            return None, f"Block {spec.get('label', spec['smiles'])}: {err}"
        seg = Chem.RemoveHs(Chem.MolFromMolBlock(mb, removeHs=False))
        segments.append(seg)

    mol = segments[0]
    for i in range(1, len(segments)):
        seg = segments[i]
        N1  = mol.GetNumAtoms()

        def _terminals(m) -> list[int]:
            deg1 = [a.GetIdx() for a in m.GetAtoms()
                    if a.GetAtomicNum() > 1 and a.GetDegree() == 1]
            if deg1:
                return deg1
            heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
            return [heavy[0], heavy[-1]] if len(heavy) >= 2 else heavy

        terms1 = _terminals(mol)
        terms2 = _terminals(seg)
        if not terms1 or not terms2:
            return None, f"Block {i} has no linkable terminal atoms"

        combined = Chem.CombineMols(mol, seg)
        rw = Chem.RWMol(combined)
        rw.AddBond(terms1[-1], terms2[0] + N1, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw)
        except Exception as exc:
            return None, f"Copolymer sanitization failed: {exc}"
        mol = rw.GetMol()

    mol = Chem.AddHs(mol)
    ps  = AllChem.ETKDGv3()
    ps.randomSeed = 42; ps.maxIterations = 3000
    if AllChem.EmbedMolecule(mol, ps) == -1:
        ps2 = AllChem.ETKDG()
        ps2.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps2) == -1:
            return None, "Copolymer 3D embedding failed. Try smaller n values."

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return Chem.MolToMolBlock(mol), None


def parse_bigsmiles(bigsmiles_str: str) -> tuple[str | None, str | None]:
    """
    Attempt to parse a BigSMILES or SMiPoly notation string into a SMILES.
    Uses SMiPoly if available, otherwise strips BigSMILES stochastic brackets.
    """
    if SMIPOLY_OK:
        try:
            sp  = _SMiPoly(bigsmiles_str)
            smi = sp.to_smiles()
            return smi, None
        except Exception as e:
            pass
    # Fallback: strip { } [ ] repeat brackets naively, extract core SMILES
    core = re.sub(r"[\{\}]", "", bigsmiles_str)
    core = re.sub(r"\[>[^\]]*\]|\[<[^\]]*\]", "", core).strip()
    if RDKIT_OK and Chem.MolFromSmiles(core):
        return core, None
    return None, "BigSMILES parsing failed. Install smipoly: pip install smipoly"


# ══════════════════════════════════════════════════════════════════════════════
# UFF OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def uff_optimize(molblock: str) -> tuple[str | None, float | None, float | None, str | None]:
    if not RDKIT_OK:
        return None, None, None, "RDKit not installed"
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            return None, None, None, "Cannot read mol block"
        ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is None:
            return None, None, None, (
                "UFF force field init failed. "
                "Molecule may contain atom types not supported by UFF."
            )
        e_before = round(ff.CalcEnergy(), 4)
        ff.Minimize(maxIts=500)
        e_after  = round(ff.CalcEnergy(), 4)
        return Chem.MolToMolBlock(mol), e_before, e_after, None
    except Exception as exc:
        return None, None, None, str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

def calc_radius_of_gyration(molblock: str) -> float | None:
    if not (RDKIT_OK and NUMPY_OK):
        return None
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        conf      = mol.GetConformer()
        positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        masses    = np.array([atom.GetMass() for atom in mol.GetAtoms()])
        total_m   = masses.sum()
        com       = (masses[:, None] * positions).sum(axis=0) / total_m
        rg2       = (masses * np.sum((positions - com) ** 2, axis=1)).sum() / total_m
        return float(np.sqrt(rg2))
    except Exception:
        return None


def calc_uff_energy(molblock: str) -> float | None:
    if not RDKIT_OK:
        return None
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            return None
        ff = AllChem.UFFGetMoleculeForceField(mol)
        return round(ff.CalcEnergy(), 4) if ff else None
    except Exception:
        return None


def calc_mw_distribution(monomers: list[dict]) -> dict:
    if not RDKIT_OK:
        return {}
    rows     = []
    total_mw = 0.0
    for spec in monomers:
        smiles = spec.get("smiles", "").strip()
        n      = int(spec.get("n", 1))
        label  = spec.get("label", smiles[:12])
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mono_mw    = round(Descriptors.MolWt(mol), 2)
        segment_mw = round(mono_mw * n, 2)
        total_mw  += segment_mw
        rows.append({"label": label, "smiles": smiles,
                     "mono_mw": mono_mw, "n": n, "segment_mw": segment_mw})
    return {"rows": rows, "total_mw": round(total_mw, 2), "pdi": 1.0}


def get_mol_properties(smiles: str, molblock: str | None = None) -> dict:
    if not RDKIT_OK:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None and molblock:
            mol = Chem.RemoveHs(Chem.MolFromMolBlock(molblock, removeHs=False))
        if mol is None:
            return {}
        # Count bonds by type
        n_single = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1.0)
        n_double = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0)
        n_triple = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 3.0)
        n_arom   = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1.5)
        total_bonds = mol.GetNumBonds()

        props: dict = {
            "Molecular Weight (g/mol)":      round(Descriptors.MolWt(mol), 4),
            "Exact Mass (g/mol)":            round(Descriptors.ExactMolWt(mol), 4),
            "Molecular Formula":             rdMolDescriptors.CalcMolFormula(mol),
            "LogP":                          round(Descriptors.MolLogP(mol), 4),
            "TPSA (A2)":                     round(Descriptors.TPSA(mol), 4),
            "H-Bond Donors":                 rdMolDescriptors.CalcNumHBD(mol),
            "H-Bond Acceptors":              rdMolDescriptors.CalcNumHBA(mol),
            "Rotatable Bonds":               rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Heavy Atoms":                   mol.GetNumHeavyAtoms(),
            "Total Bonds":                   total_bonds,
            "Single Bonds":                  n_single,
            "Double Bonds":                  n_double,
            "Triple Bonds":                  n_triple,
            "Aromatic Bonds":                n_arom,
            "Ring Count":                    rdMolDescriptors.CalcNumRings(mol),
            "Aromatic Rings":                rdMolDescriptors.CalcNumAromaticRings(mol),
            "Stereo Centers":                len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "Fraction Csp3":                 round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
            "Molar Refractivity":            round(Descriptors.MolMR(mol), 4),
        }
        if molblock:
            e = calc_uff_energy(molblock)
            if e is not None:
                props["UFF Total Energy (kcal/mol)"] = e
            rg = calc_radius_of_gyration(molblock)
            if rg is not None:
                props["Radius of Gyration (A)"] = round(rg, 4)
        return props
    except Exception:
        return {}


def molblock_to_pdb(molblock: str) -> str | None:
    if not RDKIT_OK:
        return None
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        return Chem.MolToPDBBlock(mol) if mol else None
    except Exception:
        return None


def molblock_to_xyz(molblock: str) -> str | None:
    if not RDKIT_OK:
        return None
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=True)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        conf  = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), "Generated by M2 Molecular Simulation Engine"]
        for i in range(mol.GetNumAtoms()):
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            p   = conf.GetAtomPosition(i)
            lines.append(f"{sym:<4} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
        return "\n".join(lines)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3D VIEWER  (enhanced: bond count, click-to-inspect, zoom-at-cursor)
# ══════════════════════════════════════════════════════════════════════════════

_STYLE_JS: dict = {
    "Ball-and-Stick": None,
    "Stick":         '{"stick":{"radius":0.15}}',
    "Space-filling": '{"sphere":{"radius":0.4}}',
    "Line":          '{"line":{}}',
}


def render_3d_viewer(molblock: str, style: str, bg_color: str, spin: bool,
                     height: int = 500) -> None:
    if not RDKIT_OK:
        st.warning("RDKit required for 3D viewer")
        return

    # Pre-compute bond data and FG lookup in Python
    bonds_list, bonds_by_atom = extract_bond_data(molblock)
    fg_by_atom                = _build_fg_lookup(molblock)

    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        total_atoms = mol.GetNumAtoms() if mol else 0
        total_bonds = mol.GetNumBonds() if mol else 0
    except Exception:
        total_atoms = total_bonds = 0

    # Serialize to JS
    bonds_by_atom_js = {
        str(k): [
            {
                "partner": (b["atom2"] if b["atom1"] == k else b["atom1"]),
                "partnerSym": (b["sym2"] if b["atom1"] == k else b["sym1"]),
                "order":  b["order"],
                "label":  b["label"],
                "length": b["length"],
                "energy": b["energy"] if b["energy"] is not None else "N/A",
            }
            for b in v
        ]
        for k, v in bonds_by_atom.items()
    }
    fg_by_atom_js = {
        str(k): v for k, v in fg_by_atom.items()
    }

    # Atom symbol lookup
    atom_syms = {}
    try:
        mol2 = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol2:
            for a in mol2.GetAtoms():
                atom_syms[str(a.GetIdx())] = a.GetSymbol()
    except Exception:
        pass

    safe    = molblock.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    bg_css  = "#" + bg_color[2:]
    spin_js = "true" if spin else "false"

    if style == "Ball-and-Stick":
        set_style = "viewer.setStyle({},{stick:{radius:0.10},sphere:{scale:0.28}});"
    else:
        js_str    = _STYLE_JS.get(style, '{"stick":{"radius":0.15}}')
        set_style = f"viewer.setStyle({{}},{js_str});"

    bonds_by_atom_json = json.dumps(bonds_by_atom_js)
    fg_by_atom_json    = json.dumps(fg_by_atom_js)
    atom_syms_json     = json.dumps(atom_syms)

    html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{bg_css};font-family:'SF Mono','Consolas',monospace;overflow:hidden}}
  #container{{position:relative;width:100%;height:{height}px}}
  #v{{width:100%;height:100%}}
  .overlay-btn{{
    background:rgba(15,15,30,0.75);
    border:1px solid rgba(120,180,255,0.35);
    border-radius:4px;padding:4px 12px;cursor:pointer;
    font-size:11px;color:#aac8f0;
    backdrop-filter:blur(8px);
    transition:background 0.15s,color 0.15s;
  }}
  .overlay-btn:hover{{background:rgba(60,120,200,0.4);color:#fff}}
  #controls{{position:absolute;top:8px;right:8px;z-index:100;display:flex;gap:6px}}
  #bond-counter{{
    position:absolute;top:8px;left:8px;z-index:100;
    background:rgba(15,15,30,0.75);
    border:1px solid rgba(120,180,255,0.25);
    border-radius:4px;padding:5px 10px;
    font-size:11px;color:#7db8e8;
    backdrop-filter:blur(8px);
    line-height:1.6;
  }}
  #info-panel{{
    position:absolute;bottom:12px;left:12px;z-index:200;
    max-width:340px;max-height:320px;overflow-y:auto;
    background:rgba(8,12,26,0.92);
    border:1px solid rgba(100,160,255,0.4);
    border-radius:8px;padding:12px 14px;
    font-size:11px;color:#c8d8f0;
    backdrop-filter:blur(12px);
    display:none;
    scrollbar-width:thin;
    scrollbar-color:rgba(100,160,255,0.3) transparent;
  }}
  #info-panel h4{{color:#80c0ff;font-size:12px;margin-bottom:6px;
                  border-bottom:1px solid rgba(80,140,220,0.3);padding-bottom:4px}}
  #info-panel .bond-row{{
    display:grid;grid-template-columns:1fr 1fr 1fr;
    gap:3px;margin-bottom:3px;padding:3px 5px;
    background:rgba(40,60,100,0.3);border-radius:3px;
    font-size:10.5px;
  }}
  #info-panel .bond-row span.label{{color:#90b8e0}}
  #info-panel .fg-item{{
    margin-top:6px;padding:6px 8px;
    background:rgba(30,60,40,0.4);
    border:1px solid rgba(80,180,100,0.3);
    border-radius:4px;
  }}
  #info-panel .fg-name{{color:#88e0a0;font-weight:600;margin-bottom:3px}}
  #info-panel .fg-detail{{color:#99c0a8;font-size:10px;line-height:1.5}}
  #close-panel{{
    float:right;background:none;border:none;
    color:#80a8d0;cursor:pointer;font-size:14px;padding:0 2px;
  }}
  #close-panel:hover{{color:#fff}}
</style>
</head><body>
<div id="container">
  <div id="v"></div>
  <div id="bond-counter">
    Atoms: {total_atoms} &nbsp;|&nbsp; Bonds: {total_bonds}
  </div>
  <div id="controls">
    <button class="overlay-btn" onclick="toggleSpin()">Spin</button>
    <button class="overlay-btn" onclick="viewer.zoomTo();viewer.render()">Reset</button>
    <button class="overlay-btn" onclick="document.getElementById('info-panel').style.display='none'">Clear</button>
  </div>
  <div id="info-panel">
    <button id="close-panel" onclick="this.parentElement.style.display='none'">x</button>
    <div id="info-content"></div>
  </div>
</div>
<script>
var BONDS_BY_ATOM = {bonds_by_atom_json};
var FG_BY_ATOM    = {fg_by_atom_json};
var ATOM_SYMS     = {atom_syms_json};

var viewer = $3Dmol.createViewer(
  document.getElementById('v'),
  {{backgroundColor: "{bg_color}", zoom: 1.0}}
);
viewer.addModel(`{safe}`, 'mol');
{set_style}
viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.04, color:'white'}});
viewer.zoomTo();
var _spin = {spin_js};
viewer.spin(_spin);
viewer.render();

function toggleSpin(){{ _spin = !_spin; viewer.spin(_spin); viewer.render(); }}

// ── Atom click handler ────────────────────────────────────────────────────
viewer.setClickable({{}}, true, function(atom) {{
  var idx = atom.index;
  var sym = ATOM_SYMS[String(idx)] || atom.elem || '?';
  var bonds = BONDS_BY_ATOM[String(idx)] || [];
  var fgs   = FG_BY_ATOM[String(idx)]   || [];

  var html = '<h4>Atom #' + idx + ' &mdash; ' + sym + '</h4>';

  // Bond table
  if (bonds.length > 0) {{
    html += '<div style="color:#7ab0d8;font-size:10px;margin-bottom:4px">'
          + 'Connected bonds (' + bonds.length + '):</div>';
    html += '<div class="bond-row" style="color:#5080b0;font-size:9.5px">'
          + '<span>Partner</span><span>Type / Length</span><span>Bond Energy</span></div>';
    bonds.forEach(function(b) {{
      var eSt = (b.energy === 'N/A') ? 'N/A'
                : (parseFloat(b.energy).toFixed(1) + ' kJ/mol');
      html += '<div class="bond-row">'
           + '<span class="label">' + b.partnerSym + '#' + b.partner + '</span>'
           + '<span>' + b.label + ' / ' + b.length + ' \\u00c5</span>'
           + '<span>' + eSt + '</span>'
           + '</div>';
    }});
  }} else {{
    html += '<div style="color:#607080;font-size:10px">No bond data available.</div>';
  }}

  // Functional group info
  if (fgs.length > 0) {{
    html += '<div style="color:#7ad080;font-size:10px;margin-top:8px;margin-bottom:3px">'
          + 'Functional groups:</div>';
    fgs.forEach(function(fg) {{
      html += '<div class="fg-item">'
           + '<div class="fg-name">' + fg.name + '</div>'
           + '<div class="fg-detail">'
           + '<b>Reactions:</b> ' + fg.reactions + '<br>'
           + '<b>Reagents:</b> ' + fg.reagents + '<br>'
           + '<b>Min equiv.:</b> ' + fg.min_equiv
           + '</div></div>';
    }});
  }}

  // Highlight clicked atom
  viewer.setStyle({{index: idx}}, {{sphere:{{color:'#ffcc44', radius:0.4}}}});
  {set_style.replace("viewer.setStyle({},", "// skip reset")} 
  viewer.render();

  document.getElementById('info-content').innerHTML = html;
  document.getElementById('info-panel').style.display = 'block';
}});
</script>
</body></html>"""

    components.html(html, height=height + 10, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE EXPORT COMPONENT  (live preview + PNG download with data overlays)
# ══════════════════════════════════════════════════════════════════════════════

def render_image_export(molblock: str, style: str, bg_color: str,
                        props: dict, imf: dict, mol_name: str) -> None:
    """
    Render an in-browser image export editor:
    - Live 3D viewer (drag/rotate/zoom)
    - Checkboxes to toggle data overlays
    - Capture + download as PNG
    """
    safe    = molblock.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    bg_css  = "#" + bg_color[2:]

    if style == "Ball-and-Stick":
        set_style = "viewer.setStyle({},{stick:{radius:0.10},sphere:{scale:0.28}});"
    else:
        js_str    = _STYLE_JS.get(style, '{"stick":{"radius":0.15}}')
        set_style = f"viewer.setStyle({{}},{js_str});"

    # Build overlay data categories for the checkboxes
    # Group properties into categories
    basic_props = {k: v for k, v in props.items()
                   if k in ("Molecular Weight (g/mol)", "Molecular Formula",
                             "LogP", "TPSA (A2)", "Heavy Atoms")}
    bond_props  = {k: v for k, v in props.items()
                   if "Bond" in k or "bond" in k}
    ring_props  = {k: v for k, v in props.items()
                   if "Ring" in k or "ring" in k or "Aromatic" in k}
    energy_props = {k: v for k, v in props.items()
                    if "Energy" in k or "Gyration" in k}

    basic_js  = json.dumps({str(k): str(v) for k, v in basic_props.items()})
    bond_js   = json.dumps({str(k): str(v) for k, v in bond_props.items()})
    ring_js   = json.dumps({str(k): str(v) for k, v in ring_props.items()})
    energy_js = json.dumps({str(k): str(v) for k, v in energy_props.items()})
    name_js   = json.dumps(mol_name[:40])

    imf_js = json.dumps({
        k: f"{v['strength']} (~{v['estimated_kJ_mol']} kJ/mol)"
        for k, v in imf.items()
        if k != "_dominant" and v.get("present")
    })

    html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0a0f1e;font-family:'SF Mono',monospace;color:#c0d0e8;display:flex;
        flex-direction:row;height:620px;overflow:hidden}}
  #left{{flex:1;position:relative}}
  #v{{width:100%;height:100%}}
  #right{{width:230px;background:#0d1428;border-left:1px solid rgba(80,120,200,0.2);
          padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}}
  h3{{color:#80b8f0;font-size:12px;letter-spacing:0.05em;margin-bottom:2px}}
  .section{{background:rgba(20,35,60,0.6);border:1px solid rgba(80,130,200,0.2);
             border-radius:6px;padding:8px 10px}}
  .section-title{{color:#60a0e0;font-size:10px;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:6px;font-weight:600}}
  label{{display:flex;align-items:center;gap:6px;font-size:10.5px;color:#a8c0d8;
         cursor:pointer;margin-bottom:3px}}
  label input{{accent-color:#4090e0;width:13px;height:13px}}
  #capture-btn{{
    background:linear-gradient(135deg,#1a4a8a,#2060c0);
    border:1px solid #3070d0;border-radius:6px;
    color:#c0e0ff;font-size:11px;padding:8px 14px;
    cursor:pointer;width:100%;margin-top:4px;
    transition:background 0.2s;
  }}
  #capture-btn:hover{{background:linear-gradient(135deg,#2050a0,#3070d0)}}
  #name-input{{
    background:rgba(20,35,60,0.8);border:1px solid rgba(80,130,200,0.3);
    border-radius:4px;color:#c0d8f0;font-size:10.5px;
    padding:4px 8px;width:100%;
  }}
  #canvas-overlay{{position:absolute;top:0;left:0;pointer-events:none;
                   width:100%;height:100%}}
  .status{{color:#5080a0;font-size:9.5px;text-align:center;margin-top:4px}}
</style>
</head><body>
<div id="left">
  <div id="v"></div>
  <canvas id="canvas-overlay"></canvas>
</div>
<div id="right">
  <h3>Image Export Editor</h3>

  <div class="section">
    <div class="section-title">Filename</div>
    <input id="name-input" type="text" value={name_js} placeholder="molecule"/>
  </div>

  <div class="section">
    <div class="section-title">Basic Properties</div>
    <label><input type="checkbox" id="cb-basic" checked> Molecular properties</label>
    <label><input type="checkbox" id="cb-bonds"> Bond statistics</label>
    <label><input type="checkbox" id="cb-rings"> Ring / aromaticity</label>
    <label><input type="checkbox" id="cb-energy"> Energy / geometry</label>
    <label><input type="checkbox" id="cb-imf"> Intermolecular forces</label>
  </div>

  <div class="section">
    <div class="section-title">Appearance</div>
    <label><input type="checkbox" id="cb-title" checked> Molecule name</label>
    <label><input type="checkbox" id="cb-border"> Border frame</label>
    <label><input type="checkbox" id="cb-watermark"> Watermark (M2)</label>
  </div>

  <button id="capture-btn" onclick="captureImage()">Capture & Download PNG</button>
  <div class="status" id="status"></div>
</div>

<script>
var BASIC  = {basic_js};
var BONDS  = {bond_js};
var RINGS  = {ring_js};
var ENERGY = {energy_js};
var IMF    = {imf_js};

var viewer = $3Dmol.createViewer(
  document.getElementById('v'),
  {{backgroundColor: "{bg_color}", zoom:1.0}}
);
viewer.addModel(`{safe}`, 'mol');
{set_style}
viewer.zoomTo();
viewer.render();

function captureImage() {{
  var status = document.getElementById('status');
  status.textContent = 'Rendering...';

  var uri = viewer.pngURI(2.0);
  var img = new Image();
  img.onload = function() {{
    var W = img.width, H = img.height;
    var cvs = document.createElement('canvas');
    cvs.width = W; cvs.height = H;
    var ctx = cvs.getContext('2d');

    // Draw 3D scene
    ctx.drawImage(img, 0, 0);

    // Optional border
    if (document.getElementById('cb-border').checked) {{
      ctx.strokeStyle = 'rgba(80,140,220,0.7)';
      ctx.lineWidth = 3;
      ctx.strokeRect(1.5, 1.5, W-3, H-3);
    }}

    // Build overlay lines
    var lines = [];
    if (document.getElementById('cb-title').checked) {{
      lines.push({{ text: document.getElementById('name-input').value,
                    color:'#ffffff', size:18, bold:true }});
    }}
    function addSection(obj, color) {{
      Object.entries(obj).forEach(function([k,v]) {{
        lines.push({{text: k + ': ' + v, color: color, size:13}});
      }});
    }}
    if (document.getElementById('cb-basic').checked)  addSection(BASIC,  '#a0d0ff');
    if (document.getElementById('cb-bonds').checked)  addSection(BONDS,  '#a0e8b0');
    if (document.getElementById('cb-rings').checked)  addSection(RINGS,  '#f0c080');
    if (document.getElementById('cb-energy').checked) addSection(ENERGY, '#e090c0');
    if (document.getElementById('cb-imf').checked)    addSection(IMF,    '#90e0e0');

    // Draw text overlay
    var x = 18, y = 36;
    lines.forEach(function(l) {{
      ctx.font = (l.bold ? 'bold ' : '') + l.size + 'px "SF Mono",monospace';
      var metrics = ctx.measureText(l.text);
      // Semi-transparent background
      ctx.fillStyle = 'rgba(8,14,32,0.65)';
      ctx.fillRect(x-4, y-l.size, metrics.width+8, l.size+4);
      ctx.fillStyle = l.color;
      ctx.fillText(l.text, x, y);
      y += l.size + 6;
    }});

    if (document.getElementById('cb-watermark').checked) {{
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = 'rgba(100,150,220,0.5)';
      ctx.fillText('M2 Molecular Simulation Engine', W-230, H-10);
    }}

    var fname = (document.getElementById('name-input').value || 'molecule')
                .replace(/[^a-zA-Z0-9_-]/g, '_') + '.png';
    var link = document.createElement('a');
    link.download = fname;
    link.href = cvs.toDataURL('image/png');
    link.click();
    status.textContent = 'Downloaded: ' + fname;
  }};
  img.src = uri;
}}
</script>
</body></html>"""

    components.html(html, height=630, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD MODULE
# ══════════════════════════════════════════════════════════════════════════════

def render_download_module(molblock: str, mol_name: str, smiles: str,
                            props: dict, imf: dict, bonds_list: list[dict],
                            style: str, bg_color: str) -> None:
    st.markdown("### Download")
    tab_data, tab_img, tab_model = st.tabs(["Data", "Image", "Model"])

    # ── Data Download ─────────────────────────────────────────────────────────
    with tab_data:
        st.markdown("Select data categories to include:")
        c1, c2 = st.columns(2)
        include_basic  = c1.checkbox("Basic molecular properties", value=True,
                                     key="dl_basic")
        include_bonds  = c1.checkbox("Bond statistics", value=True,
                                     key="dl_bonds")
        include_rings  = c1.checkbox("Ring / aromaticity data", value=False,
                                     key="dl_rings")
        include_energy = c2.checkbox("Energy / geometry", value=True,
                                     key="dl_energy")
        include_imf    = c2.checkbox("Intermolecular forces", value=True,
                                     key="dl_imf")
        include_bond_list = c2.checkbox("Individual bond table", value=False,
                                        key="dl_bond_list")

        export_format = st.radio("Format", ["JSON", "CSV"], horizontal=True,
                                 key="dl_fmt")

        if st.button("Prepare Data Download", key="dl_prepare"):
            export: dict = {"molecule": mol_name, "smiles": smiles}

            if include_basic:
                basic = {k: v for k, v in props.items()
                         if k not in ("UFF Total Energy (kcal/mol)",
                                      "Radius of Gyration (A)",
                                      "Total Bonds","Single Bonds","Double Bonds",
                                      "Triple Bonds","Aromatic Bonds",
                                      "Ring Count","Aromatic Rings")}
                export["basic_properties"] = basic

            if include_bonds:
                export["bond_statistics"] = {
                    k: v for k, v in props.items()
                    if "Bond" in k or "bond" in k
                }

            if include_rings:
                export["ring_aromaticity"] = {
                    k: v for k, v in props.items()
                    if "Ring" in k or "Aromatic" in k
                }

            if include_energy:
                export["energy_geometry"] = {
                    k: v for k, v in props.items()
                    if "Energy" in k or "Gyration" in k
                }

            if include_imf and imf:
                export["intermolecular_forces"] = {
                    k: {
                        "present":  v.get("present"),
                        "strength": v.get("strength"),
                        "estimated_kJ_mol": v.get("estimated_kJ_mol"),
                        "note":     v.get("note"),
                    }
                    for k, v in imf.items() if k != "_dominant"
                }

            if include_bond_list and bonds_list:
                export["bond_table"] = [
                    {
                        "idx":    b["idx"],
                        "atom1":  f"{b['sym1']}#{b['atom1']}",
                        "atom2":  f"{b['sym2']}#{b['atom2']}",
                        "type":   b["label"],
                        "length_A": b["length"],
                        "energy_kJ_mol": b["energy"],
                    }
                    for b in bonds_list
                ]

            if export_format == "JSON":
                data_bytes = json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")
                fname = f"{mol_name or 'molecule'}_data.json"
                mime  = "application/json"
            else:
                # CSV: flatten
                rows = []
                for section, content in export.items():
                    if isinstance(content, dict):
                        for k, v in content.items():
                            if isinstance(v, dict):
                                for sk, sv in v.items():
                                    rows.append(f"{section},{k} / {sk},{sv}")
                            else:
                                rows.append(f"{section},{k},{v}")
                    elif isinstance(content, list):
                        for item in content:
                            row = ",".join(str(item.get(h, "")) for h in
                                          ["idx","atom1","atom2","type","length_A","energy_kJ_mol"])
                            rows.append(f"{section}," + row)
                    else:
                        rows.append(f"meta,{section},{content}")
                data_bytes = ("section,key,value\n" + "\n".join(rows)).encode("utf-8")
                fname = f"{mol_name or 'molecule'}_data.csv"
                mime  = "text/csv"

            st.download_button(
                label=f"Download {export_format}",
                data=data_bytes,
                file_name=fname,
                mime=mime,
                use_container_width=True,
            )

    # ── Image Download ────────────────────────────────────────────────────────
    with tab_img:
        st.caption(
            "Drag to rotate, scroll to zoom, then configure overlays and capture."
        )
        render_image_export(
            molblock  = molblock,
            style     = style,
            bg_color  = bg_color,
            props     = props,
            imf       = imf,
            mol_name  = mol_name or "molecule",
        )

    # ── Model Download ────────────────────────────────────────────────────────
    with tab_model:
        st.caption("Download the 3D structure in standard chemical file formats.")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.download_button(
                "SDF",
                data      = molblock.encode("utf-8"),
                file_name = f"{mol_name or 'molecule'}.sdf",
                mime      = "chemical/x-mdl-sdfile",
                use_container_width=True,
            )

        with m2:
            pdb_data = molblock_to_pdb(molblock)
            if pdb_data:
                st.download_button(
                    "PDB",
                    data      = pdb_data.encode("utf-8"),
                    file_name = f"{mol_name or 'molecule'}.pdb",
                    mime      = "chemical/x-pdb",
                    use_container_width=True,
                )
            else:
                st.button("PDB  (unavailable)", disabled=True, use_container_width=True)

        with m3:
            xyz_data = molblock_to_xyz(molblock)
            if xyz_data:
                st.download_button(
                    "XYZ",
                    data      = xyz_data.encode("utf-8"),
                    file_name = f"{mol_name or 'molecule'}.xyz",
                    mime      = "chemical/x-xyz",
                    use_container_width=True,
                )
            else:
                st.button("XYZ  (unavailable)", disabled=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

_SS_DEFAULTS: dict = {
    "m2_molblock":       None,
    "m2_smiles":         "",
    "m2_name":           "",
    "m2_formula":        "",
    "m2_e_before":       None,
    "m2_e_after":        None,
    "m2_copoly_rows":    [
        {"label": "Block 1", "smiles": "C=Cc1ccccc1", "n": 3},
        {"label": "Block 2", "smiles": "C=CC#N",      "n": 2},
    ],
    "m2_polymer_db":     None,
    "m2_use_pysoftk":    False,
    "m2_viewer_style":   "Ball-and-Stick",
    "m2_bg_color":       "#1a1a2e",
    "m2_spin":           False,
}


def _ss_init() -> None:
    for k, v in _SS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state["m2_polymer_db"] is None:
        st.session_state["m2_polymer_db"] = _init_polymer_db()


def _ss(k):
    return st.session_state[k]


def _ss_set(**kw) -> None:
    for k, v in kw.items():
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    _init_db()
    _ss_init()
    poly_db = _ss("m2_polymer_db")

    st.markdown(
        '<div class="module-header">M2  ·  Molecular Simulation Engine</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Intelligent input  ·  Homopolymer / Copolymer builder  ·  "
        "UFF geometry optimization  ·  Interactive 3D viewer with bond inspector  ·  "
        "Intermolecular forces analysis  ·  Data / Image / Model export"
    )

    if not RDKIT_OK:
        st.error("RDKit not installed.  Run: pip install rdkit")
        return

    # Tool availability badges
    badges = []
    if PYSOFTK_OK:  badges.append("PySoftK")
    if SMIPOLY_OK:  badges.append("SMiPoly")
    if PUBCHEM_OK:  badges.append("PubChem")
    if ASE_OK:      badges.append("ASE")
    if badges:
        st.caption("Available tools: " + "  |  ".join(badges))

    col_ctrl, col_main = st.columns([1, 2], gap="large")

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT — CONTROL PANEL
    # ══════════════════════════════════════════════════════════════════════════
    with col_ctrl:
        mode = st.radio(
            "Mode",
            ["Small Molecule", "Homopolymer", "Copolymer", "Polymer Builder"],
        )
        st.markdown("---")

        # ── Small Molecule ────────────────────────────────────────────────────
        if mode == "Small Molecule":
            st.markdown("##### Input")
            src = st.radio(
                "Source",
                ["Name Search", "SMILES", "Upload CIF / PDB"],
                label_visibility="collapsed",
            )

            if src == "Name Search":
                query = st.text_input(
                    "Molecule name",
                    placeholder="styrene / aspirin / 苯乙烯",
                )
                if st.button("Search", use_container_width=True) and query:
                    with st.spinner("Searching..."):
                        result = get_structure_logic(query, poly_db)
                    if result["source"] == "failed":
                        st.error(result["error"])
                    else:
                        src_labels = {
                            "smiles_direct": "Direct SMILES",
                            "polymer_db":    "Local Polymer DB",
                            "db_cache":      "SQLite Cache",
                            "qwen":          "Qwen3-32b LLM",
                            "pubchem":       "PubChem",
                        }
                        st.success(
                            f"Source: {src_labels.get(result['source'], result['source'])}  "
                            f"|  {result['english_name']}"
                        )
                        _ss_set(m2_smiles=result["smiles"],
                                m2_name=result["english_name"] or query,
                                m2_formula=result["formula"],
                                m2_e_before=None, m2_e_after=None)
                        with st.spinner("Generating 3D coordinates..."):
                            mb, err = smiles_to_3d(result["smiles"])
                        if err:
                            st.error(err)
                        else:
                            _ss_set(m2_molblock=mb)
                            st.rerun()

            elif src == "SMILES":
                smiles_in = st.text_input(
                    "SMILES",
                    value=_ss("m2_smiles") or "",
                    placeholder="CC(=O)Oc1ccccc1C(=O)O",
                )
                if st.button("Generate 3D", type="primary", use_container_width=True):
                    if not smiles_in.strip():
                        st.error("Please enter a SMILES string.")
                    else:
                        _ss_set(m2_smiles=smiles_in, m2_name=smiles_in[:32],
                                m2_formula="", m2_e_before=None, m2_e_after=None)
                        with st.spinner("Generating 3D coordinates..."):
                            mb, err = smiles_to_3d(smiles_in)
                        if err:
                            st.error(err)
                        else:
                            _ss_set(m2_molblock=mb)
                            st.rerun()

            else:  # Upload CIF/PDB
                st.caption("Catalyst / crystal structures: .cif  |  Protein / complex: .pdb")
                uploaded = st.file_uploader(
                    "Upload structure file",
                    type=["cif", "pdb"],
                )
                if uploaded and st.button("Parse File", type="primary",
                                          use_container_width=True):
                    ext = Path(uploaded.name).suffix
                    with st.spinner("Parsing..."):
                        mb, err = parse_structure_file(uploaded.read(), ext)
                    if err:
                        st.error(err)
                    else:
                        _ss_set(m2_molblock=mb, m2_smiles="",
                                m2_name=uploaded.name, m2_formula="",
                                m2_e_before=None, m2_e_after=None)
                        st.success(f"Parsed: {uploaded.name}")
                        st.rerun()

        # ── Homopolymer ───────────────────────────────────────────────────────
        elif mode == "Homopolymer":
            st.markdown("##### Monomer")
            poly_src = st.radio(
                "Source",
                ["Polymer Library", "Name Search", "SMILES"],
                label_visibility="collapsed",
            )
            mono_smiles = ""
            mono_label  = ""

            if poly_src == "Polymer Library":
                options = {
                    f"{k}  {v['zh']} / {v['en']}": v["monomer_smiles"]
                    for k, v in poly_db.items()
                }
                sel         = st.selectbox("Polymer", list(options.keys()))
                mono_smiles = options[sel]
                mono_label  = sel.split()[0]
                st.caption(f"Monomer SMILES: `{mono_smiles}`")

            elif poly_src == "Name Search":
                q = st.text_input("Name", placeholder="polystyrene / 聚苯乙烯")
                if st.button("Search Monomer") and q:
                    with st.spinner("Searching..."):
                        r = get_structure_logic(q, poly_db)
                    if r["source"] == "failed":
                        st.error(r["error"])
                    else:
                        mono_smiles = r["smiles"]
                        mono_label  = r["english_name"] or q
                        _ss_set(m2_smiles=mono_smiles)
                        st.success(f"SMILES: `{mono_smiles}`")
                mono_smiles = _ss("m2_smiles") or mono_smiles
            else:
                mono_smiles = st.text_input(
                    "Monomer SMILES", value="C=Cc1ccccc1"
                )
                mono_label = mono_smiles[:12]

            n_units = st.slider("Repeat units (n)", 2, 25, 5)
            use_psk = PYSOFTK_OK and st.checkbox("Use PySoftK builder", value=False)

            if st.button("Build Homopolymer", type="primary", use_container_width=True):
                if not mono_smiles.strip():
                    st.error("Please provide a monomer SMILES.")
                else:
                    with st.spinner(f"Building n={n_units} chain..."):
                        mb, err = _build_single_polymer(mono_smiles, n_units,
                                                        use_pysoftk=use_psk)
                    if err:
                        st.error(err)
                    else:
                        _ss_set(m2_molblock=mb, m2_smiles=mono_smiles,
                                m2_name=f"{mono_label}_n{n_units}",
                                m2_formula="", m2_e_before=None, m2_e_after=None)
                        st.success(f"Built (n={n_units})")
                        st.rerun()

        # ── Copolymer ─────────────────────────────────────────────────────────
        elif mode == "Copolymer":
            st.markdown("##### Copolymer Builder")
            st.caption("Block order: A(n1) — B(n2) — ...")
            rows: list[dict] = _ss("m2_copoly_rows")
            to_delete = None

            for i, row in enumerate(rows):
                st.markdown(f"**Block {i + 1}**")
                c1, c2 = st.columns([3, 1])
                row["smiles"] = c1.text_input(
                    "SMILES", value=row.get("smiles", ""),
                    key=f"cp_sm_{i}", label_visibility="collapsed",
                    placeholder="monomer SMILES",
                )
                row["n"] = int(c2.number_input(
                    "n", value=int(row.get("n", 3)),
                    min_value=1, max_value=20,
                    key=f"cp_n_{i}", label_visibility="collapsed",
                ))
                row["label"] = f"Block{i + 1}"
                if len(rows) > 1 and st.button(f"Remove block {i + 1}",
                                                key=f"cp_del_{i}"):
                    to_delete = i

            if to_delete is not None:
                rows.pop(to_delete)
                _ss_set(m2_copoly_rows=rows)
                st.rerun()

            if st.button("+ Add Block"):
                rows.append({"label": "", "smiles": "", "n": 3})
                _ss_set(m2_copoly_rows=rows)
                st.rerun()

            use_psk = PYSOFTK_OK and st.checkbox("Use PySoftK builder", value=False)
            st.markdown("---")
            if st.button("Build Copolymer", type="primary", use_container_width=True):
                with st.spinner("Building copolymer chain..."):
                    mb, err = build_copolymer(rows, use_pysoftk=use_psk)
                if err:
                    st.error(err)
                else:
                    tag = "+".join(
                        f"({r['smiles'][:6]})n{r['n']}"
                        for r in rows if r.get("smiles")
                    )
                    _ss_set(m2_molblock=mb, m2_smiles="",
                            m2_name=f"copoly_{tag}",
                            m2_formula="", m2_e_before=None, m2_e_after=None)
                    st.success("Copolymer built")
                    st.rerun()

        # ── Polymer Builder (advanced) ─────────────────────────────────────────
        else:
            st.markdown("##### Advanced Polymer Builder")
            st.caption(
                "Supports BigSMILES notation (SMiPoly), complex catalysts, "
                "and branched topologies via PySoftK."
            )

            adv_mode = st.radio(
                "Input type",
                ["BigSMILES / SMiPoly notation",
                 "SMILES with custom topology",
                 "Upload PDB / CIF (catalyst / complex)"],
                label_visibility="collapsed",
            )

            if adv_mode == "BigSMILES / SMiPoly notation":
                if not SMIPOLY_OK:
                    st.info(
                        "SMiPoly not installed. BigSMILES parsing uses basic fallback. "
                        "Install with: pip install smipoly"
                    )
                bigsmiles_in = st.text_area(
                    "BigSMILES string",
                    value="{[>][<]CC[>][<]}",
                    height=80,
                    help="Example: {[>][<]CC[>][<]}  for polyethylene repeating unit",
                )
                n_rep = st.number_input("Repeat units (n)", 2, 20, 5, key="adv_n")
                if st.button("Parse and Build", type="primary", use_container_width=True):
                    with st.spinner("Parsing BigSMILES..."):
                        core_smi, err = parse_bigsmiles(bigsmiles_in)
                    if err or not core_smi:
                        st.error(err or "Parse failed")
                    else:
                        st.success(f"Extracted SMILES: `{core_smi}`")
                        with st.spinner("Building polymer chain..."):
                            mb, err2 = _build_single_polymer(core_smi, int(n_rep),
                                                             use_pysoftk=PYSOFTK_OK)
                        if err2:
                            st.error(err2)
                        else:
                            _ss_set(m2_molblock=mb, m2_smiles=core_smi,
                                    m2_name=f"bigsmiles_n{n_rep}",
                                    m2_formula="", m2_e_before=None, m2_e_after=None)
                            st.success("Structure built")
                            st.rerun()

            elif adv_mode == "SMILES with custom topology":
                st.caption(
                    "For complex catalysts with metal centers, ligands, "
                    "or unusual connectivity."
                )
                smi_in = st.text_area(
                    "SMILES",
                    value="",
                    height=80,
                    placeholder="[Rh](Cl)(Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1",
                )
                if st.button("Generate 3D", type="primary", use_container_width=True):
                    if not smi_in.strip():
                        st.error("Please enter a SMILES string.")
                    else:
                        with st.spinner("Generating 3D..."):
                            mb, err = smiles_to_3d(smi_in)
                        if err:
                            st.error(err)
                        else:
                            _ss_set(m2_molblock=mb, m2_smiles=smi_in,
                                    m2_name=smi_in[:30], m2_formula="",
                                    m2_e_before=None, m2_e_after=None)
                            st.success("3D structure generated")
                            st.rerun()

            else:  # Upload PDB/CIF
                st.caption(
                    "Upload catalyst / complex structures from crystallography databases."
                )
                up2 = st.file_uploader("Upload .cif or .pdb", type=["cif", "pdb"])
                if up2 and st.button("Parse", type="primary", use_container_width=True):
                    ext = Path(up2.name).suffix
                    with st.spinner("Parsing..."):
                        mb, err = parse_structure_file(up2.read(), ext)
                    if err:
                        st.error(err)
                    else:
                        _ss_set(m2_molblock=mb, m2_smiles="",
                                m2_name=up2.name, m2_formula="",
                                m2_e_before=None, m2_e_after=None)
                        st.success(f"Parsed: {up2.name}")
                        st.rerun()

        # ── UFF Optimization ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### Geometry Optimization")
        st.caption("UFF Universal Force Field — 500-step energy minimization")

        if st.button("Run Optimization", use_container_width=True,
                     disabled=_ss("m2_molblock") is None):
            with st.spinner("UFF optimizing..."):
                mb_opt, e_b, e_a, err = uff_optimize(_ss("m2_molblock"))
            if err:
                st.error(err)
            else:
                _ss_set(m2_molblock=mb_opt, m2_e_before=e_b, m2_e_after=e_a)
                st.success("Optimization complete")

        eb, ea = _ss("m2_e_before"), _ss("m2_e_after")
        if eb is not None and ea is not None:
            c1, c2 = st.columns(2)
            c1.metric("Before (kcal/mol)", f"{eb:.2f}")
            c2.metric("After (kcal/mol)",  f"{ea:.2f}",
                      delta=f"{ea - eb:+.2f}", delta_color="inverse")

        # ── Render settings ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### Render Settings")
        vis_style = st.selectbox("Display style", list(_STYLE_JS.keys()),
                                 key="m2_vis_style_sel")
        bg_raw    = st.color_picker("Background", _ss("m2_bg_color"),
                                    key="m2_bg_picker")
        bg_hex    = "0x" + bg_raw[1:]
        auto_spin = st.toggle("Auto spin", value=_ss("m2_spin"), key="m2_spin_toggle")
        _ss_set(m2_viewer_style=vis_style, m2_bg_color=bg_raw, m2_spin=auto_spin)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT — VIEWER + PROPERTIES
    # ══════════════════════════════════════════════════════════════════════════
    with col_main:
        st.markdown("##### 3D Structure")
        st.caption(
            "Drag to rotate  |  Scroll to zoom at cursor  |  "
            "Click atom to inspect bonds and functional groups"
        )

        molblock = _ss("m2_molblock")
        smiles   = _ss("m2_smiles") or ""
        mol_name = _ss("m2_name")   or "molecule"

        if molblock:
            render_3d_viewer(molblock, vis_style, bg_hex, auto_spin, height=500)

            # ── Molecular Properties ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("##### Molecular Properties")
            props = get_mol_properties(smiles, molblock)
            if props:
                n_cols = 3
                cols   = st.columns(n_cols)
                for i, (label, val) in enumerate(props.items()):
                    cols[i % n_cols].metric(label, val)

            # ── Functional Groups ─────────────────────────────────────────────
            fg_by_atom = _build_fg_lookup(molblock)
            all_fgs: dict[str, dict] = {}
            for fgs in fg_by_atom.values():
                for fg in fgs:
                    all_fgs[fg["name"]] = fg
            if all_fgs:
                st.markdown("---")
                st.markdown("##### Detected Functional Groups")
                for fg_name, fg_info in all_fgs.items():
                    with st.expander(fg_name):
                        st.markdown(f"**Reactions:** {fg_info['reactions']}")
                        st.markdown(f"**Typical reagents:** {fg_info['reagents']}")
                        st.markdown(f"**Min equiv. to react:** {fg_info['min_equiv']}")

            # ── Intermolecular Forces ─────────────────────────────────────────
            imf = calc_intermolecular_forces(smiles, molblock)
            if imf:
                st.markdown("---")
                st.markdown("##### Intermolecular Forces")
                if "_dominant" in imf:
                    st.caption(f"Dominant force: {imf['_dominant']}")

                imf_cols = st.columns(len([k for k in imf if k != "_dominant"]))
                for col_i, (force_name, fdata) in enumerate(
                    (k, v) for k, v in imf.items() if k != "_dominant"
                ):
                    _strength_color = {
                        "Weak": "#f0c070", "Moderate": "#70b8f0",
                        "Strong": "#70e090", "None": "#707880",
                        "Present": "#70e090", "Absent": "#707880",
                    }
                    color = _strength_color.get(fdata["strength"], "#a0b0c0")
                    with imf_cols[col_i]:
                        st.markdown(f"**{force_name}**")
                        st.markdown(
                            f"<span style='color:{color}'>{fdata['strength']}</span>",
                            unsafe_allow_html=True,
                        )
                        if fdata["present"]:
                            st.caption(f"~{fdata['estimated_kJ_mol']} kJ/mol")
                        st.caption(fdata["note"])

            # ── Physical Properties Panel ─────────────────────────────────────
            st.markdown("---")
            st.markdown("##### Physical Properties")
            pcol1, pcol2, pcol3 = st.columns(3)

            with pcol1:
                st.markdown("**UFF Total Potential Energy**")
                energy = calc_uff_energy(molblock)
                if energy is not None:
                    st.metric("Energy (kcal/mol)", f"{energy:.4f}")
                    if eb is not None and ea is not None and eb != 0:
                        reduction = eb - ea
                        st.caption(
                            f"Reduction: {reduction:.2f} kcal/mol "
                            f"({reduction / eb * 100:.1f}%)"
                        )
                else:
                    st.caption("UFF does not support some atom types in this molecule.")

            with pcol2:
                st.markdown("**Radius of Gyration**")
                rg = calc_radius_of_gyration(molblock)
                if rg is not None:
                    st.metric("Rg (Angstrom)", f"{rg:.3f}")
                    size_hint = (
                        "Small molecule" if rg < 3
                        else "Oligomer"       if rg < 8
                        else "Medium chain"   if rg < 20
                        else "Long-chain polymer"
                    )
                    st.caption(size_hint)
                else:
                    st.caption("Requires numpy")

            with pcol3:
                st.markdown("**MW Distribution**")
                if mode == "Copolymer":
                    mw_input = _ss("m2_copoly_rows")
                elif mode in ("Homopolymer", "Polymer Builder") and smiles:
                    name_part = mol_name or ""
                    try:
                        n_part = int(name_part.rsplit("n", 1)[-1]) if "_n" in name_part else 5
                    except ValueError:
                        n_part = 5
                    mw_input = [{"smiles": smiles, "n": n_part, "label": "monomer"}]
                else:
                    mw_input = [{"smiles": smiles or "", "n": 1, "label": "molecule"}]

                mw_data = calc_mw_distribution(mw_input)
                if mw_data.get("total_mw", 0) > 0:
                    st.metric("Chain Mn (g/mol)", f"{mw_data['total_mw']:.1f}")
                    st.caption(f"PDI = {mw_data['pdi']:.1f}  (single deterministic chain)")
                    if len(mw_data["rows"]) > 1:
                        for row in mw_data["rows"]:
                            frac = row["segment_mw"] / mw_data["total_mw"] * 100
                            st.progress(
                                int(frac),
                                text=f"{row['label']}  {row['segment_mw']:.1f} g/mol  ({frac:.1f}%)",
                            )
                else:
                    st.caption("Cannot calculate MW")

            # ── Save ──────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("##### Save to Database")
            save_name = st.text_input(
                "Record name",
                value=mol_name or "molecule",
                key="m2_save_name_input",
            )
            if st.button("Save to local DB", use_container_width=True):
                sdf_path = _save_sdf(save_name, molblock)
                _db_save(
                    name      = save_name,
                    smiles    = smiles,
                    energy    = _ss("m2_e_after"),
                    formula   = _ss("m2_formula"),
                    file_path = str(sdf_path),
                )
                if _UTILS_OK:
                    try:
                        _legacy_save(
                            module="M2", name=save_name,
                            file_path=str(sdf_path),
                            metadata={"smiles": smiles, "energy": _ss("m2_e_after")},
                        )
                    except Exception:
                        pass
                st.success(f"Saved: `{sdf_path.name}`")

            # ── Download Module ───────────────────────────────────────────────
            st.markdown("---")
            bonds_list, _ = extract_bond_data(molblock)
            imf_for_dl    = imf if imf else {}
            render_download_module(
                molblock   = molblock,
                mol_name   = mol_name,
                smiles     = smiles,
                props      = props,
                imf        = imf_for_dl,
                bonds_list = bonds_list,
                style      = vis_style,
                bg_color   = bg_hex,
            )

        else:
            st.info(
                "Enter a molecule name, SMILES, or upload a structure file on the left, "
                "then generate a 3D model."
            )
