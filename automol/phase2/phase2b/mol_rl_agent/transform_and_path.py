import os
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
from rdkit import DataStructs
import sascorer  # Ensure sascorer is installed
from docking_scorer import predict_docking_score  # Ensure this module is available
import csv
import os
from datetime import datetime

RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings


fragments = [
    'C',  # methyl group
    'O',  # oxygen atom
    'N',  # nitrogen atom
    'S',  # sulfur atom
    'Cl',  # chlorine atom
    'F',  # fluorine atom
    'Br',  # bromine atom
    'I',  # iodine atom
    'C=O',  # carbonyl group
    'C#N',  # nitrile group
    'C=C',  # alkene group
    'C#C',  # alkyne group
    'C1CC1',  # cyclopropane ring
    'C1CCC1',  # cyclobutane ring
    'C1CCCC1',  # cyclopentane ring
    'C1CCCCC1',  # cyclohexane ring
    'c1ccccc1',  # benzene ring
    'C(=O)O',  # carboxylic acid group
    'C(=O)N',  # amide group
    'C1=CC=CC=C1',  # another benzene ring
    # More fragments can be added here
]

# Define Aging pathways
AGING_PATHWAYS = [
    "Autophagy Induction",
    "Epigenetic Modulation",
    "Mitochondrial Function Enhancement",
    "Extracellular Matrix Modulation",
    "Stem Cell Niche Enhancement",
    "Senomorphic Effects",
    "Exosome Modulation",
    "Cellular Reprogramming",
    "Telomere Protection",
    "Cellular Senescence Pathway Modulation",
    "mTOR Inhibition",
    "Sirtuin Activation",
    "Senolytic Activity",
    "DNA Repair Enhancement",
    "Proteostasis Enhancement",
    "Circadian Rhythm Regulation",
    "Hormesis Induction",
    "Reprogramming Factor Mimetics",
    "NAD+ Metabolism Modulation",
    "Inflammaging Reduction",
    "Oxidative Stress Mitigation",
    "Genomic Instability Prevention",
    "Cellular Energy Metabolism Optimization",
    "Protein Aggregation Inhibition",
    "Stem Cell Exhaustion Prevention",
    "Intercellular Communication Enhancement",
    "Lipid Metabolism Regulation",
    "Cellular Plasticity Promotion",
    "Epigenetic Clock Modulation",
    "Mitochondrial Dynamics Regulation",
    "Cellular Senescence-Associated Secretory Phenotype (SASP) Modulation",
    "Autophagy-Lysosomal Pathway Enhancement",
    "Proteolysis Modulation",
    "Metabolic Flexibility Enhancement",
    "Cellular Stress Response Optimization"
]

# Define the known active molecules for each pathway

# Define the known active molecules for each pathway
PATHWAY_ACTIVES = {
    "Autophagy Induction": [
        "CC1=CC=C(C=C1)NC(=O)C",
        "COC1=C(C=C(C=C1)NC(=O)C)OC",
        "CC(=O)NC1=CC=C(C=C1)O",
        "CC1=C(C(=CC=C1)C)NC(=O)CC2=CC=CC=C2",
        "COC1=CC(=CC(=C1OC)OC)C(=O)NC2=CC=CC=C2",
        "CC1=CC=C(C=C1)NC(=O)CCCN2CCC(CC2)N3C(=O)NC4=CC=CC=C43",
        "COC1=C(C=C(C=C1)NC(=O)C2CC3=C(NC2=O)C=CC(=C3)Cl)OCCCN4CCOCC4",
    ],
    "Epigenetic Modulation": [
        "CC(=O)NC1=CC=CC=C1",
        "CC1=C(C=C(C=C1)S(=O)(=O)NC2=CC=CC=C2)C",
        "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
        "CC1=C(C(=O)NC(=O)N1)C2=CC=CC=C2",
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3",
        "CC1=C(NC(=O)C2=C1C=CC(=C2)C#N)C3=CC=C(C=C3)NC(=O)C4=CC=CC=C4",
    ],
    "Mitochondrial Function Enhancement": [
        "C1=CC(=CC=C1C(=O)O)O",
        "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=O)O)C",
        "CC1=C(C(=O)C2=C(C1=O)C(=CC=C2)O)O",
        "CC1=CC=C(C=C1)OCC(C)(C)NC2=NC=NC3=C2C=CN3",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCOCC3)OC",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Extracellular Matrix Modulation": [
        "CC(=O)Nc1ccc(cc1)C(=O)O",
        "C1=CC=CC=C1C(=O)O",
        "CCN(CC)C(=O)C1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)N",
        "CCOCC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Stem Cell Niche Enhancement": [
        "CC1=C(C(=O)N2CCCC2=N1)C3=CC=C(C=C3)Cl",
        "CC1=NC(=C(N1CC2=CC=C(C=C2)C(=O)O)C(=O)NC3=CC=CC=C3)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Senomorphic Effects": [
        "CC1=C(C=C(C=C1)C(=O)NC2=CC=CC=C2)NC(=O)OC3CCCC3",
        "CC1=CC=C(C=C1)C2=CC(=NO2)C(=O)NC3=CC=CC=C3C(F)(F)F",
        "COC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C3=CN=C(N=C3N)N",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
    ],
    "Exosome Modulation": [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)N1CCC(CC1)C(=O)O",
        "C1=CC=CC=C1NC(=O)O",
        "CC(=O)N(C)C1=CC=CC=C1",
        "CCC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Cellular Reprogramming": [
        "CC(=O)Nc1ccc(cc1)N",
        "CC1=CC=CC=C1NC(=O)O",
        "CCC(=O)Nc1ccc(cc1)C(=O)O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)C#N",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Telomere Protection": [
        "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C2=CC=CC=C2N1)C(=O)NC3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
    ],
    "Cellular Senescence Pathway Modulation": [
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
        "C1=CC=C(C=C1)C2=C(C(=O)OC3=CC=CC=C23)O",
        "CCC(=O)Nc1ccccc1O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "mTOR Inhibition": [
        "CC1=CC=C(C=C1)C2=CN=C(N=C2N)N",
        "CCC(=O)Nc1ccc(cc1)O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)C#N",
        "CC(=O)Nc1ccccc1O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Sirtuin Activation": [
        "CC1=CC(=CC=C1O)C=CC(=O)C2=CC=C(C=C2)O",
        "C1=CC=C(C=C1O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O",
        "CC(=O)Nc1ccc(cc1)O",
        "CC(=O)NCC1=CC=CC=C1",
        "CCC(=O)Nc1ccccc1O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Senolytic Activity": [
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
        "C1=CC=C(C=C1)C2=C(C(=O)OC3=CC=CC=C23)O",
        "CCC(=O)Nc1ccc(cc1)O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "DNA Repair Enhancement": [
        "C1=NC2=C(N1)C(=O)NC(=O)N2",
        "C1=NC(=O)NC(=O)C1",
        "CC(=O)Nc1ccc(cc1)O",
        "CCC(=O)Nc1ccccc1O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Proteostasis Enhancement": [
        "CC1=C(C=C(C=C1)C(=O)NC2=CC=CC=C2)C",
        "COC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C3=CN=C(N=C3N)N",
        "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Circadian Rhythm Regulation": [
        "C1=CC2=C(C=C1)NC(=O)C=C2",
        "C1=CC=C2C(=C1)C(=O)CC(=O)N2",
        "CCC(=O)Nc1ccccc1O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Hormesis Induction": [
        "C1=CC=C(C=C1)O",
        "C1=CC(=CC=C1O)O",
        "CC(=O)Nc1ccccc1O",
        "CCC(=O)Nc1ccc(cc1)O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Reprogramming Factor Mimetics": [
        "C1=NC2=C(N1)C(=O)NC(=O)N2",
        "C1=NC(=O)NC(=O)C1",
        "CCC(=O)Nc1ccccc1O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "NAD+ Metabolism Modulation": [
        "C1=NC(=C2C(=N1)C(=O)NC(=O)N2)N",
        "C1=NC2=C(N1)C(=O)NC(=O)N2",
        "CC(=O)NC1=CC=C(C=C1)O",
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Inflammaging Reduction": [
        "CC1=C(C=C(C=C1)C(=O)O)O",
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
    ],
}

# Define transformations and ensure they are correct
TRANSFORMATIONS = [
    ('Add Hydroxyl', '[C:1][H]>>[C:1][OH]'),
    ('Add Amine', '[C:1][H]>>[C:1][NH2]'),
    ('Add Methyl', '[C:1][H]>>[C:1][CH3]'),
    ('Add Carboxyl', '[C:1][H]>>[C:1][C](=O)[OH]'),
    ('Replace OH with NH2', '[C:1][OH]>>[C:1][NH2]'),
    ('Replace CH3 with OH', '[C:1][CH3]>>[C:1][OH]'),
    ('Add Nitro', '[C:1][H]>>[C:1][N+](=O)[O-]'),
    ('Add Sulfhydryl', '[C:1][H]>>[C:1][SH]'),
    ('Add Ether', '[C:1][H]>>[C:1][O][C:2]'),
    ('Form Ring', '[C:1]-[C:2]>>[C:1]1-[C:2]-[C]-[C]-[C]-[C]-1'),
    ('Break Ring', '[C:1]1-[C:2]-[C]-[C]-[C]-[C]-1>>[C:1]-[C:2]'),
    ('Add Benzene Ring', '[C:1][H]>>[C:1]C1=CC=CC=C1'),
    ('Modify Nitrogen', '[N:1]=[C:2]>>[N:1][C:2]'),
    ('Add Halogen', '[C:1][H]>>[C:1][Cl]'),
    ('Add Phenol', '[c:1][H]>>[c:1][OH]'),
    ('Add Thiol', '[C:1][H]>>[C:1][SH]'),
    ('Add Fluorine', '[C:1][H]>>[C:1][F]'),
    ('Add Bromine', '[C:1][H]>>[C:1][Br]'),
    ('Add Iodine', '[C:1][H]>>[C:1][I]'),
    ('Add Aldehyde', '[C:1][H]>>[C:1][C](=O)[H]'),
    ('Add Ketone', '[C:1][H]>>[C:1][C](=O)[C:2]'),
    ('Add Amide', '[C:1][H]>>[C:1][C](=O)[N:2]'),
    ('Add Ester', '[C:1][H]>>[C:1][C](=O)[O][C:2]'),
    ('Add Sulfone', '[C:1][H]>>[C:1][S](=O)(=O)[C:2]'),
    ('Add Sulfonamide', '[C:1][H]>>[C:1][S](=O)(=O)[N:2]'),
    ('Add Phosphate', '[C:1][H]>>[C:1][P](=O)([O-])[O-]'),
    ('Add Trifluoromethyl', '[C:1][H]>>[C:1][C](F)(F)F'),
('Add Isopropyl', '[C:1][H]>>[C:1][CH](C)'),
('Add tert-Butyl', '[C:1][H]>>[C:1]C(C)C'),]




INITIAL_PATHWAYS = [
            'C1=CC=CC=C1',        # Benzene
            'CCO',                # Ethanol
            'CCN',                # Ethylamine
            'CCC',                # Propane
            'CC(=O)O',            # Acetic acid
            'CC(=O)N',            # Acetamide
            'CC#N',               # Acetonitrile
            'C1CCCCC1',           # Cyclohexane
            'C1=CC=CC=C1O',       # Phenol
            'CC(C)O',             # Isopropanol

            "CC1=CC=C(C=C1)NC(=O)C",
            "COC1=C(C=C(C=C1)NC(=O)C)OC",
            "CC(=O)NC1=CC=C(C=C1)O",
            "CC1=C(C(=CC=C1)C)NC(=O)CC2=CC=CC=C2",
            "COC1=CC(=CC(=C1OC)OC)C(=O)NC2=CC=CC=C2",
            "CC1=CC=C(C=C1)NC(=O)CCCN2CCC(CC2)N3C(=O)NC4=CC=CC=C43",
            "COC1=C(C=C(C=C1)NC(=O)C2CC3=C(NC2=O)C=CC(=C3)Cl)OCCCN4CCOCC4",
            "CC(=O)NC1=CC=CC=C1",
            "CC1=C(C=C(C=C1)S(=O)(=O)NC2=CC=CC=C2)C",
            "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
            "CC1=C(C(=O)NC(=O)N1)C2=CC=CC=C2",
            "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3",
            "CC1=C(NC(=O)C2=C1C=CC(=C2)C#N)C3=CC=C(C=C3)NC(=O)C4=CC=CC=C4",
            "C1=CC(=CC=C1C(=O)O)O",
            "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=O)O)C",
            "CC1=C(C(=O)C2=C(C1=O)C(=CC=C2)O)O",
            "CC1=CC=C(C=C1)OCC(C)(C)NC2=NC=NC3=C2C=CN3",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCOCC3)OC",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC(=O)Nc1ccc(cc1)C(=O)O",
            "C1=CC=CC=C1C(=O)O",
            "CCN(CC)C(=O)C1=CC=CC=C1",
            "CC(=O)Nc1ccc(cc1)N",
            "CCOCC(=O)Nc1ccc(cc1)O",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC1=C(C(=O)N2CCCC2=N1)C3=CC=C(C=C3)Cl",
            "CC1=NC(=C(N1CC2=CC=C(C=C2)C(=O)O)C(=O)NC3=CC=CC=C3)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
            "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "C1=NC2=C(N1)C(=O)NC(=O)N2",
            "C1=NC(=O)NC(=O)C1",
            "CCC(=O)Nc1ccccc1O",
            "CC(=O)NCC1=CC=CC=C1",
            "CC(=O)Nc1ccc(cc1)O",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "C1=NC(=C2C(=N1)C(=O)NC(=O)N2)N",
            "C1=NC2=C(N1)C(=O)NC(=O)N2",
            "CC(=O)NC1=CC=C(C=C1)O",
            "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC"
        ]



REACTIONS = []
for name, smarts in TRANSFORMATIONS:
    try:
        reaction = AllChem.ReactionFromSmarts(smarts)
        REACTIONS.append((name, reaction))
    except ValueError as e:
        print(f"Warning: Could not create reaction for '{name}': {e}")
        
        
def create_similarity_scoring_function(pathway):
    # Precompute fingerprints for known actives
    active_smiles = PATHWAY_ACTIVES.get(pathway, [])
    active_mols = [Chem.MolFromSmiles(smiles) for smiles in active_smiles]
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in active_mols if mol is not None]

    def score_function(mol):
        if not active_fps:
            return None  # Return None instead of 0
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        similarities = [DataStructs.TanimotoSimilarity(mol_fp, active_fp) for active_fp in active_fps]
        max_similarity = max(similarities)
        return max_similarity  # Return the maximum similarity score

    return score_function

# Initialize similarity scoring functions for all pathways
PATHWAY_SCORING_FUNCTIONS = {
    pathway: create_similarity_scoring_function(pathway)
    for pathway in AGING_PATHWAYS
}