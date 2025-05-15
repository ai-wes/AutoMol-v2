
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import os


import asyncio
from esm.sdk.api import ESMProteinError
import os
import asyncio
import logging
from Bio.PDB import PDBParser, DSSP
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import os
import asyncio
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import truncnorm
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm3_model:ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")  # or "cpu"


import os
import logging
from typing import List, Union, Dict, Any
from esm.sdk.api import ESMProteinError
from esm.sdk.api import ESMProtein, GenerationConfig
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)

# Ensure esm3_model and ESM3InferenceClient are globally loaded
from esm.models.esm3 import ESM3
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm3_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)

valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

def predict_protein_function(sequence: str) -> float:
    try:
        sequence = ''.join(char.upper() for char in sequence if char.upper() in valid_aa)
        if not sequence:
            logger.warning("Invalid or empty amino acid sequence.")
            return 0.5

        analysis = ProteinAnalysis(sequence)
        mw = analysis.molecular_weight()
        arom = analysis.aromaticity()
        instab = analysis.instability_index()
        pI = analysis.isoelectric_point()

        norm_weight = truncnorm.cdf((mw - 25000) / 10000, -2, 2)
        norm_instability = 1 - truncnorm.cdf((instab - 40) / 10, -2, 2)
        norm_isoelectric = truncnorm.cdf((pI - 7) / 2, -2, 2)

        aa_count = {aa: sequence.count(aa) for aa in valid_aa}
        total = len(sequence)
        comp_balance = 1 - sum(abs(count / total - 0.05) for count in aa_count.values()) / 2

        weights = [0.25, 0.15, 0.25, 0.15, 0.2]
        score = sum(w * v for w, v in zip(weights, [norm_weight, arom, norm_instability, norm_isoelectric, comp_balance]))

        score = max(0, min(1, score))
        logger.info(f"Predicted function score: {score:.3f}")
        return score

    except Exception as e:
        logger.error(f"Error in predict_protein_function: {e}")
        return 0.5

def predict_properties(sequence: str) -> Dict[str, Any]:
    try:
        analysis = ProteinAnalysis(sequence)
        return {
            "molecular_weight": analysis.molecular_weight(),
            "aromaticity": analysis.aromaticity(),
            "instability_index": analysis.instability_index(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy": analysis.gravy(),
            "secondary_structure_fraction": analysis.secondary_structure_fraction()
        }
    except Exception as e:
        logger.error(f"Error in predict_properties: {e}")
        return {}

def predict_structure(sequence: str, output_dir: str) -> Union[str, None]:
    try:
        protein = ESMProtein(sequence=sequence)
        config = GenerationConfig(track="structure", num_steps=8)
        result = esm3_model.generate(protein, config)

        os.makedirs(output_dir, exist_ok=True)
        pdb_path = os.path.join(output_dir, f"{sequence[:10]}_structure.pdb")

        if hasattr(result, "to_pdb"):
            result.to_pdb(pdb_path)
            logger.info(f"Structure saved to: {pdb_path}")
            return pdb_path
        else:
            logger.error("Generated result does not support to_pdb().")
            return None
    except ESMProteinError as e:
        logger.error(f"ESMProteinError in predict_structure: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in predict_structure: {e}")
        return None




def esm3_refinement(sequence):
    try:
        protein = ESMProtein(sequence=sequence)
        refined_protein = esm3_model.generate(
            protein, 
            GenerationConfig(track="sequence", num_steps=8, temperature=0.7)
        )
        
        if isinstance(refined_protein, (ESMProtein, str)):
            refined_sequence = (
                refined_protein.sequence 
                if isinstance(refined_protein, ESMProtein) 
                else refined_protein
            )
        else:
            logger.warning(f"Unexpected refined protein type: {type(refined_protein)}. Using original sequence.")
            refined_sequence = sequence
        
        logger.info(f"Refined sequence: {refined_sequence[:50]}...")
        return refined_sequence
    except ESMProteinError as e:
        logger.warning(f"ESM Protein Error in ESM3 refinement: {str(e)}. Using original sequence.")
        return sequence
    except Exception as e:
        logger.error(f"Unexpected error in ESM3 refinement: {str(e)}. Using original sequence.")
        return sequence
    
        




def run_prediction_pipeline(sequences: List[Union[str, Dict[str, Any]]], output_dir: str) -> List[Dict[str, Any]]:
    results = []

    for i, entry in enumerate(sequences):
        try:
            sequence = entry["optimized_sequence"] if isinstance(entry, dict) else entry
            logger.info(f"Predicting sequence {i + 1}/{len(sequences)}")

            score = predict_protein_function(sequence)
            properties = predict_properties(sequence)
            pdb_file = predict_structure(sequence, output_dir)

            result = {
                "sequence": sequence,
                "score": score,
                "properties": properties,
                "pdb_file": pdb_file if pdb_file and os.path.exists(pdb_file) else None
            }

            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process sequence index {i}: {e}")
            results.append({
                "sequence": sequence if 'sequence' in locals() else "unknown",
                "score": None,
                "properties": {},
                "pdb_file": None
            })

    return results
