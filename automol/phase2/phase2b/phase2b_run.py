import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from SMILESLigandPipeline import SMILESLigandPipeline

import tenacity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the SMILESLigandPipeline
smiles_ligand_pipeline = SMILESLigandPipeline()

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=2, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
async def run_Phase_2b(
    technical_descriptions: List[str],
    predicted_structures_dir: str,
    results_dir: str,
    num_sequences: int,
    optimization_steps: int,
    score_threshold: float
) -> List[Dict[str, Any]]:
    try:
        print("Running Phase 2b")
        return await smiles_ligand_pipeline.run_smiles_ligand_pipeline(
            technical_descriptions,
            predicted_structures_dir,
            results_dir,
            num_sequences,
            optimization_steps,
            score_threshold
        )
    except Exception as e:
        logger.error(f"An error occurred in run_Phase_2b: {e}")
        raise e

async def main():
    technical_descriptions = [
        "Design a small molecule that acts as a selective agonist for tissue-specific telomerase-associated proteins, such as those found in skin stem cells, to promote wound healing and reduce skin aging.",
    ]
    predicted_structures_dir = "predicted_structures"
    results_dir = "results"
    num_sequences = 2
    optimization_steps = 100
    score_threshold = 0.8

    # Ensure output directories exist
    Path(predicted_structures_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    try:
        ligand_sequences = await run_Phase_2b(
            technical_descriptions,
            predicted_structures_dir,
            results_dir,
            num_sequences,
            optimization_steps,
            score_threshold
        )
        print(f"Generated ligand sequences: {ligand_sequences}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())