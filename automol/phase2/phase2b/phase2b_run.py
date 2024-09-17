# optimize_ligand.py

# Necessary imports (rdkit, torch, asyncio, etc.)
import asyncio
import logging
import os
from pathlib import Path
from .SMILESLigandPipeline import SMILESGenerator, SMILESOptimizer, StructurePredictor, EnsembleDocking, DockingAnalyzer, LigandScorer, validate_smiles
import torch
import json
from typing import List, Dict, Any
from utils.shared_state import get_protein_sequences


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tenacity


class SMILESLigandPipeline:
    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.ensemble_docker = EnsembleDocking()
        self.analyzer = DockingAnalyzer()
        self.scorer = LigandScorer()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def run_Phase_2b(self, technical_instruction: str, output_dir: str) -> Dict[str, Any]:
        try:
            os.makedirs(output_dir, exist_ok=True)

            # SMILES generation and optimization
            logger.info("Starting SMILES Generation")
            smiles = await self.generator.generate(technical_instruction)
            logger.info(f"SMILES Generation completed: {smiles}")

            logger.info("Starting SMILES Optimization")
            optimized_smiles = await self.optimizer.optimize(smiles)
            logger.info(f"SMILES Optimization completed: {optimized_smiles}")

            if not validate_smiles(optimized_smiles):
                raise ValueError("Invalid SMILES string after optimization")

            logger.info("Starting 3D Structure Prediction")
            ligand_pdb = await self.predictor.predict(optimized_smiles, output_dir)
            logger.info(f"3D Structure Prediction completed: {ligand_pdb}")

            # Retrieve protein sequences from shared state
            protein_sequences = await get_protein_sequences()
            logger.info(f"Retrieved {len(protein_sequences)} protein sequences from shared state")

            # Generate protein structures (this step might be part of the protein pipeline)
            protein_ensemble = await self.generate_protein_structures(protein_sequences, output_dir)

            # Continue with ensemble docking using the generated protein structures
            logger.info("Starting Ensemble Docking")
            docking_results = await self.ensemble_docker.dock_ensemble(ligand_pdb, protein_ensemble, output_dir)
            logger.info("Ensemble Docking completed")


            logger.info("Starting Docked Ligands Analysis")
            analysis_tasks = [
                self.analyzer.analyze(docking_result['docked_ligand'], protein_ensemble[docking_result['index']], output_dir)
                for docking_result in docking_results if 'docked_ligand' in docking_result
            ]
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Filter out failed analyses
            valid_analysis = []
            for result in analysis_results:
                if isinstance(result, dict):
                    valid_analysis.append(result)
                else:
                    logger.warning(f"Analysis task failed: {result}")

            logger.info("Docked Ligands Analysis completed")

            logger.info("Starting Ligand Scoring")
            best_ligand = await self.scorer.score(valid_analysis, docking_results)
            logger.info(f"Ligand Scoring completed: {best_ligand}")

            logger.info("Pipeline completed successfully")
            return best_ligand
        except Exception as e:
            logger.error(f"An error occurred in the pipeline: {e}")
            raise e


    async def generate_protein_structures(self, protein_sequences: List[str], output_dir: str) -> List[str]:
        # This method should generate protein structures from sequences
        # For now, we'll just return dummy PDB paths
        return [os.path.join(output_dir, f"protein_{i}.pdb") for i in range(len(protein_sequences))]



