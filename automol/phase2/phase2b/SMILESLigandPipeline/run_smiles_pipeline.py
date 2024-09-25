import os
import random
import logging
import subprocess
from colorama import Fore
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import Crippen

from generate_smiles import SMILESGenerator
from optimize_smiles import SMILESOptimizer
from predict_smiles import StructurePredictor
from validate_smiles import AdditionalLigandOptimizer
from smiles_mcts import MCTSNode  # Assuming MCTS is part of the pipeline

logger = logging.getLogger(__name__)

def checkpoint(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting step: {step_name}")
            print(Fore.CYAN + f"Starting step: {step_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed step: {step_name}")
                print(Fore.GREEN + f"Completed step: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Error in step {step_name}: {str(e)}")
                print(Fore.RED + f"Error in step {step_name}: {str(e)}")
                raise
        return wrapper
    return decorator

class EnsembleDocking:

    def __init__(self, vina_executable: str = "vina"):
        self.vina_executable = vina_executable

    @checkpoint("Ensemble Docking")
    def dock_ensemble(self, ligand_pdb: str, protein_ensemble: List[str], results_dir: str) -> List[Dict[str, Any]]:
        docking_results = []
        for protein_pdb in protein_ensemble:
            try:
                # Prepare output files
                protein_name = os.path.splitext(os.path.basename(protein_pdb))[0]
                output_pdbqt = os.path.join(results_dir, f"{protein_name}_docked.pdbqt")
                log_file = os.path.join(results_dir, f"{protein_name}_docking.log")

                # Convert PDB to PDBQT using prepare_ligand4.py and prepare_receptor4.py from AutoDockTools
                receptor_pdbqt = self.convert_pdb_to_pdbqt(protein_pdb, is_receptor=True, results_dir=results_dir)
                ligand_pdbqt = self.convert_pdb_to_pdbqt(ligand_pdb, is_receptor=False, results_dir=results_dir)

                # Define docking parameters
                center_x, center_y, center_z = self.get_grid_center(receptor_pdbqt)
                size_x, size_y, size_z = self.get_grid_size(receptor_pdbqt)

                # Run Vina docking
                vina_command = [
                    self.vina_executable,
                    "--receptor", receptor_pdbqt,
                    "--ligand", ligand_pdbqt,
                    "--center_x", str(center_x),
                    "--center_y", str(center_y),
                    "--center_z", str(center_z),
                    "--size_x", str(size_x),
                    "--size_y", str(size_y),
                    "--size_z", str(size_z),
                    "--out", output_pdbqt,
                    "--log", log_file,
                    "--cpu", "4",  # Adjust based on your CPU
                    "--exhaustiveness", "8",  # Adjust for thoroughness
                ]

                logger.info(f"Running Vina docking for {protein_pdb}...")
                print(Fore.BLUE + f"Running Vina docking for {protein_pdb}...")
                subprocess.run(vina_command, check=True)

                # Parse docking score from log file
                docking_score = self.parse_vina_log(log_file)

                docking_result = {
                    "protein_pdb": protein_pdb,
                    "ligand_pdbqt": output_pdbqt,
                    "score": docking_score
                }
                docking_results.append(docking_result)
                logger.info(f"Docked {ligand_pdb} with {protein_pdb}, score: {docking_score}")
                print(Fore.BLUE + f"Docked {ligand_pdb} with {protein_pdb}, score: {docking_score}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Vina docking failed for {protein_pdb}: {e}")
                print(Fore.RED + f"Vina docking failed for {protein_pdb}: {e}")
            except Exception as e:
                logger.error(f"Error during docking with {protein_pdb}: {e}")
                print(Fore.RED + f"Error during docking with {protein_pdb}: {e}")

        return docking_results

    def convert_pdb_to_pdbqt(self, pdb_file: str, is_receptor: bool, results_dir: str) -> str:

        pdbname = os.path.splitext(os.path.basename(pdb_file))[0]
        pdbqt_file = os.path.join(results_dir, f"{pdbname}.pdbqt")

        prepare_script = "prepare_receptor4.py" if is_receptor else "prepare_ligand4.py"
        prepare_script_path = os.path.join(os.getenv("AUTODOCK_TOOLS_PATH", ""), prepare_script)  # Ensure AUTODOCK_TOOLS_PATH is set

        if not os.path.isfile(prepare_script_path):
            raise FileNotFoundError(f"{prepare_script} not found at {prepare_script_path}. Please set AUTODOCK_TOOLS_PATH environment variable correctly.")

        command = [
            "python",
            prepare_script_path,
            "-r", pdb_file,
            "-o", pdbqt_file
        ]

        logger.info(f"Converting {pdb_file} to PDBQT...")
        print(Fore.BLUE + f"Converting {pdb_file} to PDBQT...")
        subprocess.run(command, check=True)

        return pdbqt_file

    def get_grid_center(self, receptor_pdbqt: str) -> List[float]:

        # Parsing PDBQT to find the centroid
        mol = Chem.MolFromPDBFile(receptor_pdbqt, removeHs=False)
        if mol is None:
            raise ValueError(f"Could not parse PDBQT file: {receptor_pdbqt}")

        conf = mol.GetConformer()
        positions = conf.GetPositions()
        centroid = positions.mean(axis=0)
        return centroid.tolist()

    def get_grid_size(self, receptor_pdbqt: str, buffer: float = 10.0) -> List[float]:

        mol = Chem.MolFromPDBFile(receptor_pdbqt, removeHs=False)
        if mol is None:
            raise ValueError(f"Could not parse PDBQT file: {receptor_pdbqt}")

        conf = mol.GetConformer()
        positions = conf.GetPositions()
        min_coords = positions.min(axis=0) - buffer
        max_coords = positions.max(axis=0) + buffer
        size = (max_coords - min_coords).tolist()
        return size

    def parse_vina_log(self, log_file: str) -> float:
        best_score = None
        with open(log_file, 'r') as file:
            for line in file:
                if "REMARK VINA RESULT" in line:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        best_score = float(parts[3])
                        break
        if best_score is None:
            raise ValueError(f"Could not parse docking score from log file: {log_file}")
        return best_score

class Analyzer:

    def __init__(self):
        pass

    @checkpoint("Docking Analysis")
    def analyze(self, docking_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not docking_results:
            logger.warning("No docking results to analyze.")
            return {}

        # Find the best docking result
        best_result = min(docking_results, key=lambda x: x['score'])
        analysis = {
            "best_protein_pdb": best_result['protein_pdb'],
            "best_score": best_result['score']
        }
        logger.info(f"Best docking score: {analysis['best_score']} with {analysis['best_protein_pdb']}")
        print(Fore.GREEN + f"Best docking score: {analysis['best_score']} with {analysis['best_protein_pdb']}")
        return analysis

class Scorer:

    def __init__(self):
        pass

    @checkpoint("Scoring Ligand")
    def calculate_docking_score(self, docking_results: List[Dict[str, Any]], protein_ensemble: List[str]) -> float:

        if not docking_results:
            return 0.0

        # Example: Average docking score
        total_score = sum(result['score'] for result in docking_results)
        average_score = total_score / len(docking_results)
        logger.info(f"Average docking score: {average_score}")
        print(Fore.BLUE + f"Average docking score: {average_score}")
        return average_score

class SMILESLigandPipeline:
    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.additional_optimizer = AdditionalLigandOptimizer()
        self.docking = EnsembleDocking()
        self.analyzer = Analyzer()
        self.scorer = Scorer()

    @checkpoint("Pipeline Execution")
    def run_pipeline(self, protein_sequences: List[str], num_sequences: int, predicted_structures_dir: str, results_dir: str, score_threshold: float) -> List[Dict[str, Any]]:
        all_results = []
        os.makedirs(predicted_structures_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        for protein_sequence in protein_sequences:
            try:
                smiles_list = self.generate_novel_smiles(protein_sequence, num_sequences)
                logger.info(f"Generated SMILES: {smiles_list}")
                print(Fore.CYAN + f"Generated SMILES: {smiles_list}")

                for smiles in smiles_list:
                    result = self.process_single_smiles(smiles, protein_sequence, predicted_structures_dir, results_dir, score_threshold)
                    if result:
                        all_results.append(result)

            except Exception as e:
                logger.error(f"Error processing protein sequence {protein_sequence}: {e}", exc_info=True)
                print(Fore.RED + f"Error processing protein sequence {protein_sequence}: {e}")

        logger.info("Pipeline completed")
        print(Fore.CYAN + "Pipeline completed")

        return all_results

    def generate_novel_smiles(self, protein_sequence: str, num_sequences: int) -> List[str]:
        valid_smiles = []
        attempts = 0
        max_attempts = num_sequences * 10  # Allow more attempts for novel molecules
        while len(valid_smiles) < num_sequences and attempts < max_attempts:
            smiles = self.generator.generate(protein_sequence, 1)[0]
            if self.validate_novel_smiles(smiles):
                valid_smiles.append(smiles)
            attempts += 1
            if attempts % 10 == 0:
                logger.info(f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
                print(Fore.YELLOW + f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
        return valid_smiles

    def validate_novel_smiles(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            # Additional novelty checks can be added here
            return True
        except:
            return False

    def adjust_logp(self, smiles: str, target_logp=2.5, tolerance=0.5) -> str:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        current_logp = Crippen.MolLogP(mol)

        while abs(current_logp - target_logp) > tolerance:
            if current_logp < target_logp:
                # Add a lipophilic group (e.g., methyl)
                mol = self.add_group(mol, 'C')
            else:
                # Add a hydrophilic group (e.g., hydroxyl)
                mol = self.add_group(mol, 'O')

            current_logp = Crippen.MolLogP(mol)

        return Chem.MolToSmiles(mol)

    def add_group(self, mol: Chem.Mol, group: str) -> Chem.Mol:
        rwmol = Chem.RWMol(mol)
        if group == 'C':
            atom = Chem.Atom(6)  # Carbon
            idx = rwmol.AddAtom(atom)
            # Attach to a random atom
            attach_atom = random.randint(0, rwmol.GetNumAtoms() - 1)
            rwmol.AddBond(attach_atom, idx, Chem.BondType.SINGLE)
        elif group == 'O':
            atom = Chem.Atom(8)  # Oxygen
            idx = rwmol.AddAtom(atom)
            # Attach to a random atom
            attach_atom = random.randint(0, rwmol.GetNumAtoms() - 1)
            rwmol.AddBond(attach_atom, idx, Chem.BondType.SINGLE)
        else:
            return mol  # Unsupported group

        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    def process_single_smiles(self, smiles: str, protein_sequence: str, predicted_structures_dir: str, results_dir: str, score_threshold: float) -> Dict[str, Any]:
        result = {"smiles": smiles}

        # 1. Optimize SMILES
        try:
            optimized_smiles = self.optimizer.optimize_smiles(smiles)
            result['optimized_smiles'] = optimized_smiles
            logger.info(f"Optimized SMILES: {optimized_smiles}")
            print(Fore.GREEN + f"Optimized SMILES: {optimized_smiles}")
        except Exception as e:
            logger.warning(f"Error during SMILES optimization: {e}")
            print(Fore.YELLOW + f"Error during SMILES optimization: {e}")
            return None

        # 2. Validate and Adjust LogP
        try:
            adjusted_smiles = self.adjust_logp(optimized_smiles)
            result['adjusted_smiles'] = adjusted_smiles
            logger.info(f"Adjusted SMILES: {adjusted_smiles}")
            print(Fore.GREEN + f"Adjusted SMILES: {adjusted_smiles}")
        except Exception as e:
            logger.warning(f"Error during LogP adjustment: {e}")
            print(Fore.YELLOW + f"Error during LogP adjustment: {e}")
            return None

        # 3. 3D Structure Prediction
        try:
            ligand_pdb = self.predictor.predict_3d_ligand_structure(adjusted_smiles, predicted_structures_dir)
            result['ligand_pdb'] = ligand_pdb
            logger.info(f"3D Structure Prediction completed: {ligand_pdb}")
            print(Fore.GREEN + f"3D Structure Prediction completed: {ligand_pdb}")
        except Exception as e:
            logger.warning(f"Error in 3D structure prediction: {e}")
            print(Fore.YELLOW + f"Error in 3D structure prediction: {e}")
            return None

        # 4. Generate Protein Structures
        try:
            protein_pdb = self.predictor.predict_protein_structure(protein_sequence, predicted_structures_dir)
            protein_ensemble = [protein_pdb]  # Assuming single structure; modify if multiple structures are generated
            logger.info(f"Generated protein structure: {protein_pdb}")
            print(Fore.GREEN + f"Generated protein structure: {protein_pdb}")
        except Exception as e:
            logger.warning(f"Error in protein structure generation: {e}")
            print(Fore.YELLOW + f"Error in protein structure generation: {e}")
            return None

        # 5. Ensemble Docking
        if ligand_pdb and protein_ensemble:
            try:
                docking_results = self.docking.dock_ensemble(ligand_pdb, protein_ensemble, results_dir)
                result['docking_results'] = docking_results
                logger.info("Ensemble Docking completed")
                print(Fore.GREEN + "Ensemble Docking completed")
            except Exception as e:
                logger.warning(f"Error in ensemble docking: {e}")
                print(Fore.YELLOW + f"Error in ensemble docking: {e}")
                return None

        # 6. Analyze Docking Results
        if 'docking_results' in result:
            try:
                analysis = self.analyzer.analyze(result['docking_results'])
                result['analysis'] = analysis
                logger.info("Docking Analysis completed")
                print(Fore.GREEN + "Docking Analysis completed")
            except Exception as e:
                logger.warning(f"Error in docking analysis: {e}")
                print(Fore.YELLOW + f"Error in docking analysis: {e}")
                return None

        # 7. Scoring
        try:
            score = self.scorer.calculate_docking_score(result['docking_results'], protein_ensemble)
            result['score'] = score
            logger.info(f"Ligand scored: {score}")
            print(Fore.GREEN + f"Ligand scored: {score}")
        except Exception as e:
            logger.warning(f"Error in scoring: {e}")
            print(Fore.YELLOW + f"Error in scoring: {e}")
            return None

        # 8. Filter based on score
        if score >= score_threshold:
            logger.info(f"Ligand {smiles} passed with score {score}")
            print(Fore.GREEN + f"Ligand {smiles} passed with score {score}")
            return result
        else:
            logger.info(f"Ligand {smiles} did not meet the score threshold ({score_threshold}).")
            print(Fore.YELLOW + f"Ligand {smiles} did not meet the score threshold ({score_threshold}).")
            return None