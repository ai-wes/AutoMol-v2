from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

import unittest
from unittest.mock import MagicMock, patch
import sys

# Add the main project directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))
sys.path.append(project_root)

from automol.phase2.phase2b.phase2b_run import Phase2bRunner
from automol.phase2.phase2b.pre_screen_compounds import pre_screen_ligand

class TestPhase2bRunner(unittest.TestCase):
    def setUp(self):
        # Create a mock SMILESLigandPipeline
        self.mock_pipeline = MagicMock()
        
        # Configure the mock methods
        self.mock_pipeline.generate_novel_smiles.return_value = ['CCO', 'CCN', 'CCC']
        self.mock_pipeline.iterative_optimization.side_effect = [
            'CCO_optimized', 'CCO_optimized', 'CCO_optimized', 'CCO_optimized', 'CCO_optimized',
            'CCN_optimized', 'CCN_optimized', 'CCN_optimized', 'CCN_optimized', 'CCN_optimized',
            'CCC_optimized', 'CCC_optimized', 'CCC_optimized', 'CCC_optimized', 'CCC_optimized'
        ]
        self.mock_pipeline.process_single_smiles.return_value = {
            'smiles': 'CCO',
            'optimized_smiles': 'CCO',
            'adjusted_smiles': 'CCO',
            'ligand_pdb': 'path/to/ligand.pdb',
            'docking_results': [{'protein_pdb': 'protein1.pdb', 'ligand_pdbqt': 'ligand1.pdbqt', 'score': -7.5}],
            'analysis': {'best_protein_pdb': 'protein1.pdb', 'best_score': -7.5},
            'score': -7.5
        }
        
        self.runner = Phase2bRunner(pipeline=self.mock_pipeline)
    
    def test_run_phase2b_success(self):
        # Define test inputs
        predicted_structures_dir = 'test_predicted_structures'
        results_dir = 'test_results'
        num_sequences = 3
        optimization_steps = 5
        score_threshold = -8.0
        protein_sequences = ['MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS']

        # Mock pre_screen_ligand to always pass
        with patch('automol.utils.pre_screen_compounds.pre_screen_ligand', return_value=(True, "Valid ligand")):
            # Run the pipeline
            result = self.runner.run_Phase_2b(
                predicted_structures_dir=predicted_structures_dir,
                results_dir=results_dir,
                num_sequences=num_sequences,
                optimization_steps=optimization_steps,
                score_threshold=score_threshold,
                protein_sequences=protein_sequences
            )
        
        # Assertions
        self.assertIn('phase2b_results', result)
        self.assertEqual(len(result['phase2b_results']), 3)  # 3 ligands generated
        for docking_result in result['phase2b_results']:
            self.assertEqual(docking_result['smiles'], 'CCO')
            self.assertEqual(docking_result['score'], -7.5)
        
        # Verify that pipeline methods were called correctly
        self.mock_pipeline.generate_novel_smiles.assert_called_once_with(
            'MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS', 3
        )
        self.assertEqual(self.mock_pipeline.iterative_optimization.call_count, 0)  # No optimization needed
        self.mock_pipeline.process_single_smiles.assert_called_with(
            smiles='CCO',
            protein_sequence='MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS',
            predicted_structures_dir=predicted_structures_dir,
            results_dir=results_dir,
            score_threshold=score_threshold
        )
    
    def test_run_phase2b_with_invalid_ligands(self):
        # Define test inputs
        predicted_structures_dir = 'test_predicted_structures'
        results_dir = 'test_results'
        num_sequences = 2
        optimization_steps = 5
        score_threshold = -8.0
        protein_sequences = ['MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS']

        # Mock pre_screen_ligand to fail the first attempt and pass the second
        def pre_screen_mock(smiles):
            if smiles == 'CCO':
                return (False, "Failed validation")
            elif smiles == 'CCN':
                return (True, "Valid after optimization")
            else:
                return (False, "Unknown SMILES")
        
        with patch('automol.utils.pre_screen_compounds.pre_screen_ligand', side_effect=pre_screen_mock):
            # Adjust the pipeline's iterative_optimization to return optimized smiles
            self.mock_pipeline.iterative_optimization.side_effect = [
                'CCO_optimized', 'CCO_optimized', 'CCO_optimized', 'CCO_optimized', 'CCO_optimized',
                'CCN_optimized', 'CCN_optimized', 'CCN_optimized', 'CCN_optimized', 'CCN_optimized',
                'CCC_optimized', 'CCC_optimized', 'CCC_optimized', 'CCC_optimized', 'CCC_optimized'
            ]
            
            # Also, mock process_single_smiles for optimized ligands
            self.mock_pipeline.process_single_smiles.side_effect = [
                {
                    'smiles': 'CCO_optimized',
                    'optimized_smiles': 'CCO_optimized',
                    'adjusted_smiles': 'CCO_optimized',
                    'ligand_pdb': 'path/to/ligand_optimized.pdb',
                    'docking_results': [{'protein_pdb': 'protein1.pdb', 'ligand_pdbqt': 'ligand_optimized.pdbqt', 'score': -6.5}],
                    'analysis': {'best_protein_pdb': 'protein1.pdb', 'best_score': -6.5},
                    'score': -6.5
                },
                {
                    'smiles': 'CCN',
                    'optimized_smiles': 'CCN',
                    'adjusted_smiles': 'CCN',
                    'ligand_pdb': 'path/to/ligand_ccn.pdb',
                    'docking_results': [{'protein_pdb': 'protein1.pdb', 'ligand_pdbqt': 'ligand_ccn.pdbqt', 'score': -8.5}],
                    'analysis': {'best_protein_pdb': 'protein1.pdb', 'best_score': -8.5},
                    'score': -8.5
                }
            ]
            
            # Run the pipeline
            result = self.runner.run_Phase_2b(
                predicted_structures_dir=predicted_structures_dir,
                results_dir=results_dir,
                num_sequences=num_sequences,
                optimization_steps=optimization_steps,
                score_threshold=score_threshold,
                protein_sequences=protein_sequences
            )
        
        # Assertions
        self.assertIn('phase2b_results', result)
        self.assertEqual(len(result['phase2b_results']), 2)  # 2 ligands generated after optimization
        self.assertEqual(result['phase2b_results'][0]['smiles'], 'CCO_optimized')
        self.assertEqual(result['phase2b_results'][0]['score'], -6.5)
        self.assertEqual(result['phase2b_results'][1]['smiles'], 'CCN')
        self.assertEqual(result['phase2b_results'][1]['score'], -8.5)
        
        # Verify that pipeline methods were called correctly
        self.mock_pipeline.generate_novel_smiles.assert_called_once_with(
            'MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS', 2
        )
        self.assertEqual(self.mock_pipeline.iterative_optimization.call_count, 2)  # Two optimizations needed
        self.assertEqual(self.mock_pipeline.process_single_smiles.call_count, 2)

if __name__ == '__main__':
    unittest.main()