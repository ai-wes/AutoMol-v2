import json
import os
from pathlib import Path
from datetime import datetime

def load_results(run_dir: str, phase_name: str) -> dict:
    """Helper function to load JSON results from a given phase."""
    result_file = Path(run_dir) / f"{phase_name}_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Results file for {phase_name} not found.")
        return {}

def analyze_phase1_results(phase1_results: dict) -> dict:
    """Analyze Phase 1 results and extract key insights."""
    # Example: Extract the technical description and initial hypothesis
    return {
        'technical_description': phase1_results.get('technical_description', ''),
        'initial_hypothesis': phase1_results.get('initial_hypothesis', '')
    }

def analyze_phase2a_results(phase2a_results: list) -> dict:
    """Analyze Phase 2a results and extract key metrics."""
    # Example: Extract optimized sequences and scores
    sequences = [result.get('sequence') for result in phase2a_results]
    scores = [result.get('score') for result in phase2a_results]
    return {
        'sequences': sequences,
        'scores': scores,
        'average_score': sum(scores) / len(scores) if scores else 0
    }

def analyze_phase2b_results(phase2b_results: dict) -> dict:
    """Analyze Phase 2b results and extract key metrics."""
    # Assuming phase2b_results is a dict with 'phase2b_results' key
    ligands = phase2b_results.get('phase2b_results', [])
    smiles_list = [ligand.get('smiles') for ligand in ligands]
    return {
        'ligands': smiles_list,
        'total_ligands': len(smiles_list)
    }

def analyze_phase3_results(phase3_results: dict) -> dict:
    """Analyze Phase 3 results and extract key metrics."""
    simulation_results = phase3_results.get('simulation_results', {})
    docking_results = phase3_results.get('docking_results', {})
    analysis_results = phase3_results.get('analysis_results', {})
    return {
        'simulation_summary': simulation_results,
        'docking_summary': docking_results,
        'analysis_summary': analysis_results
    }

def analyze_phase4_results(phase4_results: dict) -> dict:
    """Analyze Phase 4 results and extract key metrics."""
    # Assuming phase4_results contains the final analysis report
    return phase4_results

from datetime import datetime

def generate_final_report(analysis: dict, output_path: str):
    """Generates a comprehensive report from the analysis data."""
    report_lines = []
    report_lines.append("# Final Comprehensive Report\n")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Phase 1
    report_lines.append("## Phase 1: Hypothesis Generation\n")
    phase1 = analysis.get('phase1', {})
    report_lines.append(f"**Technical Description:** {phase1.get('technical_description', '')}\n")
    report_lines.append(f"**Initial Hypothesis:** {phase1.get('initial_hypothesis', '')}\n")

    # Phase 2a
    report_lines.append("## Phase 2a: Protein Generation and Optimization\n")
    phase2a = analysis.get('phase2a', {})
    sequences = phase2a.get('sequences', [])
    scores = phase2a.get('scores', [])
    report_lines.append(f"**Number of Sequences Generated:** {len(sequences)}\n")
    report_lines.append(f"**Average Optimization Score:** {phase2a.get('average_score', 0):.2f}\n")
    report_lines.append("### Sequences and Scores:\n")
    for seq, score in zip(sequences, scores):
        report_lines.append(f"- **Sequence:** {seq}\n  - **Score:** {score:.2f}\n")

    # Phase 2b
    report_lines.append("## Phase 2b: Ligand Generation and Optimization\n")
    phase2b = analysis.get('phase2b', {})
    smiles_list = phase2b.get('ligands', [])
    report_lines.append(f"**Total Valid Ligands Generated:** {len(smiles_list)}\n")
    report_lines.append("### Ligand SMILES:\n")
    for smiles in smiles_list:
        report_lines.append(f"- {smiles}\n")

    # Phase 3
    report_lines.append("## Phase 3: Simulation\n")
    phase3 = analysis.get('phase3', {})
    report_lines.append("### Simulation Summary:\n")
    report_lines.append("```json\n")
    report_lines.append(json.dumps(phase3.get('simulation_summary', {}), indent=2))
    report_lines.append("\n```\n")
    report_lines.append("### Docking Summary:\n")
    report_lines.append("```json\n")
    report_lines.append(json.dumps(phase3.get('docking_summary', {}), indent=2))
    report_lines.append("\n```\n")
    report_lines.append("### Analysis Summary:\n")
    report_lines.append("```json\n")
    report_lines.append(json.dumps(phase3.get('analysis_summary', {}), indent=2))
    report_lines.append("\n```\n")

    # Phase 4
    report_lines.append("## Phase 4: Digital Twin Simulation and Analysis\n")
    report_lines.append("### Final Analysis Results:\n")
    report_lines.append("```json\n")
    report_lines.append(json.dumps(analysis.get('phase4', {}), indent=2))
    report_lines.append("\n```\n")

    # Write the report to a file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Final report generated at {output_path}")





def has_significant_improvement(current_best: float, previous_best: float, improvement_threshold: float = 0.05) -> bool:
    if previous_best == 0:
        return current_best > 0
    improvement = (current_best - previous_best) / previous_best
    return improvement >= improvement_threshold



def main():
    # Assuming the results are saved in a directory named 'results' with subdirectories for each run
    base_output_dir = 'results'  # Or get it from config
    # Find the latest run directory
    run_dirs = sorted(Path(base_output_dir).glob('run_*'), key=os.path.getmtime)
    if not run_dirs:
        print("No run directories found.")
        return
    run_dir = run_dirs[-1]  # Use the latest run directory

    # Load results from each phase
    phase1_results = load_results(run_dir, 'phase1_results')
    phase2a_results = load_results(run_dir, 'phase2a_results')
    phase2b_results = load_results(run_dir, 'phase2b_results')
    phase3_results = load_results(run_dir, 'phase3_results')
    phase4_results = load_results(run_dir, 'phase4_results')

    # Collect and analyze results
    analysis = {
        'phase1': analyze_phase1_results(phase1_results),
        'phase2a': analyze_phase2a_results(phase2a_results),
        'phase2b': analyze_phase2b_results(phase2b_results),
        'phase3': analyze_phase3_results(phase3_results),
        'phase4': analyze_phase4_results(phase4_results)
    }

    # Generate the report
    output_path = os.path.join(run_dir, 'final_comprehensive_report.md')
    generate_final_report(analysis, output_path)



if __name__ == '__main__':
    main()
