import os
import logging
from pathlib import Path
import json
from generate_final_report import load_results, analyze_results
from server.app import emit_progress
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.phase3_run import run_Phase_3
from phase4.phase4_run import run_Phase_4

# Configure logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_phase(phase_function, phase_name, *args, **kwargs):
    """Execute a phase function with error handling."""
    try:
        emit_progress(f"Starting {phase_name}")
        result = phase_function(*args, **kwargs)
        emit_progress(f"{phase_name} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in {phase_name}: {str(e)}", exc_info=True)
        emit_progress(f"Error in {phase_name}: {str(e)}")
        return None
    
    
def decision_making_process(analysis: dict, config: dict) -> str:
    """Decide the next steps based on the analysis results."""
    # Define thresholds (could be moved to a config file)
    MIN_AVERAGE_SCORE = config.get('MIN_AVERAGE_SCORE', 0.8)
    MIN_LIGANDS = config.get('MIN_LIGANDS', 5)
    MIN_SIMULATION_SCORE = config.get('MIN_SIMULATION_SCORE', 0.75)
    MAX_ITERATIONS = config.get('MAX_ITERATIONS', 3)
    IMPROVEMENT_THRESHOLD = config.get('IMPROVEMENT_THRESHOLD', 0.05)
    CLOSE_TO_TARGET_THRESHOLD = config.get('CLOSE_TO_TARGET_THRESHOLD', 0.9)

    iteration = 1
    previous_overall_score = 0

    while True:
        emit_progress(f"Starting decision-making process (Iteration {iteration}).")
        logger.info(f"Starting decision-making process (Iteration {iteration}).")
        logger.debug(f"Analysis data: {json.dumps(analysis, indent=2)}")

        # Retrieve metrics
        average_sequence_score = analysis.get('phase2a', {}).get('average_score', 0)
        total_ligands = analysis.get('phase2b', {}).get('total_ligands', 0)
        simulation_score = analysis.get('phase3', {}).get('simulation_summary', {}).get('average_score', 0)

        # Calculate overall score
        overall_score = (average_sequence_score + simulation_score) / 2

        emit_progress(f"Average Sequence Score: {average_sequence_score}")
        emit_progress(f"Total Ligands: {total_ligands}")
        emit_progress(f"Simulation Score: {simulation_score}")
        emit_progress(f"Overall Score: {overall_score}")

        logger.info(f"Average Sequence Score: {average_sequence_score}")
        logger.info(f"Total Ligands: {total_ligands}")
        logger.info(f"Simulation Score: {simulation_score}")
        logger.info(f"Overall Score: {overall_score}")

        # Decision logic
        decision = ""
        phases_to_rerun = []

        if average_sequence_score < MIN_AVERAGE_SCORE:
            decision = "Iterate Phase 2a: Protein sequences need further optimization."
            phases_to_rerun.extend(['phase2a', 'phase2b', 'phase3', 'phase4'])
        elif total_ligands < MIN_LIGANDS:
            decision = "Iterate Phase 2b: Generate more valid ligands."
            phases_to_rerun.extend(['phase2b', 'phase3', 'phase4'])
        elif simulation_score < MIN_SIMULATION_SCORE:
            decision = "Iterate Phase 3: Simulation results are not satisfactory."
            phases_to_rerun.extend(['phase3', 'phase4'])
        else:
            decision = "Proceed to experimental validation of promising molecules."

        emit_progress(f"Decision: {decision}")
        logger.info(f"Decision: {decision}")

        # Check if we've reached the desired metrics
        if not phases_to_rerun:
            return decision

        # Check if we're close to the target and improvement is minimal
        close_to_target = (
            average_sequence_score >= MIN_AVERAGE_SCORE * CLOSE_TO_TARGET_THRESHOLD and
            total_ligands >= MIN_LIGANDS * CLOSE_TO_TARGET_THRESHOLD and
            simulation_score >= MIN_SIMULATION_SCORE * CLOSE_TO_TARGET_THRESHOLD
        )

        if iteration > 1 and close_to_target:
            improvement = (overall_score - previous_overall_score) / previous_overall_score
            if improvement < IMPROVEMENT_THRESHOLD:
                return f"Stopping iterations due to minimal improvement (< {IMPROVEMENT_THRESHOLD*100}%) while close to target metrics."

        if iteration >= MAX_ITERATIONS:
            return f"Maximum iterations ({MAX_ITERATIONS}) reached. Proceeding with best available results."

        if phases_to_rerun:
            emit_progress(f"Re-running phases: {', '.join(phases_to_rerun)}")
            logger.info(f"Re-running phases: {', '.join(phases_to_rerun)}")

            # Execute the necessary phases
            if 'phase2a' in phases_to_rerun:
                execute_phase(run_Phase_2a, "Phase 2a", config['phase2a'])
            if 'phase2b' in phases_to_rerun:
                execute_phase(run_Phase_2b, "Phase 2b", config['phase2b'])
            if 'phase3' in phases_to_rerun:
                execute_phase(run_Phase_3, "Phase 3", config['phase3'])
            if 'phase4' in phases_to_rerun:
                execute_phase(run_Phase_4, "Phase 4", config['phase4'])

            # Load new results
            run_dir = config.get('run_dir', 'results/latest_run')
            new_analysis = {
                'phase1': analysis.get('phase1', {}),
                'phase2a': analyze_results(load_results(run_dir, 'phase2a'), 'phase2a'),
                'phase2b': analyze_results(load_results(run_dir, 'phase2b'), 'phase2b'),
                'phase3': analyze_results(load_results(run_dir, 'phase3'), 'phase3'),
                'phase4': analyze_results(load_results(run_dir, 'phase4'), 'phase4'),
            }

            previous_overall_score = overall_score
            analysis = new_analysis
            iteration += 1
        else:
            # Final decision to proceed
            return decision

def main(config_path: str):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load results from the latest run
    base_output_dir = config.get('base_output_dir', 'results')
    run_dirs = sorted(Path(base_output_dir).glob('run_*'), key=os.path.getmtime)
    if not run_dirs:
        logger.error("No run directories found.")
        return
    
    run_dir = run_dirs[-1]  # Use the latest run directory
    
    # Load and analyze results
    analysis = {
        'phase2a': load_results(run_dir, 'phase2a_results'),
        'phase2b': load_results(run_dir, 'phase2b_results'),
        'phase3': load_results(run_dir, 'phase3_results'),
        'phase4': load_results(run_dir, 'phase4_results')
    }
    
    # Make decision and execute phases if necessary
    decision = decision_making_process(analysis)
    
    # Save the decision to a file
    decision_file = run_dir / 'decision.txt'
    with open(decision_file, 'w') as f:
        f.write(decision)
    
    logger.info(f"Decision made: {decision}")
    logger.info(f"Decision saved to {decision_file}")

if __name__ == "__main__":
    config_path = 'path/to/your/config.json'  # Update this path
    main(config_path)