import os
from pathlib import Path
from generate_final_report import load_results, analyze_phase1_results, analyze_phase2a_results, analyze_phase2b_results, analyze_phase3_results, analyze_phase4_results, generate_final_report
from decision_making_process import decision_making_process, main as decision_main
from server.app import emit_progress

def run_Phase_5(base_output_dir: str, config_path: str):
    emit_progress("Starting Phase 5", 0)
    
    # Load results
    run_dirs = sorted(Path(base_output_dir).glob('run_*'), key=os.path.getmtime)
    if not run_dirs:
        print("No run directories found.")
        emit_progress("No run directories found", 100)
        return
    run_dir = str(run_dirs[-1])  # Use the latest run directory

    emit_progress("Loading results from previous phases", 10)
    phase1_results = load_results(run_dir, 'phase1')
    phase2a_results = load_results(run_dir, 'phase2a')
    phase2b_results = load_results(run_dir, 'phase2b')
    phase3_results = load_results(run_dir, 'phase3')
    phase4_results = load_results(run_dir, 'phase4')

    # Analyze results
    emit_progress("Analyzing results from previous phases", 30)
    analysis = {
        'phase1': analyze_phase1_results(phase1_results),
        'phase2a': analyze_phase2a_results(phase2a_results),
        'phase2b': analyze_phase2b_results(phase2b_results),
        'phase3': analyze_phase3_results(phase3_results),
        'phase4': analyze_phase4_results(phase4_results)
    }

    # Generate final report
    emit_progress("Generating final comprehensive report", 50)
    output_path = os.path.join(run_dir, 'final_comprehensive_report.md')
    generate_final_report(analysis, output_path)

    # Decision-making process
    emit_progress("Running decision-making process", 70)
    next_steps = decision_making_process(analysis)
    print("Decision on Next Steps:")
    print(next_steps)

    # Optionally, save the decision to a file
    emit_progress("Saving decision to file", 80)
    decision_path = os.path.join(run_dir, 'next_steps.txt')
    with open(decision_path, 'w') as f:
        f.write(next_steps)
    print(f"Decision saved at {decision_path}")

    # Call the main function from decision_making_process
    emit_progress("Running decision-making process main function", 90)
    print("Running decision-making process main function...")
    decision_main(config_path)

    emit_progress("Phase 5 completed", 100)

if __name__ == "__main__":
    base_output_dir = 'path/to/your/output/directory'  # Update this path
    config_path = 'path/to/your/config.json'  # Update this path
    run_Phase_5(base_output_dir, config_path)