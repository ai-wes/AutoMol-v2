
import os   
import sys
# Add the parent directory to the Python path

from pathlib import Path



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import os
from phase5.generate_final_report import load_results, analyze_phase1_results, analyze_phase2a_results, analyze_phase2b_results, analyze_phase3_results, analyze_phase4_results, generate_final_report
from phase5.decision_making_process import decision_making_process, main as decision_main

def run_Phase_5(run_dir):
    print("Starting Phase 5")
    

    print("Loading results from previous phases")
    phase1_results = load_results(run_dir, 'phase1')
    phase2a_results = load_results(run_dir, 'phase2a')
    phase2b_results = load_results(run_dir, 'phase2b')
    phase3_results = load_results(run_dir, 'phase3')
    phase4_results = load_results(run_dir, 'phase4')

    # Analyze results
    print("Analyzing results from previous phases")
    analysis = {
        'phase1': analyze_phase1_results(phase1_results),
        'phase2a': analyze_phase2a_results(phase2a_results),
        'phase2b': analyze_phase2b_results(phase2b_results),
        'phase3': analyze_phase3_results(phase3_results),
        'phase4': analyze_phase4_results(phase4_results)
    }

    # Generate final report
    print("Generating final comprehensive report")
    output_path = os.path.join(run_dir, 'final_comprehensive_report.md')
    generate_final_report(analysis, output_path)

    # Decision-making process
    print("Running decision-making process")
    next_steps = decision_making_process(analysis)
    print("Decision on Next Steps:")
    print(next_steps)

    # Optionally, save the decision to a file
    print("Saving decision to file")
    decision_path = os.path.join(run_dir, 'next_steps.txt')
    with open(decision_path, 'w') as f:
        f.write(next_steps)
    print(f"Decision saved at {decision_path}")

    # Call the main function from decision_making_process
    print("Running decision-making process main function")
    print("Running decision-making process main function...")
    decision_main(config_path)

    print("Phase 5 completed")

if __name__ == "__main__":
    base_output_dir = 'path/to/your/output/directory'  # Update this path
    config_path = 'path/to/your/config.json'  # Update this path
    run_Phase_5(base_output_dir, config_path)