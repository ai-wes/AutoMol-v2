# AutoMol-v2/automol/phase3/pymol_analysis.py

import os
import logging
from pymol import cmd, finish_launching
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def run_pymol_analysis(structure_file, trajectory_file, analysis_dir):
    """
    Perform PyMOL analysis on the given structure and trajectory.
    
    Parameters:
    - structure_file: Path to the fixed PDB file.
    - trajectory_file: Path to the trajectory PDB file.
    - analysis_dir: Directory to store analysis outputs.
    
    Returns:
    - Dictionary containing analysis output paths.
    """
    try:
        logger.info(f"Starting PyMOL analysis for {structure_file} and {trajectory_file}")
        
        # Initialize PyMOL
        finish_launching(['pymol', '-qc'])  # Quiet and no GUI
        
        cmd.reinitialize()
        cmd.load(structure_file, 'structure')
        cmd.load_traj(trajectory_file, 'structure')
        
        # Hide everything and show cartoons
        cmd.hide('everything', 'structure')
        cmd.show('cartoon', 'structure')
        cmd.color('blue', 'structure')
        
        # RMSD Calculation
        rmsd_values = []
        cmd.align('structure', 'structure', cycles=5, object='aligned')
        for state in range(1, cmd.count_states('structure') + 1):
            rmsd = cmd.rms_cur(f'structure and state {state}', 'aligned and state {state}', matchmaker=1)
            rmsd_values.append(rmsd)
            logger.debug(f"RMSD for state {state}: {rmsd}")
        
        # Save RMSD plot
        rmsd_plot_path = os.path.join(analysis_dir, 'rmsd_plot.png')
        plt.figure(figsize=(10, 6))
        plt.plot(rmsd_values, marker='o')
        plt.title('RMSD over Frames')
        plt.xlabel('Frame')
        plt.ylabel('RMSD (Å)')
        plt.savefig(rmsd_plot_path)
        plt.close()
        logger.info(f"RMSD plot saved to {rmsd_plot_path}")
        
        # Generate and save an image of the structure
        image_path = os.path.join(analysis_dir, 'structure_image.png')
        cmd.png(image_path, width=1024, height=768, dpi=300, ray=1)
        logger.info(f"Structure image saved to {image_path}")
        
        # Save PyMOL session
        session_path = os.path.join(analysis_dir, 'analysis_session.pse')
        cmd.save(session_path)
        logger.info(f"PyMOL session saved to {session_path}")
        
        # Quit PyMOL
        cmd.quit()
        
        return {
            "rmsd_plot": rmsd_plot_path,
            "structure_image": image_path,
            "session_file": session_path
        }
    
    except Exception as e:
        logger.error(f"Error during PyMOL analysis: {str(e)}", exc_info=True)
        return {
            "rmsd_plot": None,
            "structure_image": None,
            "session_file": None,
            "error": str(e)
        }

def run_pymol_analysis_pipeline(simulation_results, analysis_dir):
    """
    Run PyMOL analysis for all simulation results.
    
    Parameters:
    - simulation_results: List of simulation result dictionaries.
    - analysis_dir: Base directory for analysis outputs.
    
    Returns:
    - List of analysis result dictionaries.
    """
    logger.info("Starting PyMOL Analysis Pipeline...")
    
    def analyze(sim_result):
        structure_file = sim_result.get("fixed_pdb")
        trajectory_file = sim_result.get("trajectory_file")
        protein_id = os.path.basename(structure_file).split('.')[0]
        protein_analysis_dir = os.path.join(analysis_dir, f"protein_{protein_id}")
        os.makedirs(protein_analysis_dir, exist_ok=True)
        return run_pymol_analysis(structure_file, trajectory_file, protein_analysis_dir)
    
    analysis_results = [analyze(sim) for sim in simulation_results]
    
    logger.info("PyMOL Analysis Pipeline completed.")
    return analysis_results
