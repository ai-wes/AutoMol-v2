import torch
import os
import logging
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from Bio import PDB

logger = logging.getLogger(__name__)

last_residue_number = 0

def custom_read_residue_number(pdb_line, field, read_bytes, read_code):
    global last_residue_number
    num_str = pdb_line[22:26].strip()
    if not num_str:
        last_residue_number += 1
        logger.warning(f"Empty residue number encountered in line: {pdb_line.strip()}. Assigning number: {last_residue_number}")
        return last_residue_number
    try:
        residue_number = int(num_str)
        last_residue_number = residue_number
        return residue_number
    except ValueError:
        last_residue_number += 1
        logger.warning(f"Non-numeric residue number encountered: '{num_str}' in line: {pdb_line.strip()}. Assigning number: {last_residue_number}")
        return last_residue_number

# Monkey-patch the MDTraj function
from mdtraj.formats import pdb
pdb.pdbstructure._read_residue_number = custom_read_residue_number

def validate_pdb_file(pdb_file):
    """Validate the PDB file and print detailed information about its structure."""
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in PDB file: {len(lines)}")
    
    atom_lines = [line for line in lines if line.startswith('ATOM')]
    print(f"Total ATOM lines: {len(atom_lines)}")
    
    residues = set()
    for line in atom_lines:
        residue_num = line[22:26].strip()
        residues.add(residue_num)
    
    print(f"Unique residue numbers: {len(residues)}")
    print(f"First few residue numbers: {list(residues)[:10]}")
    
    if '' in residues:
        print("WARNING: Empty residue numbers detected!")
        empty_lines = [line for line in atom_lines if line[22:26].strip() == '']
        print(f"Number of lines with empty residue numbers: {len(empty_lines)}")
        print("First few lines with empty residue numbers:")
        for line in empty_lines[:5]:
            print(line.strip())

def load_trajectory(trajectory_file, topology_file):
    """Load the trajectory using MDTraj's built-in functions."""
    print("Loading trajectory...")
    try:
        # Validate the PDB file before loading
        validate_pdb_file(topology_file)
        
        # Load the topology file
        topology = md.load_pdb(topology_file).topology
        print(f"Topology loaded. Atoms: {topology.n_atoms}, Residues: {topology.n_residues}")

        # Load the trajectory using the topology
        traj = md.load(trajectory_file, top=topology)
        print(f"Trajectory loaded successfully. Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
        return traj
    except Exception as e:
        print(f"Error loading trajectory: {str(e)}")
        print(f"Trajectory file size: {os.path.getsize(trajectory_file)} bytes")
        return None

def calculate_rmsd(positions):
    if isinstance(positions, torch.Tensor):
        return torch.sqrt(3 * torch.mean(torch.sum(torch.square(positions - torch.mean(positions, dim=0)), dim=2), dim=1)).cpu().numpy()
    else:
        return np.sqrt(3 * np.mean(np.sum(np.square(positions - np.mean(positions, axis=0)), axis=2), axis=1))

def calculate_radius_of_gyration(positions):
    if isinstance(positions, torch.Tensor):
        center_of_mass = torch.mean(positions, dim=1, keepdim=True)
        rg = torch.sqrt(torch.mean(torch.sum(torch.square(positions - center_of_mass), dim=2), dim=1))
        return rg.cpu().numpy()
    else:
        return md.compute_rg(positions)

def calculate_secondary_structure(traj):
    """Calculate secondary structure of the trajectory."""
    logger.info("Calculating secondary structure...")
    ss = md.compute_dssp(traj)
    
    # Convert string labels to numerical values
    ss_map = {'H': 0, 'E': 1, 'C': 2}  # Helix, Sheet, Coil
    ss_numeric = np.array([[ss_map.get(s, 2) for s in frame] for frame in ss])
    
    return ss_numeric

def calculate_rmsf(traj):
    """Calculate RMSF of the trajectory."""
    logger.info("Calculating RMSF...")
    return np.sqrt(3*np.mean(np.square(traj.xyz - np.mean(traj.xyz, axis=0)), axis=0))

def generate_rmsd_plot(rmsd, output_dir):
    """Generate RMSD plot."""
    plt.figure()
    plt.plot(rmsd)
    plt.xlabel('Frame')
    plt.ylabel('RMSD (nm)')
    plt.title('Root Mean Square Deviation')
    plt.savefig(os.path.join(output_dir, 'rmsd_plot.png'))
    plt.close()

def generate_rmsf_plot(rmsf, output_dir):
    """Generate RMSF plot."""
    plt.figure()
    plt.plot(rmsf)
    plt.xlabel('Residue')
    plt.ylabel('RMSF (nm)')
    plt.title('Root Mean Square Fluctuation')
    plt.savefig(os.path.join(output_dir, 'rmsf_plot.png'))
    plt.close()

def generate_rg_plot(rg, output_dir):
    """Generate radius of gyration plot."""
    plt.figure()
    plt.plot(rg)
    plt.xlabel('Frame')
    plt.ylabel('Radius of Gyration (nm)')
    plt.title('Radius of Gyration')
    plt.savefig(os.path.join(output_dir, 'rg_plot.png'))
    plt.close()

def generate_ss_plot(ss, output_dir):
    """Generate secondary structure plot."""
    plt.figure(figsize=(10, 6))
    plt.imshow(ss.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.xlabel('Frame')
    plt.ylabel('Residue')
    plt.colorbar(label='Secondary Structure')
    plt.title('Secondary Structure Evolution')
    
    # Add custom colorbar ticks
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Helix', 'Sheet', 'Coil'])
    
    plt.savefig(os.path.join(output_dir, 'ss_plot.png'))
    plt.close()

def calculate_final_score(rmsd, rmsf, rg, ss):
    """Calculate a final score based on the analysis results."""
    try:
        # 1. RMSD stability score (lower is better)
        rmsd_mean = np.mean(rmsd)
        rmsd_std = np.std(rmsd)
        rmsd_score = 1 / (1 + rmsd_mean + rmsd_std)  # Normalized between 0 and 1
        logger.info(f"RMSD score: {rmsd_score}, Mean: {rmsd_mean}, Std: {rmsd_std}")

        # 2. RMSF flexibility score (balance between flexibility and rigidity)
        rmsf_mean = np.mean(rmsf)
        rmsf_score = np.exp(-(rmsf_mean - 0.1)**2 / 0.02)  # Gaussian centered at 0.1 nm
        logger.info(f"RMSF score: {rmsf_score}, Mean: {rmsf_mean}")

        # 3. Radius of gyration compactness score (prefer compact structures)
        rg_mean = np.mean(rg)
        rg_score = 1 / (1 + rg_mean)  # Normalized between 0 and 1
        logger.info(f"Radius of Gyration score: {rg_score}, Mean: {rg_mean}")

        # 4. Secondary structure stability score
        ss_counts = np.sum(ss, axis=0)
        total_counts = np.sum(ss_counts)
        if total_counts > 0:
            helix_percent = ss_counts[0] / total_counts
            sheet_percent = ss_counts[1] / total_counts
            ss_score = (helix_percent + sheet_percent) / 2  # Prefer more structured elements
        else:
            ss_score = 0
        logger.info(f"Secondary Structure score: {ss_score}, Helix %: {helix_percent}, Sheet %: {sheet_percent}")

        # Calculate final score (weighted average)
        weights = [0.3, 0.2, 0.2, 0.3]  # Adjust these weights as needed
        final_score = np.dot([rmsd_score, rmsf_score, rg_score, ss_score], weights)
        logger.info(f"Weighted score before normalization: {final_score}")

        # Normalize final score between 0 and 1
        final_score = (final_score - 0.5) * 2  # Assuming 0.5 is an average score
        final_score = max(0, min(1, final_score))  # Clamp between 0 and 1
        logger.info(f"Final normalized score: {final_score}")

        return final_score

    except Exception as e:
        logger.error(f"Error calculating final score: {str(e)}")
        logger.error(f"RMSD shape: {rmsd.shape}, RMSF shape: {rmsf.shape}, RG shape: {rg.shape}, SS shape: {ss.shape}")
        logger.error(f"RMSD: {rmsd}")
        logger.error(f"RMSF: {rmsf}")
        logger.error(f"RG: {rg}")
        logger.error(f"SS: {ss}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 0  # Return 0 if there's an error

def run_analysis_pipeline(trajectory_file, final_pdb, output_dir, device):
    try:
        print(f"Starting analysis pipeline")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")

        print(f"Loading trajectory: {trajectory_file}")
        print(f"Using topology file: {final_pdb}")

        traj = load_trajectory(trajectory_file, final_pdb)
        if traj is None:
            print("Failed to load trajectory. Skipping analysis.")
            return None

        print(f"Trajectory loaded successfully. Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")

        # Ensure trajectory contains data
        if traj.n_frames == 0 or traj.n_atoms == 0:
            print("Trajectory contains no frames or atoms. Exiting analysis.")
            return None

        # Calculate RMSD
        print("Calculating RMSD...")
        rmsd = md.rmsd(traj, traj, 0)
        if rmsd.size == 0:
            print("RMSD calculation returned empty array.")
            return None

        # Calculate RMSF
        print("Calculating RMSF...")
        rmsf = md.rmsf(traj, traj, 0)
        if rmsf.size == 0:
            print("RMSF calculation returned empty array.")
            return None

        # Calculate radius of gyration
        print("Calculating radius of gyration...")
        rg = md.compute_rg(traj)
        if rg.size == 0:
            print("Radius of gyration calculation returned empty array.")
            return None

        # Calculate secondary structure
        print("Calculating secondary structure...")
        ss = md.compute_dssp(traj)
        if ss.size == 0:
            print("Secondary structure calculation returned empty array.")
            return None

        print("Generating plots...")
        generate_rmsd_plot(rmsd, output_dir)
        generate_rmsf_plot(rmsf, output_dir)
        generate_rg_plot(rg, output_dir)
        generate_ss_plot(ss, output_dir)
        print("Plots generated")

        print("Calculating final score...")
        final_score = calculate_final_score(rmsd, rmsf, rg, ss)
        print(f"Final score calculated: {final_score}")

        return {
            "rmsd": rmsd.tolist(),
            "rmsf": rmsf.tolist(),
            "radius_of_gyration": rg.tolist(),
            "secondary_structure": ss.tolist(),
            "final_score": final_score
        }

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        print(f"Trajectory file: {trajectory_file}")
        print(f"Topology file: {final_pdb}")
        print(f"Output directory: {output_dir}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    trajectory_file = r"C:\users\wes\AutoMol-v2\results\phase3\protein_0\simulation\trajectory.dcd"
    final_pdb = r"C:\users\wes\AutoMol-v2\results\phase3\protein_0\simulation\final.pdb"
    output_dir = r"C:\users\wes\AutoMol-v2\results\phase3\protein_0\simulation\analysis"
    device = "cuda"
    run_analysis_pipeline(trajectory_file, final_pdb, output_dir, device)