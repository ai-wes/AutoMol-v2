# utils/eval.py  –  one place only!
from time import perf_counter
import mdtraj as md
import torch

WEIGHTS = (0.5, 0.3, 0.2)  # ΔΔG, RMSD, time

def evaluate_kernel(kernel_fn, batch):
    """
    Args
        kernel_fn: callable that predicts ΔG binding for a batch of (prot, lig)
        batch:     list[(protein_pdb_path, ligand_sdf_path, exp_ddg)]
    Returns
        score_dict with the three atomic metrics + weighted scalar
    """
    t0 = perf_counter()
    pred = torch.stack([kernel_fn(p,l) for p,l,_ in batch])
    ddg_err = (pred - torch.tensor([x[2] for x in batch])).abs().mean().item()

    # quick 50 ps MD check on 1st complex
    traj = md.load_simulation(kernel_fn.last_complex_top_file, n_steps=2500)  # 50 ps @ 20 fs
    rmsd = md.rmsd(traj, traj, 0).max()        # drift from starting structure

    gpu_sec = perf_counter() - t0

    scalar = sum(w*m for w,m in zip(WEIGHTS, (ddg_err, rmsd, gpu_sec)))
    return {
        "ddg_mae": ddg_err,
        "md_rmsd": rmsd,
        "gpu_s":   gpu_sec,
        "scalar":  scalar
    }
