# phase2a/run_phase2a.py
"""
Phase 2a – generate, mutate / optimise proteins, evaluate with a canonical
three-component metric, and persist the best candidates.

Changes vs. original
--------------------
✅  single logging.basicConfig call
✅  f-strings instead of latent literals
✅  best_score reset per technical description
✅  evaluation uses utils.eval.evaluate_kernel → returns ΔΔG-MAE, MD-RMSD,
   GPU-seconds, scalar score
✅  stores full metric dict next to each sequence
✅  no silent overwrite in set_protein_sequences
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from colorama import Fore, init

# --------------------------------------------------------------------------- #
# 3rd-party / project imports
# --------------------------------------------------------------------------- #
from phase2a.generate import generate_protein_sequence
from phase2a.optimize_new import run_optimization_pipeline
from phase2a.predict import run_prediction_pipeline
from phase2a.shared_state import set_protein_sequences
from utils.eval import evaluate_kernel, BENCHMARK_BATCH            #  <-- NEW
from utils.save_utils import create_sequence_directories, save_json

# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# main entry-point
# --------------------------------------------------------------------------- #
def run_Phase_2a(
    technical_descriptions: List[str],
    predicted_structures_dir: str,
    results_dir: str,
    num_sequences: int,
    optimization_steps: int,
    score_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns
    -------
    all_analysis_results : list[dict]
        Each dict holds sequence, pdb path and *full* eval metrics.
    all_generated_sequences : list[str]
    """

    logger.info("▶ Phase 2a – generating & optimising novel proteins")
    print(Fore.CYAN + "\nStarting Phase 2a: Generating and Optimising novel proteins")

    all_analysis_results: List[Dict[str, Any]] = []
    all_generated_sequences: List[str] = []

    # ------------------------------------------------------------------- #
    # iterate over problem descriptions
    # ------------------------------------------------------------------- #
    for i, desc in enumerate(technical_descriptions, start=1):
        print(Fore.YELLOW + f"\nProcessing Technical Description {i}:")
        print(Fore.WHITE + f"  {desc}")
        logger.info(f"[{i}/{len(technical_descriptions)}] {desc}")

        best_score = float("inf")          # lower = better after harness change
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            try:
                # ----------------------------------------------------- #
                # 1) generate raw candidates
                # ----------------------------------------------------- #
                sequences = generate_protein_sequence(desc, num_sequences)
                if not sequences:
                    logger.warning("⇢ generation returned no sequences; retrying")
                    attempts += 1
                    continue

                generated_sequence = sequences[0]          # take top-1 for now
                all_generated_sequences.append(generated_sequence)

                logger.info(f"generated seq (attempt {attempts+1}): "
                            f"{generated_sequence[:40]}…")

                # ----------------------------------------------------- #
                # 2) optimise / mutate
                # ----------------------------------------------------- #
                optimised_results = run_optimization_pipeline(
                    [generated_sequence],
                    iterations=optimization_steps,
                    score_threshold=score_threshold,
                )
                if not optimised_results:
                    logger.warning("⇢ optimisation produced no results; retrying")
                    attempts += 1
                    continue

                for opt in optimised_results:
                    optimised_seq: str = opt["optimized_sequence"]
                    best_method = opt.get("best_method", "N/A")

                    # ------------------------------------------------- #
                    # 3) evaluate deterministically with harness
                    # ------------------------------------------------- #
                    eval_metrics = evaluate_kernel(
                        kernel_fn=opt["kernel_fn"],       # ensure pipeline returns
                        batch=BENCHMARK_BATCH,
                    )
                    scalar_score = eval_metrics["scalar"]

                    logger.info(
                        f"{Fore.BLUE}Eval | ΔΔG={eval_metrics['ddg_mae']:.3f} "
                        f"RMSD={eval_metrics['md_rmsd']:.3f}Å "
                        f"GPU={eval_metrics['gpu_s']:.2f}s → "
                        f"score={scalar_score:.4f}"
                    )

                    # track best
                    if scalar_score >= best_score:
                        continue
                    best_score = scalar_score

                    # ------------------------------------------------- #
                    # 4) structure prediction → PDB
                    # ------------------------------------------------- #
                    prediction = run_prediction_pipeline(
                        [optimised_seq],
                        output_dir=predicted_structures_dir,
                    )
                    pdb_file = (
                        prediction[0]["pdb_file"] if prediction else None
                    )
                    if pdb_file is None:
                        logger.error("prediction failed – skipping")
                        continue

                    # copy PDB to run dir
                    analysis_dir, _ = create_sequence_directories(
                        results_dir, len(all_analysis_results)
                    )
                    new_pdb_path = Path(analysis_dir) / Path(pdb_file).name
                    shutil.copy(pdb_file, new_pdb_path)

                    # ------------------------------------------------- #
                    # 5) persist and log
                    # ------------------------------------------------- #
                    result_record = {
                        "sequence": optimised_seq,
                        "pdb_file": str(new_pdb_path),
                        "best_method": best_method,
                        **eval_metrics,                  # ddG, RMSD, gpu_s, scalar
                    }
                    all_analysis_results.append(result_record)

                    save_json(
                        result_record,
                        Path(analysis_dir) / "eval_metrics.json",
                    )

                    print(
                        Fore.GREEN + f"✓ Seq saved (score={scalar_score:.4f})"
                    )
                    logger.info(f"saved analysis to {analysis_dir}")

                attempts += 1

            except Exception as exc:
                logger.exception(f"❌ error during Phase 2a loop: {exc}")
                attempts += 1

        if attempts == max_attempts:
            logger.info(f"max attempts reached for description {i}")

        # ------------------------------------------------------------------- #
        # end loop over technical description
        # ------------------------------------------------------------------- #

    # final bookkeeping – don’t overwrite scores
    set_protein_sequences(
        sequences=all_generated_sequences,
        scores=[r["scalar"] for r in all_analysis_results],
        score_threshold=score_threshold,
    )

    logger.info("✔ Phase 2a completed")
    print(Fore.GREEN + "Phase 2a completed")

    return all_analysis_results, all_generated_sequences


# guard for manual runs
if __name__ == "__main__":
    raise RuntimeError(
        "Phase 2a is intended to be called from the top-level pipeline."
    )
