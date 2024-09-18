import os
from typing import List

_protein_sequences: List[str] = []
_sequences_ready = False
def set_protein_sequences(sequences: List[str], scores: List[float], score_threshold: float):
    """
    Set the protein sequences that meet the score threshold and mark them as ready.

    Args:
        sequences (List[str]): List of protein sequences.
        scores (List[float]): List of scores corresponding to the sequences.
        score_threshold (float): The minimum score required to set a sequence.
    """
    global _protein_sequences, _sequences_ready
    _protein_sequences = [seq for seq, score in zip(sequences, scores) if score >= score_threshold]
    _sequences_ready = True
    print(f"Protein sequences meeting the score threshold of {score_threshold} have been set and are ready.")
    print(f"Number of sequences set: {len(_protein_sequences)}")

def get_protein_sequences() -> List[str]:
    """
    Retrieve the protein sequences.

    Returns:
        List[str]: List of protein sequences.
    """
    if not _sequences_ready:
        print("Protein sequences are not yet set.")
        return []
    print(f"Retrieving {_protein_sequences.__len__()} protein sequences.")
    return _protein_sequences