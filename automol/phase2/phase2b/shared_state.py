import os
from typing import List

_protein_sequences: List[str] = []
_sequences_ready = False

def set_protein_sequences(sequences: List[str]):
    """
    Set the protein sequences and mark them as ready.

    Args:
        sequences (List[str]): List of protein sequences.
    """
    global _protein_sequences, _sequences_ready
    _protein_sequences = sequences
    _sequences_ready = True
    print("Protein sequences have been set and are ready.")

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