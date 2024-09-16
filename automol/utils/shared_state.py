import asyncio
from typing import List

_protein_sequences: List[str] = []
_sequences_ready = asyncio.Event()

async def set_protein_sequences(sequences: List[str]):
    global _protein_sequences
    _protein_sequences = sequences
    _sequences_ready.set()

async def get_protein_sequences() -> List[str]:
    await _sequences_ready.wait()
    return _protein_sequences