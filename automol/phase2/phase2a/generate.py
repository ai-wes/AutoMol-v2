import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large",
                                             torch_dtype=torch.float16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("hugohrban/progen2-large", trust_remote_code=True)
logger.info("Model and tokenizer loaded")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Set pad_token to eos_token")

# Define valid amino acids
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def extract_amino_acids(sequence):
    """Extract only valid amino acids from the sequence."""
    return ''.join(char for char in sequence if char in VALID_AMINO_ACIDS)

async def generate_protein_sequence(input_text, num_sequences=1):
    """Generate protein sequences based on the input description."""
    model.eval()
    max_length = 256
    input_text = f"Input Text: {input_text} Sequence:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    logger.info(f"Generating {num_sequences} sequence(s) for input: {input_text}")

    try:
        with torch.no_grad():
            output = await asyncio.to_thread(
                model.generate,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_sequences,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_sequences = []
        for i in range(num_sequences):
            generated_sequence = tokenizer.decode(output[i], skip_special_tokens=True)
            sequence = generated_sequence.split("Sequence:")[-1].strip()
            extracted_sequence = extract_amino_acids(sequence)
            generated_sequences.append(extracted_sequence)
            logger.info(f"Generated and extracted sequence {i+1}: {extracted_sequence[:50]}...")  # Log first 50 amino acids
            print(f"Extracted Sequence {i+1}:", extracted_sequence)

        return generated_sequences
    except Exception as e:
        logger.error(f"Error in generate_protein_sequence: {str(e)}")
        return None