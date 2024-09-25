import gc
from typing import List, Dict, Any
import logging
from colorama import Fore
from transformers import EncoderDecoderModel, RobertaTokenizer
import torch
from rdkit import Chem

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()

def checkpoint(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting step: {step_name}")
            print(Fore.CYAN + f"Starting step: {step_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed step: {step_name}")
                print(Fore.GREEN + f"Completed step: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Error in step {step_name}: {str(e)}")
                print(Fore.RED + f"Error in step {step_name}: {str(e)}")
                raise
        return wrapper
    return decorator

class SMILESGenerator:
    """Generates SMILES strings using a pretrained model."""

    def __init__(self, model_name: str = "gokceuludogan/WarmMolGenTwo"):
        self.model_name = model_name
        self.protein_tokenizer = None
        self.mol_tokenizer = None
        self.model = None

    def load_model(self):
        """Load the tokenizer and model."""
        if self.protein_tokenizer is None:
            self.protein_tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            logger.info("Protein tokenizer loaded.")
            print(Fore.YELLOW + "Protein tokenizer loaded.")

        if self.mol_tokenizer is None:
            self.mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
            logger.info("Molecule tokenizer loaded.")
            print(Fore.YELLOW + "Molecule tokenizer loaded.")

        if self.model is None:
            self.model = EncoderDecoderModel.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            if USE_CUDA:
                self.model = self.model.cuda()
                logger.info("Model loaded on CUDA.")
                print(Fore.YELLOW + "Model loaded on CUDA.")
            else:
                logger.info("Model loaded on CPU.")
                print(Fore.YELLOW + "Model loaded on CPU.")

    def unload_model(self):
        """Unload the model and tokenizer to free up memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded.")
            print(Fore.YELLOW + "Model unloaded.")
        if self.protein_tokenizer is not None:
            del self.protein_tokenizer
            self.protein_tokenizer = None
            logger.info("Protein tokenizer unloaded.")
            print(Fore.YELLOW + "Protein tokenizer unloaded.")
        if self.mol_tokenizer is not None:
            del self.mol_tokenizer
            self.mol_tokenizer = None
            logger.info("Molecule tokenizer unloaded.")
            print(Fore.YELLOW + "Molecule tokenizer unloaded.")
        if USE_CUDA:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
            print(Fore.YELLOW + "CUDA cache cleared.")

    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string is valid."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    @checkpoint("SMILES Generation")
    def generate(self, protein_sequence: str, num_sequences: int = 5) -> List[str]:
        """Generate SMILES strings based on a protein sequence."""
        self.load_model()
        try:
            inputs = self.protein_tokenizer(protein_sequence, return_tensors="pt")
            if USE_CUDA:
                inputs = inputs.to('cuda')

            outputs = self.model.generate(
                **inputs,
                decoder_start_token_id=self.mol_tokenizer.bos_token_id,
                eos_token_id=self.mol_tokenizer.eos_token_id,
                pad_token_id=self.mol_tokenizer.eos_token_id,
                max_length=256,
                num_return_sequences=num_sequences,
                do_sample=True,
                top_p=0.95
            )

            generated_smiles = self.mol_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            logger.info("Generated SMILES:")
            valid_smiles = []
            for smiles in generated_smiles:
                is_valid = self.is_valid_smiles(smiles)
                logger.info(f"{smiles}: {is_valid}")
                print(f"{smiles}: {is_valid}")
                if is_valid:
                    valid_smiles.append(smiles)

            return valid_smiles

        except Exception as e:
            logger.error(f"Error in SMILES generation: {str(e)}")
            print(Fore.RED + f"Error in SMILES generation: {str(e)}")
            return []

        finally:
            print(Fore.YELLOW + "Unloading model...")
            self.unload_model()
            torch.cuda.empty_cache()
            gc.collect()
            print(Fore.YELLOW + "CUDA cache cleared.")