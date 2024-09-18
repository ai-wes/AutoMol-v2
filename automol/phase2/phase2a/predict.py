
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import os


import subprocess
import platform
import asyncio
import aioconsole
from esm.sdk.api import ESMProteinError
import os
import asyncio
import logging
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, DSSP
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from pymol import cmd
import pymol
import os
import asyncio
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import truncnorm
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm3_model:ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")  # or "cpu"




def predict_protein_function(sequence):
    """Predict the function score of a given protein sequence."""
    try:
        print(f"Received sequence for prediction: {sequence}")
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence = ''.join(char.upper() for char in sequence if char.upper() in valid_aa)
        print(f"Cleaned sequence: {sequence}")
        if not sequence:
            print("Empty or invalid sequence for protein function prediction")
            logger.error("Empty or invalid sequence for protein function prediction")
            return 0.5

        analysis = ProteinAnalysis(sequence)
        molecular_weight = analysis.molecular_weight()
        aromaticity = analysis.aromaticity()
        instability_index = analysis.instability_index()
        isoelectric_point = analysis.isoelectric_point()

        norm_weight = truncnorm.cdf((molecular_weight - 25000) / 10000, -2, 2)
        norm_aromaticity = aromaticity
        norm_instability = 1 - truncnorm.cdf((instability_index - 40) / 10, -2, 2)
        norm_isoelectric = truncnorm.cdf((isoelectric_point - 7) / 2, -2, 2)
        aa_count = {aa: sequence.count(aa) for aa in valid_aa}
        total_aa = len(sequence)
        composition_balance = 1 - sum(abs(count/total_aa - 0.05) for count in aa_count.values()) / 2

        weights = [0.25, 0.15, 0.25, 0.15, 0.2]
        score = sum(w * v for w, v in zip(weights, [norm_weight, norm_aromaticity, norm_instability, norm_isoelectric, composition_balance]))
        print(f"Predicted function score for sequence: {score}")
        logger.info(f"Predicted function score for sequence: {score}")
        return max(0, min(1, score))
    except Exception as e:
        print(f"Error in predict_protein_function: {str(e)}")
        logger.error(f"Error in predict_protein_function: {str(e)}")
        return 0.5
    

    
    
def predict_properties(sequence):
    """Predict various properties of a protein sequence."""
    try:
        analysis = ProteinAnalysis(sequence)
        properties = {
            "molecular_weight": analysis.molecular_weight(),
            "aromaticity": analysis.aromaticity(),
            "instability_index": analysis.instability_index(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy": analysis.gravy(),
            "secondary_structure_fraction": analysis.secondary_structure_fraction()
        }
        logger.info(f"Predicted properties for sequence: {properties}")
        return properties
    except Exception as e:
        logger.error(f"Error in predict_properties: {str(e)}")
        return {}



import pdb

def create_3d_model_esm3(protein_pdb, prompt_string, output_dir):
    """Create a 3D model visualization using PyMOL and provide instructions to open it."""
    print(f"Starting 3D model creation for {protein_pdb}")
    pymol.finish_launching(['pymol', '-cq'])  # '-cq' for command line only and quiet mode
    
    try:
        print(f"Loading PDB file: {protein_pdb}")
        cmd.load(protein_pdb)
        
        print("Applying visualization settings")
        cmd.spectrum("count", "rainbow", "all")
        cmd.show("cartoon")
        cmd.bg_color("black")
        
        print("Saving image")
        image_filename = os.path.join(output_dir, f"{prompt_string}_model.png")
        cmd.png(image_filename)
        print(f"Image saved as {image_filename}")
        
        print("Saving PyMOL session")
        session_filename = os.path.join(output_dir, f"{prompt_string}_model.pse")
        cmd.save(session_filename)
        print(f"PyMOL session saved as {session_filename}")
        
        print("Closing PyMOL")
        cmd.quit()
        
        print("\nTo view the 3D structure:")
        print(f"1. Open PyMOL")
        print(f"2. In PyMOL, go to File -> Open")
        print(f"3. Navigate to and select this file: {os.path.abspath(session_filename)}")
        
        
        try:
            user_input = input("Press Enter when you're ready to continue, or type 'skip' to move on: ")
            if user_input.lower() == 'skip':
                print("User chose to skip opening PyMOL.")
            else:
                print("User acknowledged PyMOL instructions.")
        except Exception:
            print("No input received. Continuing.")
        
    except pymol.CmdException as e:
        print(f"PyMOL command failed - {e}")
        
        

def predict_structure(sequence):
    """Predict the structure of a protein sequence using ESM-3."""
    try:
        protein = ESMProtein(sequence=sequence)
        config = GenerationConfig(track="structure", num_steps=8)
        result = esm3_model.generate(protein, config)

        logger.info(f"Structure generation result type: {type(result)}")
        logger.info(f"Structure generation result attributes: {dir(result)}")

        # Save the PDB file
        output_dir = "predicted_structures"
        os.makedirs(output_dir, exist_ok=True)
        pdb_file = os.path.join(output_dir, f"{sequence[:10]}_structure.pdb")
        result.to_pdb(pdb_file)

        logger.info(f"Structure predicted and saved to {pdb_file}")
        return pdb_file
    except ESMProteinError as e:
        logger.error(f"ESM Protein Error in predict_structure: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in predict_structure: {str(e)}")
        return None

def esm3_refinement(sequence):
    try:
        protein = ESMProtein(sequence=sequence)
        refined_protein = esm3_model.generate(
            protein, 
            GenerationConfig(track="sequence", num_steps=8, temperature=0.7)
        )
        
        if isinstance(refined_protein, (ESMProtein, str)):
            refined_sequence = (
                refined_protein.sequence 
                if isinstance(refined_protein, ESMProtein) 
                else refined_protein
            )
        else:
            logger.warning(f"Unexpected refined protein type: {type(refined_protein)}. Using original sequence.")
            refined_sequence = sequence
        
        logger.info(f"Refined sequence: {refined_sequence[:50]}...")
        return refined_sequence
    except ESMProteinError as e:
        logger.warning(f"ESM Protein Error in ESM3 refinement: {str(e)}. Using original sequence.")
        return sequence
    except Exception as e:
        logger.error(f"Unexpected error in ESM3 refinement: {str(e)}. Using original sequence.")
        return sequence
    
        

def run_prediction_pipeline(sequences, output_dir):
    try:
        results = []
        for i, sequence in enumerate(sequences):
            if isinstance(sequence, dict):
                sequence = sequence['optimized_sequence']
            logger.info(f"Processing sequence {i+1}/{len(sequences)}: {sequence}")
            
            try:
                score = predict_protein_function(sequence)
                properties = predict_properties(sequence)
                pdb_file = os.path.join(output_dir, f"{sequence[:10]}_structure.pdb") 
                
                pdb_file = predict_structure(sequence)
            except Exception as e:
                logger.error(f"Error processing sequence {i+1}: {str(e)}")
                score, properties, pdb_file = None, None, None

            results.append({
                "sequence": sequence,
                "score": score,
                "properties": properties,
                "pdb_file": pdb_file
            })

        return results
    except Exception as e:
        logger.error(f"Error in run_prediction_pipeline: {str(e)}")
        logger.error(f"Sequences: {sequences}")
        logger.error(f"Output directory: {output_dir}")
        return None


        
if __name__ == "__main__":
    sequences = ["HMPYHFQGTNWFGCIASVRNGMRTKWFELSFAWYERSMMYQWNKVWWTKFYWYVWFLWQKLWQQMLYWAVGWRHFHPTRHPSFHVKKFFVQKAVSFAICHPWYKWKHPQQRFIQGRISMQNPWHPMNVNTFHLDWKLIFKYQNLWIGNEWMLITWKDWRFNPYWIWKSKLGIWWWYWYTWFRHMFRNAFQHMVTQYYINNFYRLMMFVLDSFKELITYWFRFRKHGQGRCNWQTAYWFYKYTFDHERRVGPIS",
        "KFRWSWNRYYKWNWQTWWNTQWFVQMWTTIFFYWSFMMGRTWGWRTRYSYAFFFTFMLLYKIHTWAEMWPMIYWTCGAMMYQHFWVFSWWPHKLYIWQRIWRHRKMRYWIHDWWYQCFYKHVSNWQLLREKSTFKGNNFFPWNWAVFRRPCYRPRHWIPERMVVRPYHDNVFFRKLLRTDQKSIRYNSLVRFCIKHLYFKMQNPNGAVPYFNHFWYWYYTFTQWMKIVYWHYWKLCVFDKFWPNYLHMPAQFSAII",
        "FKVFTRVFFWTWKLYEYEFHAHWWTWPGHYYLYWYRVIKEKEWWNNFTLFWRMWGWFPLRRWKPPWKSWLYWWTQQMPCSFILFWRFSFRAAMMWHMRGENRARLKPIMTYWQVWDQVWEAWRIRPWYKQCANHDRFWWEPHSPLMRFDWWQIIFFKPRNTSLNFHWRYSKAQYNPLFKWYYLTWWFVMVNWQHVCYFAYQKFGKWCHYWIHLTFWIAMPTNEVLHCRNAHFIRYRMFWWRCITARNNQISHD",
        "LLRTQVRMIPNYAAWYMAHMWWYIVMGRDWFFAWYRQFVLGWPRIRYKPKGLHRYIQHNNKHTWFSQHFTFQCKWWWKKHWPWPRLHMWKWWLYTNYVFMIMDVGATWMNHEKYTLACRHWDWMKKIKWWDNYNTRVWTMQFFPGFFKLTYWIIQNWWHWLAWRYQWFQIVYPFTWWQYWVSQFLENPWPRDAWRHRGSFLPHISRSWFWNTWGGFVRKMVFNMKNIWRFSSVFDRIMQLWPWWTLSGRFSFIPFK",
        "FGTLKVQMYWWNGWKGAWWLYFRTHRRHGKSKYFFFTWWFPKWWNSPRQPMYSWWAPIFCRYSMEGNVMEYNWAKWREYEWDQWFWLEKLIGFEVFNARNHLRNRQWKSNFQPWSLEHPYWLDFFRFGTIMYWWSHWCFWIWFWRMNYTNYQQHFVWTTHPPPWPLQWTQMFQIYWEVAIKFRLMEHRMHFKWWRATPWIVMARKNMGFVSIYKYWRKVRVWVRTNFWHLYFAMRMSFKYLRHHCEWIWRWIHTQA"
    ]
    asyncio.run(run_prediction_pipeline(sequences, "predicted_structures"))
    