Certainly! I'll provide a detailed usage guide for AlphaFold, including how to get the weights and the necessary databases. Here's a step-by-step guide:

## AlphaFold Usage Guide

### Step 1: Obtain AlphaFold Weights and Databases

1. Download the AlphaFold parameters:

   ```bash
   wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
   tar -xvf alphafold_params_2022-12-06.tar
   ```

2. Download and prepare the genetic databases:
   - UniRef90 (requires ~100GB disk space):
     ```bash
     wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
     gunzip uniref90.fasta.gz
     ```
   - BFD (requires ~270GB disk space):
     ```bash
     wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
     tar -xzvf bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
     ```
   - MGnify (requires ~64GB disk space):
     ```bash
     wget https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/mgy_clusters.fa.gz
     gunzip mgy_clusters.fa.gz
     ```
   - PDB70 (requires ~19GB disk space):
     ```bash
     wget http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pdb70_from_mmcif_latest.tar.gz
     tar -xzvf pdb70_from_mmcif_latest.tar.gz
     ```
   - UniRef30 (requires ~86GB disk space):
     ```bash
     wget https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz
     tar -xzvf uniclust30_2018_08_hhsuite.tar.gz
     ```

### Step 2: Set Up the Environment

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   export ALPHAFOLD_DATA_DIR=/path/to/alphafold/data
   export PATH=$PATH:/path/to/alphafold
   ```

### Step 3: Prepare Your Input

1. Create a FASTA file with your protein sequence:
   ```
   >protein_name
   SEQUENCE
   ```

### Step 4: Run AlphaFold

1. Basic command structure:

   ```bash
   python run_alphafold.py \
     --fasta_paths=/path/to/your/input.fasta \
     --output_dir=/path/to/output \
     --data_dir=$ALPHAFOLD_DATA_DIR \
     --uniref90_database_path=$ALPHAFOLD_DATA_DIR/uniref90/uniref90.fasta \
     --mgnify_database_path=$ALPHAFOLD_DATA_DIR/mgnify/mgy_clusters.fa \
     --bfd_database_path=$ALPHAFOLD_DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
     --uniclust30_database_path=$ALPHAFOLD_DATA_DIR/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
     --pdb70_database_path=$ALPHAFOLD_DATA_DIR/pdb70/pdb70 \
     --template_mmcif_dir=$ALPHAFOLD_DATA_DIR/pdb_mmcif/mmcif_files \
     --max_template_date=2022-12-06 \
     --model_preset=monomer \
     --db_preset=full_dbs
   ```

2. For multimer prediction, use:
   ```bash
   --model_preset=multimer
   ```

### Step 5: Analyze Results

1. The output directory will contain:

   - Predicted PDB structures
   - Predicted aligned error (PAE) plots
   - Per-residue confidence scores

2. Use visualization tools like PyMOL or ChimeraX to view the predicted structures.

### Additional Tips:

- Use `--db_preset=reduced_dbs` for faster predictions with slightly lower accuracy.
- Adjust `--max_template_date` to control which PDB templates are used.
- Use `--num_multimer_predictions_per_model` to control the number of predictions for multimers.

Remember to cite the AlphaFold papers when using the software for research:

1. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021).
2. Mirdita, M. et al. ColabFold: Making protein folding accessible to all. Nature Methods 19, 679–682 (2022).

This guide should help you get started with AlphaFold. Always refer to the official documentation for the most up-to-date information and advanced usage options[1][8].

Citations:
[1] https://github.com/google-deepmind/alphafold/activity
[2] https://alphafold.ebi.ac.uk
[3] https://arxiv.org/html/2404.11068v1
[4] https://www.cgl.ucsf.edu/chimerax/docs/user/tools/alphafold.html
[5] https://bcrf.biochem.wisc.edu/2024/03/15/alphafold-a-practical-guide-online-tutorial-ebi-embl/
[6] https://docs.tacc.utexas.edu/software/alphafold/
[7] https://github.com/lipan6461188/AlphaFold-StepByStep
[8] https://github.com/google-deepmind/alphafold/blob/main/README.md
[9] https://www.osc.edu/book/export/html/5857
