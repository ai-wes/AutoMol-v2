# AutoProt Framework: Advanced Protein and Ligand Design System

The AutoProt Framework is a multi-phase computational pipeline designed for the automated generation, analysis, and testing of novel molecular structures. This system integrates machine learning techniques, molecular dynamics simulations, and bioinformatics tools to create and optimize molecules tailored to specific biological functions.

## Table of Contents

- [Features](#features)
- [Pipeline](#pipeline)
- [Setup](#setup)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

## Pipeline Overview

The framework consists of four main phases:

1. **Natural Language instruction to Technical Description Generation (Phase 1)**: This initial phase uses LLMs and a molecular database to generate and refine detailed "technical descriptions" for protein-ligand pairs. It employs an iterative process to ensure the descriptions are feasible, novel, and aligned with the user's input.

2. **Molecule Generation and Optimization (Phase 2)**: Split into two parallel tracks:

   - **Protein Generation (Phase 2a)**: Utilizes NL instruction to sequence generation model (ProGen) to generate protein sequences. The output is passed to the optimization pipeline where it undergoes numerous optimization techniques (monte carlo, gradient descent, replica exchange, etc.) to generate a large number of high quality sequences. The final sequence is then passed to a structure and function prediction pipeline (ESM3) to predict the 3D structure of the generated sequences. Lastly, the protein is subjected to molecular dynamics simulation using OpenMM to ensure stability and functional integrity.

   - **Ligand Generation (Phase 2b)**: The Ligand Generation pipeline is similar to the protein pipeline. It uses a SMILESGenerator(Caption2Smiles) to generate a large number of SMILES strings. These strings are then passed to an Optimizer pipeline that uses a custom trained GNN(Telomerase-focused GNN) to optimize the ligands. The final ligand is then passed to a StructurePredictor(ESM3) to predict the 3D structure of the generated ligands. It also undergoes structural prediction, ensemble docking simulations, and analysis and scoring.

3. **Comprehensive Analysis and Simulation (Phase 3)**: This final phase conducts in-depth analysis of the generated molecules, including structural visualization, cellular environment simulation using a digital twin approach, and virtual high-throughput screening.

4. **Virtual Lab Simulation and Automation (Phase 4) (planned)**:
   This phase simulates laboratory experiments and automates the process of testing and validating the generated protein-ligand pairs in a virtual environment.

   - **Virtual Assay Development**
   - **Automated Experimental Design**
   - **In Silico Cell Culture Simulation**
   - **Virtual Safety and Toxicity Testing**
   - **Automated Data Analysis and Reporting**
   - **Integration with Wet Lab Validation**

The system aims to accelerate the molecular discovery process by automating the design and initial testing of potential protein-ligand interactions, providing researchers with high-quality candidates for further experimental validation.

# Setup

## Prerequisites

Ensure the following software and dependencies are installed and configured:

- Python 3.8 or later
- CUDA (if using GPU acceleration for PyTorch)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/autoprot-framework.git
   cd autoprot-framework
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

This will install the following main dependencies:

- fair-esm and esm for protein language models
- biopython for biological sequence analysis
- transformers and torch for machine learning models
- rdkit for cheminformatics
- langchain and related packages for language model integration
- mdtraj for molecular dynamics trajectory analysis
- openmmpi and pdbfixer for molecular dynamics simulations
- nglview for molecular visualization

3. Ensure CUDA is properly set up if you plan to use GPU acceleration for PyTorch.

4. Some packages may require additional system-level dependencies. Please refer to their respective documentation for any extra setup steps.

## Usage

1. Run the Pipeline: Execute the main script to start the full pipeline:

   ```bash
   python main.py
   ```

2. Monitor Directory for Changes (Optional): Use the provided PowerShell script or inotifywait-based script to automatically trigger rsync transfers when files are saved in the designated directories.

## Pipeline Steps

1. **Generate Protein Sequence with ProGen**:

   - Generates protein sequences based on provided descriptions using ProGen's pretrained models.

2. **Predict Protein Structure with ESM3**:

   - Uses ESM3 to predict the 3D structure of the generated sequences, saving the output in PDB format.

3. **Run Molecular Dynamics Simulation with OpenMM**:

   - Loads the predicted structures and performs energy minimization and dynamics simulation to refine the protein models.

4. **Automate Mutagenesis and Screening with PyRosetta**:

   - Applies mutagenesis and automated relaxation to optimize the protein structures for stability and function.

5. **Perform Docking Simulations with HADDOCK**:

   - Simulates docking of the optimized proteins with potential ligands or other proteins to explore interactions and binding affinities.

6. **Visualize Final Structures with PyMOL**:
   - Generates visual representations of the final optimized structures, providing insight into their geometrical and functional properties.

## Configuration

- **AlphaFold Model Path**: Ensure the AlphaFold model weights and configuration files are correctly set in the script (`alphafold_model_path`).
- **HADDOCK Configuration**: Verify that `dock.cfg` and any other necessary configuration files for HADDOCK are in place and properly configured.
- **Directory Paths**: Update the script with the correct paths for your local and remote directories for rsync operations.

## Troubleshooting

- **Permission Issues**: Ensure all scripts have the necessary execution permissions and that SSH keys are correctly set up for secure access between machines.
- **Dependency Errors**: Verify that all dependencies are installed and compatible with your system's architecture and Python version.
- **Resource Limitations**: Adjust resource allocations or run individual steps in isolation if encountering performance issues due to hardware limitations.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Accreditation

This project was developed with contributions from various individuals and organizations. We would like to acknowledge and thank the following:

1. **Conceptualization, Design, and Implementation**:

   - Wes Lagarde

2. **Institutional Accreditation**:

   - Uniprot, PDB, ChEMBL, EMBL-EBI, MDAnalysis, PyMOL

3. **Open Source Libraries and Tools**:

   - ProGen
   - ESM3
   - OpenMM
   - PyMOL
   - HuggingFace

4. **Funding Sources**:

   - [List of funding agencies or grants]

5. **External Collaborators**:

   - [Names of external collaborators or advisors]

6. **Computational Resources**:
   - [Names of computing facilities or cloud services used]

We are grateful for the support and contributions of all individuals and organizations involved in this project. Their expertise and resources have been invaluable in the development and success of this pipeline.

If you use this pipeline in your research, please cite our work and the relevant tools and libraries mentioned above.
