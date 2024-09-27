# AutoMol-v2/automol/server/app.py

import os
from datetime import datetime
import threading
import time
import json
import subprocess
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import logging
from pymongo import MongoClient
sys.path.append(os.path.dirname(parent_dir))
# Install eventlet for asynchronous support
# AutoMol-v2/automol/server/main.py

import os
from datetime import datetime
import json
import sys
import logging
from pathlib import Path
import subprocess
import argparse
import socketio  # Import Socket.IO client
from automol.emit_progress import emit_progress
from phase1.phase1_run import run_Phase_1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.phase3_run import run_Phase_3
from phase4.phase4_run import run_Phase_4
from phase5.phase5_run import run_Phase_5
from utils.save_utils import save_json, create_organized_directory_structure
import os
from datetime import datetime
import json
import sys
from pathlib import Path

from flask import jsonify, request, send_from_directory, make_response
from flask_cors import cross_origin
import logging
from pymongo import MongoClient

from socket_manager import socketio, app  # Import socketio and app from socket_manager

# Placeholder for database connection (update with actual connection details)
client = MongoClient('mongodb://localhost:27017/')
db = client['autoprotdb']
molecules_collection = db['molecules']

logger = logging.getLogger(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CONFIG_PATH =   'config.json'

# Pipeline status tracking
pipeline_status = {
    "Phase 1 - Research and Hypothesize": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 2 - Experimentation and Data Collection": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 3 - Analysis and Interpretation": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 4 - Validation and Verification": {"status": "idle", "progress": 0, "message": "IDLE"},
}

notifications = []

@socketio.on('connect')
def handle_connect():
    logger = logging.getLogger(__name__)
    logger.info('Client connected')
    
    # Generate current progress data
    current_progress = {}
    phases = [
        "Phase 1 - Research and Hypothesize",
        "Phase 2 - Experimentation and Data Collection",
        "Phase 3 - Analysis and Interpretation",
        "Phase 4 - Validation and Verification"
    ]
    for phase in phases:
        current_progress[phase] = {
            'progress': pipeline_status[phase]['progress'],
            'message': pipeline_status[phase]['message']
        }
    
    # Generate a sample CLI log message
    log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Server log: Client connected"
    
    # Compile all data into a single object
    initial_data = {
        'pipeline_status': pipeline_status,
        'notifications': notifications,
        'progress': current_progress,
        'status_message': 'Connected to server',
        'cli_log': log_message
    }
    
    # Emit all data in a single event
    emit('initial_state', initial_data)

@socketio.on('disconnect')
def handle_disconnect():
    logger = logging.getLogger(__name__)
    logger.info("Client disconnected")



def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin', '*'))
    response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
    response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def build_actual_response(response, status=200):
    response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin', '*'))
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response, status

@app.route('/api/notifications', methods=['GET', 'OPTIONS'])
def get_notifications():
    logger = logging.getLogger(__name__)
    if request.method == "OPTIONS":
        return build_preflight_response()
    try:
        return jsonify(notifications), 200
    except Exception as e:
        logger.error(f"Error in get_notifications: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/upload_pdb', methods=['POST'])
def upload_pdb():
    logger = logging.getLogger(__name__)
    try:
        if 'file' not in request.files:
            logger.warning("No file part in upload_pdb request.")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file in upload_pdb request.")
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            logger.info(f"PDB file {file.filename} uploaded successfully.")
            return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 201
        else:
            logger.warning("File type not allowed in upload_pdb request.")
            return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        logger.error(f"Error in upload_pdb: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/pdb_files/<filename>', methods=['GET'])
def get_pdb_file(filename):
    logger = logging.getLogger(__name__)
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename), 200
    except FileNotFoundError:
        logger.warning(f"PDB file {filename} not found.")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error in get_pdb_file: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/pipeline_status', methods=['GET'])
@cross_origin()
def get_pipeline_status():
    logger = logging.getLogger(__name__)
    try:
        return jsonify(pipeline_status), 200
    except Exception as e:
        logger.error(f"Error in get_pipeline_status: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdb', 'json'}

def send_idle_message():
    while True:
        socketio.emit('log_message', {'message': 'IDLE: Server is running'})
        socketio.sleep(10)  # Use socketio.sleep for compatibility with eventlet



def merge_config_with_args(config, args):
    """Merge command-line arguments into the configuration dictionary."""
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoMol-v2: Novel molecule generation and analysis pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--input_text", type=str, help="Input text describing the desired molecule function")
    parser.add_argument("--num_sequences", type=int, help="Number of molecule sequences to generate initially")
    parser.add_argument("--optimization_steps", type=int, help="Number of optimization steps to perform")
    parser.add_argument("--score_threshold", type=float, help="Minimum score threshold for accepting generated sequences")
    parser.add_argument("--device", type=str, help="Device to use for computations (cuda or cpu)")
    parser.add_argument("--skip_description_gen", action="store_true", help="Skip the description generation phase")
    return parser.parse_args()

def run_main_pipeline(config_data):
    logger = logging.getLogger(__name__)
    if not config_data:
        logger.warning("No configuration data provided.")
        socketio.emit('pipeline_control_response', {"error": "No configuration data provided."}, namespace='/')
        return
    
    CONFIG_PATH = 'config.json'
    
    # Save configuration to config.json
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info("Configuration saved successfully.")
    emit_progress(phase="System", progress=10, message=f"Configuration saved successfully at {CONFIG_PATH}")
    
    # Use the config_data directly instead of loading from file
    config = config_data
    
    logger.info("Pipeline started successfully.")
    emit_progress(phase="System", progress=20, message="Pipeline Initialized from server.")

    print("Initial Config: ", config)

    # We'll skip merging with command-line arguments as we want to prioritize the passed config_data
    # args = parse_arguments()
    # config = merge_config_with_args(config, args)
    
    print("Final Config: ", config)
    logger.info("Configuration prepared.")
    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=10, message="Configuration prepared.")
    # Load configuration
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config = json.load(config_file)
        print("Config: ", config)
        logger.info("Configuration loaded successfully.")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=5, message="Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Config file not found at {CONFIG_PATH}.")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message=f"Config file not found at {CONFIG_PATH}.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message="Invalid JSON in config file.")
        return

    # Ensure all required keys are present in the config
    required_keys = [
        'base_output_dir', 'input_text', 'num_sequences', 'optimization_steps',
        'score_threshold', 'device', 'output_paths', 'phase1', 'phase2a',
        'phase2b', 'phase3', 'phase4', 'mongodb'
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required configuration key(s): {', '.join(missing_keys)}")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message=f"Missing required configuration key(s): {', '.join(missing_keys)}")
        return

    # Set up logging after verifying base_output_dir
    log_file_path = Path(config['base_output_dir']) / config['output_paths']['log_file']
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create log directory at {log_file_path.parent}: {e}")
        emit_progress(phase="System", progress=0, message=f"Failed to create log directory: {e}")
        return

    # Reconfigure logging to include the new log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path)
        ]
    )
    logger = logging.getLogger(__name__)

    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=15, message="Creating organized directory structure...")
    try:
        run_dir, phase_dirs, log_file_path = create_organized_directory_structure(config['base_output_dir'])
        if not run_dir or not phase_dirs:
            raise ValueError("Failed to create directory structure")
        logger.info(f"Organized directory structure created at {run_dir}.")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=20, message=f"Organized directory structure created at {run_dir}.")
    except Exception as e:
        logger.error(f"Failed to create directory structure: {str(e)}")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message=f"Failed to create directory structure: {str(e)}")
        return

    # Update config with run_dir and phase directories
    config['run_dir'] = run_dir
    config['phase_dirs'] = phase_dirs
    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=25, message="Updated config with run_dir and phase directories.")

    try:
        if config.get('skip_description_gen', False):
            phase1_results = {
                'technical_description': config.get('input_text', '')
            }
            save_json(phase1_results, Path(run_dir) / "phase1_results.json")
            logger.info("Phase 1 skipped. Results saved successfully.")
            emit_progress(phase="Phase 1 - Skipped Research and Hypothesize", progress=100, message="Phase 1 skipped. Results saved successfully.")
        else:
            # Phase 1: Generate Hypothesis
            logger.info("Starting Phase 1: Generate Hypothesis")
            emit_progress(phase="Phase 1 - Research and Hypothesize", progress=30, message="Starting Phase 1: Generate Hypothesis")
            
            if 'phase1' not in config:
                logger.error("Phase 1 configuration is missing.")
                emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message="Phase 1 configuration is missing.")
                return

            phase1_results = run_Phase_1(config['phase1'])
            save_json(phase1_results, Path(run_dir) / "phase1_results.json")
            logger.info("Phase 1 results saved successfully.")
            emit_progress(phase="Phase 1 - Research and Hypothesize", progress=40, message="Phase 1 results saved successfully.")
        

        # Phase 2a: Generate and Optimize Proteins
        logger.info("Starting Phase 2a: Generate and Optimize Proteins")
        phase2a_config = config['phase2a'].copy()
        phase2a_config.update({
            'technical_descriptions': [phase1_results['technical_description']],
            'predicted_structures_dir': os.path.join(run_dir, phase_dirs['phase2a'][0]),  # 'generated_sequences'
            'results_dir': os.path.join(run_dir, 'phase2a'),
            'num_sequences': config['num_sequences'],
            'optimization_steps': config['optimization_steps'],
            'score_threshold': config['score_threshold']
        })

        try:
            phase2a_results, all_generated_sequences = run_Phase_2a(**phase2a_config)
            save_json(phase2a_results, Path(run_dir) / "phase2a_results.json")
            logger.info("Phase 2a results saved successfully.")
            emit_progress(phase="Phase 2 - Experimentation and Data Collection", progress=50, message="Phase 2a results saved successfully.")

            # Extract protein sequences from phase2a_results
        except TypeError as e:
            logger.error(f"Error in run_Phase_2a: {str(e)}")
            logger.error(f"Phase 2a config: {phase2a_config}")
            emit_progress(phase="Phase 2 - Experimentation and Data Collection", progress=50, message=f"Error in Phase 2a: {str(e)}")
            raise

        # Phase 2b: Generate and Optimize Ligands
        logger.info("Starting Phase 2b: Generate and Optimize Ligands")
        phase2b_config = config['phase2b'].copy()

        phase2b_config.update({
            'predicted_structures_dir': os.path.join(run_dir, phase_dirs['phase2b'][0]),  # 'ligands'
            'results_dir': os.path.join(run_dir, 'phase2b'),
            'num_sequences': config['num_sequences'],
            'optimization_steps': config['optimization_steps'],
            'score_threshold': config['score_threshold'],
            'protein_sequences': [result['sequence'] for result in phase2a_results]
        })

        try:
            phase2b_results = run_Phase_2b(**phase2b_config)
            save_json(phase2b_results, Path(run_dir) / "phase2b_results.json")
            logger.info("Phase 2b results saved successfully.")
            emit_progress(phase="Phase 2 - Experimentation and Data Collection", progress=75, message="Phase 2b results saved successfully.")
        except TypeError as e:
            logger.error(f"Error in run_Phase_2b: {str(e)}")
            logger.error(f"Phase 2b config: {phase2b_config}")
            emit_progress(phase="Phase 2 - Experimentation and Data Collection", progress=75, message=f"Error in Phase 2b: {str(e)}")
            raise


        # Phase 3: Simulation
        logger.info("Starting Phase 3: Simulation")
        phase3_config = config['phase3']
        phase3_config.update({
            'protein_results': phase2a_results,
            'ligand_results': phase2b_results,
            'output_dir': os.path.join(run_dir, "phase3"),
        })
        phase3_results = run_Phase_3(**phase3_config)
        save_json(phase3_results, Path(run_dir) / "phase3" / "phase3_results.json")
        logger.info("Phase 3 results saved successfully.")
        emit_progress(phase="Phase 3 - Analysis and Interpretation", progress=90, message="Phase 3 results saved successfully.")

        # Phase 4: Final Analysis and Reporting
        logger.info("Starting Phase 4: Final Analysis and Reporting")
        phase4_config = config['phase4']
        phase4_config.update({
            'simulation_results': phase3_results,
            'output_dir': os.path.join(run_dir, "phase4")
        })
        phase4_results = run_Phase_4(phase3_results, config = phase4_config)
        save_json(phase4_results, Path(run_dir) / "phase4_results.json")
        logger.info("Phase 4 results saved successfully.")
        emit_progress(phase="Phase 4 - Validation and Verification", progress=100, message="Phase 4 results saved successfully.")

        # Save All Results Consolidated
        all_results = {
          'phase1': phase1_results,
          'phase2a': phase2a_results,
          'phase2b': phase2b_results,
          'phase3': phase3_results,
          'phase4': phase4_results
        }
        save_json(all_results, Path(run_dir) / "final_results.json")
        logger.info("All phase results saved successfully.")
        emit_progress(phase="Phase 4 - Validation and Verification", progress=100, message="All phase results saved successfully.")

        # Phase 5: Final Report and Decision Making Process
        logger.info("Starting Phase 5: Decision Making Process")
        phase5_config = config['phase5']
        base_output_dir = config['base_output_dir']
        phase5_config.update({
            'base_output_dir': base_output_dir  
        })
        phase5_results = run_Phase_5(phase5_config)
        save_json(phase5_results, Path(run_dir) / "phase5_results.json")
        logger.info("Phase 5 results saved successfully.")
        emit_progress(phase="Phase 5 - Final Report and Decision Making Process", progress=100, message="Phase 5 results saved successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        emit_progress(phase="System", progress=0, message=f"An unexpected error occurred: {e}")

# Modify the handle_pipeline_control function
@socketio.on('pipeline_control')
def handle_pipeline_control(data):
    action = data.get('action')
    if action == 'start':
        config_data = data.get('config')
        emit_progress(phase="System", progress=0, message="Pipeline control received in server.")
        socketio.start_background_task(run_main_pipeline, config_data)
    elif action == 'stop':
        # Implement stop logic if needed
        logger.info("Stop request received. Implementing stop logic.")
        emit('pipeline_control_response', {"message": "Stop request received"})
    else:
        logger.warning("Invalid action received in pipeline_control.")
        emit('pipeline_control_response', {"error": "Invalid action"})
        
        
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)