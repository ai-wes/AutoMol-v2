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
import main
# Install eventlet for asynchronous support


# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://dashboard.automol-ai.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    },
    r"/socket.io/*": {
        "origins": ["http://localhost:3000", "https://dashboard.automol-ai.com"],
        "supports_credentials": True
    }
}, supports_credentials=True)

# Setup Flask-SocketIO with eventlet
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "https://dashboard.automol-ai.com"], 
                     allow_credentials=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder for database connection (update with actual connection details)
client = MongoClient('mongodb://localhost:27017/')
db = client['autoprotdb']
molecules_collection = db['molecules']

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CONFIG_PATH = Path(__file__).parent.parent / 'config.json'

# Pipeline status tracking
pipeline_status = {
    "Phase 1 - Research and Hypothesize": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 2 - Experimentation and Data Collection": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 3 - Analysis and Interpretation": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 4 - Validation and Verification": {"status": "idle", "progress": 0, "message": "IDLE"},
}

notifications = []

def emit_progress(phase=None, progress=None, message=None):
    logger.info(f"Emitting progress: phase={phase}, progress={progress}, message={message}")
    socketio.emit('progress_update', {
        'phase': phase,
        'progress': progress,
        'message': message
    })

@socketio.on('connect')
def handle_connect():
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
    logger.info("Client disconnected")


@socketio.on('pipeline_control')
def handle_pipeline_control(data):
    action = data.get('action')
    if action == 'start':
        config_data = data.get('config')
        emit_progress(phase="System", progress=0, message="Pipeline control received in server.")
        socketio.start_background_task(run_pipeline, config_data)
    else:
        logger.warning("Invalid action received in pipeline_control.")
        emit('pipeline_control_response', {"error": "Invalid action"})

def run_pipeline(config_data):
    try:
        if not config_data:
            logger.warning("No configuration data provided.")
            socketio.emit('pipeline_control_response', {"error": "No configuration data provided."}, namespace='/')
            return
        
        
        # Save configuration to config.json
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info("Configuration saved successfully.")
        emit_progress(phase="System", progress=10, message="Configuration saved successfully at " + str(CONFIG_PATH))
        
        # Execute the main script in a non-blocking manner
        socketio.start_background_task(run_main_pipeline)
        logger.info("Pipeline started successfully.")
        emit_progress(phase="System", progress=20, message="Pipeline Initialized from server.")
    except Exception as e:
        logger.error(f"Error in run_pipeline: {e}")
        emit_progress(phase="System", progress=0, message="Internal Server Error.")

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
    if request.method == "OPTIONS":
        return build_preflight_response()
    try:
        return jsonify(notifications), 200
    except Exception as e:
        logger.error(f"Error in get_notifications: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/upload_pdb', methods=['POST'])
def upload_pdb():
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

# Start the idle message thread when the server starts
socketio.start_background_task(send_idle_message)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)