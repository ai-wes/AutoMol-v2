import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import threading
import time
from datetime import datetime

from bson import ObjectId
import json
import subprocess
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import os
import logging
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://dashboard.automol-ai.com"], "supports_credentials": True}})

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

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

# Pipeline status tracking
pipeline_status = {
    "Phase 1 - Research and Hypothesize": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 2 - Experimentation and Data Collection": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 3 - Analysis and Interpretation": {"status": "idle", "progress": 0, "message": "IDLE"},
    "Phase 4 - Validation and Verification": {"status": "idle", "progress": 0, "message": "IDLE"},
}

class SocketIOHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        print(f"Emitting log: {log_entry}")  # Debug print
        socketio.emit('log_message', {'message': log_entry})




notifications = []

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('initial_state', {'pipeline_status': pipeline_status, 'notifications': notifications})
    
@socketio.on('disconnect')
def handle_disconnect():
    print("Client connected")
    emit('initial_state', {'pipeline_status': pipeline_status, 'notifications': notifications})
    emit('status_message', {'message': 'Connected to server'})


def emit_progress(phase=None, progress=None):
    socketio.emit('progress_update', {
        'phase': phase,
        'progress': progress
    })

def emit_message(message):
    socketio.emit('message', {'message': message})
    
def emit_notification(notification):
    socketio.emit('notification', {'notification': notification})
    
def emit_status(status):
    socketio.emit('status', {'status': status})

# The socketio instance is already defined at the top of the file,
# so there's no need to add it here. The function is correctly using
# the global socketio instance.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdb', 'json'}

@app.route('/api/pipeline_status', methods=['GET', 'OPTIONS'])
def get_pipeline_status():
    try:
        return jsonify(pipeline_status), 200
    except Exception as e:
        logger.error(f"Error in get_pipeline_status: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/key-metrics', methods=['GET', 'OPTIONS'])
def get_key_metrics():
    try:
        total_sequences = molecules_collection.count_documents({})
        passed_sequences = molecules_collection.count_documents({"pass_fail": "Pass"})
        failed_sequences = molecules_collection.count_documents({"pass_fail": "Fail"})
        return jsonify({
            "total_sequences": total_sequences,
            "passed_sequences": passed_sequences,
            "failed_sequences": failed_sequences
        }), 200
    except Exception as e:
        logger.error(f"Error in get_key_metrics: {e}")
        return jsonify({"error": "Internal Server Error"}), 500





@app.route('/api/notifications', methods=['GET', 'OPTIONS'])
def get_notifications():
    try:
        return jsonify(notifications), 200
    except Exception as e:
        logger.error(f"Error in get_notifications: {e}")
        return jsonify({"error": "Internal Server Error"}), 500




@app.route('/api/graph_data', methods=['GET', 'OPTIONS'])
def get_graph_data():
    try:
        # Example graph data
        data = {
            "labels": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "values": [10, 20, 30]
        }
        return jsonify(data), 200
    except KeyError as e:
        logger.error(f"Missing key in graph_data: {e}")
        return jsonify({"error": "Bad Request"}), 400
    except Exception as e:
        logger.error(f"Error in get_graph_data: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/api/configuration', methods=['POST'])
def save_configuration():
    try:
        config = request.json
        # Save the configuration to a file
        with open('./automol/config.json', 'w') as f:
            json.dump(config, f)
        return jsonify({"message": "Configuration saved successfully"}), 200
    except Exception as e:
        logger.error(f"Error in save_configuration: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/pipeline-control', methods=['POST'])
def control_pipeline():
    try:
        data = request.json
        action = data.get('action')
        if action == 'start':
            # Execute the main script
            subprocess.Popen([sys.executable, 'main.py'], 
                             cwd=str(Path(__file__).parent.parent))
            return jsonify({"message": "Pipeline started successfully"}), 200
        else:
            return jsonify({"error": "Invalid action"}), 400
    except Exception as e:
        logger.error(f"Error in control_pipeline: {e}")
        return jsonify({"error": "Internal Server Error"}), 500




@app.route('/api/molecules/<id>', methods=['DELETE'])
def delete_molecule(id):
    try:
        result = molecules_collection.delete_one({'_id': ObjectId(id)})
        if result.deleted_count == 1:
            emit('moleculeDeleted', id, broadcast=True)
            return jsonify({"message": "Molecule deleted successfully"}), 200
        else:
            return jsonify({"error": "Molecule not found"}), 404
    except Exception as e:
        logger.error(f"Error in delete_molecule: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/api/upload_pdb', methods=['POST'])
def upload_pdb():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 201
        else:
            return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        logger.error(f"Error in upload_pdb: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/pdb_files/<filename>', methods=['GET'])
def get_pdb_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename), 200
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error in get_pdb_file: {e}")
        return jsonify({"error": "Internal Server Error"}), 500




def send_idle_message():
    while True:
        socketio.emit('log_message', {'message': 'IDLE: Server is running'})
        time.sleep(10)  # Send message every 10 seconds

# Start the idle message thread when the server starts
idle_thread = threading.Thread(target=send_idle_message, daemon=True)
idle_thread.start()


    
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)