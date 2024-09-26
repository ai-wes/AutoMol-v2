# AutoMol-v2/automol/server/app.py
import os
import time
import json
import logging
from flask import Flask, jsonify, request, send_from_directory, cross_origin
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": "https://chat.build-a-bf.com"}}, supports_credentials=True)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="https://chat.build-a-bf.com", async_mode='eventlet')

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

notifications = []

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    emit('initial_state', {'pipeline_status': pipeline_status, 'notifications': notifications})
    emit('status_message', {'message': 'Connected to server', 'status': 'IDLE'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdb', 'json'}

@app.route('/api/pipeline_status', methods=['GET'])
@cross_origin()
def get_pipeline_status():
    try:
        return jsonify(pipeline_status), 200
    except Exception as e:
        logger.error(f"Error in get_pipeline_status: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/key-metrics', methods=['GET'])
@cross_origin()
def get_key_metrics():
    try:
        total_sequences = molecules_collection.count_documents({})
        passed_sequences = molecules_collection.count_documents({"pass_fail": "Pass"})
        failed_sequences = molecules_collection.count_documents({"pass_fail": "Fail"})
        return jsonify({
            "total_sequences": 3,
            "passed_sequences": 6,
            "failed_sequences": 7
        }), 200
    except Exception as e:
        logger.error(f"Error in get_key_metrics: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/notifications', methods=['GET'])
@cross_origin()
def get_notifications():
    try:
        return jsonify(notifications), 200
    except Exception as e:
        logger.error(f"Error in get_notifications: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/graph_data', methods=['GET'])
@cross_origin()
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

@app.route('/api/molecules/<id>', methods=['DELETE'])
@cross_origin()
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
@cross_origin()
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
@cross_origin()
def get_pdb_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename), 200
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error in get_pdb_file: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/configuration', methods=['POST'])
@cross_origin()
def save_configuration():
    try:
        config = request.json
        # Implement your configuration saving logic here
        # For example, save to a config collection in the database
        db.configurations.insert_one(config)
        return jsonify({"message": "Configuration saved successfully"}), 200
    except Exception as e:
        logger.error(f"Error in save_configuration: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/pipeline-control', methods=['POST'])
@cross_origin()
def control_pipeline():
    try:
        data = request.json
        action = data.get('action')
        if action not in ['start', 'stop', 'pause']:
            return jsonify({"error": "Invalid action"}), 400

        # Update pipeline_status based on action
        for phase in pipeline_status:
            if action == 'start':
                pipeline_status[phase]['status'] = 'running'
                pipeline_status[phase]['message'] = 'RUNNING'
                pipeline_status[phase]['progress'] = 0
            elif action == 'stop':
                pipeline_status[phase]['status'] = 'error'
                pipeline_status[phase]['message'] = 'STOPPED'
                pipeline_status[phase]['progress'] = 0
            elif action == 'pause':
                pipeline_status[phase]['status'] = 'paused'
                pipeline_status[phase]['message'] = 'PAUSED'
                pipeline_status[phase]['progress'] = 0

        emit('pipeline_control', {'action': action}, broadcast=True)
        return jsonify({"message": f"Pipeline {action} requested"}), 200
    except Exception as e:
        logger.error(f"Error in control_pipeline: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)