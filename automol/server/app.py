import random
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
from main import main as run_pipeline  # Import your pipeline's main function

import os
import time
import threading
import pandas as pd
import plotly.express as px
from db import db  # Assuming you have a db.py file with your database connection
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
from pymongo import MongoClient
from bson import ObjectId
from flask import jsonify
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from transformers import AutoTokenizer, AutoModel
import torch
import json
import pyarrow.parquet as pq
import logging
from io import BytesIO
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

molecules_collection = db['molecules']



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://chat.build-a-bf.com"]}})
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdb', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB connection

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pipeline_status = {
    "Phase 1 - Research and Hypothesize": {"status": "stopped", "progress": 0, "output": []},
    "Phase 2 - Protein Generation and Optimization": {"status": "stopped", "progress": 0, "output": []},
    "Phase 3 - Ligand Generation and Optimization": {"status": "stopped", "progress": 0, "output": []},
    "Phase 4 - Analysis and Simulation": {"status": "stopped", "progress": 0, "output": []}
}

notifications = []

def emit_progress(phase, progress, message):
    socketio.emit('progress_update', {
        'phase': phase,
        'progress': progress,
        'message': message
    })


# Example usage in a pipeline function
def run_pipeline_phase(phase):
    for i in range(100):
        # ... do some work ...
        emit_progress(phase, i, f"Completed step {i} of phase {phase}")
        time.sleep(0.1)  # Simulate work

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from flask import send_from_directory

@app.route('/api/files/<filename>', methods=['GET'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/pipeline_status', methods=['GET'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def get_pipeline_status():
    return jsonify(pipeline_status)

@app.route('/api/key-metrics', methods=['GET'])
def get_key_metrics():
    pipeline_collection = db['pipeline_log']
    return jsonify({
        "total_sequences": pipeline_collection.count_documents({}),
        "passed_sequences": pipeline_collection.count_documents({"pass_fail": "Pass"}),
        "failed_sequences": pipeline_collection.count_documents({"pass_fail": "Fail"})
    })

@app.route('/api/pipeline_data', methods=['GET'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def get_pipeline_data():
    df = load_data()
    return jsonify(df.to_dict('records'))


@app.route('/api/pdb_files/<filename>', methods=['GET'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def download_pdb(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/pipeline_control', methods=['POST'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def control_pipeline():
    data = request.json
    phase, action = data.get('phase'), data.get('action')
    if action == 'start' and pipeline_status[phase]["status"] in ['stopped', 'completed']:
        pipeline_status[phase]["status"], pipeline_status[phase]["progress"], pipeline_status[phase]["output"] = "starting", 0, []
        threading.Thread(target=run_pipeline_phase, args=(phase,)).start()
        return jsonify({"status": "starting"})
    elif action in ['stop', 'pause']:
        pipeline_status[phase]["status"] = action
        return jsonify({"status": action})
    return jsonify({"error": "Invalid action or phase is already running"}), 400



@app.route('/api/notifications', methods=['GET'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def get_notifications():
    return jsonify(notifications)




def read_file(filename, contents):
    if filename.endswith('.csv'):
        return pd.read_csv(BytesIO(contents))
    elif filename.endswith('.tsv'):
        return pd.read_csv(BytesIO(contents), sep='\t')
    elif filename.endswith('.parquet'):
        return pq.read_table(BytesIO(contents)).to_pandas()
    elif filename.endswith('.json'):
        return pd.read_json(BytesIO(contents))
    else:
        raise ValueError("Unsupported file format")

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        contents = file.read()
        df = read_file(file.filename, contents)
        return jsonify({"preview": df.head().to_dict(), "columns": df.columns.tolist()})
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
    
    
    
@app.route('/process', methods=['POST'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def process_data():
    try:
        file = request.files['file']
        options = json.loads(request.form['options'])
        
        contents = file.read()
        df = read_file(file.filename, contents)

        # Apply selected processing options
        if 'clean' in options['selectedOptions']:
            df = clean_data(df)
        if 'normalize' in options['selectedOptions']:
            df = normalize_data(df)
        if 'align' in options['selectedOptions']:
            df = align_to_reference(df)
        if 'map' in options['selectedOptions']:
            df = map_features(df)

        # Apply advanced preprocessing
        if options['advancedPreprocessing'].get('featureSelection', False):
            df = feature_selection(df)
        if options['advancedPreprocessing'].get('dimensionalityReduction', False):
            df = dimensionality_reduction(df)
        if options['advancedPreprocessing'].get('outlierDetection', False):
            df = outlier_detection(df)
        if options['advancedPreprocessing'].get('missingDataImputation', False):
            df = missing_data_imputation(df)

        # Apply AI processing if selected
        if options['useAI']:
            df = ai_processing(df, options['selectedAIModel'], options['harmonizationLevel'])

        return jsonify({"processed_data": df.to_dict(), "columns": df.columns.tolist()})
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        return jsonify({"error": str(e)}), 400

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')
    df = df.dropna()
    return df

def normalize_data(df):
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def align_to_reference(df):
    # Placeholder for align_to_reference functionality
    return df

def map_features(df):
    # Placeholder for map_features functionality
    return df

def feature_selection(df):
    selector = VarianceThreshold()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = selector.fit_transform(df[numeric_columns])
    return df

def dimensionality_reduction(df):
    pca = PCA(n_components=0.95)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = pca.fit_transform(df[numeric_columns])
    return df

def outlier_detection(df):
    iso_forest = IsolationForest(contamination=0.1)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = iso_forest.fit_predict(df[numeric_columns])
    df = df[outliers != -1]
    return df

def missing_data_imputation(df):
    imputer = IterativeImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

def ai_processing(df, model_name, harmonization_level):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        inputs = tokenizer(df[col].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        harmonized_embeddings = embeddings * (harmonization_level / 100)
        
        for i in range(embeddings.shape[1]):
            df[f"{col}_embed_{i}"] = harmonized_embeddings[:, i].numpy()
        
        df = df.drop(columns=[col])

    return df

@app.route('/api/graph_data')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def get_graph_data():
    df = load_data()
    
    data = {
        'labels': df['timestamp'].tolist(),
        'datasets': [{
            'label': 'Score',
            'data': df['score'].tolist(),
            'borderColor': 'rgb(75, 192, 192)',
            'backgroundColor': 'rgba(75, 192, 192, 0.5)',
            'tension': 0.1
        }]
    }
    return jsonify(data)

def load_data():
    # Replace this with your actual data loading logic
    # This is just a placeholder example
    data = {
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'score': [75, 80, 85, 90, 88]
    }
    return pd.DataFrame(data)


@app.route('/stream-logs')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def stream_logs():
    def generate():
        with open('app.log', 'r') as f:
            while True:
                yield f.read()
                time.sleep(1)
    return Response(generate(), mimetype='text/plain')



def load_data():
    pipeline_collection = db['pipeline_log']
    data = list(pipeline_collection.find())
    for item in data:
        item['_id'] = str(item['_id'])
    return pd.DataFrame(data)

# ... (existing imports)
from main import main as run_pipeline  # Import your pipeline's main function


@socketio.on('start_pipeline')
def handle_start_pipeline(data):
    input_text = data.get('input_text', '')
    config = {
        'input_text': input_text,
        'max_generations': data.get('max_generations', 2),
        'num_reflections': data.get('num_reflections', 2),
        'num_sequences': data.get('num_sequences', 2),
        'optimization_steps': data.get('optimization_steps', 15),
        'score_threshold': data.get('score_threshold', 0.55),
        'output_dir': data.get('output_dir', 'results'),
        'skip_description_gen': data.get('skip_description_gen', False)
    }
    
    def pipeline_wrapper():
        try:
            results = run_pipeline(config)
            socketio.emit('pipeline_complete', results)
        except Exception as e:
            socketio.emit('pipeline_error', {'error': str(e)})

    threading.Thread(target=pipeline_wrapper).start()
    return {'status': 'Pipeline started'}


def run_pipeline_phase(phase):
    pipeline_status[phase]["status"] = "running"
    for i in range(1, 101):
        time.sleep(0.1)
        pipeline_status[phase]["progress"] = i
        output = f"{phase} progress: {i}%"
        pipeline_status[phase]["output"].append(output)
        socketio.emit('pipeline_status', {phase: pipeline_status[phase]})
        socketio.emit('pipeline_output', {'phase': phase, 'output': output})
        if i == 100:
            notifications.append({"message": f"{phase} completed successfully.", "type": "success"})
            socketio.emit('notification', {"message": f"{phase} completed successfully.", "type": "success"})
    pipeline_status[phase]["status"] = "completed"
    socketio.emit('pipeline_status', {phase: pipeline_status[phase]})



@app.route('/api/molecules', methods=['GET'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def get_molecules():
    molecules = list(molecules_collection.find())
    for molecule in molecules:
        molecule['_id'] = str(molecule['_id'])
    return jsonify(molecules)

@app.route('/api/molecules', methods=['POST'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def add_molecule():
    new_molecule = request.json
    result = molecules_collection.insert_one(new_molecule)
    new_molecule['_id'] = str(result.inserted_id)
    socketio.emit('moleculeUpdate', {'type': 'moleculeUpdate', 'molecule': new_molecule})
    return jsonify(new_molecule), 201

@app.route('/api/molecules/<id>', methods=['PUT'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def update_molecule(id):
    updated_molecule = request.json
    molecules_collection.update_one({'_id': ObjectId(id)}, {'$set': updated_molecule})
    updated_molecule['_id'] = id
    socketio.emit('moleculeUpdate', {'type': 'moleculeUpdate', 'molecule': updated_molecule})
    return jsonify(updated_molecule)


@app.route('/api/upload_pdb', methods=['POST'])
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def upload_pdb():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        return jsonify({"message": "File uploaded successfully", "filename": file.filename})


@cross_origin()  # Add this decorator to ensure CORS is applied to this route


# Emit progress to WebSocket
def emit_progress(phase, progress, message):
    log_message = f"{phase} progress: {progress}% - {message}"
    logger.info(log_message)
    socketio.emit('progress_update', {'phase': phase, 'progress': progress, 'message': message})
    socketio.emit('log', {'message': log_message})

# Example pipeline with logging
def run_pipeline_phase(phase):
    pipeline_status[phase]["status"] = "running"
    for i in range(100):
        time.sleep(0.1)
        progress_message = f"Completed step {i} of phase {phase}"
        emit_progress(phase, i, progress_message)
    pipeline_status[phase]["status"] = "completed"
    emit_progress(phase, 100, f"{phase} completed successfully!")

# Start RL updates and emit logs
def simulate_rl_updates():
    while True:
        data = {
            "Toxicity": round(random.uniform(0, 1), 2),
            "Structure": round(random.uniform(0, 1), 2),
            "Viability": round(random.uniform(0, 1), 2),
            "Efficiency": round(random.uniform(0, 1), 2),
            "Cost": round(random.uniform(0, 1), 2)
        }
        log_message = f"RL update: {data}"
        logger.info(log_message)
        socketio.emit('update', data)
        socketio.emit('log', {'message': log_message})
        time.sleep(2)


@socketio.on('connect')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_disconnect():
    print('Client disconnected')




@socketio.on('moleculeAdded')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_molecule_added(newMolecule):
    socketio.emit('moleculeAdded', newMolecule)

@socketio.on('moleculeUpdated')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_molecule_updated(updatedMolecule):
    socketio.emit('moleculeUpdated', updatedMolecule)

@socketio.on('moleculeDeleted')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_molecule_deleted(deletedMoleculeId):
    socketio.emit('moleculeDeleted', deletedMoleculeId)

@socketio.on('connect')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
@cross_origin()  # Add this decorator to ensure CORS is applied to this route

def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)