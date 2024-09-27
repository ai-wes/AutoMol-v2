import logging
from flask_socketio import SocketIO
from flask import Flask
from flask_cors import CORS

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
socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:3000", "https://dashboard.automol-ai.com"],
    allow_credentials=True
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)