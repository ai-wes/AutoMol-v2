import logging
from socket_manager import socketio  # Import socketio from socket_manager

logger = logging.getLogger(__name__)

def emit_progress(phase=None, progress=None, message=None):
    logger.info(f"Emitting progress: phase={phase}, progress={progress}, message={message}")
    socketio.emit('progress_update', {
        'phase': phase,
        'progress': progress,
        'message': message
    })