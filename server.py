from flask import Flask, request, jsonify
import logging
from typing import Dict, Any
from functools import wraps

app = Flask(__name__)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_request(f):
    """Decorator to validate incoming requests from Apple Shortcuts"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        return f(*args, **kwargs)
    return decorated_function

@app.route("/siri", methods=["POST"])
@validate_request
def handle_siri_request() -> Dict[Any, Any]:
    """
    Handles incoming requests from Apple Shortcuts/Siri
    
    Expected JSON format:
    {
        "input": "spoken text from Siri",
        "parameters": {} # optional additional parameters
    }
    
    Returns:
        JSON response with processed result
    """
    try:
        data = request.get_json()
        input_text = data.get("input", "")
        parameters = data.get("parameters", {})
        
        if not input_text:
            return jsonify({"error": "Input text is required"}), 400
            
        # Process the input text here
        # This is where you would integrate your existing speech processing logic
        processed_result = process_input(input_text, parameters)
        
        return jsonify({
            "status": "success",
            "result": processed_result
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def process_input(text: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for text processing logic.
    Replace with your existing speech processing system.
    
    Args:
        text: Input text from Siri
        params: Additional parameters from the request
        
    Returns:
        Processed result as a dictionary
    """
    # Integration point for existing system
    return {"processed_text": text}

def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """
    Starts the Flask server
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        debug: Enable debug mode
    """
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server()
