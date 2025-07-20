# app.py
# Make sure to set the python path to the embedded python executable.
# Set the python io encoding to utf8.
import sys
import logging

# 在程序启动时启用底层崩溃dump

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("BACKEND_START_IMPORTS", flush=True)

try:
    from flask import Flask
    from flask_cors import CORS
    from routes import api_blueprint
except ImportError as e:
    print(f"Error importing dependencies: {e}", flush=True)
    # If flask isn't installed in the embedded python, we can't run.
    # This might happen if the python folder wasn't packaged correctly.
    sys.exit(1)


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)
log = logging.getLogger('werkzeug')
log.disabled = True
# Register the blueprint from routes.py
app.register_blueprint(api_blueprint)

print("BACKEND_IMPORTS_DONE", flush=True)

# --- Main Execution ---
if __name__ == '__main__':
    # The host must be 0.0.0.0 to be accessible from outside the container
    print("Backend is now listening for requests...", flush=True)

    # use_reloader=False is critical for packaged apps
    app.run(debug=False, host='0.0.0.0', port=1234, use_reloader=False)

