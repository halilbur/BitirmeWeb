import os
import sys

# Get the directory where this wsgi.py file is located (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to the Python path
src_path = os.path.join(project_root, 'src')

# Clear any existing src paths to avoid duplicates
sys.path = [p for p in sys.path if not p.endswith('/src') and not p.endswith('\\src')]

# Add paths in the correct order
if src_path not in sys.path:
    sys.path.insert(0, src_path)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug info (will be visible in Render logs)
print(f"WSGI Debug - Project root: {project_root}")
print(f"WSGI Debug - Src path: {src_path}")
print(f"WSGI Debug - Working directory: {os.getcwd()}")
print(f"WSGI Debug - Src exists: {os.path.exists(src_path)}")
print(f"WSGI Debug - Models exists: {os.path.exists(os.path.join(src_path, 'models'))}")

# Import the app from the src directory
from main import app

if __name__ == "__main__":
    app.run()
