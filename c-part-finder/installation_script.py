import subprocess
import sys

# List of required libraries
required_libraries = [
    "opencv-python",  # For image processing
    "numpy",  # For numerical computations
    "matplotlib",  # For plotting
    "python-dotenv",  # For environment variable management
    "Pillow",  # For image handling
    "google",  # For Google search (installing the parent package for `googlesearch`)
    "googlesearch-python",  # Updated googlesearch package
    "requests",  # For HTTP requests
    "pandas",  # For handling CSV and data
    "streamlit",  # For web app interface
]

# Install each library if not already installed
for library in required_libraries:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])
        print(f"'{library}' installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error installing '{library}'. Please install it manually.")
