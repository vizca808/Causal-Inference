# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run the pipeline
python main.py
