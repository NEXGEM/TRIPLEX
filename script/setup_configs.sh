#!/bin/bash
# Setup configs directory and make config script executable

# Create configs directory if it doesn't exist
mkdir -p ../configs

# Make run_pipeline.py executable
chmod +x run_pipeline.py

echo "Configuration directory created at ../configs"
echo "You can now run the pipeline with: ./run_pipeline.py <config_name>"
