# TRIPLEX Pipeline Configurations

This directory contains YAML configuration files for the TRIPLEX pipeline.

## Available Configurations

- `default.yaml`: Default configuration with standard settings
- `inference.yaml`: Configuration optimized for inference mode
- `hest.yaml`: Configuration for HEST data processing
- `BC1.yaml`: Configuration for BC1 dataset
- `BC2.yaml`: Configuration for BC2 dataset
- `SCC.yaml`: Configuration for SCC dataset
- `lunit.yaml`: Configuration for Lunit data processing

## Usage

You can run the pipeline with a specific configuration using:

```bash
python script/run_pipeline.py BC1
```

Or override specific settings:

```bash
python script/run_pipeline.py inference --input_dir /path/to/data --total_gpus 4
```

## Configuration Format

Each configuration file follows this structure:

```yaml
# Basic settings
mode: train  # train, inference, or hest
platform: visium  # visium, xenium, cosmx, etc.

# Path settings
input_dir: /path/to/input
output_dir: /path/to/output

# Slide settings
slide_ext: .svs
patch_size: 224
slide_level: 0

# Additional settings...
```

## Creating Custom Configurations

You can create your own configuration files by copying and modifying existing ones.
Place them in this directory with a `.yaml` extension.
