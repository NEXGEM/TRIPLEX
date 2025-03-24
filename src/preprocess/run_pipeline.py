#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.pipeline import TriplexPipeline
from src.pipeline.config import get_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run TRIPLEX pipeline with predefined configurations")
    parser.add_argument("config_name", type=str, help="Name of YAML configuration file to use")
    
    # Override options
    parser.add_argument("--input_dir", type=str, help="Override input directory")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--total_gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--overwrite", action="store_true", help="Override existing results")
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Get configuration from file
    try:
        config = get_config(args.config_name)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_name}' not found")
        print("Available configurations:")
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
        configs = [os.path.splitext(f)[0] for f in os.listdir(config_dir) if f.endswith('.yaml')]
        for config_name in sorted(configs):
            print(f"  - {config_name}")
        sys.exit(1)
    
    # Override with command line arguments if provided
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.total_gpus:
        config["total_gpus"] = args.total_gpus
    if args.overwrite:
        config["overwrite"] = True
    
    # Check if required paths are provided
    if "input_dir" not in config or config["input_dir"] is None:
        print("Error: input_dir is required but not provided")
        sys.exit(1)
    if "output_dir" not in config or config["output_dir"] is None:
        print("Error: output_dir is required but not provided")
        sys.exit(1)
    
    # Add timestamp to logs
    print(f"\n=== Starting TRIPLEX Pipeline ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {args.config_name}")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    
    # Run the pipeline
    pipeline = TriplexPipeline(config)
    pipeline.run_pipeline()
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
