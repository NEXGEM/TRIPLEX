"""
Configuration utility for TRIPLEX pipeline.
Handles loading and merging YAML configuration files.
"""

import os
import yaml
from typing import Dict, Optional, Any, Union


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict, override_config: Optional[Dict] = None) -> Dict:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base configuration
        
    Returns:
        Merged configuration
    """
    if not override_config:
        return base_config
    
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged_config and isinstance(value, dict) and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config


def get_config(config_dir: str, config_name: str, override_config: Optional[Dict] = None) -> Dict:
    """
    Get configuration by name, with optional overrides.
    
    Args:
        config_name: Name of the configuration (filename without extension)
        override_config: Optional configuration to override the named configuration
        
    Returns:
        Configuration dictionary
    """
    # Find config files directory
    config_dir = os.path.join(config_dir, 'config')
    
    # Find specific config file
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"
    
    config_path = os.path.join(config_dir, config_name)
    
    # Load config
    config = load_yaml_config(config_path)
    
    # Apply overrides if provided
    if override_config:
        config = merge_configs(config, override_config)
    
    return config


def update_pipeline_config(pipeline, config_name: Optional[str] = None, **kwargs) -> None:
    """
    Update pipeline configuration from a named config and/or keyword arguments.
    
    Args:
        pipeline: TriplexPipeline instance
        config_name: Optional name of configuration to load
        **kwargs: Override configuration values
    """
    if config_name:
        config = get_config(config_name)
        pipeline.config = merge_configs(pipeline.config, config)
    
    # Apply kwargs overrides
    if kwargs:
        pipeline.config = merge_configs(pipeline.config, kwargs)
    
    # Re-setup directories after config update
    pipeline._setup_dirs()
