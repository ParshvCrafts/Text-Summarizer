import os
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
from src.text_summarizer.logger import logger
from pathlib import Path
from typing import List, Union

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns ConfigBox
    Args:
        path_to_yaml (Path): Path to the yaml file
    Returns:
        ConfigBox: ConfigBox type object
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise BoxValueError(f"Error while converting yaml to config box {e}")
    except Exception as e:
        raise e

def create_directories(path_to_directories: List[Path]) -> None:
    """Creates list of directories
    Args:
        path_to_directories (List[Path]): List of paths to directories
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created at: {path}")

def get_size(path: Path) -> str:
    """Get size in KB
    Args:
        path (Path): Path to the file
    Returns:
        str: Size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"
