"""
Shared utilities for image processing scripts.

This module contains common functions used by multiple image processing scripts
(align.py, create_gif.py, etc.)
"""

import logging
import re
from pathlib import Path


def setup_logger(name: str, debug: bool = False) -> logging.Logger:
    """
    Configure and return a logger with appropriate level.
    
    Args:
        name: Logger name (e.g., "align", "gif_creator")
        debug: If True, use DEBUG level; otherwise use CRITICAL level
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if debug else logging.CRITICAL
    logging.basicConfig(level=level)
    return logging.getLogger(name)


def natural_key(filepath: Path) -> list:
    """
    Generate a sorting key that orders filenames numerically.
    
    This allows natural sorting where file2.jpg comes before file10.jpg,
    rather than the lexicographic ordering where file10.jpg comes first.
    
    Args:
        filepath: Path object for the file
        
    Returns:
        List of mixed integers and strings for proper natural sorting
        
    Example:
        >>> files = [Path('file10.jpg'), Path('file2.jpg'), Path('file1.jpg')]
        >>> sorted(files, key=natural_key)
        [Path('file1.jpg'), Path('file2.jpg'), Path('file10.jpg')]
    """
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', filepath.name)]


# Supported image formats
VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}


def get_image_files(folder: str | Path) -> list[Path]:
    """
    Retrieve and naturally sort valid image files from a directory.
    
    Args:
        folder: Path to directory containing images
        
    Returns:
        List of Path objects for valid image files, sorted naturally.
        Returns empty list if folder doesn't exist or contains no valid images.
        
    Supported formats:
        .png, .jpg, .jpeg, .tif, .bmp
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []
    
    files = [f for f in folder_path.iterdir() 
             if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    return sorted(files, key=natural_key)
