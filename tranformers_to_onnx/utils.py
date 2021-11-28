#!/usr/bin/env python3
""" ⚙️ Transformers to ONNX - Utils

    Utils for Transformers to ONNX.

    Author:
        Thomas Chaigneau
"""

import os


def save_file(dir: str, files):
    """
    Function to save files from user uploads.
    """

    for file in files:
        with open(os.path.join(dir, file.name), "wb") as f:
            f.write(file.getbuffer())


def create_dir(dir: str):
    """
    Function to create a directory if it doesn't exist.

    Parameters
    ----------
    dir : str
        Directory to create.
    """

    if not os.path.exists(dir):
        os.makedirs(dir)
