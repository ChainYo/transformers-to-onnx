#!/usr/bin/env python3
""" ⚙️ Transformers to ONNX - Streamlit App

    This app is a simple wrapper around the Transformers to ONNX library.
    It allows you to convert a PyTorch model to ONNX format and save it to a file.

    Author:
        - Thomas Chaigneau
"""

import streamlit as st


st.set_page_config(
    page_title="Transformers Converter",
    page_icon="🤗",
    layout="centered",
)
st.title("Transformers to ONNX")