#!/usr/bin/env python3
""" ⚙️ Transformers to ONNX - Streamlit App

    This app is a simple wrapper around the Transformers to ONNX library.
    It allows you to convert a PyTorch model to ONNX format and save it 
    to a file.

    Author:
        Thomas Chaigneau
"""

import streamlit as st

from convert import run_conversion
from utils import save_file


st.set_page_config(
    page_title="Transformers Converter",
    page_icon="🤗",
    layout="centered",
)
st.title("Transformers to ONNX")

with st.expander("📝 HOW TO"):
    st.markdown(
        """
        @TODO
        """
    )

uploaded_files = st.file_uploader(
    label="📁 Upload your model", 
    accept_multiple_files=True,
    help="Drop your files here."
)

if st.button(
    label="🚀 Launch Conversion",
    key="launch_conversion",
):
    try:    
        if uploaded_files is not None:
            save_file("temp", uploaded_files)
            run_conversion("temp", "output")
            st.success("Conversion successful.")
    except Exception as e:
        st.error(e)

# st.text_input(
#     label="Model Name",
#     value="",
#     key="model_name",
#     help="Type the model's name.",
#     placeholder="e.g. google/mt5-small"
# )