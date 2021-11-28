#!/usr/bin/env python3
""" ⚙️ Transformers to ONNX - Conversion

    Conversion service for Transformers to ONNX.

    Author:
        Thomas Chaigneau
"""

import torch 
from torch.onnx import export
from typing import List, Tuple

from transformers.models.auto import AutoTokenizer
from transformers.onnx.config import OnnxConfig
from transformers.onnx.convert import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TensorType,
    TFPreTrainedModel,
    is_torch_available,
)


def run_conversion(
    model_files:str, 
    output: str, 
    atol: float = 1e-4,
    opset: int = None, 
    feature: str = "default",
):
    tokenizer = AutoTokenizer.from_pretrained(model_files)
    model = FeaturesManager.get_model_from_feature(feature, model_files)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model_files, feature)
    onnx_config = model_onnx_config(model.config)

    if opset:
        while opset < onnx_config.default_onnx_opset:
            opset += 1
    else: opset = onnx_config.default_onnx_opset
    
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, opset, output)

    validate_model_outputs(onnx_config, tokenizer, model, output, onnx_outputs, atol)
    print(f"All good, model saved at: {output.as_posix()}")
