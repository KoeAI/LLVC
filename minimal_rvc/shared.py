# This module is based on code from ddPn08, liujing04, and teftef6220
# https://github.com/ddPn08/rvc-webui
# https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# https://github.com/teftef6220/Voice_Separation_and_Selection
# These modules are licensed under the MIT License.

import os
import sys

import torch

from .cmd_opts import opts

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "llvc_models", "models")


def has_mps():
    if sys.platform != "darwin":
        return False
    else:
        if not getattr(torch, "has_mps", False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False


is_half = opts.precision == "fp16"
half_support = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 5.3
)

if not half_support:
    print("WARNING: FP16 is not supported on this GPU")
    is_half = False

device = "cuda:0"

if not torch.cuda.is_available():
    if has_mps():
        print("Using MPS")
        device = "mps"
    else:
        print("Using CPU")
        device = "cpu"

device = torch.device(device)
