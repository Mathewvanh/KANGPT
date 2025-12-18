"""
KANGPT - Kolmogorov-Arnold Networks for GPT

A KAN-based language model that replaces MLP layers with polynomial transformations.
"""

from .config import KANGPTConfig
from .model import KANGPT, KANLayer, KANMLP
from .lightning import KANGPTLM

__version__ = "0.1.0"
__all__ = ["KANGPT", "KANGPTConfig", "KANLayer", "KANMLP", "KANGPTLM"]
