"""Model definitions for AnatCoder and benchmark baselines."""

from .anatcoder.network import AnatCoderNetwork
from .vanilla_inr.network import VanillaINRNetwork

__all__ = ["AnatCoderNetwork", "VanillaINRNetwork"]
