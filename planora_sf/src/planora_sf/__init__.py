"""PLANORA-SF research package (Stage B scaffold).

Implements the mechanics specified in ``PLANORA_SF_Method_Spec.md``: Fourier
views, latent partition, SelEx/FourEx losses, hierarchical pseudo-labels and
the GCD evaluator. It contains no training loop, no data loading and no
checkpoints, and is independent of the PLANORA production software.
"""
from planora_sf.config import PlanoraSFConfig

__version__ = "0.1.0"

__all__ = ["PlanoraSFConfig", "__version__"]
