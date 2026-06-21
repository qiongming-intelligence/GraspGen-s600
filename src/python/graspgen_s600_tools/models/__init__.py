"""
Models module for GraspGen-s600.

Contains ONNX-compatible model implementations.
"""

from .pointnet_encoder import PointNetEncoder

__all__ = ['PointNetEncoder']
