import torch
import torch.nn.functional as nnf
from complexPyTorch import complexFunctions as cf
from comPolar64 import ComPolar64

def polar_relu(polar: ComPolar64) -> ComPolar64:
    """
    Apply ReLU activation to the magnitude, keep phase unchanged.
    """
    activated_mag = nnf.relu(polar.get_magnitude())
    return ComPolar64(activated_mag, polar.get_phase())

def polar_sigmoid(polar: ComPolar64) -> ComPolar64:
    """
    Apply sigmoid activation to the magnitude, keep phase unchanged.
    """
    activated_mag = nnf.sigmoid(polar.get_magnitude())
    return ComPolar64(activated_mag, polar.get_phase())

def polar_tanh(polar: ComPolar64) -> ComPolar64:
    """
    Apply tanh activation to the magnitude, keep phase unchanged.
    """
    activated_mag = nnf.tanh(polar.get_magnitude())
    return ComPolar64(activated_mag, polar.get_phase())


def polar_avg_pool2d(polar: ComPolar64, *args, **kwargs) -> ComPolar64:
    """
    Perform average pooling on magnitude; phase is pooled by averaging as well.
    """
    pooled_mag = nnf.avg_pool2d(polar.get_magnitude(), *args, **kwargs)
    # Optional: you could also just copy the phase unchanged (or use center pixel phase)
    pooled_phase = nnf.avg_pool2d(polar.get_phase(), *args, **kwargs)
    return ComPolar64(pooled_mag, pooled_phase)


def polar_max_pool2d(polar: ComPolar64, *args, **kwargs) -> ComPolar64:
    """
    Perform max pooling on magnitude; phase is selected from location of pooled magnitude.
    """
    pooled_mag, indices = nnf.max_pool2d(
        polar.get_magnitude(), *args, return_indices=True, **kwargs
    )
    pooled_phase = cf._retrieve_elements_from_indices(polar.get_phase(), indices)
    return ComPolar64(pooled_mag, pooled_phase)

def polar_dropout(polar: ComPolar64, p=0.5, training=True) -> ComPolar64:
    """
    Apply dropout to magnitude; phase remains unchanged.
    """
    mask = nnf.dropout(torch.ones_like(polar.get_magnitude()), p, training)
    dropped_mag = polar.get_magnitude() * mask
    return ComPolar64(dropped_mag, polar.get_phase())


def polar_dropout2d(polar: ComPolar64, p=0.5, training=True) -> ComPolar64:
    """
    Apply 2D dropout to magnitude; phase remains unchanged.
    """
    mask = nnf.dropout2d(torch.ones_like(polar.get_magnitude()), p, training)
    dropped_mag = polar.get_magnitude() * mask
    return ComPolar64(dropped_mag, polar.get_phase())

def polar_upsample(
    polar: ComPolar64,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None
) -> ComPolar64:
    """
    Upsample magnitude and phase independently, then recombine.
    """
    upsampled_mag = nnf.interpolate(
        polar.get_magnitude(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    upsampled_phase = nnf.interpolate(
        polar.get_phase(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return ComPolar64(upsampled_mag, upsampled_phase)

def polar_stack(polar_list: list[ComPolar64], dim: int) -> ComPolar64:
    """
    Stack a list of ComPolar64 instances along the specified dimension.
    Magnitude and phase are stacked separately.
    """
    mags = [p.get_magnitude() for p in polar_list]
    phases = [p.get_phase() for p in polar_list]
    stacked_mag = torch.stack(mags, dim=dim)
    stacked_phase = torch.stack(phases, dim=dim)
    return ComPolar64(stacked_mag, stacked_phase)
