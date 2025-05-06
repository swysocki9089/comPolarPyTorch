import torch
import math
import pytest
from complexFunctions import ComPolar64 # Adjust import path as needed

# def test_polar_normalize_scalar():
#     z = ComPolar64(torch.tensor(3.0), torch.tensor(math.pi / 4))
#     z_norm = polar_normalize(z)

#     assert torch.allclose(z_norm.phase, z.phase)
#     assert torch.allclose(z_norm.magnitude, torch.tensor(0.0), atol=1e-5)

# def test_polar_normalize_batch():
#     magnitudes = torch.tensor([1.0, 2.0, 3.0, 4.0])
#     phases = torch.tensor([0.0, 0.5, 1.0, 1.5])
#     z = ComPolar64(magnitudes, phases)

#     z_norm = polar_normalize(z)

#     mean = z_norm.magnitude.mean()
#     std = z_norm.magnitude.std()

#     assert torch.allclose(mean, torch.tensor(0.0), atol=1e-5)
#     assert torch.allclose(std, torch.tensor(1.0), atol=1e-5)
#     assert torch.allclose(z_norm.phase, phases)

# def test_polar_normalize_zero_std_safe():
#     magnitude = torch.tensor([5.0, 5.0, 5.0])
#     phase = torch.tensor([0.0, 0.0, 0.0])
#     z = ComPolar64(magnitude, phase)

#     z_norm = polar_normalize(z)

#     assert torch.allclose(z_norm.magnitude, torch.tensor([0.0, 0.0, 0.0]), atol=1e-5)
#     assert torch.allclose(z_norm.phase, phase)

# === Phase comparison helper ===
def assert_coterminal_tensor(a, b, atol=1e-5):
    diff = ((a - b + math.pi) % (2 * math.pi)) - math.pi
    assert torch.allclose(diff, torch.zeros_like(diff), atol=atol)

# === Test 1: Single vector rotation ===
def test_polar_single_rotation():
    z = ComPolar64(torch.tensor(2.0), torch.tensor(math.pi / 4))
    rotated = z.rotate(math.pi / 2)

    expected_phase = ((math.pi / 4 + math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi
    assert torch.allclose(rotated.magnitude, torch.tensor(2.0), atol=1e-5)
    assert_coterminal_tensor(rotated.phase, torch.tensor(expected_phase))

# === Test 2: Batch vector rotation === 
def test_polar_batch_rotation():
    mag = torch.tensor([1.0, 1.0, 1.0])
    phase = torch.tensor([0.0, math.pi / 2, -math.pi / 2])
    z = ComPolar64(mag, phase)
    rotated = z.rotate(math.pi / 2)

    expected_phase = ((phase + math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi
    assert_coterminal_tensor(rotated.phase, expected_phase)
    assert torch.allclose(rotated.magnitude, mag, atol=1e-5)


# === Test 3: Batch scalar multiplication ===
def test_polar_batch_scalar_multiplication():
    mag = torch.tensor([2.0, 3.0, 4.0])
    phase = torch.tensor([0.0, math.pi / 2, math.pi])
    z = ComPolar64(mag, phase)
    scaled = z.scale(2.0)

    assert torch.allclose(scaled.magnitude, mag * 2.0, atol=1e-5)
    assert_coterminal_tensor(scaled.phase, phase)

# === Test 4: Batch addition ===
def test_polar_batch_arithmetic_addition():
    a = ComPolar64.from_complex(torch.tensor([1 + 1j, 2 + 2j]))
    b = ComPolar64.from_complex(torch.tensor([1 - 1j, 1 - 2j]))
    result = a + b
    expected = torch.tensor([2 + 0j, 3 + 0j])
    assert torch.allclose(result.to_cartesian(), expected, atol=1e-5)


