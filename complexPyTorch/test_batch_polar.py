import torch
import math
import pytest
from complexPyTorch import complexFunctions as cf
import polarComplexFunctions as pcf
from comPolar64 import ComPolar64

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
    a = ComPolar64.from_cartesian(torch.tensor([1 + 1j, 2 + 2j]))
    b = ComPolar64.from_cartesian(torch.tensor([1 - 1j, 1 - 2j]))
    result = a + b
    expected = torch.tensor([2 + 0j, 3 + 0j])
    assert torch.allclose(result.to_cartesian(), expected, atol=1e-5)


def test_polar_relu():
    a = ComPolar64(torch.tensor([-1.0, 0.0, 2.0]), torch.tensor([0.0, 0.0, 0.0]))
    result = pcf.polar_relu(a)
    expected_mag = torch.nn.functional.relu(a.get_magnitude())
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), a.get_phase())

def test_polar_sigmoid():
    a = ComPolar64(torch.tensor([-1.0, 0.0, 2.0]), torch.tensor([0.5, 1.0, 1.5]))
    result = pcf.polar_sigmoid(a)
    expected_mag = torch.sigmoid(a.get_magnitude())
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), a.get_phase())

def test_polar_tanh():
    a = ComPolar64(torch.tensor([-1.0, 0.0, 2.0]), torch.tensor([0.5, 1.0, 1.5]))
    result = pcf.polar_tanh(a)
    expected_mag = torch.tanh(a.get_magnitude())
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), a.get_phase())

def test_polar_avg_pool2d():
    mag = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    phase = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    polar = ComPolar64(mag, phase)
    result = pcf.polar_avg_pool2d(polar, kernel_size=2)
    expected_mag = torch.nn.functional.avg_pool2d(mag, kernel_size=2)
    expected_phase = torch.nn.functional.avg_pool2d(phase, kernel_size=2)
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), expected_phase)

def test_polar_max_pool2d():
    mag = torch.tensor([[[[1.0, 5.0], [3.0, 2.0]]]])
    phase = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    polar = ComPolar64(mag, phase)
    result = pcf.polar_max_pool2d(polar, kernel_size=2)
    expected_mag, indices = torch.nn.functional.max_pool2d(mag, kernel_size=2, return_indices=True)
    selected_phase = cf._retrieve_elements_from_indices(phase, indices)
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), selected_phase)

def test_polar_dropout():
    mag = torch.ones(10)
    phase = torch.linspace(0, 1, 10)
    polar = ComPolar64(mag, phase)
    result = pcf.polar_dropout(polar, p=0.5, training=True)

    scale = 1.0 / (1.0 - 0.5)  # inverse of (1-p)
    expected_max = mag.max() * scale
    assert (result.get_magnitude() <= expected_max).all()
    # mask should zero out about half the values (stochastic → check valid range)
    assert (result.get_magnitude() >= 0).all()
    assert torch.allclose(result.get_phase(), phase)

def test_polar_dropout2d():
    mag = torch.ones(1, 1, 4, 4)
    phase = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4)
    polar = ComPolar64(mag, phase)
    result = pcf.polar_dropout2d(polar, p=0.5, training=True)
    scale = 1.0 / (1.0 - 0.5)  # 2.0
    expected_max = mag.max() * scale
    assert (result.get_magnitude() <= expected_max).all()
    assert (result.get_magnitude() >= 0).all()
    assert torch.allclose(result.get_phase(), phase)

def test_polar_upsample():
    mag = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    phase = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    polar = ComPolar64(mag, phase)
    result = pcf.polar_upsample(polar, scale_factor=2, mode='nearest')
    expected_mag = torch.nn.functional.interpolate(mag, scale_factor=2, mode='nearest')
    expected_phase = torch.nn.functional.interpolate(phase, scale_factor=2, mode='nearest')
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), expected_phase)

def test_polar_stack():
    a = ComPolar64(torch.tensor([1.0]), torch.tensor([0.1]))
    b = ComPolar64(torch.tensor([2.0]), torch.tensor([0.2]))
    result = pcf.polar_stack([a, b], dim=0)
    expected_mag = torch.stack([a.get_magnitude(), b.get_magnitude()], dim=0)
    expected_phase = torch.stack([a.get_phase(), b.get_phase()], dim=0)
    assert torch.allclose(result.get_magnitude(), expected_mag)
    assert torch.allclose(result.get_phase(), expected_phase)