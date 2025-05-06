import torch
import math
import pytest
from comPolar64 import ComPolar64

# === Helper: Coterminal phase equality ===
def assert_coterminal(phase1, phase2, atol=1e-5):
    diff = ((phase1 - phase2 + math.pi) % (2 * math.pi)) - math.pi
    if isinstance(diff, torch.Tensor):
        assert torch.allclose(diff, torch.zeros_like(diff), atol=atol)
    else:
        assert abs(diff) < atol

# ------------------------------------------------------------------------------
# 01. Conversion and Representation Tests
# ------------------------------------------------------------------------------

def test_01_cartesian_encoding_and_decoding():
    c = torch.tensor(3.0 + 4.0j)
    polar = ComPolar64.from_cartesian(c)
    cartesian = polar.to_cartesian()
    assert torch.allclose(cartesian, c, atol=1e-5)

def test_02_cartesian_to_polar_to_cartesian_cycle():
    c = torch.tensor(-1.0 - 1.0j)
    z = ComPolar64.from_cartesian(c)
    c2 = z.to_cartesian()
    assert torch.allclose(c, c2, atol=1e-5)

def test_03_polar_encoding_and_decoding():
    original = ComPolar64(torch.tensor(2.0), torch.tensor(math.pi / 2))
    encoded = original._polar
    decoded = ComPolar64.from_polar(encoded)
    assert torch.allclose(decoded.magnitude, original.magnitude, atol=1e-5)
    assert torch.allclose(decoded.phase, original.phase, atol=1e-5)

# ------------------------------------------------------------------------------
# 02. Arithmetic Tests (Addition and Multiplication)
# ------------------------------------------------------------------------------

def test_04_addition_of_polar_instances():
    a = ComPolar64.from_cartesian(torch.tensor(1.0 + 1.0j))
    b = ComPolar64.from_cartesian(torch.tensor(1.0 - 1.0j))
    result = a + b
    expected = torch.tensor(2.0 + 0.0j)
    assert torch.allclose(result.to_cartesian(), expected, atol=1e-5)

def test_05_addition_with_native_complex():
    a = ComPolar64.from_cartesian(torch.tensor(1.0 + 1.0j))
    result = a + torch.tensor(0.0 + 2.0j)
    expected = torch.tensor(1.0 + 3.0j)
    assert torch.allclose(result.to_cartesian(), expected, atol=1e-5)

def test_06_multiplication_with_scalar():
    a = ComPolar64(torch.tensor(2.0), torch.tensor(math.pi / 4))
    scaled = a * 3
    assert torch.allclose(scaled.magnitude, torch.tensor(6.0), atol=1e-5)
    assert_coterminal(scaled.phase.item(), a.phase.item())

def test_07_multiplication_with_polar():
    a = ComPolar64(torch.tensor(2.0), torch.tensor(math.pi / 4))
    b = ComPolar64(torch.tensor(3.0), torch.tensor(math.pi / 6))
    result = a * b
    expected_mag = 6.0
    expected_phase = math.pi / 4 + math.pi / 6
    wrapped_phase = (expected_phase + math.pi) % (2 * math.pi) - math.pi
    assert torch.allclose(result.magnitude, torch.tensor(expected_mag), atol=1e-5)
    assert_coterminal(result.phase.item(), wrapped_phase)

# ------------------------------------------------------------------------------
# 03. Phase Wrapping & Sign Handling
# ------------------------------------------------------------------------------

def test_08_negative_magnitude_flips_phase():
    z = ComPolar64(torch.tensor(-1.0), torch.tensor(0.0))
    assert torch.allclose(z.magnitude, torch.tensor(1.0), atol=1e-5)
    assert_coterminal(z.phase.item(), math.pi)

def test_09_phase_wrapping_to_negative_pi_to_pi():
    z = ComPolar64(torch.tensor(1.0), torch.tensor(3 * math.pi))
    assert -math.pi - 1e-6 <= z.phase.item() <= math.pi + 1e-6

# ------------------------------------------------------------------------------
# 04. Type Safety Tests
# ------------------------------------------------------------------------------

def test_10_addition_invalid_type():
    z = ComPolar64(1.0, 0.0)
    with pytest.raises(TypeError):
        _ = z + "invalid"

def test_11_multiplication_invalid_type():
    z = ComPolar64(1.0, 0.0)
    with pytest.raises(TypeError):
        _ = z * "invalid"
