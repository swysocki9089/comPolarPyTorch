import math
import torch
class ComPolar64:
    ###
    # Wrapper for both cartesian and polar representations.
    ###

    ###
    # INITIALIZATION AND CONSTRUCTORS
    # These methods handle creation from magnitude/phase or other formats
    ###
    def __init__(self, magnitude, phase):
        # Ensure both are tensors
        magnitude = torch.as_tensor(magnitude)
        phase = torch.as_tensor(phase)

        # Handle negative magnitude
        negative_mask = magnitude < 0
        if negative_mask.any():
            magnitude = torch.abs(magnitude)
            phase = phase + math.pi * negative_mask.float()

        # Wrap phase to (-π, π]
        phase = (phase + math.pi) % (2 * math.pi) - math.pi

        self.magnitude = magnitude
        self.phase = phase
        self._cartesian = self.to_cartesian()
        self._polar = self.polar_encoded()

    @classmethod
    def from_cartesian(cls, c: torch.Tensor):
        assert torch.is_complex(c), "Input must be a complex tensor"
        magnitude = torch.abs(c)
        phase = torch.angle(c)
        return cls(magnitude, phase)

    @classmethod
    def from_polar(cls, encoded: torch.Tensor):
        magnitude = encoded.real
        phase = encoded.imag
        return cls(magnitude, phase)

    ###
    # ENCODING AND CONVERSION FUNCTIONS
    # Methods for translating between representations
    ###
    def polar_encoded(self):
        """Returns a complex64 where real=magnitude and imag=phase"""
        return torch.complex(self.magnitude, self.phase).to(torch.complex64)

    def to_cartesian(self):
        """Converts internal polar data to cartesian and caches it"""
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        self._cartesian = torch.complex(real, imag).to(torch.complex64)
        return self._cartesian

    ###
    # ATTRIBUTE ACCESSORS
    # Basic getters for magnitude and phase
    ###
    def get_magnitude(self):
        return self.magnitude

    def get_phase(self):
        return self.phase

    ###
    # ARITHMETIC OPERATIONS
    # Operator overloads for +, *, == 
    ###
    def __add__(self, other):
        self_cartesian = self._cartesian

        if isinstance(other, ComPolar64):
            other_cartesian = other._cartesian
            result = self_cartesian + other_cartesian
        elif isinstance(other, torch.Tensor) and torch.is_complex(other):
            result = self_cartesian + other
        else:
            return NotImplemented

        return ComPolar64.from_cartesian(result)

    def __mul__(self, other):
        if isinstance(other, ComPolar64):
            magnitude = self.get_magnitude() * other.get_magnitude()
            phase = self.get_phase() + other.get_phase()
            return ComPolar64(magnitude, phase)
        elif isinstance(other, (int, float, torch.Tensor)):
            return ComPolar64(self.magnitude * other, self.phase)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, ComPolar64):
            return torch.allclose(self._cartesian, other._cartesian)
        elif torch.is_complex(other):
            return torch.allclose(self._cartesian, other)
        return NotImplemented

    ###
    # REPRESENTATION AND DEBUGGING
    ###
    def __repr__(self):
        return f"ComPolar64(magnitude={self.magnitude}, phase={self.phase})"

    ###
    # MODIFICATION / TRANSFORMATION METHODS
    # These return a new instance with modified magnitude/phase
    ###
    def scale(self, scalar):
        """Return a new instance scaled in magnitude"""
        new_magnitude = self.magnitude * scalar
        return ComPolar64(new_magnitude, self.phase)

    def rotate(self, angle):
        """Return a new instance rotated by given angle (radians)"""
        new_phase = self.phase + angle
        return ComPolar64(self.magnitude, new_phase)

    def set_magnitude(self, new_magnitude):
        """Return a new instance with magnitude replaced"""
        return ComPolar64(new_magnitude, self.phase)

    def set_phase(self, new_phase):
        """Return a new instance with phase replaced"""
        return ComPolar64(self.magnitude, new_phase)
