"""Plane state: everything the GUI manipulates lives here as (center, u, v, lu, lv).
All units Bohr internally; Angstrom only at the UI boundary (1 A = 1.889726 Bohr).
Multiwfn expert-mode plane (type 6) = corner + vec1 + vec2, where
corner = center - (lu*u + lv*v)/2, vec1 = lu*u, vec2 = lv*v.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

BOHR_PER_A = 1.88972612462577


@dataclass(frozen=True)
class Plane:
    center: np.ndarray    # window center, Bohr
    u: np.ndarray         # unit in-plane axis (map +x)
    v: np.ndarray         # unit in-plane axis (map +y); n = cross(u, v)
    lu: float             # window full length along u, Bohr
    lv: float             # window full length along v, Bohr

    @property
    def n(self) -> np.ndarray:
        return np.cross(self.u, self.v)

    def corner(self) -> np.ndarray:
        return self.center - 0.5 * (self.lu * self.u + self.lv * self.v)

    def vec1(self) -> np.ndarray:
        return self.lu * self.u

    def vec2(self) -> np.ndarray:
        return self.lv * self.v

    # ---- transforms: each returns a new Plane ------------------------------
    def step(self, d_bohr: float) -> "Plane":
        """Move parallel to itself along +n (VESTA-style slice stepping)."""
        return replace(self, center=self.center + d_bohr * self.n)

    def pan(self, du_bohr: float, dv_bohr: float) -> "Plane":
        return replace(self, center=self.center + du_bohr * self.u + dv_bohr * self.v)

    def zoom(self, fu: float, fv: float | None = None) -> "Plane":
        fv = fu if fv is None else fv
        return replace(self, lu=self.lu * fu, lv=self.lv * fv)

    def spin(self, deg: float) -> "Plane":
        """Rotate the in-plane axes around n (map spins inside the fixed frame)."""
        t = np.deg2rad(deg)
        u2 = np.cos(t) * self.u + np.sin(t) * self.v
        v2 = -np.sin(t) * self.u + np.cos(t) * self.v
        return replace(self, u=u2, v=v2)

    def tilt_u(self, deg: float) -> "Plane":
        """Tip the plane around its u-axis (v swings toward the normal);
        center and window size stay, the section orientation changes."""
        t = np.deg2rad(deg)
        return replace(self, v=np.cos(t) * self.v + np.sin(t) * self.n)

    def tilt_v(self, deg: float) -> "Plane":
        """Tip the plane around its v-axis (u swings toward the normal)."""
        t = np.deg2rad(deg)
        return replace(self, u=np.cos(t) * self.u + np.sin(t) * self.n)

    def offset_bohr(self, reference: "Plane") -> float:
        """Signed displacement of center along reference normal (for readouts)."""
        return float(np.dot(self.center - reference.center, reference.n))

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_atoms(cls, atoms_xyz: list[np.ndarray], margin: float = 1.0) -> "Plane":
        a1, a2, a3 = (np.asarray(a, dtype=float) for a in atoms_xyz[:3])
        n = np.cross(a2 - a1, a3 - a1)
        if np.linalg.norm(n) < 1e-8:
            raise ValueError("the three atoms are collinear - no plane defined")
        n = n / np.linalg.norm(n)
        u = a2 - a1
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        center = (a1 + a2 + a3) / 3.0
        du = max(abs(np.dot(a - center, u)) for a in (a1, a2, a3))
        dv = max(abs(np.dot(a - center, v)) for a in (a1, a2, a3))
        return cls(center=center, u=u, v=v, lu=2 * du * margin, lv=2 * dv * margin)

    def fmt(self) -> str:
        c = self.corner()
        return (f"center={np.round(self.center,3).tolist()} "
                f"lu={self.lu:.2f} lv={self.lv:.2f} Bohr "
                f"offset_dir={np.round(self.n,3).tolist()}")
