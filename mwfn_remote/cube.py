"""Gaussian cube header parsing (atoms only - the data block is never read)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Cube:
    natoms: int
    origin: np.ndarray            # Bohr
    n: np.ndarray                 # grid counts (3,)
    vec: np.ndarray               # grid vectors, rows (3,3), Bohr
    atoms: list = field(default_factory=list)   # (Z, charge, xyz Bohr)

    @classmethod
    def parse(cls, path: str) -> "Cube":
        with open(path) as f:
            f.readline(); f.readline()                    # two title lines
            p = f.readline().split()
            nat = int(p[0])
            origin = np.array([float(x) for x in p[1:4]])
            n = np.zeros(3, dtype=int); vec = np.zeros((3, 3))
            for i in range(3):
                q = f.readline().split()
                n[i] = int(q[0])
                vec[i] = [float(x) for x in q[1:4]]
            atoms = []
            for _ in range(nat):
                q = f.readline().split()
                atoms.append((int(q[0]), float(q[1]),
                              np.array([float(x) for x in q[2:5]])))
        return cls(nat, origin, n, vec, atoms)
