"""Persistent Multiwfn session: one subprocess, cube loaded once, renders on demand.

Talks to Multiwfn purely through its documented interactive menus (same as a
human typing); Multiwfn itself is never modified or redistributed.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .plane import Plane

# ---------------------------------------------------------------- markers --
MAIN_MENU = "Main function menu"
FUNCTIONS_LIST = "Available real space functions"
MAP_KIND = "Draw which kind of map"
GRIDS = "How many grids"
PLANE_MENU = "Please define the plane"
ORIGIN_PROMPT = "Input origin"
VEC1_PROMPT = "transitional vector 1"
VEC2_PROMPT = "transitional vector 2"
POST_MENU = "Post-processing menu"
SAVED = "Graph have been saved"
TOPO_MENU = "Topology analysis"
MODIFY_CPS = "Modify or export found CPs"
CPS_PATH_PROMPT = "Input file path"
CPS_RETRIEVED = "retrieved"
ATOMS_PROMPT = "Input index of three atoms"
EXTENSION_PROMPT = "Input extension distance"
CP_SUB_MENU = "Disable showing (3,-3) CPs"    # CP-display submenu after save
PATHS_DONE = "Delete interbasin paths"          # option 6 flips to this when generated
LABEL_VALUE_PROMPT = 'suffix, e.g. 0.45 A'      # option 17 value prompt
LABEL_LIGHT_PROMPT = "light face type"          # option 17 follow-up y/n


class DesyncError(RuntimeError):
    """Multiwfn said something unexpected - session must be rebuilt."""


@dataclass
class RenderStyle:
    map_type: int = 6          # 1 filled, 2 contours, 6 gradient lines+contours
    grid: str = "300,300"
    bonds: bool = True
    bond_color: int = 3        # blue
    angstrom: bool = True      # -8 in post-menu
    paths: bool = False        # draw interbasin (bond) paths - needs CPs loaded
    label_bohr: float | None = None   # atom-label reach in Bohr (opt 17, light face)


@dataclass
class MultiwfnSession:
    multiwfn: str
    cube: str
    settings_ini: str
    session_dir: str = ".session"
    nthreads_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_dir = os.path.abspath(self.session_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        self.render_count = 0
        self._buf = ""
        self._scan = 0
        self._menu = "boot"
        self._styled = False
        self._bonds_on = False
        self._labels_on = True
        self._bond_color = 3
        self.label_threshold = None            # re-apply after session rebuilds
        self._proc = None
        self.cube_loaded = False
        self.dots_file = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._prepare_settings()
        self.start()

    # ------------------------------------------------------------ plumbing --
    def _prepare_settings(self) -> None:
        txt = open(self.settings_ini).read()
        txt = re.sub(r"^  isilent=.*$", "  isilent= 1", txt, flags=re.M)
        txt = re.sub(r"^  graphformat=.*$", "  graphformat= png", txt, flags=re.M)
        with open(os.path.join(self.session_dir, "settings.ini"), "w") as f:
            f.write(txt)

    def start(self) -> None:
        env = dict(os.environ)
        env["Multiwfnpath"] = self.session_dir
        env.setdefault("OMP_STACKSIZE", "64000000")
        if "DISLIN" not in env and os.environ.get("DISLIN"):
            env["DISLIN"] = os.environ["DISLIN"]
        env.update(self.nthreads_env)
        self._proc = subprocess.Popen(
            [self.multiwfn], cwd=self.session_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        self._buf, self._scan = "", 0
        self._menu = "boot"
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for chunk in iter(lambda: self._proc.stdout.read(1), ""):
            with self._lock:
                self._buf += chunk
                if len(self._buf) > 400_000:
                    self._buf = self._buf[-200_000:]
                    self._scan = max(0, self._scan - 200_000)
                self._cond.notify_all()

    def send(self, line: str) -> None:
        if not self._proc or self._proc.poll() is not None:
            raise DesyncError("multiwfn process is gone")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def wait_either(self, markers: list, timeout: float = 30.0) -> str:
        """Wait until any of `markers` appears in new output; return that marker."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                hits = [(self._buf.find(m, self._scan), m) for m in markers]
                hits = [(i, m) for i, m in hits if i >= 0]
                if hits:
                    i, m = min(hits, key=lambda t: t[0])
                    self._scan = i + len(m)
                    return m
                if time.time() > deadline:
                    tail = self._buf[-1500:]
                    raise DesyncError(
                        f"timeout waiting for any of {markers}; tail:\n{tail}")
                self._cond.wait(0.25)

    def wait_for(self, marker: str, timeout: float = 60.0) -> str:
        """Wait until `marker` appears in *new* output; return consumed text."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                i = self._buf.find(marker, self._scan)
                if i >= 0:
                    end = i + len(marker)
                    chunk = self._buf[self._scan:end]
                    self._scan = end
                    return chunk
                if time.time() > deadline:
                    tail = self._buf[-1500:]
                    raise DesyncError(
                        f"timeout waiting for {marker!r}; buffer tail:\n{tail}")
                self._cond.wait(0.25)

    # ------------------------------------------------------------- actions --
    def load_cube(self, timeout: float = 300.0) -> None:
        self.send(self.cube)
        try:
            self.wait_for(MAIN_MENU, timeout)
        except DesyncError:
            self.send("")                       # e.g. non-orthogonal-cell warning
            self.wait_for(MAIN_MENU, timeout)
        self._menu = "main"
        self.cube_loaded = True

    def load_dots(self, cps_path: str) -> None:
        """Install a CPs file as CPs.txt in the session dir and load it."""
        shutil.copy(cps_path, os.path.join(self.session_dir, "CPs.txt"))
        self.send("2");  self.wait_for(TOPO_MENU)
        self.send("-4"); self.wait_for(MODIFY_CPS)
        self.send("5");  self.wait_for(CPS_PATH_PROMPT)
        self.send("");   self.wait_for(CPS_RETRIEVED)
        self.send("0");  self.wait_for(TOPO_MENU)
        self.send("-10"); self.wait_for(MAIN_MENU)
        self._menu = "main"
        self.dots_file = cps_path

    def _to_main(self) -> None:
        """Navigate back to the main menu from wherever we are.
        Menu chain after a save: CP-display submenu -> post-menu -> main."""
        if self._menu == "cp_sub":
            self.send("0"); self.wait_for(POST_MENU)
            self._menu = "post"
        if self._menu == "post":
            self.send("-5"); self.wait_for(MAIN_MENU)
        self._menu = "main"

    def _post_styling(self, style: RenderStyle) -> None:
        """Bonds/units are TOGGLES - apply once per session, never re-send."""
        if self._styled:
            return
        self._bond_color = style.bond_color
        if style.bonds:
            self.send("8")
            self.send(str(style.bond_color))
            self._bonds_on = True
        if style.angstrom:
            self.send("-8")
        self.wait_for(POST_MENU)
        self._styled = True

    def _finish_save(self, out_path: str) -> float:
        """Save via the proven pattern: first `0` save (may write the previous
        frame when isilent=1 - discarded), move its file AWAY, then `-1` redraw
        + `0` save = the current plane. Moving the intermediate file away is
        essential (dislin.png replacement is unreliable otherwise)."""
        t0 = time.time()
        src = os.path.join(self.session_dir, "dislin.png")
        self.send("0"); self.wait_for(SAVED, timeout=90)
        for _ in range(50):
            if os.path.exists(src):
                break
            time.sleep(0.1)
        if os.path.exists(src):                  # discard stale frame
            shutil.move(src, os.path.join(self.session_dir, "dislin_prev.png"))
        self.send("-1"); time.sleep(1.5)         # redraw with current plane
        self.send("0"); self.wait_for(SAVED, timeout=90)
        for _ in range(50):
            if os.path.exists(src):
                break
            time.sleep(0.1)
        shutil.move(src, out_path)
        self.render_count += 1
        # Post-save menu DEPENDS on map type: gradient maps drop into the
        # CP-display submenu, filled/contour maps reprint the post-menu.
        try:
            which = self.wait_either([CP_SUB_MENU, POST_MENU], timeout=30)
            self._menu = "cp_sub" if which == CP_SUB_MENU else "post"
        except DesyncError:
            self._menu = "post"
        return time.time() - t0

    def _start_render(self, style: RenderStyle) -> None:
        self._to_main()
        self.send("4");   self.wait_for(FUNCTIONS_LIST)
        self.send("100"); self.wait_for(MAP_KIND)
        self.send(str(style.map_type)); self.wait_for(GRIDS)
        self.send(style.grid); self.wait_for(PLANE_MENU)

    def render(self, plane: Plane, style: RenderStyle, out_path: str) -> float:
        """Expert-mode (type 6) render. Returns save-stage wall time (s).
        NB: Multiwfn expects per-grid-point STEP vectors (see 0123dim.f90:
        dist = |v| * ngrid), so full window vectors are divided by ngrid-1."""
        self._start_render(style)
        n1, n2 = (int(x) for x in style.grid.split(","))
        self.send("6");  self.wait_for(ORIGIN_PROMPT)
        c = plane.corner()
        self.send(", ".join(f"{x:.6f}" for x in c))
        self.wait_for(VEC1_PROMPT)
        self.send(", ".join(f"{x:.6f}" for x in plane.vec1() / (n1 - 1)))
        self.wait_for(VEC2_PROMPT)
        self.send(", ".join(f"{x:.6f}" for x in plane.vec2() / (n2 - 1)))
        self.wait_for(POST_MENU, timeout=120)
        self._menu = "post"
        self._post_styling(style)
        if style.paths and self.dots_file:
            # interbasin paths are PLANE-SPECIFIC: regenerate every render
            self.send("6")
            self.wait_for(PATHS_DONE, timeout=120)
        if style.label_bohr is not None and self.label_threshold != style.label_bohr:
            self.set_label_threshold(style.label_bohr)
            self.label_threshold = style.label_bohr
        return self._finish_save(out_path)

    # ------------------------------------------------- live plot options --
    def ensure_post(self) -> None:
        """Be at the post-menu (navigate out of the CP submenu if needed)."""
        if self._menu == "cp_sub":
            self.send("0"); self.wait_for(POST_MENU)
            self._menu = "post"
        elif self._menu != "post":
            raise DesyncError("not at a post-menu - render something first")

    def set_label_threshold(self, bohr: float, light: bool | None = None) -> None:
        """Post-menu 17: distance (Bohr) at which atom labels (and CP dots /
        paths) appear. light=True adds faint labels for farther atoms; by
        default only when the reach is deliberately wide (>= 2 Bohr)."""
        if light is None:
            light = bohr >= 2.0
        self.ensure_post()
        self.send("17"); self.wait_for(LABEL_VALUE_PROMPT)
        self.send(f"{bohr}")
        self.wait_for(LABEL_LIGHT_PROMPT)
        self.send("y" if light else "n")
        self.wait_for(POST_MENU)

    def toggle_bonds(self, on: bool) -> None:
        """Post-menu 8 is a toggle; enabling additionally asks for a color."""
        self.ensure_post()
        if on and not self._bonds_on:
            self.send("8")
            self.send(str(self._bond_color))
            self.wait_for(POST_MENU)
            self._bonds_on = True
        elif not on and self._bonds_on:
            self.send("8")
            self.wait_for(POST_MENU)
            self._bonds_on = False

    def resave(self, out_path: str) -> float:
        """Re-save the current graph (after live option changes), no re-render."""
        return self._finish_save(out_path)

    def render_by_atoms(self, atoms: tuple[int, int, int], style: RenderStyle,
                        out_path: str, extension: float = 0.0) -> float:
        """Reference render: plane type 4 (three atoms) + extension."""
        self._start_render(style)
        self.send("0"); self.wait_for(EXTENSION_PROMPT)
        self.send(f"{extension:.5f}")
        self.wait_for(PLANE_MENU)
        self.send("4"); self.wait_for(ATOMS_PROMPT)
        self.send(",".join(str(a) for a in atoms))
        self.wait_for(POST_MENU, timeout=120)
        self._menu = "post"
        self._post_styling(style)
        return self._finish_save(out_path)

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self.send("q"); self.send("q"); self.send("q")
            except OSError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
