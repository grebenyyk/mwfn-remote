"""mwfn-remote CLI (spike stage): headless renders + self-verification.

  python -m mwfn_remote verify mwfn.toml
      render the same section via expert mode (type 6) and via three atoms
      (type 4, extension 0), compare the PNGs - proves the plane math.

  python -m mwfn_remote render mwfn.toml [--offset 0,0.5] [--zoom 1.4] \
         [--spin 15] [--dots all|bonded|none] [--grid 300,300] [--out name]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .config import load_config
from .cube import Cube
from .plane import Plane, BOHR_PER_A
from .session import MultiwfnSession, RenderStyle


def make_session(cfg: dict) -> MultiwfnSession:
    return MultiwfnSession(
        multiwfn=cfg["multiwfn"]["path"],
        cube=cfg["data"]["cube"],
        settings_ini=cfg["multiwfn"]["settings_ini"],
        session_dir=os.path.join(os.path.dirname(os.path.abspath(cfg["_path"])),
                                 ".session"),
    )


def base_plane(cfg: dict, cube: Cube) -> Plane:
    i1, i2, i3 = cfg["plane"]["atoms"]
    margin = float(cfg["plane"].get("margin", 1.0))
    return Plane.from_atoms([cube.atoms[i1 - 1][2], cube.atoms[i2 - 1][2],
                             cube.atoms[i3 - 1][2]], margin=margin)


def style_from(cfg: dict, args) -> RenderStyle:
    r = cfg.get("render", {})
    return RenderStyle(
        map_type=int(getattr(args, "map_type", None) or r.get("map_type", 6)),
        grid=getattr(args, "grid", None) or r.get("grid", "300,300"),
        bonds=bool(r.get("bonds", True)),
        bond_color=int(r.get("bond_color", 3)),
        angstrom=bool(r.get("angstrom", True)),
    )


def dots_path(cfg: dict, which: str):
    if which == "none":
        return None
    key = f"cps_{which}"
    p = cfg["data"].get(key)
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.join(cfg["data"].get("workdir", "."), p)
    return p if os.path.exists(p) else None


def cmd_verify(cfg: dict, args) -> None:
    cube = Cube.parse(cfg["data"]["cube"])
    plane = base_plane(cfg, cube)
    style = style_from(cfg, args)
    atoms = tuple(cfg["plane"]["atoms"])
    s = make_session(cfg)
    try:
        t0 = time.time(); s.load_cube(); t_load = time.time() - t0
        print(f"cube loaded in {t_load:.1f}s ({cube.natoms} atoms)")
        print(f"plane: {plane.fmt()}")
        t0 = time.time()
        s.render_by_atoms(atoms, style, "renders/verify_type4.png", extension=0.0)
        t4 = time.time() - t0
        t0 = time.time()
        s.render(plane, style, "renders/verify_type6.png")
        t6 = time.time() - t0
        import matplotlib.image as mpimg
        a = mpimg.imread("renders/verify_type4.png").astype(float)
        b = mpimg.imread("renders/verify_type6.png").astype(float)
        if a.shape != b.shape:
            print(f"IMAGE SHAPES DIFFER: {a.shape} vs {b.shape}")
            return
        diff = np.mean(np.abs(a - b))
        rel = diff / max(np.mean(a), 1e-9)
        print(f"type-4 render {t4:.1f}s | type-6 render {t6:.1f}s")
        print(f"mean |pixel diff| = {diff:.5f} ({rel * 100:.2f}% of mean intensity)")
        verdict = "MATCH - plane math OK" if rel < 0.05 else \
                  "DIFFERENT - check corner/vector semantics!"
        print(verdict)
    finally:
        s.close()


def cmd_render(cfg: dict, args) -> None:
    cube = Cube.parse(cfg["data"]["cube"])
    plane = base_plane(cfg, cube)
    style = style_from(cfg, args)
    offsets = [float(x) for x in str(args.offset).split(",")]
    s = make_session(cfg)
    try:
        t0 = time.time(); s.load_cube(); print(f"cube loaded in {time.time()-t0:.1f}s")
        dots = dots_path(cfg, args.dots)
        if dots and dots != s.dots_file:
            t0 = time.time(); s.load_dots(dots)
            print(f"dots loaded ({os.path.basename(dots)}) in {time.time()-t0:.1f}s")
        zoom = float(args.zoom)
        spin = float(args.spin)
        for off_a in offsets:
            p = plane.step(off_a * BOHR_PER_A)
            if zoom != 1.0:
                p = p.zoom(1.0 / zoom)          # zoom>1 = magnify = smaller window
            if spin != 0.0:
                p = p.spin(spin)
            name = args.out or (
                f"render_{os.path.splitext(os.path.basename(cfg['data']['cube']))[0]}"
                f"_off{off_a:+.2f}A" + (f"_z{zoom:g}" if zoom != 1 else "")
                + (f"_s{spin:g}" if spin else "") + ".png")
            out = os.path.join("renders", name)
            t0 = time.time(); s.render(p, style, out)
            print(f"{out}  (render {time.time()-t0:.1f}s)")
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="mwfn-remote")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gui", help="interactive plane-steering GUI")
    g.add_argument("config")
    for name in ("verify", "render"):
        p = sub.add_parser(name)
        p.add_argument("config")
        if name == "render":
            p.add_argument("--offset", default="0", help="A along normal, comma list ok")
            p.add_argument("--zoom", type=float, default=1.0)
            p.add_argument("--spin", type=float, default=0.0, help="degrees")
            p.add_argument("--dots", default="none", choices=["none", "all", "bonded"])
            p.add_argument("--grid", default=None)
            p.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["_path"] = os.path.abspath(args.config)
    if args.cmd == "gui":
        from .gui import run_gui
        os.makedirs(os.path.join(os.path.dirname(cfg["_path"]), "renders"),
                    exist_ok=True)
        run_gui(args.config)
        return
    os.makedirs("renders", exist_ok=True)
    {"verify": cmd_verify, "render": cmd_render}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
