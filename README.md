# mwfn-remote

**An unofficial remote-control GUI for [Multiwfn](http://sobereva.com/multiwfn) plane maps.**

mwfn-remote keeps one Multiwfn process alive with your cube loaded, and lets you
steer the plotting plane interactively — every gesture re-renders through
Multiwfn itself, so what you see is exactly what Multiwfn computes.

```
┌─────────────────────────────┬──────────────┐
│                             │ CP dots      │
│      rendered section       │  ○ none      │
│      (Multiwfn output)      │  ○ all       │
│                             │  ○ bonded    │
│  drag = pan   wheel = zoom  │ Map type     │
│  ⇧drag / ↑↓ = step plane    │  ... radios  │
│  t/T y/Y = tilt   r/R = spin│ bonds □      │
│                             │ label reach  │
│                             │ zoom/tilt/pan│
└─────────────────────────────┴──────────────┘
```

## Why

Multiwfn's plane maps (gradient lines + contours + critical-point dots) are the
reference QTAIM section plots, but driving them means typing obscure menu codes
in a terminal, one render at a time. mwfn-remote gives that loop a throttle:
zoom, pan, tilt and parallel-step the section until it shows what you want, then
save.

## What it drives

| control | effect |
|---|---|
| drag / `←→` / pan buttons | pan the section under the fixed frame |
| wheel / `+` `−` | zoom **the data window** (re-render, not image scaling) |
| `↑` / `↓` (⇧ = ×4) | move the plane parallel to itself (VESTA-style, Å) |
| `t`/`T`, `y`/`Y` | tilt the plane around its in-plane axes (5° steps) |
| `r` / `R` | spin the map in-plane |
| `d` | cycle CP-dot sets (none / all / bonded) |
| `m` | cycle map type (gradient lines, contours, color-filled, relief) |
| `b` | chemical bonds on/off |
| `s` | save PNG (auto-named with plane params) |
| `Esc` | reset to the configured base plane |

Typical render time is 5–10 s per step (unit-cell cube); the cube loads once
per session. A session watchdog restarts Multiwfn silently if the menu
dialogue ever desyncs.

## Install

Requirements: Python ≥ 3.10 with `numpy`, `matplotlib` (+ Tk), and your own
[Multiwfn](http://sobereva.com/multiwfn/download.html) (tested with 2026.7.x;
the binary is **not** bundled — see Ethics).

```bash
git clone https://github.com/grebenyyk/mwfn-remote
cd mwfn-remote
cp mwfn.example.toml mwfn.toml     # then edit paths + plane atoms
python -m mwfn_remote gui mwfn.toml
```

Headless batch use:

```bash
python -m mwfn_remote render mwfn.toml --offset 0,0.5,1.0 --dots all
python -m mwfn_remote verify mwfn.toml   # self-check of the plane math
```

## Config

TOML, see `mwfn.example.toml`: Multiwfn path, cube file, CP-dot set files
(Multiwfn `CPs.txt` format), base plane as three atom indices (1-based, cube
order), optional window size, map type, grid, view step sizes.

## How it talks to Multiwfn

Only through its documented interactive menus — the same keystrokes a human
would type — via a persistent subprocess. No patching, no linking, no
redistribution of Multiwfn. Some behaviours worth knowing if you hack on this
(reverse-engineered from Multiwfn's shipped Fortran, version 2026.7.15):

- expert-mode plane vectors are **per-grid-point steps** (`span = |v|·ngrid`)
- post-menu options `8` (bonds) / `-8` (Å units) are **toggles** — apply once
  per session
- with `isilent= 1` the first graph save can emit the **previous frame**;
  mwfn-remote saves, moves the file away, redraws (`-1`) and saves again
- the menu reached after a save **depends on map type** (CP submenu vs
  post-menu) and is detected dynamically
- interbasin-path overlays draw as straight stick segments, not AIM-style
  gradient trajectories (feature deliberately not used)

## Ethics & citation

Multiwfn is freeware by Tian Lu (Sobereva). This is an unofficial third-party
driver: it bundles nothing of Multiwfn, and merely automates a user-installed
copy. If you publish figures made through this tool, cite Multiwfn:

- Tian Lu, Feiwu Chen, *J. Comput. Chem.*, **33**, 580 (2012). DOI: 10.1002/jcc.22885
- Tian Lu, *J. Chem. Phys.*, **161**, 082503 (2024). DOI: 10.1063/5.0216272

## License

MIT (this code only) — see [LICENSE](LICENSE).
