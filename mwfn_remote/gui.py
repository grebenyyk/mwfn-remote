"""mwfn-remote GUI: steer a live Multiwfn plane-map session.

Gestures (act on the DATA WINDOW, not the picture; each triggers a re-render
through Multiwfn, ~6-9 s on a unit-cell cube):
  drag            pan the section under the fixed frame (with live preview)
  wheel           zoom in/out, anchored at the cursor
  shift+drag      move the plane parallel to itself (VESTA-style, in Angstrom)
  ArrowUp/Down    step the plane by +-step_a (shift+Arrow = 1 A)
  r / R           spin the map in-plane by -/+ spin_deg
"""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

from .config import load_config
from .cube import Cube
from .plane import Plane, BOHR_PER_A
from .session import MultiwfnSession, RenderStyle, DesyncError

MAP_TYPES = {"gradient lines + contours": 6, "contour lines": 2,
             "color-filled": 1, "relief": 3}


# --------------------------------------------------------------------------
class RenderWorker(threading.Thread):
    """Owns the Multiwfn session; always renders the newest requested state."""

    def __init__(self, cfg, plane0, style, dots):
        super().__init__(daemon=True)
        self.cfg, self.style, self.dots = cfg, style, dots
        self.req, self.req_evt = None, threading.Event()
        self.results = queue.Queue()
        self.cmds = queue.Queue()          # callables run in the worker thread
        self.session = None
        self.dots_count = None
        self.last_plane = plane0
        self.render_dir = os.path.join(os.path.dirname(cfg["_path"]), "renders")

    def out_path(self):
        return os.path.join(self.render_dir, "_current.png")

    def request(self, plane):
        self.req = plane
        self.last_plane = plane
        self.req_evt.set()

    def submit(self, fn):
        """Run fn(worker) in the worker thread, then refresh the display."""
        self.cmds.put(fn)
        self.req_evt.set()

    def stop(self):
        self.req_evt.set()

    def _post(self, *msg):
        self.results.put(msg)

    def _dots_path(self):
        p = self.cfg["data"].get(f"cps_{self.dots}")
        if not p:
            return None
        if not os.path.isabs(p):
            p = os.path.join(self.cfg["data"].get("workdir", "."), p)
        return p if os.path.exists(p) else None

    def _new_session(self):
        self._post("status", "starting Multiwfn, loading cube ...")
        self.session = MultiwfnSession(
            multiwfn=self.cfg["multiwfn"]["path"],
            cube=self.cfg["data"]["cube"],
            settings_ini=self.cfg["multiwfn"]["settings_ini"],
            session_dir=os.path.join(os.path.dirname(self.cfg["_path"]), ".session"))
        self.session.load_cube()
        self.dots_count = None
        if self.dots != "none":
            dp = self._dots_path()
            if dp:
                self.session.load_dots(dp)
                with open(dp) as f:
                    self.dots_count = int(f.readline().split()[0])
        self._post("status", "session ready")

    def run(self):
        consecutive_failures = 0
        while True:
            self.req_evt.wait()
            self.req_evt.clear()
            plane, self.req = self.req, None
            while self.req_evt.is_set():           # coalesce: newest wins
                self.req_evt.clear()
                plane, self.req = self.req, None
            # 1) commands (bonds toggle, label threshold, ...)
            while True:
                try:
                    fn = self.cmds.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn(self)
                    if self.session is not None:
                        self._post("png", self.out_path(), 0.0, self.last_plane)
                except Exception as e:
                    self._post("error", f"{type(e).__name__}: {e}")
            # 2) render request
            if plane is None:
                continue
            t0 = time.time()
            try:
                if self.session is None:
                    self._new_session()
                elif self.dots != "none" and self.session.dots_file != self._dots_path():
                    self._rebuild()
                elif self.dots == "none" and self.session.dots_file:
                    self._rebuild()
                self.session.render(plane, self.style, self.out_path())
                self._post("png", self.out_path(), time.time() - t0, plane)
                consecutive_failures = 0
            except DesyncError as e:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    self._post("error", f"render failed twice - giving up "
                                        f"(see ~/.mwfn-remote.log). Tail: "
                                        f"{str(e)[-300:]}")
                    consecutive_failures = 0
                    continue
                self._post("status", "desync - rebuilding session, retrying ...")
                self._rebuild()
                self.req = plane
                self.req_evt.set()                 # retry with fresh session
            except Exception as e:
                self._post("error", f"{type(e).__name__}: {e}")

    def _rebuild(self):
        try:
            self.session.close()
        except Exception:
            pass
        self.session = None
        self._new_session()


# --------------------------------------------------------------------------
def detect_plot_box(arr):
    """Pixel bounds (x0,x1,y0,y1) of the DISLIN axis frame in the render."""
    g = np.asarray(arr)[..., :3].mean(-1)
    dark = g < 0.5
    h, w = dark.shape
    rows = np.where(dark.sum(1) > w * 0.45)[0]
    cols = np.where(dark.sum(0) > h * 0.45)[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


class App:
    def __init__(self, root: tk.Tk, cfg: dict):
        self.root, self.cfg = root, cfg
        view = cfg.get("view", {})
        self.step_a = float(view.get("step_a", 0.25))
        self.zoom_factor = float(view.get("zoom_factor", 1.2))
        self.spin_deg = float(view.get("spin_deg", 15.0))
        self.tilt_deg = float(view.get("tilt_deg", 5.0))
        self.tilt_u_total = 0.0
        self.tilt_v_total = 0.0

        r = cfg.get("render", {})
        self.style = RenderStyle(
            map_type=int(r.get("map_type", 6)),
            grid=str(r.get("grid", "300,300")),
            bonds=bool(r.get("bonds", True)),
            bond_color=int(r.get("bond_color", 3)),
            angstrom=bool(r.get("angstrom", True)),
            label_bohr=float(r.get("label_bohr", 0.5)))
        self.dots = "none"
        self.dots_avail = {d: cfg["data"].get(f"cps_{d}") for d in ("all", "bonded")}

        cube = Cube.parse(cfg["data"]["cube"])
        i1, i2, i3 = cfg["plane"]["atoms"]
        self.base = Plane.from_atoms(
            [cube.atoms[i - 1][2] for i in (i1, i2, i3)],
            margin=float(cfg["plane"].get("margin", 1.0)))
        # Window size: explicit size_a (Angstrom) wins; otherwise SQUARE with
        # the max triangle extent - avoids thin-sliver maps from sliver triangles.
        size_a = cfg["plane"].get("size_a")
        if size_a:
            self.base = replace(self.base, lu=float(size_a[0]) * BOHR_PER_A,
                                lv=float(size_a[1]) * BOHR_PER_A)
        else:
            L = max(self.base.lu, self.base.lv)
            self.base = replace(self.base, lu=L, lv=L)
        self.plane = self.base
        self.box = None                     # plot-frame px bounds
        self.render_t = None

        root.title("mwfn-remote")
        root.geometry("1200x820")            # fits the whole side panel + canvas
        root.minsize(980, 760)
        self._build_widgets()
        self._bind_gestures()
        # make sure macOS considers the window frontmost so taps register as
        # normal clicks (a background-process Tk window eats light taps)
        root.lift(); root.focus_force()
        root.after(300, lambda: (root.lift(), root.focus_force()))

        self.worker = RenderWorker(cfg, self.plane, self.style, self.dots)
        self.worker.start()
        self.worker.request(self.plane)
        root.after(120, self._poll)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI --
    def _build_widgets(self):
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=(9.2, 7.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        side = ttk.Frame(main, width=230, padding=(8, 0))
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        ttk.Label(side, text="CP dots").pack(anchor="w", pady=(4, 0))
        self.dots_var = tk.StringVar(value="none")
        for d in ("none", "all", "bonded"):
            tt = ttk.Radiobutton(side, text=d, value=d, variable=self.dots_var,
                                 command=self._on_dots)
            if d != "none" and not self.dots_avail.get(d):
                tt.state(["disabled"])
            tt.pack(anchor="w")

        ttk.Label(side, text="Map type").pack(anchor="w", pady=(10, 0))
        # radios, NOT a combobox: the macOS Tk combobox popup freezes the app
        self.map_var = tk.StringVar(value=[k for k, v in MAP_TYPES.items()
                                           if v == self.style.map_type][0])
        for label in MAP_TYPES:
            ttk.Radiobutton(side, text=label, value=label, variable=self.map_var,
                            command=self._on_maptype).pack(anchor="w")

        ttk.Label(side, text="Grid (n1,n2)").pack(anchor="w", pady=(10, 0))
        self.grid_var = tk.StringVar(value=self.style.grid)
        ge = ttk.Entry(side, textvariable=self.grid_var, width=12)
        ge.pack(fill="x"); ge.bind("<Return>", self._on_grid)

        ttk.Label(side, text="Overlays").pack(anchor="w", pady=(10, 0))
        self.bonds_var = tk.BooleanVar(value=self.style.bonds)
        ttk.Checkbutton(side, text="chemical bonds",
                        variable=self.bonds_var,
                        command=self._on_bonds).pack(anchor="w")

        ttk.Label(side, text="Label/dot/path reach (Bohr)").pack(anchor="w", pady=(10, 0))
        row = ttk.Frame(side); row.pack(fill="x")
        self.label_var = tk.StringVar(value=f"{self.style.label_bohr:g}")
        le = ttk.Entry(row, textvariable=self.label_var, width=6)
        le.pack(side="left"); le.bind("<Return>", self._on_label)
        ttk.Button(row, text="Apply", width=6, command=self._on_label).pack(side="left", padx=4)

        sep = ttk.Separator(side); sep.pack(fill="x", pady=10)
        ttk.Button(side, text="Zoom +", command=lambda: self._zoom(1 / self.zoom_factor)
                   ).pack(fill="x", pady=1)
        ttk.Button(side, text="Zoom −", command=lambda: self._zoom(self.zoom_factor)
                   ).pack(fill="x", pady=1)
        ttk.Button(side, text=f"Step +{self.step_a:g} Å",
                   command=lambda: self._step(+self.step_a)).pack(fill="x", pady=1)
        ttk.Button(side, text=f"Step −{self.step_a:g} Å",
                   command=lambda: self._step(-self.step_a)).pack(fill="x", pady=1)
        ttk.Button(side, text="Spin ↻", command=lambda: self._spin(+self.spin_deg)
                   ).pack(fill="x", pady=1)
        ttk.Button(side, text="Spin ↺", command=lambda: self._spin(-self.spin_deg)
                   ).pack(fill="x", pady=1)
        trow = ttk.Frame(side); trow.pack(fill="x", pady=1)
        ttk.Button(trow, text="Tilt u −", width=7,
                   command=lambda: self._tilt(-self.tilt_deg, 0)).pack(side="left", expand=True, fill="x")
        ttk.Button(trow, text="Tilt u +", width=7,
                   command=lambda: self._tilt(+self.tilt_deg, 0)).pack(side="left", expand=True, fill="x")
        trow2 = ttk.Frame(side); trow2.pack(fill="x", pady=1)
        ttk.Button(trow2, text="Tilt v −", width=7,
                   command=lambda: self._tilt(0, -self.tilt_deg)).pack(side="left", expand=True, fill="x")
        ttk.Button(trow2, text="Tilt v +", width=7,
                   command=lambda: self._tilt(0, +self.tilt_deg)).pack(side="left", expand=True, fill="x")
        # pan: content moves in the window, axes stay
        prow = ttk.Frame(side); prow.pack(fill="x", pady=(6, 1))
        ttk.Button(prow, text="◀", width=4,
                   command=lambda: self._pan_frac(-0.25, 0)).pack(side="left", expand=True, fill="x")
        updn = ttk.Frame(prow); updn.pack(side="left", expand=True, fill="x")
        ttk.Button(updn, text="▲", width=4,
                   command=lambda: self._pan_frac(0, +0.25)).pack(fill="x", pady=1)
        ttk.Button(updn, text="▼", width=4,
                   command=lambda: self._pan_frac(0, -0.25)).pack(fill="x", pady=1)
        ttk.Button(prow, text="▶", width=4,
                   command=lambda: self._pan_frac(+0.25, 0)).pack(side="left", expand=True, fill="x")
        ttk.Button(side, text="Reset view",
                   command=self._reset).pack(fill="x", pady=(1, 10))
        ttk.Button(side, text="Save PNG ...", command=self._save_png).pack(fill="x")

        self.status = tk.StringVar(value="starting ...")
        # inset + flat: a full-width sunken bar pokes past the rounded window
        # corners on macOS and leaves ugly white wedges; compact vertical padding
        ttk.Label(self.root, textvariable=self.status, relief="flat",
                  anchor="w", padding=(8, 1)
                  ).pack(fill="x", side="bottom", padx=10, pady=(0, 4))

    # ---------------------------------------------------------- gestures --
    def _bind_gestures(self):
        w = self.canvas.mpl_connect
        w("button_press_event", self._press)
        w("motion_notify_event", self._motion)
        w("button_release_event", self._release)
        w("scroll_event", self._scroll)
        # all keys on the ROOT window (work regardless of widget focus, and
        # immune to the macOS tap-vs-press issue on Tk widgets)
        B = self.root.bind
        B("<r>", lambda e: self._spin(-self.spin_deg))
        B("<R>", lambda e: self._spin(+self.spin_deg))
        B("<t>", lambda e: self._tilt(-self.tilt_deg, 0))
        B("<T>", lambda e: self._tilt(+self.tilt_deg, 0))
        B("<y>", lambda e: self._tilt(0, -self.tilt_deg))
        B("<Y>", lambda e: self._tilt(0, +self.tilt_deg))
        B("<Up>", lambda e: self._step(+self.step_a))
        B("<Down>", lambda e: self._step(-self.step_a))
        B("<Shift-Up>", lambda e: self._step(+1.0))
        B("<Shift-Down>", lambda e: self._step(-1.0))
        B("<plus>", lambda e: self._zoom(1 / self.zoom_factor))
        B("<minus>", lambda e: self._zoom(self.zoom_factor))
        B("<Left>", lambda e: self._pan_frac(-0.25, 0))
        B("<Right>", lambda e: self._pan_frac(+0.25, 0))
        B("<b>", self._key_bonds)
        B("<d>", self._key_dots)
        B("<m>", self._key_maptype)
        B("<s>", lambda e: self._save_png())
        B("<Escape>", lambda e: self._reset())

    # ------------------------------------------------- keyboard toggles --
    def _key_bonds(self, _=None):
        self.bonds_var.set(not self.bonds_var.get())
        self._on_bonds()

    def _key_dots(self, _=None):
        cycle = {"none": "all", "all": "bonded", "bonded": "none"}
        self.dots_var.set(cycle[self.dots_var.get()])
        self._on_dots()

    def _key_maptype(self, _=None):
        keys = list(MAP_TYPES)
        i = keys.index(self.map_var.get())
        self.map_var.set(keys[(i + 1) % len(keys)])
        self._on_maptype()

    def _px_per_bohr(self):
        if self.box and self.im is not None:
            x0, x1, y0, y1 = self.box
            return (x1 - x0) / self.plane.lu, (y1 - y0) / self.plane.lv
        return None

    def _press(self, ev):
        if ev.in_axes and ev.button == 1:
            self._drag = (ev.x, ev.y, bool(ev.guiEvent.state & 0x0001))
            self._moved = 0.0

    def _motion(self, ev):
        d = getattr(self, "_drag", None)
        if not d or self.im is None:
            return
        x0, y0, shift = d
        dx, dy = ev.x - x0, ev.y - y0
        if shift:      # parallel-plane drag: accumulate vertical motion (px)
            self._moved = -dy
            return
            # pan preview: slide the picture with the cursor
            self.im.set_extent([dx, W + dx, H + dy, dy])
            self.canvas.draw_idle()
            self._drag = (ev.x, ev.y, shift)
            self._pan_acc = (self._pan_acc if hasattr(self, "_pan_acc") else (0, 0))
            self._pan_acc = (self._pan_acc[0] + dx, self._pan_acc[1] + dy)

    def _release(self, ev):
        d = getattr(self, "_drag", None)
        if not d:
            return
        self._drag = None
        if self.im is None or not self.box:
            return
        if d[2]:                                    # shift-drag: parallel step
            dA = self._moved / self._px_per_bohr()[1] / BOHR_PER_A
            if abs(dA) > 1e-3:
                self._step(dA)
        else:                                       # pan in the plane
            ax, ay = getattr(self, "_pan_acc", (0, 0))
            self._pan_acc = (0, 0)
            pxu, pxv = self._px_per_bohr()
            du, dv = -ax / pxu, ay / pxv
            if abs(du) + abs(dv) > 1e-3:
                self.plane = self.plane.pan(du, dv)
                self._request("pan")

    def _scroll(self, ev):
        if ev.in_axes:
            f = 1 / self.zoom_factor if ev.button == "up" else self.zoom_factor
            self._zoom(f, anchor=(ev.xdata, ev.ydata))

    # ------------------------------------------------------------ actions --
    def _zoom(self, f, anchor=None):
        self.plane = self.plane.zoom(f)
        self._request(f"zoom ×{1 / f:.2f}")

    def _step(self, d_a):
        self.plane = self.plane.step(d_a * BOHR_PER_A)
        self._request(f"step {d_a:+.2f} Å")

    def _spin(self, deg):
        self.plane = self.plane.spin(deg)
        self._request(f"spin {deg:+g}°")

    def _tilt(self, du, dv):
        if du:
            self.plane = self.plane.tilt_u(du)
            self.tilt_u_total += du
        if dv:
            self.plane = self.plane.tilt_v(dv)
            self.tilt_v_total += dv
        self._request(f"tilt u:{self.tilt_u_total:+g}° v:{self.tilt_v_total:+g}°")

    def _pan_frac(self, fu, fv):
        """Slide the CONTENT in the indicated direction (arrow = content motion);
        fractions are of the current window width/height. Sign determined
        empirically - the window centre moves opposite to the content."""
        self.plane = self.plane.pan(+fu * self.plane.lu, +fv * self.plane.lv)
        self._request(f"pan {fu:+.0%},{fv:+.0%}")

    def _reset(self):
        self.plane = self.base
        self.tilt_u_total = self.tilt_v_total = 0.0
        self._request("reset")

    def _request(self, what):
        self.status.set(f"rendering ({what}) ...")
        self.worker.request(self.plane)

    def _on_dots(self):
        self.worker.dots = self.dots_var.get()
        if self.worker.session is not None:
            self.status.set(f"switching dots to '{self.worker.dots}': "
                            f"rebuilding session (~10 s), then rendering ...")
        self._request(f"dots: {self.worker.dots}")

    def _on_maptype(self, _=None):
        self.style.map_type = MAP_TYPES[self.map_var.get()]
        self._request(f"map: {self.map_var.get()}")

    def _on_grid(self, _=None):
        try:
            n1, n2 = (int(x) for x in self.grid_var.get().split(","))
            if n1 < 20 or n2 < 20:
                raise ValueError
            self.style.grid = f"{n1},{n2}"
            self._request(f"grid {self.style.grid}")
        except ValueError:
            self.status.set("grid must be like 300,300")

    def _on_paths(self):
        pass    # bond-path overlay removed: Multiwfn's interbasin-path overlay
                # draws straight stick-like segments, not AIM-style gradient
                # paths - the gradient-lines map (type 6) already IS the AIM
                # gradient-line picture.

    def _on_bonds(self):
        on = bool(self.bonds_var.get())
        self.status.set(f"bonds {'on' if on else 'off'} - refreshing ...")

        def cmd(w):
            w.session.toggle_bonds(on)
            w.session.resave(w.out_path())
        self.worker.submit(cmd)

    def _on_labels(self):
        pass

    def _on_label(self, _=None):
        try:
            v = float(self.label_var.get())
            if not 0.1 <= v <= 6.0:
                raise ValueError
        except ValueError:
            self.status.set("reach must be 0.1..6.0 Bohr")
            return
        self.style.label_bohr = v
        self._request(f"label/dot/path reach {v:g} Bohr")

    def _save_png(self):
        if self.im is None:
            return
        tag = os.path.splitext(os.path.basename(self.cfg["data"]["cube"]))[0]
        i1, i2, i3 = self.cfg["plane"]["atoms"]
        off = self.plane.offset_bohr(self.base) / BOHR_PER_A
        sugg = (f"{tag}_{i1}-{i2}-{i3}_off{off:+.2f}A_"
                f"z{self.base.lu / self.plane.lu:.2f}.png")
        fn = filedialog.asksaveasfilename(initialfile=sugg,
                                          defaultextension=".png")
        if fn:
            shutil.copy(os.path.join(self.worker.render_dir, "_current.png"), fn)
            self.status.set(f"saved {fn}")

    # -------------------------------------------------------------- pump --
    im = None

    def _poll(self):
        try:
            while True:
                msg = self.worker.results.get_nowait()
                kind = msg[0]
                if kind == "png":
                    _, path, dt, plane = msg
                    arr = mpimg.imread(path)
                    if self.im is None:
                        self.im = self.ax.imshow(arr)
                    else:
                        self.im.set_data(arr)
                        self.im.set_extent([0, arr.shape[1], arr.shape[0], 0])
                    box = detect_plot_box(arr)
                    if box:
                        self.box = box
                    off = plane.offset_bohr(self.base) / BOHR_PER_A
                    zoom = self.base.lu / plane.lu
                    dots_note = ""
                    if self.worker.dots_count:
                        dots_note = (f"  ·  {self.worker.dots_count} CPs in set "
                                     f"(only those within 0.5 bohr of the plane are drawn)")
                    self.status.set(
                        f"plane {self.cfg['plane']['atoms']}  offset {off:+.2f} Å  "
                        f"tilt u:{self.tilt_u_total:+.0f}° v:{self.tilt_v_total:+.0f}°  "
                        f"window {plane.lu/BOHR_PER_A:.1f}×{plane.lv/BOHR_PER_A:.1f} Å  "
                        f"zoom ×{zoom:.2f}  render {dt:.1f}s  "
                        f"dots: {self.worker.dots}  map: {self.style.map_type}"
                        + dots_note)
                    self.canvas.draw_idle()
                elif kind == "status":
                    self.status.set(msg[1])
                elif kind == "error":
                    self.status.set(f"ERROR: {msg[1]}")
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _on_close(self):
        self.worker.stop()
        # kill the Multiwfn child, else it orphan-persists and keeps the
        # app's dock icon alive after the window is gone
        try:
            proc = self.worker.session._proc
            if proc is not None:
                proc.kill()
        except Exception:
            pass
        self.root.destroy()


def run_gui(config_path: str) -> None:
    cfg = load_config(config_path)
    cfg["_path"] = os.path.abspath(config_path)
    root = tk.Tk()
    app = App(root, cfg)
    root.mainloop()
    # post-mainloop safety net: no orphaned Multiwfn, no lingering dock icon
    try:
        proc = app.worker.session._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
    except Exception:
        pass
