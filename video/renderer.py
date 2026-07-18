"""Khan-Academy-style chalkboard renderer.

Dark board, colorful hand-drawn strokes revealed progressively while the
narrator speaks. Scenes are lists of timed events; every event knows how to
paint itself at a given progress in [0, 1]. Rendering is done at 2x and
downsampled for smooth anti-aliased lines.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
SS = 2  # supersampling factor
FPS = 30
TAIL = 1.0  # seconds of hold after narration ends
FADE = 0.5  # fade-out duration at scene end

# ---------------------------------------------------------------- palette
BG = (13, 17, 23)
CHALK = (232, 230, 221)
YELLOW = (255, 215, 94)
CYAN = (83, 200, 240)
GREEN = (124, 217, 146)
RED = (240, 113, 120)
PINK = (242, 143, 202)
PURPLE = (186, 156, 240)
ORANGE = (255, 169, 77)
TEAL = (78, 201, 176)
DIM = (138, 147, 160)
GRID = (44, 52, 63)

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_font_cache: dict = {}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def hand_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Kalam-Bold.ttf" if bold else "Kalam-Regular.ttf"
    return _font(str(FONT_DIR / name), size)


def sym_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return _font(DEJAVU_BOLD if bold else DEJAVU, size)


# characters routed to the symbol font (handwriting fonts lack these glyphs)
_SYM = set("·×→←↑↓−≈∝‖αβμπΔθℓ∈✓✗ℝ⋅≥≤⟨⟩ᵀ⁺⁻₊₋⋮∅✦⇒")


def _runs(s: str):
    """Split string into (text, is_symbol) runs for mixed-font drawing."""
    out, cur, cur_sym = [], "", False
    for ch in s:
        is_sym = ch in _SYM
        if cur and is_sym != cur_sym:
            out.append((cur, cur_sym))
            cur = ""
        cur += ch
        cur_sym = is_sym
    if cur:
        out.append((cur, cur_sym))
    return out


def text_width(s: str, size: int, bold: bool = False) -> float:
    w = 0.0
    for run, is_sym in _runs(s):
        f = sym_font(int(size * 0.92), bold) if is_sym else hand_font(size, bold)
        w += f.getlength(run)
    return w


def draw_text_mixed(draw: ImageDraw.ImageDraw, xy, s: str, size: int, fill,
                    bold: bool = False, anchor_middle: bool = False):
    """Draw text mixing the handwriting font with DejaVu for symbols."""
    x, y = xy
    if anchor_middle:
        x -= text_width(s, size, bold) / 2
    for run, is_sym in _runs(s):
        if is_sym:
            f = sym_font(int(size * 0.92), bold)
            dy = size * 0.10
        else:
            f = hand_font(size, bold)
            dy = 0
        draw.text((x, y + dy), run, font=f, fill=fill)
        x += f.getlength(run)


# ---------------------------------------------------------------- geometry
def _seed_from(*parts) -> int:
    h = hashlib.md5(repr(parts).encode()).hexdigest()
    return int(h[:8], 16)


def _resample(points, step=6.0):
    """Resample a polyline to roughly even arclength spacing."""
    pts = [np.asarray(p, float) for p in points]
    if not pts:
        return []
    if len(pts) < 2:
        return [tuple(pts[0]), tuple(pts[0])]
    out = [pts[0]]
    carry = 0.0  # distance already covered since last emitted point
    for a, b in zip(pts, pts[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg < 1e-9:
            continue
        d = (b - a) / seg
        pos = -carry  # position of last emitted point relative to a
        while pos + step <= seg:
            pos += step
            out.append(a + d * pos)
        carry = seg - pos
    if not np.allclose(out[-1], pts[-1]):
        out.append(pts[-1])
    return [tuple(p) for p in out]


def wobble(points, seed, amp=2.2, waves=2):
    """Displace a polyline perpendicular to itself with smooth pseudo-noise."""
    pts = _resample(points, step=7.0)
    n = len(pts)
    if n < 3:
        return pts
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * math.pi, waves)
    freqs = rng.uniform(1.0, 2.6, waves)
    arr = np.array(pts)
    out = []
    for i, p in enumerate(arr):
        t = i / (n - 1)
        if 0 < i < n - 1:
            d = arr[i + 1] - arr[i - 1]
        elif i == 0:
            d = arr[1] - arr[0]
        else:
            d = arr[-1] - arr[-2]
        L = np.linalg.norm(d)
        perp = np.array([-d[1], d[0]]) / (L + 1e-9)
        disp = sum(math.sin(2 * math.pi * f * t + ph) for f, ph in zip(freqs, phases))
        disp *= amp / waves
        # taper wobble at the very ends so joints meet cleanly
        taper = min(1.0, 6 * t, 6 * (1 - t))
        out.append(tuple(p + perp * disp * taper))
    return out


def _arc_prefix(pts, frac):
    """Prefix of polyline covering `frac` of its arclength; also tip point."""
    if frac >= 1.0 or len(pts) < 2:
        return pts, pts[-1] if pts else None
    arr = np.array(pts)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    total = seg.sum()
    target = total * max(frac, 0.0)
    acc = 0.0
    out = [pts[0]]
    for i, s in enumerate(seg):
        if acc + s >= target:
            r = (target - acc) / (s + 1e-12)
            tip = tuple(arr[i] + (arr[i + 1] - arr[i]) * r)
            out.append(tip)
            return out, tip
        acc += s
        out.append(pts[i + 1])
    return out, pts[-1]


def ease_out(p):
    return 1 - (1 - p) ** 3


# ---------------------------------------------------------------- events
class Event:
    """One animated element. kind-specific params in self.p"""

    def __init__(self, kind, t0, dur, **p):
        self.kind = kind
        self.t0 = t0
        self.dur = max(dur, 1e-3)
        self.p = p
        self.seed = _seed_from(kind, sorted(p.items(), key=lambda kv: kv[0])[:4], t0)

    def progress(self, t):
        return min(1.0, max(0.0, (t - self.t0) / self.dur))

    # -- painting -------------------------------------------------------
    def paint(self, draw: ImageDraw.ImageDraw, t: float, pen_tips: list, ctx=None):
        pr = self.progress(t)
        if pr <= 0:
            return
        if self.kind == "clear":
            self._paint_clear(draw, pr, pen_tips, ctx)
            return
        fn = getattr(self, f"_paint_{self.kind}")
        fn(draw, pr, pen_tips)

    def _sc(self, v):
        """scale logical coord(s) to supersampled space"""
        if isinstance(v, (tuple, list)) and len(v) and isinstance(v[0], (tuple, list, np.ndarray)):
            return [tuple(np.asarray(q, float) * SS) for q in v]
        if isinstance(v, (tuple, list, np.ndarray)):
            return tuple(np.asarray(v, float) * SS)
        return v * SS

    def _stroke_path(self, draw, pts, pr, color, width, pen_tips, wob=True, amp=2.2):
        pts = self._sc(pts)
        if wob:
            pts = wobble(pts, self.seed, amp=amp * SS)
        vis, tip = _arc_prefix(pts, ease_out(pr))
        if len(vis) >= 2:
            draw.line(vis, fill=color, width=int(width * SS), joint="curve")
            r = width * SS / 2
            for q in (vis[0], vis[-1]):
                draw.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=color)
        if pr < 1 and tip is not None:
            pen_tips.append(tip)

    def _paint_stroke(self, draw, pr, pen_tips):
        self._stroke_path(draw, self.p["points"], pr, self.p["color"],
                          self.p.get("width", 4), pen_tips,
                          wob=self.p.get("wob", True), amp=self.p.get("amp", 2.2))

    def _paint_line(self, draw, pr, pen_tips):  # straight helper
        a, b = self.p["a"], self.p["b"]
        self._stroke_path(draw, [a, b], pr, self.p["color"],
                          self.p.get("width", 4), pen_tips, amp=self.p.get("amp", 1.6))

    def _paint_dash(self, draw, pr, pen_tips):
        a = np.asarray(self.p["a"], float)
        b = np.asarray(self.p["b"], float)
        on = self.p.get("on", 12)
        gap = self.p.get("off_len", 9)
        L = float(np.linalg.norm(b - a))
        d = (b - a) / (L + 1e-9)
        n = max(1, int(L // (on + gap)) + 1)
        shown = ease_out(pr) * L
        pos = 0.0
        for _ in range(n):
            seg_end = min(pos + on, L, shown)
            if seg_end <= pos:
                break
            p0 = a + d * pos
            p1 = a + d * seg_end
            self._stroke_path(draw, [tuple(p0), tuple(p1)], 1.0, self.p["color"],
                              self.p.get("width", 3), [], wob=False)
            pos += on + gap
        if pr < 1:
            pen_tips.append(tuple(np.asarray(self._sc(tuple(a + d * min(shown, L))))))

    def _paint_clear(self, draw, pr, pen_tips, ctx):
        """Eraser wipe: restore the pristine background, sweeping left to right."""
        x, y, w, h = self.p.get("rect", (0, 0, W, H))
        ww = int(w * ease_out(pr) * SS)
        if ww <= 0 or ctx is None:
            return
        box = (int(x * SS), int(y * SS), int(x * SS) + ww, int((y + h) * SS))
        ctx["img"].paste(ctx["bg"].crop(box), box[:2])

    def _paint_arrow(self, draw, pr, pen_tips):
        a = np.asarray(self.p["a"], float)
        b = np.asarray(self.p["b"], float)
        color = self.p["color"]
        width = self.p.get("width", 4)
        curve = self.p.get("curve", 0.0)  # perpendicular bulge in px
        mid = (a + b) / 2
        d = b - a
        L = np.linalg.norm(d) + 1e-9
        perp = np.array([-d[1], d[0]]) / L
        ctrl = mid + perp * curve
        ts = np.linspace(0, 1, 24)
        pts = [tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b) for t in ts]
        # shaft takes 80% of the animation, head the rest
        shaft_pr = min(1.0, pr / 0.8)
        self._stroke_path(draw, pts, shaft_pr, color, width, pen_tips, amp=1.4)
        if pr > 0.8:
            head_pr = (pr - 0.8) / 0.2
            tang = (b - ctrl)
            tang /= np.linalg.norm(tang) + 1e-9
            hp = np.array([-tang[1], tang[0]])
            hl = self.p.get("head", 14)
            for sgn in (1, -1):
                tipseg = [tuple(b), tuple(b - tang * hl + hp * sgn * hl * 0.55)]
                self._stroke_path(draw, tipseg, head_pr, color, width, pen_tips, wob=False)

    def _paint_text(self, draw, pr, pen_tips):
        s = self.p["s"]
        n = max(1, int(len(s) * pr))
        draw_text_mixed(draw, self._sc(self.p["xy"]), s[:n], self.p["size"] * SS,
                        self.p["color"], bold=self.p.get("bold", False),
                        anchor_middle=False)

    def _paint_ctext(self, draw, pr, pen_tips):  # centered text
        s = self.p["s"]
        n = max(1, int(len(s) * pr))
        size = self.p["size"] * SS
        x, y = self._sc(self.p["xy"])
        full_w = text_width(s, size, self.p.get("bold", False))
        draw_text_mixed(draw, (x - full_w / 2, y), s[:n], size,
                        self.p["color"], bold=self.p.get("bold", False))

    def _paint_box(self, draw, pr, pen_tips):
        x, y, w, h = self.p["rect"]
        r = self.p.get("r", 14)
        pts = _rounded_rect_path(x, y, w, h, r)
        self._stroke_path(draw, pts, pr, self.p["color"],
                          self.p.get("width", 4), pen_tips, amp=1.8)

    def _paint_ellipse(self, draw, pr, pen_tips):
        cx, cy = self.p["c"]
        rx, ry = self.p["rx"], self.p["ry"]
        start = self.p.get("start", -0.5 * math.pi)
        pts = [(cx + rx * math.cos(start + t), cy + ry * math.sin(start + t))
               for t in np.linspace(0, 2 * math.pi * 1.02, 48)]
        self._stroke_path(draw, pts, pr, self.p["color"],
                          self.p.get("width", 4), pen_tips, amp=2.0)

    def _paint_dot(self, draw, pr, pen_tips):
        cx, cy = self._sc(self.p["c"])
        r = self.p.get("r", 6) * SS * ease_out(pr)
        c = self.p["color"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)

    def _paint_dots(self, draw, pr, pen_tips):
        pts = self.p["points"]
        n = len(pts)
        r0 = self.p.get("r", 6)
        for i, q in enumerate(pts):
            # staggered pop-in
            local = min(1.0, max(0.0, (pr * (n + 2) - i) / 2.0))
            if local <= 0:
                continue
            cx, cy = self._sc(q)
            r = r0 * SS * ease_out(local)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self.p["color"])

    def _paint_bar(self, draw, pr, pen_tips):
        x, y0, w, h = self.p["rect"]  # y0 = baseline, h = full height (grows up)
        hh = h * ease_out(pr)
        x, y0, w, hh = x * SS, y0 * SS, w * SS, hh * SS
        c = self.p["color"]
        fill = tuple(int(v * 0.42) for v in c)
        draw.rectangle([x, y0 - hh, x + w, y0], fill=fill, outline=c, width=2 * SS)

    def _paint_fill(self, draw, pr, pen_tips):
        x, y, w, h = [v * SS for v in self.p["rect"]]
        c = self.p["color"]
        a = self.p.get("alpha", 0.18) * ease_out(pr)
        fill = tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))
        rad = min(10 * SS, w / 2, h / 2)
        draw.rounded_rectangle([x, y, x + w, y + h], radius=rad, fill=fill)

    def _paint_wave(self, draw, pr, pen_tips):
        """placeholder squiggly 'text lines' inside cards"""
        x, y, w = self.p["x"], self.p["y"], self.p["w"]
        rows = self.p.get("rows", 3)
        gap = self.p.get("gap", 16)
        color = self.p.get("color", DIM)
        for r in range(rows):
            row_pr = min(1.0, max(0.0, pr * rows - r))
            if row_pr <= 0:
                break
            ww = w * (1.0 if (rows == 1 or r < rows - 1) else 0.6)
            pts = [(x + ww * t, y + r * gap + 2.2 * math.sin(t * 12 + r))
                   for t in np.linspace(0, 1, 20)]
            self._stroke_path(draw, pts, row_pr, color, 2, pen_tips, amp=1.0)


def _rounded_rect_path(x, y, w, h, r):
    """Closed rounded-rectangle polyline starting at top-left arc end."""
    pts = []

    def arc(cx, cy, a0, a1):
        for t in np.linspace(a0, a1, 7):
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))

    pts.append((x + r, y))
    pts.append((x + w - r, y))
    arc(x + w - r, y + r, -math.pi / 2, 0)
    pts.append((x + w, y + h - r))
    arc(x + w - r, y + h - r, 0, math.pi / 2)
    pts.append((x + r, y + h))
    arc(x + r, y + h - r, math.pi / 2, math.pi)
    pts.append((x, y + r))
    arc(x + r, y + r, math.pi, 1.5 * math.pi)
    return pts


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, sid: str, audio_dir: Path):
        self.sid = sid
        self.events: list[Event] = []
        align = json.loads((audio_dir / f"{sid}.align.json").read_text())
        self.chars = align["characters"]
        self.starts = align["character_start_times_seconds"]
        self.ends = align["character_end_times_seconds"]
        self.text = "".join(self.chars)
        self.audio_end = self.ends[-1]

    # -- narration time lookup -----------------------------------------
    def T(self, anchor, occ: int = 1, end: bool = False, off: float = 0.0) -> float:
        if isinstance(anchor, (int, float)):
            return float(anchor) + off
        idx = -1
        for _ in range(occ):
            idx = self.text.find(anchor, idx + 1)
            if idx < 0:
                raise ValueError(f"[{self.sid}] anchor not found: {anchor!r}")
        if end:
            return self.ends[min(idx + len(anchor) - 1, len(self.ends) - 1)] + off
        return self.starts[idx] + off

    def add(self, kind, at, dur, **p) -> Event:
        occ = p.pop("occ", 1)
        off = p.pop("off", 0.0)
        t0 = self.T(at, occ=occ, off=off)
        ev = Event(kind, t0, dur, **p)
        self.events.append(ev)
        return ev

    @property
    def duration(self):
        return self.audio_end + TAIL


# ---------------------------------------------------------------- render
def _make_bg() -> Image.Image:
    img = Image.new("RGB", (W * SS, H * SS), BG)
    # vignette
    v = Image.new("L", (W, H), 0)
    dv = ImageDraw.Draw(v)
    dv.ellipse([-W * 0.25, -H * 0.35, W * 1.25, H * 1.35], fill=70)
    v = v.filter(ImageFilter.GaussianBlur(120)).resize((W * SS, H * SS))
    dark = Image.new("RGB", (W * SS, H * SS), tuple(int(c * 0.55) for c in BG))
    img = Image.composite(img, dark, v)
    return img


class SceneRenderer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.bg = _make_bg()
        self.base = self.bg.copy()
        self.base_n = 0  # events fully painted into base (prefix in z-order)

    def frame(self, t: float) -> Image.Image:
        evs = self.scene.events
        # grow the cached prefix: leading events already finished by time t
        while self.base_n < len(evs) and evs[self.base_n].progress(t) >= 1.0:
            d = ImageDraw.Draw(self.base)
            evs[self.base_n].paint(d, t, [], ctx={"img": self.base, "bg": self.bg})
            self.base_n += 1
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        ctx = {"img": img, "bg": self.bg}
        pen_tips: list = []
        for ev in evs[self.base_n:]:
            ev.paint(d, t, pen_tips, ctx=ctx)
        # pen tip glow: little bright dot where "the pen" currently is
        for tip in pen_tips[-2:]:
            x, y = tip
            r = 5 * SS
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 245))
        return img

    def render_to_mp4(self, out_path: Path, audio_path: Path):
        dur = self.scene.duration
        n_frames = int(math.ceil(dur * FPS))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
            "-r", str(FPS), "-i", "pipe:0",
            "-i", str(audio_path),
            "-filter_complex", f"[1:a]apad=whole_dur={dur:.3f}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "21",
            "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
            "-t", f"{dur:.3f}", str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(n_frames):
                t = i / FPS
                img = self.frame(t)
                img = img.resize((W, H), Image.LANCZOS)
                # fade to black at the very end
                rem = dur - t
                if rem < FADE:
                    f = max(0.0, rem / FADE)
                    img = Image.eval(img, (lambda v, f=f: int(v * f)))
                proc.stdin.write(img.tobytes())
        finally:
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed for {out_path}")
