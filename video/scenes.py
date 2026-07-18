"""Visual choreography for every scene.

Each build_* function receives a Scene and adds timed events. `at=` anchors are
exact substrings of the narration in narration.py; the event starts when the
narrator reaches that phrase (via ElevenLabs character alignment).
"""

from __future__ import annotations

import numpy as np

from renderer import (
    CHALK, CYAN, DIM, GREEN, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW,
    Scene, text_width,
)

CH = CHALK


# ------------------------------------------------------------------ helpers
def title(S: Scene, s: str, at=0.2, color=CH):
    S.add("text", at, 1.2, s=s, xy=(42, 22), size=40, color=color, bold=True)
    S.add("line", at, 0.8, off=0.7, a=(44, 80), b=(44 + text_width(s, 40, True) + 8, 80),
          color=YELLOW, width=3)


def label(S, at, s, xy, size=22, color=DIM, dur=0.9, bold=False, center=False):
    S.add("ctext" if center else "text", at, dur, s=s, xy=xy, size=size,
          color=color, bold=bold)


def card(S, at, rect, name, color, dur=1.0, rows=3):
    x, y, w, h = rect
    S.add("box", at, dur, rect=rect, color=color, width=4)
    S.add("text", at, 0.7, off=0.35, s=name, xy=(x + 14, y + 8), size=24,
          color=color, bold=True)
    S.add("wave", at, 1.0, off=0.7, x=x + 16, y=y + 52, w=w - 40, rows=rows, gap=20)


def barchart_bar(S, at, x, base_y, w, val, scale, color, lab, val_s=None, dur=1.0):
    h = val * scale
    S.add("bar", at, dur, rect=(x, base_y, w, h), color=color)
    S.add("ctext", at, 0.6, off=dur * 0.7, s=val_s or f"{val:g}%",
          xy=(x + w / 2, base_y - h - 34), size=24, color=color, bold=True)
    S.add("ctext", at, 0.6, off=dur * 0.5, s=lab, xy=(x + w / 2, base_y + 8),
          size=20, color=DIM)


# ------------------------------------------------------------------ scene 1
def build_01(S: Scene):
    title(S, "Ranking content: which one wins?", at="Say you've got")

    card(S, "two posts sitting", (120, 120, 300, 190), "Post A", CYAN)
    card(S, "two posts sitting", (860, 120, 300, 190), "Post B", PINK)

    # outcomes
    S.add("stroke", "take off", 0.9,
          points=[(150, 390), (210, 374), (260, 380), (330, 350), (392, 326)],
          color=GREEN, width=4)
    S.add("arrow", "take off", 0.5, off=0.7, a=(352, 340), b=(400, 320),
          color=GREEN, width=4)
    label(S, "thousands of reactions", "thousands of reactions", (150, 402),
          color=GREEN, size=22)
    S.add("dash", "sit there quietly", 0.9, a=(890, 350), b=(1130, 350),
          color=RED, width=3)
    label(S, "almost none", "almost none", (940, 362), color=RED, size=22)

    # the central question
    S.add("ellipse", "which one will win", 0.8, c=(640, 215), rx=52, ry=52,
          color=YELLOW, width=5)
    S.add("ctext", "which one will win", 0.7, off=0.3, s="?", xy=(640, 178),
          size=64, color=YELLOW, bold=True)
    S.add("arrow", "which one will win", 0.6, off=0.5, a=(586, 215), b=(432, 215),
          color=YELLOW, width=3)
    S.add("arrow", "which one will win", 0.6, off=0.5, a=(694, 215), b=(848, 215),
          color=YELLOW, width=3)

    # pairwise ranking banner
    S.add("box", "pairwise ranking", 0.9, rect=(430, 425, 420, 56), color=TEAL, width=4)
    S.add("ctext", "pairwise ranking", 0.9, s="PAIRWISE  RANKING", xy=(640, 434),
          size=32, color=TEAL, bold=True)
    S.add("ctext", "A or B?", 0.8, s="just answer:  A or B ?", xy=(640, 495),
          size=28, color=CH)

    # ranking chain
    chain_y = 585
    for i, x in enumerate([80, 190, 300, 410, 520]):
        S.add("box", "you can rank anything", 0.5, off=i * 0.16,
              rect=(x, chain_y, 64, 64), color=CH, width=3, r=8)
        if i < 4:
            S.add("ctext", "you can rank anything", 0.4, off=0.2 + i * 0.16, s=">",
                  xy=(x + 88, chain_y + 12), size=34, color=YELLOW, bold=True)
    label(S, "many pairwise comparisons", "a full ranking = lots of little A-vs-B games",
          (80, 662), size=22, color=DIM)

    # teaser bars
    S.add("line", "coin-flip", 0.5, a=(940, 660), b=(1220, 660), color=DIM, width=3)
    barchart_bar(S, "coin-flip", 970, 660, 80, 50, 1.5, CH, "guess", "50%", dur=0.8)
    barchart_bar(S, "almost ninety percent", 1110, 660, 80, 88, 1.5, GREEN,
                 "this video", "≈90%", dur=0.9)
    S.add("arrow", "almost ninety percent", 0.6, off=0.6, a=(1052, 570), b=(1104, 535),
          color=YELLOW, width=4, curve=-24)

    # look inside the model
    S.add("box", "language model's head", 0.8, rect=(680, 570, 150, 84), color=PURPLE,
          width=4)
    S.add("ctext", "language model's head", 0.6, off=0.3, s="LLM", xy=(755, 590),
          size=30, color=PURPLE, bold=True)
    S.add("arrow", "looking", 0.7, a=(892, 560), b=(834, 594), color=YELLOW,
          width=3, curve=14)
    label(S, "looking", "look inside", (880, 538), color=YELLOW, size=22)


# ------------------------------------------------------------------ scene 2
def build_02(S: Scene):
    title(S, "Why not just ask the model?")

    card(S, "just ask a language model", (70, 120, 330, 240), "prompt", CH, rows=2)
    S.add("text", "paste in both posts", 0.8, s="Post A: …", xy=(90, 210),
          size=24, color=CYAN)
    S.add("text", "paste in both posts", 0.8, off=0.4, s="Post B: …", xy=(90, 246),
          size=24, color=PINK)
    S.add("text", "which of these got more engagement", 1.0,
          s="which got more engagement?", xy=(90, 292), size=22, color=YELLOW)

    S.add("arrow", "Qwen two point five", 0.6, a=(410, 240), b=(490, 240),
          color=CH, width=4)
    S.add("box", "Qwen two point five", 0.9, rect=(500, 185, 180, 110), color=PURPLE,
          width=4)
    S.add("ctext", "Qwen two point five", 0.7, off=0.4, s="Qwen 2.5 · 3B",
          xy=(590, 222), size=26, color=PURPLE, bold=True)

    S.add("ellipse", "And the model answers", 0.7, c=(770, 195), rx=56, ry=40,
          color=CH, width=3)
    S.add("ctext", "And the model answers", 0.5, off=0.4, s="B", xy=(770, 168),
          size=42, color=CH, bold=True)

    # accuracy chart
    base_y, scale = 420, 4.2
    S.add("line", "three hundred and eighteen", 0.6, a=(880, base_y), b=(1240, base_y),
          color=DIM, width=3)
    S.add("dash", "three hundred and eighteen", 0.8, off=0.4,
          a=(880, base_y - (50 - 40) * 7.2), b=(1240, base_y - (50 - 40) * 7.2),
          color=DIM, width=2)
    label(S, "three hundred and eighteen", "50%",
          (846, base_y - (50 - 40) * 7.2 - 14), size=18, color=DIM)
    # bars drawn versus a 50%-anchored axis: height above baseline = (v-40)*scale
    barchart_bar(S, "sixty-two percent", 930, base_y, 90, 62.3 - 40, 7.2, CYAN,
                 "LinkedIn", "62.3%")
    barchart_bar(S, "fifty-six percent", 1090, base_y, 90, 56.5 - 40, 7.2, PINK,
                 "X (same author)", "56.5%")

    # one bit out of an enormous computation
    S.add("ellipse", "one-bit summary", 1.0, c=(500, 520), rx=150, ry=85,
          color=TEAL, width=4)
    S.add("wave", "one-bit summary", 1.2, off=0.5, x=390, y=485, w=210, rows=4, gap=22,
          color=TEAL)
    label(S, "enormous internal computation", "an enormous computation",
          (380, 620), size=22, color=TEAL)
    S.add("arrow", "If we only read the output", 0.7, a=(655, 520), b=(760, 520),
          color=CH, width=4)
    S.add("box", "If we only read the output", 0.6, off=0.3, rect=(768, 492, 56, 56),
          color=CH, width=3, r=8)
    S.add("ctext", "If we only read the output", 0.5, off=0.6, s="B", xy=(796, 500),
          size=30, color=CH, bold=True)
    label(S, "throwing almost all", "1 bit survives", (770, 560), size=22, color=RED)
    S.add("stroke", "throwing almost all", 0.8, off=0.3,
          points=[(842, 545), (876, 520), (908, 545)], color=RED, width=3)

    S.add("arrow", "start reading the state", 0.8, a=(940, 640), b=(628, 560),
          color=YELLOW, width=4, curve=-40)
    label(S, "start reading the state", "read the state itself", (950, 630),
          size=26, color=YELLOW, bold=True)


# ------------------------------------------------------------------ scene 3
def build_03(S: Scene):
    title(S, "Activations: the model's working memory")

    # residual stream column (added first so boxes/tokens draw over it)
    S.add("fill", "residual stream", 1.2, rect=(352, 145, 60, 430), color=CYAN,
          alpha=0.30)

    # token row
    words = ["Which", "post", "got", "more", "…", "Answer:"]
    x = 120
    for i, wd in enumerate(words):
        w = 86 if wd != "…" else 52
        S.add("box", "chopped into tokens", 0.45, off=i * 0.14, rect=(x, 606, w, 46),
              color=CH, width=3, r=8)
        S.add("ctext", "chopped into tokens", 0.4, off=0.1 + i * 0.14, s=wd,
              xy=(x + w / 2, 616), size=20, color=CH)
        x += w + 10
    label(S, "chopped into tokens", "tokens", (34, 618), size=22, color=DIM)

    # arrows into the stack
    for i, ax in enumerate([180, 330, 480]):
        S.add("arrow", "every token becomes a vector", 0.5, off=i * 0.12,
              a=(ax, 600), b=(ax, 572), color=DIM, width=3)

    # layer stack
    ys = [515, 450, 385, 320, 255]
    for i, y in enumerate(ys):
        S.add("box", "stack of layers", 0.5, off=i * 0.2, rect=(150, y, 480, 48),
              color=PURPLE, width=3)
    S.add("ctext", "thirty-six of them", 0.5, s="⋮", xy=(390, 200), size=36,
          color=PURPLE, bold=True)
    S.add("box", "thirty-six of them", 0.5, off=0.4, rect=(150, 150, 480, 44),
          color=PURPLE, width=3)
    S.add("stroke", "thirty-six of them", 0.7, off=0.6,
          points=[(650, 150), (668, 170), (668, 350), (668, 540), (650, 563)],
          color=DIM, width=3)
    label(S, "thirty-six of them", "36 layers", (680, 340), size=24, color=PURPLE)

    # residual stream labels
    S.add("arrow", "residual stream", 0.7, off=0.5, a=(300, 118), b=(368, 152),
          color=CYAN, width=3, curve=14)
    label(S, "residual stream", "the residual stream", (128, 100), size=24,
          color=CYAN, bold=True)
    label(S, "adds its result back in", "layers read it, add back",
          (682, 432), size=20, color=DIM)

    S.add("ctext", "two thousand numbers", 0.9, s="one vector ∈ ℝ^2048  per token, per layer",
          xy=(400, 674), size=22, color=CH)

    # the hook
    S.add("dot", "a little hook", 0.5, c=(382, 279), r=9, color=RED)
    S.add("stroke", "a little hook", 0.9, off=0.3,
          points=[(391, 279), (520, 240), (700, 250), (860, 300)], color=RED, width=3)
    label(S, "the eighth layer from the end", "layer −8", (440, 292), size=24,
          color=YELLOW, bold=True)
    S.add("box", "we save the activation", 0.8, rect=(870, 280, 250, 130),
          color=RED, width=3)
    label(S, "we save the activation", "recorded activations", (890, 288), size=20,
          color=RED)
    S.add("wave", "we save the activation", 1.0, off=0.5, x=890, y=330, w=200,
          rows=3, gap=20, color=RED)

    # response tokens
    rx = 760
    for i, wd in enumerate(["I'd", "say", "B"]):
        S.add("box", "writes its answer", 0.4, off=i * 0.15, rect=(rx, 606, 70, 46),
              color=GREEN, width=3, r=8)
        S.add("ctext", "writes its answer", 0.4, off=0.1 + i * 0.15, s=wd,
              xy=(rx + 35, 616), size=20, color=GREEN)
        rx += 80
    label(S, "writes its answer", "response", (770, 662), size=20, color=GREEN)

    # mean pooling
    for i in range(3):
        S.add("box", "average them", 0.4, off=i * 0.12, rect=(880 + i * 44, 470, 28, 78),
              color=GREEN, width=3, r=6)
        if i < 2:
            S.add("ctext", "average them", 0.3, off=0.1 + i * 0.12, s="+",
                  xy=(922 + i * 44, 490), size=26, color=CH)
    S.add("arrow", "mean pooling", 0.6, a=(1016, 508), b=(1082, 508), color=CH, width=3)
    S.add("ellipse", "one single vector", 0.6, c=(1126, 508), rx=32, ry=32,
          color=YELLOW, width=4)
    S.add("ctext", "one single vector", 0.5, off=0.3, s="h", xy=(1126, 484),
          size=34, color=YELLOW, bold=True)
    label(S, "snapshot of the model's state of mind", "h = the state of mind",
          (1010, 560), size=24, color=YELLOW)


# ------------------------------------------------------------------ scene 4
def build_04(S: Scene):
    title(S, "The contrastive vector")

    # training pairs stack
    for i in range(3):
        S.add("box", "couple hundred", 0.5, off=i * 0.18,
              rect=(70 + i * 14, 130 + i * 12, 150, 90), color=DIM, width=3)
    S.add("ctext", "couple hundred", 0.7, off=0.5, s="× 200 pairs", xy=(160, 250),
          size=22, color=DIM)
    label(S, "answer was actually right", "training data → we know if it was right",
          (60, 292), size=20, color=DIM)

    # axes
    S.add("stroke", "two piles of snapshots", 1.0,
          points=[(310, 180), (310, 600), (830, 600)], color=DIM, width=3)
    label(S, "two piles of snapshots", "activation space  (2 of 2048 dims)",
          (330, 612), size=20, color=DIM)

    rng = np.random.default_rng(7)
    g_pts = [(660 + dx, 300 + dy) for dx, dy in
             zip(rng.normal(0, 48, 14), rng.normal(0, 36, 14))]
    r_pts = [(430 + dx, 480 + dy) for dx, dy in
             zip(rng.normal(0, 40, 9), rng.normal(0, 30, 9))]
    S.add("dots", "Green ones", 1.6, points=g_pts, color=GREEN, r=7)
    label(S, "hundred and sixty-one", "correct (161)", (700, 196), size=24,
          color=GREEN, bold=True)
    S.add("dots", "And red ones", 1.3, points=r_pts, color=RED, r=7)
    label(S, "thirty-nine", "wrong (39)", (330, 540), size=24, color=RED, bold=True)

    # means
    S.add("line", "mu correct", 0.4, a=(645, 285), b=(675, 315), color=GREEN, width=5)
    S.add("line", "mu correct", 0.4, off=0.2, a=(675, 285), b=(645, 315),
          color=GREEN, width=5)
    label(S, "mu correct", "μ⁺", (688, 292), size=28, color=GREEN, bold=True)
    S.add("line", "mu wrong", 0.4, a=(415, 465), b=(445, 495), color=RED, width=5)
    S.add("line", "mu wrong", 0.4, off=0.2, a=(445, 465), b=(415, 495), color=RED,
          width=5)
    label(S, "mu wrong", "μ⁻", (352, 430), size=28, color=RED, bold=True)

    # the subtraction arrow
    S.add("arrow", "And we subtract", 1.1, a=(430, 480), b=(660, 300), color=YELLOW,
          width=6, head=18)
    S.add("ctext", "And we subtract", 0.6, off=0.6, s="v", xy=(520, 340), size=40,
          color=YELLOW, bold=True)

    # formulas
    S.add("text", "v equals mu correct minus mu wrong", 1.4, s="v = μ⁺ − μ⁻",
          xy=(890, 210), size=36, color=YELLOW, bold=True)
    S.add("text", "scale v to length one", 1.0, s="v ← v / ‖v‖", xy=(890, 272),
          size=30, color=CH)

    # what cancels
    label(S, "the topic, the format", "topic · format · language",
          (890, 370), size=24, color=DIM)
    S.add("line", "and cancels out", 0.6, a=(884, 388), b=(1176, 388), color=RED,
          width=3)
    label(S, "and cancels out", "shared stuff cancels", (890, 404), size=22,
          color=RED)
    label(S, "is being right versus", "what's left = right vs wrong",
          (890, 452), size=24, color=YELLOW)

    # confused -> clear-headed
    label(S, "points from confused", "confused", (497, 516), size=22, color=RED)
    label(S, "toward clear-headed", "clear-headed", (748, 338), size=22, color=GREEN)

    # 1-D projection strip
    S.add("line", "project both piles onto v", 0.8, a=(880, 560), b=(1210, 560),
          color=CH, width=3)
    for i, tx in enumerate([900, 922, 938, 960, 985]):
        S.add("line", "project both piles onto v", 0.3, off=0.5 + i * 0.08,
              a=(tx, 548), b=(tx, 572), color=RED, width=3)
    for i, tx in enumerate([1080, 1102, 1118, 1142, 1160, 1183]):
        S.add("line", "project both piles onto v", 0.3, off=0.9 + i * 0.08,
              a=(tx, 548), b=(tx, 572), color=GREEN, width=3)
    S.add("stroke", "three units apart", 0.7,
          points=[(990, 590), (1000, 600), (1065, 600), (1075, 590)], color=YELLOW,
          width=3)
    S.add("ctext", "three units apart", 0.8, off=0.4, s="≈ 3 units apart",
          xy=(1035, 610), size=22, color=YELLOW)


# ------------------------------------------------------------------ scene 5
def build_05(S: Scene):
    title(S, "Score = one dot product")

    o = np.array([300.0, 520.0])
    v_end = np.array([840.0, 260.0])
    d = (v_end - o) / np.linalg.norm(v_end - o)
    h_tip = np.array([560.0, 270.0])
    foot = o + d * float(np.dot(h_tip - o, d))

    S.add("arrow", "this direction v", 1.0, a=tuple(o), b=tuple(v_end), color=YELLOW,
          width=5, head=16)
    S.add("ctext", "this direction v", 0.5, off=0.5, s="v", xy=(862, 236), size=36,
          color=YELLOW, bold=True)
    label(S, "measuring stick", "the measuring stick", (700, 152), size=24,
          color=YELLOW)

    S.add("arrow", "Grab its snapshot h", 0.9, a=tuple(o), b=tuple(h_tip),
          color=CYAN, width=5, head=16)
    S.add("ctext", "Grab its snapshot h", 0.6, off=0.4, s="h", xy=(548, 228),
          size=34, color=CYAN, bold=True)
    label(S, "any new answer", "(new answer, unknown if right)", (430, 196),
          size=20, color=DIM)

    S.add("dash", "dropping the shadow", 0.9, a=tuple(h_tip), b=tuple(foot),
          color=DIM, width=3)
    S.add("dot", "a projection", 0.5, c=tuple(foot), r=8, color=YELLOW)
    # measurement bracket along v from origin to foot
    off_perp = np.array([d[1], -d[0]]) * -26
    m0, m1 = o + off_perp * 1.6, foot + off_perp * 1.6
    S.add("line", "s equals h dot v", 0.8, a=tuple(m0), b=tuple(m1), color=CH,
          width=3)
    for q in (m0, m1):
        S.add("line", "s equals h dot v", 0.3, off=0.6, a=tuple(q - off_perp * 0.35),
              b=tuple(q + off_perp * 0.35), color=CH, width=3)
    mid = (o + foot) / 2 + off_perp * 3.4
    S.add("ctext", "s equals h dot v", 1.0, s="s = h · v", xy=(mid[0], mid[1]),
          size=32, color=CH, bold=True)

    label(S, "lands far along v", "far along v → looks correct", (700, 420),
          size=22, color=GREEN)
    label(S, "it lands low", "low → looks wrong", (330, 585), size=22, color=RED)

    # confidence meter
    S.add("box", "confidence meter", 1.0, rect=(1020, 170, 74, 330), color=CH,
          width=3, r=20)
    for i in range(5):
        S.add("line", "confidence meter", 0.3, off=0.5 + i * 0.08,
              a=(1026, 210 + i * 62), b=(1046, 210 + i * 62), color=DIM, width=2)
    S.add("line", "confidence meter", 0.5, off=0.9, a=(1028, 252), b=(1086, 252),
          color=YELLOW, width=5)
    S.add("ctext", "confidence meter", 0.5, off=0.3, s="right-ish", xy=(1057, 138),
          size=20, color=GREEN)
    S.add("ctext", "confidence meter", 0.5, off=0.5, s="wrong-ish", xy=(1057, 508),
          size=20, color=RED)
    S.add("arrow", "we never trained", 0.7, off=0.3, a=(1176, 156), b=(1104, 184),
          color=YELLOW, width=4, curve=-12)
    S.add("ctext", "we never trained", 0.9, s="never trained!", xy=(1180, 118),
          size=22, color=YELLOW)

    label(S, "no labels at test time", "no extra model · no labels · no fine-tuning",
          (330, 636), size=24, color=CH)

    # linear probe tag
    S.add("box", "linear probe", 0.7, rect=(600, 470, 220, 50), color=PURPLE,
          width=3, r=12)
    S.add("ctext", "linear probe", 0.7, off=0.3, s="“a linear probe”",
          xy=(710, 480), size=24, color=PURPLE)
    S.add("line", "linear probe", 0.5, off=0.4, a=(670, 468), b=(600, 400),
          color=PURPLE, width=2)


# ------------------------------------------------------------------ scene 6
def build_06(S: Scene):
    title(S, "Best-of-N with an internal judge")

    card(S, "don't generate one answer", (60, 130, 250, 150), "prompt", CH, rows=2)

    # temperature dial
    S.add("ellipse", "Turn up the temperature", 0.7, c=(185, 400), rx=34, ry=34,
          color=ORANGE, width=4)
    S.add("line", "Turn up the temperature", 0.4, off=0.4, a=(185, 400), b=(206, 378),
          color=ORANGE, width=4)
    label(S, "Turn up the temperature", "T = 0.7", (235, 386), size=24, color=ORANGE)

    letters = ["B", "A", "B", "B", "A", "B", "A", "B"]
    scores = ["−0.4", "+1.1", "+0.3", "+2.1", "−1.2", "+0.8", "+1.9", "+0.1"]
    y0 = 118
    for i in range(8):
        y = y0 + i * 54
        S.add("arrow", "say eight of them", 0.4, off=i * 0.10, a=(320, 210),
              b=(420, y + 20), color=DIM, width=2, curve=(y + 20 - 210) * -0.12)
        S.add("box", "say eight of them", 0.5, off=0.15 + i * 0.10,
              rect=(430, y, 290, 40), color=CH, width=3, r=8)
        S.add("text", "say eight of them", 0.4, off=0.3 + i * 0.10, s=letters[i],
              xy=(444, y + 4), size=24, color=(CYAN if letters[i] == "A" else PINK),
              bold=True)
        S.add("wave", "say eight of them", 0.5, off=0.35 + i * 0.10, x=480, y=y + 20,
              w=210, rows=1, gap=10)
    label(S, "some of those answers will say A", "8 samples", (560, 556), size=22,
          color=DIM)

    # scores appear
    for i in range(8):
        y = y0 + i * 54
        S.add("text", "score every sample", 0.5, off=i * 0.14, s=f"s = {scores[i]}",
              xy=(740, y + 6), size=22,
              color=(YELLOW if i == 3 else DIM), bold=(i == 3))

    # pick the argmax
    S.add("ellipse", "keep the answer", 0.8, c=(660, y0 + 3 * 54 + 20), rx=250, ry=34,
          color=YELLOW, width=4)
    S.add("arrow", "scored highest", 0.7, a=(915, y0 + 3 * 54 + 20), b=(1010, 300),
          color=YELLOW, width=4, curve=-30)
    S.add("box", "scored highest", 0.7, off=0.3, rect=(1020, 260, 170, 84),
          color=GREEN, width=5, r=14)
    S.add("ctext", "scored highest", 0.6, off=0.6, s="B  ✓", xy=(1105, 278),
          size=40, color=GREEN, bold=True)
    label(S, "internal judge", "highest internal score wins", (980, 356), size=20,
          color=GREEN)

    # majority voting comparison
    S.add("box", "majority voting", 0.8, rect=(60, 470, 280, 120), color=DIM,
          width=3)
    label(S, "majority voting", "majority vote", (76, 478), size=22, color=DIM)
    S.add("text", "counts the letters", 0.9, s="A: | | |    B: | | | | |",
          xy=(84, 520), size=24, color=CH)
    label(S, "equally trustworthy", "every vote counts the same…", (76, 558),
          size=20, color=DIM)

    # results mini-chart
    base_y = 655
    S.add("line", "On the tweet benchmark", 0.5, a=(880, base_y), b=(1240, base_y),
          color=DIM, width=3)
    S.add("dash", "On the tweet benchmark", 0.6, off=0.3, a=(880, base_y - 65),
          b=(1240, base_y - 65), color=DIM, width=2)
    label(S, "On the tweet benchmark", "50%", (846, base_y - 78), size=18, color=DIM)
    barchart_bar(S, "fifty-six and a half", 905, base_y, 80, 6.5, 10, CH,
                 "ask once", "56.5")
    barchart_bar(S, "fifty-seven and a half", 1030, base_y, 80, 7.5, 10, PINK,
                 "vote of 8", "57.5")
    barchart_bar(S, "fifty-eight for persona", 1155, base_y, 80, 8.0, 10, YELLOW,
                 "persona 8", "58.0")


# ------------------------------------------------------------------ scene 7
def build_07(S: Scene):
    title(S, "Power persona sampling  (Metropolis–Hastings)")

    # ---- trajectory strips
    cell_w, n_cells, x0, y_cur = 66, 13, 80, 160
    total_w = cell_w * n_cells
    S.add("box", "whole reasoning trajectory", 1.0, rect=(x0, y_cur, total_w, 46),
          color=CH, width=3, r=8)
    for i in range(1, n_cells):
        S.add("line", "whole reasoning trajectory", 0.25, off=0.4 + i * 0.05,
              a=(x0 + i * cell_w, y_cur), b=(x0 + i * cell_w, y_cur + 46),
              color=DIM, width=2)
    label(S, "whole reasoning trajectory", "current trajectory:  prompt | reasoning | answer",
          (x0, 126), size=20, color=DIM)
    S.add("fill", "short chain of thought", 0.8, rect=(x0, y_cur, 3 * cell_w, 46),
          color=DIM, alpha=0.35)
    S.add("ctext", "then the answer", 0.5, s="B", xy=(x0 + 12.5 * cell_w, y_cur + 8),
          size=26, color=GREEN, bold=True)

    cut_x = x0 + 7 * cell_w
    S.add("dash", "random cut point", 0.6, a=(cut_x, 140), b=(cut_x, 240), color=RED,
          width=3, on=8, off_len=7)
    S.add("arrow", "random cut point", 0.6, off=0.3, a=(cut_x + 74, 118),
          b=(cut_x + 6, 142), color=RED, width=3, curve=12)
    label(S, "random cut point", "cut!", (cut_x + 80, 100), size=24, color=RED,
          bold=True)
    S.add("line", "keep everything before", 0.7, a=(x0, 220), b=(cut_x - 6, 220),
          color=GREEN, width=4)
    label(S, "keep everything before", "keep", (x0, 228), size=20, color=GREEN)

    y_prop = 268
    S.add("arrow", "resample everything after", 0.7, a=(cut_x, 206),
          b=(cut_x + 30, y_prop + 20), color=PINK, width=3, curve=-24)
    S.add("box", "resample everything after", 0.9, rect=(cut_x, y_prop,
          total_w - 7 * cell_w, 46), color=PINK, width=3, r=8)
    S.add("wave", "resample everything after", 0.9, off=0.5, x=cut_x + 14,
          y=y_prop + 22, w=total_w - 7 * cell_w - 90, rows=1, gap=10, color=PINK)
    S.add("ctext", "resample everything after", 0.4, off=0.7, s="A",
          xy=(x0 + 12.5 * cell_w, y_prop + 8), size=26, color=PINK, bold=True)
    label(S, "let the model resample", "proposal: new suffix", (x0 + 4 * cell_w, y_prop + 8),
          size=20, color=PINK)

    # scores for both
    S.add("text", "score both trajectories", 0.8, s="s = 1.9   log p = −41",
          xy=(960, 128), size=20, color=CH)
    S.add("text", "score both trajectories", 0.8, off=0.4, s="s = 2.6   log p = −39",
          xy=(960, 320), size=20, color=PINK)

    # ---- acceptance rule
    F = "log r = (α − 1)·Δlog p  +  β·Δs"
    fx, fy, fs = 300, 380, 34
    S.add("box", "log acceptance ratio", 1.0, rect=(fx - 24, fy - 14,
          text_width(F, fs, True) + 48, 78), color=YELLOW, width=4)
    S.add("text", "log acceptance ratio", 2.2, s=F, xy=(fx, fy), size=fs,
          color=YELLOW, bold=True)
    # underline the two terms
    pre1 = text_width("log r = ", fs, True)
    w1 = text_width("(α − 1)·Δlog p", fs, True)
    pre2 = text_width("log r = (α − 1)·Δlog p  +  ", fs, True)
    w2 = text_width("β·Δs", fs, True)
    S.add("line", "change in log probability", 0.7, a=(fx + pre1, fy + 58),
          b=(fx + pre1 + w1, fy + 58), color=CYAN, width=3)
    S.add("ctext", "change in log probability", 0.8, off=0.3, s="fluency term",
          xy=(fx + pre1 + w1 / 2, fy + 66), size=20, color=CYAN)
    S.add("line", "change in persona score", 0.7, a=(fx + pre2, fy + 58),
          b=(fx + pre2 + w2, fy + 58), color=GREEN, width=3)
    S.add("ctext", "change in persona score", 0.8, off=0.3, s="state-of-mind term",
          xy=(fx + pre2 + w2 / 2, fy + 66), size=20, color=GREEN)

    # accept / reject branches
    S.add("arrow", "you tend to keep it", 0.6, a=(480, 498), b=(380, 534),
          color=GREEN, width=4, curve=16)
    label(S, "you tend to keep it", "better → accept ✓", (250, 540), size=24,
          color=GREEN)
    S.add("arrow", "sometimes keep it anyway", 0.6, a=(720, 498), b=(820, 534),
          color=ORANGE, width=4, curve=-16)
    label(S, "sometimes keep it anyway", "worse → accept sometimes (log u < log r)",
          (640, 540), size=24, color=ORANGE)
    label(S, "getting stuck in a rut", "randomness ⇒ no getting stuck",
          (700, 578), size=20, color=DIM)

    # chain plot
    S.add("stroke", "Run ten or twenty steps", 0.8,
          points=[(90, 590), (90, 700), (620, 700)], color=DIM, width=3)
    hs = [10, 24, 18, 40, 36, 60, 54, 78, 92, 102]
    acc = ["✓", "✓", "✗", "✓", "✗", "✓", "✗", "✓", "✓", "✓"]
    pts = [(120 + i * 52, 692 - h) for i, h in enumerate(hs)]
    S.add("stroke", "Run ten or twenty steps", 1.6, off=0.5, points=pts, color=TEAL,
          width=3)
    S.add("dots", "Run ten or twenty steps", 1.6, off=0.5, points=pts, color=TEAL,
          r=5)
    for i, a in enumerate(acc[:8]):
        S.add("text", "the chain drifts", 0.3, off=i * 0.1, s=a,
              xy=(112 + i * 52, 700),
              size=16, color=(GREEN if a == "✓" else RED))
    label(S, "the chain drifts", "persona score s over MH steps", (150, 596),
          size=20, color=TEAL)
    S.add("ellipse", "clear-headed region", 0.7, c=(608, 590), rx=52, ry=30,
          color=GREEN, width=3)

    # target distribution
    S.add("text", "sampling text with probability proportional", 1.6,
          s="π(y) ∝ p(y|x)^α · e^(β·s(y))", xy=(760, 622), size=30, color=PURPLE,
          bold=True)
    S.add("stroke", "the name comes from", 0.6,
          points=[(944, 668), (930, 686), (908, 690)], color=PURPLE, width=3)
    label(S, "the name comes from", "the “power”", (952, 668), size=20,
          color=PURPLE)

    # ---- results takeover
    S.add("clear", "on LinkedIn, this is dramatic", 0.8, rect=(0, 96, 1280, 624))
    base_y, scale = 600, 4.6
    S.add("line", "Baseline:", 0.5, a=(150, base_y), b=(760, base_y), color=DIM,
          width=3)
    barchart_bar(S, "sixty-two point three", 220, base_y, 150, 62.3, scale, CH,
                 "just ask", "62.3%", dur=1.2)
    barchart_bar(S, "eighty-eight point one", 500, base_y, 150, 88.1, scale, GREEN,
                 "power persona", "88.1%", dur=1.4)
    S.add("arrow", "eighty-eight point one", 0.9, off=0.3, a=(390, 300),
          b=(480, 210), color=YELLOW, width=5, curve=-30)
    label(S, "beta equal to one", "318 held-out LinkedIn pairs · β = 1",
          (240, 640), size=22, color=DIM)

    S.add("text", "ninety-five", 0.9, s="95 flips  ✗ → ✓", xy=(850, 300), size=32,
          color=GREEN, bold=True)
    S.add("text", "only thirteen", 0.9, s="13 flips  ✓ → ✗", xy=(850, 370), size=32,
          color=RED, bold=True)
    S.add("box", "nothing was trained", 0.8, rect=(830, 460, 400, 70), color=YELLOW,
          width=3, r=14)
    S.add("ctext", "nothing was trained", 1.0, off=0.3, s="same weights — zero training",
          xy=(1030, 478), size=26, color=YELLOW)


# ------------------------------------------------------------------ scene 8
def build_08(S: Scene):
    title(S, "Same recipe, new modality")

    # recipe checklist
    items = [
        ("1. record activations", "vectors between layers"),
        ("2. contrast:  v = μ⁺ − μ⁻", "subtracting two group"),
        ("3. score:  s = h · v", "a dot product"),
        ("4. search with s", "Search with that score"),
    ]
    for i, (txt, anchor) in enumerate(items):
        y = 140 + i * 56
        S.add("text", "Look back at the recipe", 0.8, off=i * 0.25, s=txt,
              xy=(70, y), size=26, color=CH)
        S.add("text", anchor, 0.4, s="✓", xy=(30, y), size=28, color=GREEN,
              bold=True)
    S.add("text", "Nothing in there cares", 1.2, s="none of it mentions text!",
          xy=(70, 372), size=28, color=YELLOW, bold=True)
    S.add("line", "Nothing in there cares", 0.6, off=0.8, a=(70, 408),
          b=(70 + text_width("none of it mentions text!", 28, True), 408),
          color=YELLOW, width=3)

    # image + patches
    ix, iy, iw, ih = 620, 120, 160, 130
    S.add("box", "takes an image", 0.8, rect=(ix, iy, iw, ih), color=CYAN, width=4)
    S.add("stroke", "takes an image", 0.8, off=0.4,
          points=[(ix + 12, iy + ih - 16), (ix + 55, iy + 55), (ix + 90, iy + ih - 16)],
          color=CYAN, width=3)
    S.add("stroke", "takes an image", 0.6, off=0.7,
          points=[(ix + 80, iy + ih - 16), (ix + 112, iy + 78), (ix + 148, iy + ih - 16)],
          color=CYAN, width=3)
    S.add("ellipse", "takes an image", 0.5, off=0.9, c=(ix + 128, iy + 34), rx=14,
          ry=14, color=YELLOW, width=3)
    for k in (1, 2):
        S.add("line", "grid of patches", 0.5, off=k * 0.12,
              a=(ix + iw * k / 3, iy), b=(ix + iw * k / 3, iy + ih), color=CH, width=2)
        S.add("line", "grid of patches", 0.5, off=0.24 + k * 0.12,
              a=(ix, iy + ih * k / 3), b=(ix + iw, iy + ih * k / 3), color=CH, width=2)
    label(S, "grid of patches", "patches", (ix + 40, iy + ih + 6), size=20, color=CH)

    S.add("arrow", "vision encoder", 0.5, a=(ix + iw + 6, iy + 60), b=(ix + iw + 60, iy + 60),
          color=CH, width=3)
    S.add("box", "vision encoder", 0.8, rect=(850, 130, 190, 84), color=TEAL, width=4)
    S.add("ctext", "vision encoder", 0.6, off=0.4, s="vision encoder", xy=(945, 158),
          size=24, color=TEAL)
    S.add("arrow", "projects them", 0.5, a=(1046, 172), b=(1096, 172), color=CH,
          width=3)
    S.add("box", "projects them", 0.6, rect=(1100, 144, 80, 58), color=TEAL, width=3)
    S.add("ctext", "projects them", 0.5, off=0.3, s="proj", xy=(1140, 158), size=22,
          color=TEAL)

    # into the same stack
    S.add("arrow", "very same residual stream", 0.7, a=(1140, 208), b=(1000, 300),
          color=CH, width=3, curve=30)
    for i in range(3):
        S.add("box", "very same residual stream", 0.6, off=0.3 + i * 0.18,
              rect=(830, 300 + i * 56, 260, 44), color=PURPLE, width=3)
    label(S, "very same residual stream", "the same transformer layers",
          (836, 470), size=20, color=PURPLE)
    # text tokens join too
    for i, wd in enumerate(["text", "tokens"]):
        S.add("box", "that text flows through", 0.5, off=i * 0.15,
              rect=(620, 316 + i * 52, 92, 40), color=CH, width=3, r=8)
        S.add("ctext", "that text flows through", 0.4, off=0.1 + i * 0.15, s=wd,
              xy=(666, 324 + i * 52), size=20, color=CH)
        S.add("arrow", "that text flows through", 0.5, off=0.2 + i * 0.15,
              a=(714, 336 + i * 52), b=(824, 336 + i * 52), color=DIM, width=2)
    S.add("ctext", "an image is just more tokens", 1.4,
          s="“an image is just more tokens”", xy=(900, 508), size=28, color=YELLOW,
          bold=True)

    # hook + score
    S.add("dot", "Hook the same layer", 0.5, c=(1090, 378), r=8, color=RED)
    S.add("stroke", "Hook the same layer", 0.7, off=0.3,
          points=[(1098, 378), (1150, 390), (1188, 420)], color=RED, width=3)
    S.add("ellipse", "pool the same way", 0.6, c=(1206, 446), rx=26, ry=26,
          color=YELLOW, width=3)
    S.add("ctext", "pool the same way", 0.5, off=0.3, s="h", xy=(1206, 428),
          size=26, color=YELLOW, bold=True)
    S.add("ctext", "score new comparisons", 0.8, s="· v → s", xy=(1180, 490),
          size=24, color=YELLOW)

    # 400+ badge
    S.add("ellipse", "four hundred images", 1.0, c=(220, 560), rx=86, ry=66,
          color=YELLOW, width=5)
    S.add("ctext", "four hundred images", 0.8, off=0.4, s="400+", xy=(220, 520),
          size=44, color=YELLOW, bold=True)
    S.add("ctext", "four hundred images", 0.8, off=0.7, s="images benchmarked",
          xy=(220, 578), size=20, color=CH)
    S.add("text", "same story holds up", 0.6, s="✓", xy=(320, 520), size=52,
          color=GREEN, bold=True)
    label(S, "beats just reading", "scoring activations  >  reading the answer",
          (420, 560), size=24, color=CH)
    S.add("line", "beats just reading", 0.6, off=0.5, a=(420, 594),
          b=(420 + text_width("scoring activations  >  reading the answer", 24), 594),
          color=GREEN, width=3)

    # video = frames
    fx0, fy0 = 460, 620
    for i in range(4):
        S.add("box", "a video is frames", 0.4, off=i * 0.12,
              rect=(fx0 + i * 96, fy0, 84, 58), color=PINK, width=3, r=6)
        S.add("dot", "a video is frames", 0.2, off=0.2 + i * 0.12,
              c=(fx0 + i * 96 + 8, fy0 + 8), r=3, color=PINK)
        S.add("dot", "a video is frames", 0.2, off=0.25 + i * 0.12,
              c=(fx0 + i * 96 + 8, fy0 + 50), r=3, color=PINK)
    S.add("arrow", "pool the scores", 0.6, a=(fx0 + 380, fy0 + 30), b=(fx0 + 460, fy0 + 30),
          color=PINK, width=3)
    label(S, "pool the scores", "rank frames, pool scores", (fx0 + 470, fy0 + 16),
          size=22, color=PINK)


# ------------------------------------------------------------------ scene 9
def build_09(S: Scene):
    title(S, "The whole picture")

    S.add("text", "models know more than they say", 1.4,
          s="1.  models know more than they say",
          xy=(70, 122), size=28, color=CH)
    S.add("text", "read the activations", 1.0,
          s="→ read the activations, not just outputs",
          xy=(110, 162), size=24, color=DIM)
    S.add("text", "use contrast", 1.4,
          s="2.  contrast right vs wrong moments",
          xy=(70, 216), size=28, color=CH)
    S.add("text", "That difference is a direction", 1.2,
          s="v = μ⁺ − μ⁻ ,   score  s = h · v",
          xy=(110, 256), size=24, color=DIM)
    S.add("text", "once you can score, you can search", 1.4,
          s="3.  once you can score, you can search",
          xy=(70, 306), size=28, color=CH)
    label(S, "Best-of-N if you want simple", "best-of-N (simple)  ·  MH power sampling (strong)",
          (110, 348), size=24, color=DIM)

    # mini results
    base_y = 560
    S.add("line", "sixty-two percent", 0.5, a=(100, base_y), b=(420, base_y),
          color=DIM, width=3)
    barchart_bar(S, "sixty-two percent", 140, base_y, 90, 62, 1.55, CH, "ask", "62%")
    barchart_bar(S, "eighty-eight", 290, base_y, 90, 88, 1.55, GREEN, "search", "88%")
    S.add("arrow", "eighty-eight", 0.6, off=0.3, a=(248, 448), b=(286, 428),
          color=YELLOW, width=4, curve=-14)

    S.add("ellipse", "four hundred plus images", 0.9, c=(560, 490), rx=80, ry=56,
          color=YELLOW, width=4)
    S.add("ctext", "four hundred plus images", 0.7, off=0.3, s="400+", xy=(560, 456),
          size=36, color=YELLOW, bold=True)
    S.add("ctext", "four hundred plus images", 0.6, off=0.6, s="images ✓",
          xy=(560, 500), size=20, color=CH)
    label(S, "brutally noisy tweet benchmark", "text ✓   images ✓   (video: frames)",
          (480, 570), size=22, color=DIM)

    # pipeline
    steps = ["pair A/B", "answers + h", "v = μ⁺−μ⁻", "s = h·v", "search", "winner"]
    colors = [CH, CYAN, YELLOW, YELLOW, TEAL, GREEN]
    px = 960
    for i, (st, c) in enumerate(zip(steps, colors)):
        y = 150 + i * 62
        S.add("box", "No fine-tuning", 0.5, off=i * 0.22, rect=(px, y, 240, 46),
              color=c, width=3, r=10)
        S.add("ctext", "No fine-tuning", 0.5, off=0.1 + i * 0.22, s=st,
              xy=(px + 120, y + 8), size=22, color=c)
        if i < 5:
            S.add("arrow", "No fine-tuning", 0.3, off=0.2 + i * 0.22,
                  a=(px + 120, y + 48), b=(px + 120, y + 60), color=DIM, width=3)
    label(S, "no labels at inference", "no fine-tuning · no reward model · no labels",
          (850, 542), size=22, color=DIM)

    # the closer
    S.add("ctext", "a subtraction and a dot product", 1.4,
          s="a subtraction   +   a dot product", xy=(400, 630), size=30,
          color=YELLOW, bold=True)
    S.add("ellipse", "a subtraction and a dot product", 0.9, off=0.6, c=(400, 650),
          rx=290, ry=44, color=YELLOW, width=4)
    S.add("ctext", "Thanks for watching", 1.0, s="thanks for watching ✦",
          xy=(1000, 640), size=26, color=CH)


BUILDERS = {
    "01_problem": build_01,
    "02_just_ask": build_02,
    "03_activations": build_03,
    "04_contrastive": build_04,
    "05_projection": build_05,
    "06_best_of_n": build_06,
    "07_power_mh": build_07,
    "08_images": build_08,
    "09_recap": build_09,
}
