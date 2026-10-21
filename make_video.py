"""生成竖屏剧情短视频《小球越狱》。

一颗被困在圈里的小球发现：每撞一次墙，自己就会变大一点。它拼命撞墙，
终于撑破了圆环——镜头拉远，原来外面还有一个更大的圈。结尾画面与开头完全一致，
可以无缝循环播放。

物理模拟 → 分镜（时间重映射、慢动作、表情、字幕）→ cairo 逐帧绘制 → ffmpeg 编码。
所有音效都由同一份时间轴合成，因此声画严格同步。运行：python3 make_video.py
"""
import colorsys
import math
import os
import random
import subprocess
import wave

import cairo
import numpy as np
from scipy import signal

W, H, FPS = 1080, 1920, 60
CX, CY, R = W / 2, 1010, 430          # 圆环中心（屏幕）与半径（世界单位）
R0, GROW = 40, 7                      # 初始球半径、每次碰撞增长
GRAVITY = 2600
SIM_HZ, TRAIL_HZ = 1200, 240          # 物理步频、拖影采样频率
END_GAP = 6
SR = 44100
FONT = "WenQuanYi Zen Hei"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

BG = (0.027, 0.027, 0.05)
WHITE, YELLOW, RED = (1, 1, 1), (1, 0.84, 0.29), (1, 0.38, 0.32)
BALL_LIGHT, BALL_DARK = (1.0, 0.86, 0.45), (0.96, 0.46, 0.10)
INK = (0.16, 0.07, 0.03)

T_HOP = 2.0                           # 开场静止多久后起跳
S_CU = 2.2                            # 开场特写的缩放
CU_BALL_Y = 1150                      # 特写时球心在屏幕上的 y
CU_CY = (R - R0) - (CU_BALL_Y - CY) / S_CU

# 《欢乐颂》（贝多芬，公有领域），MIDI 音高
ODE = [64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62,
       64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 62, 60, 60,
       62, 62, 64, 60, 62, 64, 65, 64, 60, 62, 64, 65, 64, 62, 60, 62, 55,
       64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 62, 60, 60]
MELODY = [n + 12 for n in ODE]


# ---------------------------------------------------------------- 物理模拟
def simulate(seed=7):
    """球从圆环底部起跳。返回 TRAIL_HZ 采样的 (x, y, r)、碰撞列表和结束时刻。"""
    rnd = random.Random(seed)
    x, y, vx, vy, r = 0.0, R - R0, 600.0, -1780.0, float(R0)
    e0 = 0.5 * (vx * vx + vy * vy) - GRAVITY * y   # 保持总能量，球不会停下
    dt = 1 / SIM_HZ
    step = SIM_HZ // TRAIL_HZ
    t, i = 0.0, 0
    samples, bounces = [(x, y, r)], []
    while r < R - END_GAP:
        vy += GRAVITY * dt
        x += vx * dt
        y += vy * dt
        t += dt
        i += 1
        d = math.hypot(x, y)
        if d + r >= R:
            nx, ny = x / d, y / d
            vn = vx * nx + vy * ny
            if vn > 0:
                vx -= 2 * vn * nx
                vy -= 2 * vn * ny
                r = min(r + GROW, R - END_GAP)
                x, y = nx * (R - r) * 0.999, ny * (R - r) * 0.999
                a = rnd.uniform(-0.02, 0.02)
                c, s = math.cos(a), math.sin(a)
                vx, vy = vx * c - vy * s, vx * s + vy * c
                sp = math.sqrt(max(2 * (e0 + GRAVITY * y), 900 ** 2))
                m = math.hypot(vx, vy)
                vx, vy = vx / m * sp, vy / m * sp
                bounces.append((t, nx, ny, r))
        if i % step == 0:
            samples.append((x, y, r))
    samples.append((x, y, r))
    return samples, bounces, (len(samples) - 1) / TRAIL_HZ


def sample_at(samples, tau):
    f = tau * TRAIL_HZ
    i = min(int(f), len(samples) - 2)
    a, b = samples[i], samples[i + 1]
    u = min(max(f - i, 0.0), 1.0)
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2]


# ---------------------------------------------------------------- 分镜
def plan(bounces, sim_end):
    """逐帧决定模拟时间 tau（含慢动作），并记录剧情节点的视频时间。"""
    taus, ev, hits, slow = [], {}, [], []
    tau, f, bi = 0.0, 0, 0
    while True:
        t = f / FPS
        if "end" in ev and t >= ev["end"] - 1e-9:
            return taus, ev, hits
        if t < T_HOP:
            sp = 0.0
        else:
            sp = 1.0
            for s0, dur, v in slow:
                if s0 <= t < s0 + dur:
                    sp = min(sp, v)
                elif s0 + dur <= t < s0 + dur + 0.4:
                    sp = min(sp, v + (1 - v) * (t - s0 - dur) / 0.4)
        new = min(tau + sp / FPS, sim_end)
        while bi < len(bounces) and bounces[bi][0] <= new:
            bt, _, _, r = bounces[bi]
            te = t - 1 / FPS + (bt - tau) / max(new - tau, 1e-9) / FPS
            fill = (r / R) ** 2
            hits.append((te, bi))
            if bi == 0:
                ev["first"] = te
                slow.append((te, 2.6, 0.08))
            if bi == 7:
                ev["again"] = te
            if "crack" not in ev and fill >= 0.12:
                ev["crack"] = te
                slow.append((te, 1.0, 0.25))
            if "squeeze" not in ev and fill >= 0.3:
                ev["squeeze"] = te
            if "almost" not in ev and fill >= 0.55:
                ev["almost"] = te
            bi += 1
        tau = new
        taus.append(tau)
        if "simend" not in ev and tau >= sim_end:
            ev["simend"] = t
            ev["boom"] = t + 1.0
            ev["doubt"] = ev["boom"] + 2.4          # “……吗？”
            ev["zoom0"] = ev["boom"] + 2.9
            ev["zoom1"] = ev["zoom0"] + 3.6
            ev["push0"] = ev["zoom1"] + 2.0
            ev["end"] = ev["zoom1"] + 4.8
        f += 1


def timeline(ev):
    e = ev
    moods = [
        (0, e["first"], "sad"),
        (e["first"], e["first"] + 1.3, "shock"),
        (e["first"] + 1.3, e["again"], "determined"),
        (e["again"], e["again"] + 1.5, "happy"),
        (e["again"] + 1.5, e["crack"], "determined"),
        (e["crack"], e["crack"] + 0.7, "shock"),
        (e["crack"] + 0.7, e["squeeze"], "determined"),
        (e["squeeze"], e["boom"], "strain"),
        (e["boom"], e["doubt"], "happy"),
        (e["doubt"], e["zoom1"], "shock"),
        (e["zoom1"], e["end"] + 1, "sad"),
    ]
    caps = [
        (0, e["first"], "这颗小球\n一辈子都被困在这个圈里", WHITE),
        (e["first"], e["first"] + 1.3, "等等……\n撞一下墙 它就变大了？！", YELLOW),
        (e["first"] + 1.3, e["first"] + 3.2, "那如果……\n一直撞下去呢？", WHITE),
        (e["again"], e["again"] + 1.6, "变大了！再来！！", YELLOW),
        (e["crack"], e["crack"] + 1.7, "墙……裂开了？！", YELLOW),
        (e["crack"] + 1.7, e["crack"] + 3.5, "管它呢！\n继续撞！！", WHITE),
        (e["squeeze"], e["almost"], "可是越大\n就越挤……", WHITE),
        (e["almost"], e["simend"], "坚持住！\n就差一点了！！", YELLOW),
        (e["simend"], e["boom"], "啊啊啊啊啊——", RED),
        (e["boom"] + 0.25, e["doubt"], "它终于自由了！！", YELLOW),
        (e["doubt"], e["zoom1"] - 0.3, "……吗？", WHITE),
        (e["zoom1"] - 0.3, e["push0"], "原来\n外面还有一个更大的圈", WHITE),
        (e["push0"], e["end"] + 1, "那你呢？\n你被困在第几个圈里？", YELLOW),
    ]
    return moods, caps


def active(segments, t):
    """取已开始且未结束的片段中最晚开始的一个。"""
    best = None
    for seg in segments:
        if seg[0] <= t < seg[1] and (best is None or seg[0] >= best[0]):
            best = seg
    return best


# ---------------------------------------------------------------- 绘图工具
def hue(h, s=0.85, v=1.0):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ease_out_back(p):
    p = clamp01(p)
    c1 = 1.70158
    return 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2


def ease_in_out(p):
    p = clamp01(p)
    return p * p * (3 - 2 * p)


def text(ctx, s, x, y, size, color, scale=1.0, alpha=1.0, stroke=10, line_gap=1.3):
    if alpha <= 0 or scale <= 0:
        return
    lines = s.split("\n")
    ctx.save()
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    ctx.translate(x, y)
    ctx.scale(scale, scale)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    y0 = -(len(lines) - 1) * size * line_gap / 2
    for k, line in enumerate(lines):
        ext = ctx.text_extents(line)
        ctx.move_to(-ext.x_advance / 2, y0 + k * size * line_gap + size * 0.36)
        ctx.text_path(line)
    ctx.set_source_rgba(0, 0, 0, 0.85 * alpha)
    ctx.set_line_width(stroke)
    ctx.stroke_preserve()
    ctx.set_source_rgba(*color, alpha)
    ctx.fill()
    ctx.restore()


def ring_back(ctx, x, y, r, glow=1.0, disk=1.0):
    if glow > 0:
        for w, a in ((60, 0.05), (36, 0.08), (20, 0.12)):
            ctx.arc(x, y, r + 5, 0, 2 * math.pi)
            ctx.set_source_rgba(0.6, 0.7, 1.0, a * glow)
            ctx.set_line_width(w)
            ctx.stroke()
    if disk > 0:
        ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.set_source_rgba(*BG, disk)
        ctx.fill()


def ring_stroke(ctx, x, y, r, color, width, alpha=1.0):
    ctx.arc(x, y, r + 5, 0, 2 * math.pi)
    ctx.set_source_rgba(*color, alpha)
    ctx.set_line_width(width)
    ctx.stroke()


def draw_ball(ctx, x, y, r, mood, look, blink, squash=None):
    """带表情的小球。r 为屏幕半径，look 为瞳孔方向（-1..1）。"""
    ctx.save()
    ctx.translate(x, y)
    if squash:
        a, k = squash
        ctx.rotate(a)
        ctx.scale(1 - 0.22 * k, 1 + 0.12 * k)
        ctx.rotate(-a)
    g = cairo.RadialGradient(-r * 0.35, -r * 0.4, r * 0.05, 0, 0, r)
    g.add_color_stop_rgb(0, *BALL_LIGHT)
    g.add_color_stop_rgb(1, *BALL_DARK)
    ctx.arc(0, 0, r, 0, 2 * math.pi)
    ctx.set_source(g)
    ctx.fill_preserve()
    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(max(3, min(6, r * 0.05)))
    ctx.stroke()

    lw = max(2.0, r * 0.055)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    def ink_line(pts, w=lw):
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(w)
        ctx.stroke()

    ex, ey = 0.32 * r, -0.10 * r
    # 眼睛
    if mood == "happy":
        for sgn in (-1, 1):
            ctx.arc(sgn * ex, ey + 0.05 * r, 0.13 * r, math.pi * 1.1, math.pi * 1.9)
            ctx.set_source_rgb(*INK)
            ctx.set_line_width(lw * 1.1)
            ctx.stroke()
        for sgn in (-1, 1):                      # 腮红
            ctx.save()
            ctx.translate(sgn * 0.52 * r, 0.16 * r)
            ctx.scale(1, 0.6)
            ctx.arc(0, 0, 0.12 * r, 0, 2 * math.pi)
            ctx.restore()
            ctx.set_source_rgba(1, 0.35, 0.4, 0.45)
            ctx.fill()
    elif mood == "strain":
        for sgn in (-1, 1):
            cx = sgn * ex
            ink_line([(cx - sgn * 0.12 * r, ey - 0.09 * r), (cx + sgn * 0.08 * r, ey),
                      (cx - sgn * 0.12 * r, ey + 0.09 * r)], lw * 1.1)
    elif blink and mood != "shock":
        for sgn in (-1, 1):
            ink_line([(sgn * ex - 0.12 * r, ey), (sgn * ex + 0.12 * r, ey)])
    else:
        er = 0.22 * r if mood == "shock" else 0.16 * r
        pr = 0.07 * r if mood == "shock" else 0.095 * r
        for sgn in (-1, 1):
            ctx.arc(sgn * ex, ey, er, 0, 2 * math.pi)
            ctx.set_source_rgb(1, 1, 1)
            ctx.fill_preserve()
            ctx.set_source_rgb(*INK)
            ctx.set_line_width(lw * 0.6)
            ctx.stroke()
            px = sgn * ex + look[0] * (er - pr) * 0.8
            py = ey + look[1] * (er - pr) * 0.8
            ctx.arc(px, py, pr, 0, 2 * math.pi)
            ctx.set_source_rgb(*INK)
            ctx.fill()
            ctx.arc(px - pr * 0.35, py - pr * 0.35, pr * 0.32, 0, 2 * math.pi)
            ctx.set_source_rgb(1, 1, 1)
            ctx.fill()
    # 眉毛
    for sgn in (-1, 1):
        if mood == "sad":
            ink_line([(sgn * 0.16 * r, -0.40 * r), (sgn * 0.46 * r, -0.30 * r)])
        elif mood == "shock":
            ctx.arc(sgn * ex, -0.42 * r, 0.14 * r, math.pi * 1.15, math.pi * 1.85)
            ctx.set_source_rgb(*INK)
            ctx.set_line_width(lw)
            ctx.stroke()
        elif mood in ("determined", "strain"):
            ink_line([(sgn * 0.14 * r, -0.28 * r), (sgn * 0.48 * r, -0.42 * r)], lw * 1.2)
    # 嘴
    if mood == "sad":
        ctx.arc(0, 0.46 * r, 0.17 * r, math.pi * 1.2, math.pi * 1.8)
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(lw)
        ctx.stroke()
    elif mood == "shock":
        ctx.save()
        ctx.translate(0, 0.32 * r)
        ctx.scale(0.8, 1)
        ctx.arc(0, 0, 0.12 * r, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*INK)
        ctx.fill()
    elif mood == "determined":
        ink_line([(-0.15 * r, 0.30 * r), (0.15 * r, 0.27 * r)])
    elif mood == "happy":
        ctx.move_to(-0.24 * r, 0.20 * r)
        ctx.arc_negative(0, 0.20 * r, 0.24 * r, math.pi, 0)
        ctx.close_path()
        ctx.set_source_rgb(0.45, 0.08, 0.08)
        ctx.fill_preserve()
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(lw * 0.8)
        ctx.stroke()
    elif mood == "strain":
        x0, y0, w, h = -0.22 * r, 0.22 * r, 0.44 * r, 0.15 * r
        ctx.rectangle(x0, y0, w, h)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(lw * 0.7)
        ctx.stroke()
        ink_line([(x0, y0 + h / 2), (x0 + w, y0 + h / 2)], lw * 0.5)
        for k in range(1, 4):
            ink_line([(x0 + w * k / 4, y0), (x0 + w * k / 4, y0 + h)], lw * 0.5)
    ctx.restore()


def make_crack(rnd, nx, ny):
    """以碰撞点为起点向外的锯齿裂纹（世界坐标，相对圆环中心）。"""
    a = math.atan2(ny, nx)
    lines = []
    for _ in range(rnd.randint(1, 2)):
        ang = a + rnd.uniform(-0.03, 0.03)
        rad = R + 5
        pts = [(math.cos(ang) * rad, math.sin(ang) * rad)]
        for _ in range(rnd.randint(3, 5)):
            rad += rnd.uniform(10, 28)
            ang += rnd.uniform(-0.06, 0.06)
            pts.append((math.cos(ang) * rad, math.sin(ang) * rad))
        lines.append(pts)
    side = rnd.choice((-1, 1))                   # 沿圆环延伸的一道
    ang, pts = a, []
    for _ in range(4):
        rad = R + 5 + rnd.uniform(-4, 4)
        pts.append((math.cos(ang) * rad, math.sin(ang) * rad))
        ang += side * rnd.uniform(0.03, 0.07)
    lines.append(pts)
    return lines


# ---------------------------------------------------------------- 渲染
def render(samples, bounces, sim_end, taus, ev):
    moods, caps = timeline(ev)
    n_frames = len(taus)
    px_end, py_end, r_end = samples[-1]
    k_big = r_end / R0                                     # 大圆与小圆的相似比
    big_c = (px_end, py_end - k_big * (R - R0))

    def cam_from_anchor(aw, ascr, s):
        return aw[0] - (ascr[0] - CX) / s, aw[1] - (ascr[1] - CY) / s, s

    def camera(t):
        if t < T_HOP:
            return 0.0, CU_CY, S_CU
        if t < T_HOP + 0.6:
            u = ease_in_out((t - T_HOP) / 0.6)
            s = math.exp(math.log(S_CU) * (1 - u))
            sy = (CY - CU_CY * S_CU) * (1 - u) + CY * u
            return cam_from_anchor((0, 0), (CX, sy), s)
        if t < ev["zoom0"]:
            return 0.0, 0.0, 1.0
        if t < ev["zoom1"]:
            u = ease_in_out((t - ev["zoom0"]) / (ev["zoom1"] - ev["zoom0"]))
            s = math.exp(-math.log(k_big) * u)
            a0 = (px_end + CX, py_end + CY)
            a1 = (CX, CY + R - R0)
            return cam_from_anchor((px_end, py_end),
                                   (a0[0] + (a1[0] - a0[0]) * u, a0[1] + (a1[1] - a0[1]) * u), s)
        if t < ev["push0"]:
            return big_c[0], big_c[1], 1 / k_big
        u = ease_in_out((t - ev["push0"]) / (ev["end"] - ev["push0"]))
        s = math.exp(-math.log(k_big) + math.log(S_CU) * u)
        sy = (CY + R - R0) * (1 - u) + CU_BALL_Y * u
        return cam_from_anchor((px_end, py_end), (CX, sy), s)

    bg = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    bctx = cairo.Context(bg)
    g = cairo.RadialGradient(CX, CY, 100, CX, CY, 1300)
    g.add_color_stop_rgb(0, 0.10, 0.06, 0.20)
    g.add_color_stop_rgb(1, 0.01, 0.01, 0.03)
    bctx.set_source(g)
    bctx.paint()

    trail = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)  # 世界坐标 1:1，原点在 (CX, CY)
    tctx = cairo.Context(trail)
    frame = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(frame)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
         os.path.join(OUT, "_video.mp4")], stdin=subprocess.PIPE)

    rnd = random.Random(3)
    sparks, debris, shards, sweat, stars = [], [], [], [], []
    cracks = []
    exploded = False
    bi = 0
    prev_tau = 0.0
    last_hit_t, last_hit = -10.0, None
    next_sweat = next_star = 0.0
    cover_frame = int((ev["squeeze"] + 0.6) * FPS)

    for f in range(n_frames):
        t = f / FPS
        tau = taus[f]
        dt = (tau - prev_tau) if t < ev["simend"] else 1 / FPS   # 粒子跟随慢动作
        boomed = t >= ev["boom"]
        mood = active(moods, t)[2]

        # ---- 碰撞事件
        while bi < len(bounces) and bounces[bi][0] <= tau:
            bt, nx, ny, r_after = bounces[bi]
            col = hue(bt * 0.18)
            for _ in range(14):
                a = math.atan2(-ny, -nx) + rnd.uniform(-1.2, 1.2)
                sp = rnd.uniform(250, 900)
                sparks.append([nx * R, ny * R, math.cos(a) * sp, math.sin(a) * sp, 0.45, col, 1.0])
            if t >= ev["crack"]:
                cracks.append((make_crack(rnd, nx, ny), t))
                for _ in range(8):
                    a = math.atan2(ny, nx) + rnd.uniform(-0.8, 0.8)
                    sp = rnd.uniform(150, 500)
                    sparks.append([nx * (R + 5), ny * (R + 5), math.cos(a) * sp, math.sin(a) * sp,
                                   0.6, (0.9, 0.9, 1.0), 0.7])
            last_hit_t, last_hit = t, (bt, nx, ny)
            bi += 1

        # ---- 拖影
        i0, i1 = int(prev_tau * TRAIL_HZ) + 1, int(tau * TRAIL_HZ)
        for i in range(i0, min(i1, len(samples) - 1) + 1):
            x, y, r = samples[i]
            tctx.arc(CX + x, CY + y, r, 0, 2 * math.pi)
            tctx.set_source_rgb(*BG)
            tctx.fill_preserve()
            tctx.set_source_rgb(*hue(i / TRAIL_HZ * 0.18))
            tctx.set_line_width(3)
            tctx.stroke()
        prev_tau = tau

        # ---- 镜头与震动
        cx, cy, s = camera(t)
        shake = 0.0
        if ev["almost"] <= t < ev["simend"]:
            shake = 4
        elif ev["simend"] <= t < ev["boom"]:
            shake = 4 + 10 * (t - ev["simend"]) / (ev["boom"] - ev["simend"])
        elif boomed and t < ev["boom"] + 0.6:
            shake = 30 * (1 - (t - ev["boom"]) / 0.6) ** 2
        bx, by, br = sample_at(samples, tau)
        if br > 250 and t - last_hit_t < 0.1:
            shake = max(shake, 3)
        shx, shy = rnd.uniform(-shake, shake), rnd.uniform(-shake, shake)

        def w2s(x, y):
            return (x - cx) * s + CX + shx, (y - cy) * s + CY + shy

        # ---- 爆炸
        if boomed and not exploded:
            exploded = True
            a = 0.0
            while a < 2 * math.pi - 0.05:
                span = min(rnd.uniform(0.12, 0.32), 2 * math.pi - a)
                mid = a + span / 2
                pts = [(math.cos(a + span * j / 6) * (R + 5), math.sin(a + span * j / 6) * (R + 5))
                       for j in range(7)]
                cxs = math.cos(mid) * (R + 5)
                cys = math.sin(mid) * (R + 5)
                sp = rnd.uniform(350, 1000)
                shards.append([cxs, cys, math.cos(mid) * sp, math.sin(mid) * sp, 0.0,
                               rnd.uniform(-4, 4), [(p[0] - cxs, p[1] - cys) for p in pts], 2.2])
                a += span
            for _ in range(320):
                a = rnd.uniform(0, 2 * math.pi)
                sp = rnd.uniform(400, 1500)
                debris.append([math.cos(a) * R, math.sin(a) * R, math.cos(a) * sp, math.sin(a) * sp,
                               rnd.uniform(1.0, 2.2), hue(rnd.random()), rnd.uniform(4, 10)])

        # ---- 背景、大圆、小圆
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(bg, 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        if t >= ev["doubt"]:
            a = clamp01((t - ev["doubt"]) / 0.3)
            disk = clamp01((t - ev["zoom0"]) / 1.8)
            bsx, bsy = w2s(*big_c)
            ring_back(ctx, bsx, bsy, k_big * R * s, glow=a, disk=disk)
            ring_stroke(ctx, bsx, bsy, k_big * R * s, WHITE, 10, a)

        trail_a = 1.0 if t < ev["zoom0"] else max(0.0, 1 - (t - ev["zoom0"]) / (ev["zoom1"] - ev["zoom0"]) * 1.4)
        sx0, sy0 = w2s(0, 0)
        if not boomed:
            ring_back(ctx, sx0, sy0, R * s)
        elif trail_a > 0:
            ring_back(ctx, sx0, sy0, R * s, glow=0, disk=trail_a)
        if trail_a > 0:
            ctx.save()
            ctx.translate(CX + shx, CY + shy)
            ctx.scale(s, s)
            ctx.translate(-cx - CX, -cy - CY)
            ctx.set_source_surface(trail, 0, 0)
            ctx.paint_with_alpha(trail_a)
            ctx.restore()

        if not boomed:
            tension = clamp01((t - ev["squeeze"]) / max(ev["boom"] - ev["squeeze"], 0.1)) if t >= ev["squeeze"] else 0
            flash = max(0.0, 1 - (t - last_hit_t) / 0.18)
            hc = hue(last_hit[0] * 0.18) if last_hit else WHITE
            base = [1 + (RED[j] - 1) * tension * 0.8 for j in range(3)]
            col = [base[j] + (hc[j] - base[j]) * flash for j in range(3)]
            ring_stroke(ctx, sx0, sy0, R * s, col, 10 + 8 * flash + 10 * clamp01((t - ev["simend"]) / 1.0))
            for lines, t0 in cracks:
                grow = clamp01((t - t0) / 0.15)
                for pts in lines:
                    n = max(2, int(len(pts) * grow + 0.999))
                    sp_ = [w2s(*p) for p in pts[:n]]
                    for w, c, al in ((9, (1, 0.45, 0.2), 0.25 + 0.5 * tension), (3, (1, 1, 1), 0.95)):
                        ctx.move_to(*sp_[0])
                        for p in sp_[1:]:
                            ctx.line_to(*p)
                        ctx.set_source_rgba(*c, al)
                        ctx.set_line_width(w)
                        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                        ctx.stroke()

        # ---- 碎片与粒子（世界坐标）
        alive = []
        for p in shards:
            p[0] += p[2] / FPS
            p[1] += p[3] / FPS
            p[2] *= 0.985
            p[3] = p[3] * 0.985 + 500 / FPS
            p[4] += p[5] / FPS
            p[7] -= 1 / FPS
            if p[7] > 0:
                c, sn = math.cos(p[4]), math.sin(p[4])
                pts = [w2s(p[0] + q[0] * c - q[1] * sn, p[1] + q[0] * sn + q[1] * c) for q in p[6]]
                ctx.move_to(*pts[0])
                for q in pts[1:]:
                    ctx.line_to(*q)
                ctx.set_source_rgba(1, 1, 1, clamp01(p[7]))
                ctx.set_line_width(max(1.5, 10 * s))
                ctx.stroke()
                alive.append(p)
        shards = alive

        alive = []
        for p in sparks:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 1800 * dt
            p[4] -= dt
            if p[4] > 0:
                x, y = w2s(p[0], p[1])
                ctx.arc(x, y, (2 + 6 * p[4]) * p[6] * s, 0, 2 * math.pi)
                ctx.set_source_rgba(*p[5], clamp01(p[4] * 3))
                ctx.fill()
                alive.append(p)
        sparks = alive

        alive = []
        for p in debris:
            p[0] += p[2] / FPS
            p[1] += p[3] / FPS
            p[2] *= 0.985
            p[3] = p[3] * 0.985 + 900 / FPS
            p[4] -= 1 / FPS
            if p[4] > 0:
                x, y = w2s(p[0], p[1])
                ctx.arc(x, y, p[6] * min(1, p[4]) * s, 0, 2 * math.pi)
                ctx.set_source_rgba(*p[5], clamp01(p[4] * 1.5))
                ctx.fill()
                alive.append(p)
        debris = alive

        if boomed and t - ev["boom"] < 0.7:                   # 冲击波
            te = t - ev["boom"]
            x, y = w2s(bx, by)
            ctx.arc(x, y, (r_end + 1600 * te) * s, 0, 2 * math.pi)
            ctx.set_source_rgba(1, 1, 1, 0.6 * (1 - te / 0.7))
            ctx.set_line_width(24 * (1 - te / 0.7) + 2)
            ctx.stroke()

        # ---- 情绪特效：汗珠、星星
        if mood == "strain" and t >= next_sweat:
            next_sweat = t + 0.14
            side = rnd.choice((-1, 1))
            sweat.append([bx + side * br * 0.8, by - br * 0.55, side * rnd.uniform(150, 320),
                          -rnd.uniform(250, 450), 0.6])
        if mood == "happy" and t >= next_star:
            next_star = t + 0.07
            a = rnd.uniform(0, 2 * math.pi)
            d = br * rnd.uniform(1.05, 1.35)
            stars.append([bx + math.cos(a) * d, by + math.sin(a) * d, 0.6, rnd.uniform(0.6, 1.2)])
        alive = []
        for p in sweat:
            p[0] += p[2] / FPS
            p[1] += p[3] / FPS
            p[3] += 1400 / FPS
            p[4] -= 1 / FPS
            if p[4] > 0:
                x, y = w2s(p[0], p[1])
                rr = (10 + br * 0.03) * s
                ang = math.atan2(p[3], p[2]) + math.pi
                ctx.move_to(x + math.cos(ang) * rr * 2.2, y + math.sin(ang) * rr * 2.2)
                ctx.arc(x, y, rr, ang + math.pi / 2, ang - math.pi / 2)
                ctx.close_path()
                ctx.set_source_rgba(0.6, 0.86, 1.0, clamp01(p[4] * 3))
                ctx.fill()
                alive.append(p)
        sweat = alive
        alive = []
        for p in stars:
            p[2] -= 1 / FPS
            if p[2] > 0:
                x, y = w2s(p[0], p[1])
                sz = (14 + br * 0.04) * s * p[3] * math.sin(p[2] / 0.6 * math.pi)
                ctx.move_to(x, y - sz)
                for j in range(1, 8):
                    a = j * math.pi / 4
                    rr = sz if j % 2 == 0 else sz * 0.3
                    ctx.line_to(x + math.sin(a) * rr, y - math.cos(a) * rr)
                ctx.close_path()
                ctx.set_source_rgba(1, 0.95, 0.6, 0.95)
                ctx.fill()
                alive.append(p)
        stars = alive

        # ---- 主角
        if t >= ev["simend"] and not boomed:
            jit = 2 + 8 * (t - ev["simend"])
            bx += rnd.uniform(-jit, jit)
            by += rnd.uniform(-jit, jit)
        bsr = br * s
        if boomed and t < ev["doubt"]:
            te = t - ev["boom"]
            bsr *= 1 + 0.12 * math.exp(-5 * te) * math.cos(16 * te)
        squash = None
        if last_hit and t < ev["simend"]:
            kq = 1 - (tau - last_hit[0]) / 0.07
            if kq > 0:
                squash = (math.atan2(last_hit[2], last_hit[1]), kq)
        if mood == "sad":
            look = (-0.35, 0.55)
        elif mood in ("shock", "happy") or t >= ev["simend"]:
            look = (0.0, 0.0)
        else:
            ax, ay, _ = sample_at(samples, max(tau - 0.02, 0))
            vx, vy = bx - ax, by - ay
            m = math.hypot(vx, vy)
            look = (vx / m, vy / m) if m > 1e-6 else (0.0, 0.0)
        blink = (t % 3.1) < 0.12 and t > 0.5
        sx, sy = w2s(bx, by)
        draw_ball(ctx, sx, sy, bsr, mood, look, blink, squash)

        if mood == "shock":
            st = active(moods, t)[0]
            sz = min(max(60, bsr * 0.5), 170)
            text(ctx, "!?", sx + bsr * 0.75, sy - bsr * 0.95, sz, YELLOW,
                 scale=ease_out_back((t - st) / 0.25), stroke=10)

        if boomed and t - ev["boom"] < 0.5:                   # 白闪
            ctx.set_source_rgba(1, 1, 1, (1 - (t - ev["boom"]) / 0.5) ** 2 * 0.9)
            ctx.paint()

        # ---- 封面：用挤压阶段最缤纷的画面，换上悬念标题
        if f == cover_frame:
            cover = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
            cc = cairo.Context(cover)
            cc.set_source_surface(frame, 0, 0)
            cc.paint()
            text(cc, "它被困在这个圈里一辈子", CX, 230, 66, WHITE)
            text(cc, "直到它发现了一个秘密……", CX, 340, 70, YELLOW)
            text(cc, "结局让我沉默了", CX, 1640, 84, WHITE, stroke=12)
            cover.write_to_png(os.path.join(OUT, "cover.png"))

        # ---- 字幕
        cap = active(caps, t)
        if cap:
            c0, _, s_txt, col = cap
            sc = 1.0 if c0 == 0 else ease_out_back((t - c0) / 0.25)
            dx = dy = 0.0
            if col is RED:
                dx, dy = rnd.uniform(-6, 6), rnd.uniform(-6, 6)
            text(ctx, s_txt, CX + dx, 300 + dy, 66, col, scale=sc)

        # ---- 计数
        hud_a = clamp01((t - ev["first"] - 3.2) / 0.3) * (1 - clamp01((t - ev["boom"]) / 0.3))
        if hud_a > 0:
            text(ctx, f"弹跳 {bi} 次   体积 {round((br / R) ** 2 * 100)}%", CX, 1640, 52,
                 (0.85, 0.87, 0.95), alpha=hud_a, stroke=8)

        frame.flush()
        ff.stdin.write(bytes(frame.get_data()))

    ff.stdin.close()
    ff.wait()


# ---------------------------------------------------------------- 音频
def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def tone(freq, dur, amp):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(t / 0.003, 1.0)
    s = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 4.0)
         + 0.45 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 7.0)
         + 0.20 * np.sin(2 * np.pi * freq * 3.01 * t) * np.exp(-t * 12.0)
         + 0.12 * np.sin(2 * np.pi * freq * 4.2 * t) * np.exp(-t * 25.0))   # 马林巴质感
    return s * env * amp


def pad(notes, dur, amp, attack=0.4, release=0.6):
    t = np.arange(int(dur * SR)) / SR
    s = np.zeros_like(t)
    for n in notes:
        f = midi_hz(n)
        for d in (-0.004, 0.0, 0.004):
            s += np.sin(2 * np.pi * f * (1 + d) * t) / 3
        s += 0.2 * np.sin(2 * np.pi * f * 2 * t)
    env = np.ones_like(t)
    if attack > 0:
        env *= np.minimum(t / attack, 1)
    if release > 0:
        env *= np.clip((dur - t) / release, 0, 1)
    return s * env * amp / max(len(notes), 1)


def noise(dur, seed, lo=None, hi=None):
    n = np.random.default_rng(seed).standard_normal(int(dur * SR))
    if lo and hi:
        n = signal.sosfilt(signal.butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), n)
    elif hi:
        n = signal.sosfilt(signal.butter(2, hi, "lowpass", fs=SR, output="sos"), n)
    elif lo:
        n = signal.sosfilt(signal.butter(2, lo, "highpass", fs=SR, output="sos"), n)
    return n / (np.max(np.abs(n)) + 1e-9)


def sweep(f0, f1, dur, shape=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = f0 + (f1 - f0) * (t / dur) ** shape
    return np.sin(2 * np.pi * np.cumsum(f) / SR), t


def add(buf, sig, start):
    i = int(start * SR)
    if i < 0:
        sig, i = sig[-i:], 0
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[:j - i]


def make_audio(hits, ev, path):
    e = ev
    dur = e["end"]
    buf = np.zeros(int(dur * SR) + SR)

    # 开场：忧伤的 Am 铺底（与结尾同一和弦、同一音量，循环时无缝）
    add(buf, pad((57, 60, 64), e["first"] + 0.3, 0.12, attack=0, release=0.3), 0)
    # 起跳 “嘣~”
    s, t = sweep(180, 520, 0.25, 0.6)
    add(buf, 0.25 * s * np.exp(-t * 8), T_HOP)

    # 发现的瞬间：低音重击 + “？！” 两声
    tb = np.arange(int(0.8 * SR)) / SR
    add(buf, 0.6 * np.sin(2 * np.pi * (45 * tb + 8 * (1 - np.exp(-tb * 8)))) * np.exp(-tb * 4), e["first"])
    add(buf, tone(midi_hz(81), 0.4, 0.18), e["first"] + 0.05)
    add(buf, tone(midi_hz(86), 0.5, 0.18), e["first"] + 0.17)
    # 慢动作里的悬念铺垫，渐强接入主歌
    swell = pad((48, 55, 62), 2.9, 0.08, attack=2.5, release=0.2)
    add(buf, swell, e["first"] + 0.3)

    # 主歌：C-G-Am-F 和弦进行，随紧张度渐强
    m0, m1 = e["first"] + 3.0, e["boom"]
    prog = [(36, 60, 64, 67), (31, 59, 62, 67), (33, 60, 64, 69), (29, 60, 65, 69)]
    tt, k = m0, 0
    while tt < m1:
        d = min(2.0, m1 - tt)
        amp = 0.09 + 0.08 * (tt - m0) / (m1 - m0)
        add(buf, pad(prog[k % 4][1:], d + 0.5, amp, attack=0.15, release=0.5), tt)
        add(buf, pad(prog[k % 4][:1], d + 0.3, amp * 0.9, attack=0.05, release=0.3), tt)
        tt += 2.0
        k += 1

    # 每次碰撞一个《欢乐颂》音符
    prev = -1.0
    for te, i in hits:
        gap = te - prev
        prev = te
        amp = 0.30 * min(1.0, 0.35 + gap * 2.5)
        note = MELODY[i % len(MELODY)]
        add(buf, tone(midi_hz(note), 1.2, amp), te)
        add(buf, tone(midi_hz(note - 12), 0.6, amp * 0.35), te)
        if te >= e["crack"]:                         # 裂纹的咔嚓声
            n = noise(0.08, i, lo=2000, hi=9000) * np.exp(-np.arange(int(0.08 * SR)) / SR * 60)
            add(buf, 0.12 * n, te)

    # 心跳：越挤越快
    tt = e["squeeze"]
    while tt < e["boom"] - 0.15:
        tb = np.arange(int(0.25 * SR)) / SR
        thump = np.sin(2 * np.pi * (52 * tb + 3 * (1 - np.exp(-tb * 30)))) * np.exp(-tb * 16)
        add(buf, 0.55 * thump, tt)
        add(buf, 0.38 * thump, tt + 0.16)
        u = (tt - e["squeeze"]) / max(e["boom"] - e["squeeze"], 0.1)
        tt += 0.85 - 0.45 * u

    # 爆发前的上升音
    d = e["boom"] - e["simend"]
    s, t = sweep(200, 1600, d, 2.0)
    add(buf, (0.12 * s + 0.08 * noise(d, 5, lo=1500, hi=8000)) * (t / d) ** 1.5, e["simend"])

    # 爆炸：轰鸣 + 玻璃碎裂 + 胜利和弦
    tb = np.arange(int(1.8 * SR)) / SR
    boom = np.sin(2 * np.pi * (35 * tb + 10 * (1 - np.exp(-tb * 6)))) * np.exp(-tb * 3.0)
    add(buf, 0.8 * boom + 0.3 * noise(1.8, 1, hi=3000) * np.exp(-tb * 6), e["boom"])
    rng = np.random.default_rng(9)
    for _ in range(40):
        at = e["boom"] + rng.uniform(0, 0.7)
        f = rng.uniform(2500, 6500)
        tp = np.arange(int(0.25 * SR)) / SR
        add(buf, 0.05 * np.sin(2 * np.pi * f * tp) * np.exp(-tp * 25), at)
    win = e["doubt"] - e["boom"]
    add(buf, pad((48, 60, 64, 67, 72), win, 0.16, attack=0.02, release=0.05), e["boom"])
    for k, n in enumerate((72, 76, 79, 84)):
        add(buf, tone(midi_hz(n), 1.6, 0.16), e["boom"] + 0.1 + k * 0.09)

    # “……吗？” 唱片急停
    sc_d = 0.45
    tp = np.arange(int(sc_d * SR)) / SR
    f = 900 * np.abs(np.sin(np.pi * tp / sc_d * 1.5)) + 120
    saw = 2 * ((np.cumsum(f) / SR) % 1) - 1
    scratch = signal.sosfilt(signal.butter(2, [300, 3500], "bandpass", fs=SR, output="sos"),
                             saw + 0.5 * noise(sc_d, 4))
    add(buf, 0.35 * scratch * np.minimum(tp / 0.01, 1) * np.clip((sc_d - tp) / 0.1, 0, 1), e["doubt"])

    # 镜头拉远：风声 + 下行音
    zd = e["zoom1"] - e["zoom0"]
    s, t = sweep(520, 70, zd, 0.7)
    env = np.sin(np.pi * t / zd) ** 1.5
    add(buf, (0.10 * s + 0.12 * noise(zd, 6, hi=900)) * env, e["zoom0"])

    # 结尾：A-E-C 三个孤单的音，然后回到开场的 Am
    for k, n in enumerate((69, 64, 60)):
        add(buf, tone(midi_hz(n), 2.0, 0.3), e["zoom1"] - 0.2 + k * 0.85)
    add(buf, tone(midi_hz(57), 3.0, 0.22), e["push0"])
    tail = dur - e["push0"]
    add(buf, pad((57, 60, 64), tail, 0.12, attack=1.5, release=0), e["push0"])

    # 简单混响
    ir_t = np.arange(int(1.3 * SR)) / SR
    ir = np.random.default_rng(2).standard_normal(len(ir_t)) * np.exp(-ir_t * 4.5)
    ir[0] = 0
    n = len(buf) + len(ir)
    wet = np.fft.irfft(np.fft.rfft(buf, n) * np.fft.rfft(ir, n), n)[:len(buf)]
    wet *= 0.25 * np.max(np.abs(buf)) / (np.max(np.abs(wet)) + 1e-9)
    mix = np.tanh((buf + wet) * 1.3) * 0.9
    mix = mix[:int(dur * SR)]
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    samples, bounces, sim_end = simulate()
    taus, ev, hits = plan(bounces, sim_end)
    print(f"模拟 {sim_end:.1f}s，{len(bounces)} 次碰撞；成片 {ev['end']:.1f}s")
    print("剧情节点：" + "  ".join(f"{k}={v:.1f}" for k, v in ev.items()))
    render(samples, bounces, sim_end, taus, ev)
    make_audio(hits, ev, os.path.join(OUT, "_audio.wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(OUT, "_video.mp4"),
         "-i", os.path.join(OUT, "_audio.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", os.path.join(OUT, "story.mp4")], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(OUT, "cover.png"),
         "-q:v", "2", os.path.join(OUT, "cover.jpg")], check=True)
    for tmp in ("_video.mp4", "_audio.wav", "cover.png"):
        os.remove(os.path.join(OUT, tmp))
    print("完成：output/story.mp4, output/cover.jpg")


if __name__ == "__main__":
    main()
