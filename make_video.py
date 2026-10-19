"""生成竖屏短视频《球每弹一次就变大一点》。

物理模拟 → cairo 逐帧绘制 → ffmpeg 编码；每次碰撞的音符由模拟结果直接合成，
因此画面和声音严格同步。运行：python3 make_video.py
"""
import colorsys
import math
import os
import random
import subprocess
import wave

import cairo
import numpy as np

W, H, FPS = 1080, 1920, 60
CX, CY, R = W / 2, 1010, 430          # 圆环中心与半径
R0, GROW = 22, 5                      # 初始球半径、每次碰撞增长
GRAVITY = 2600
SUB = 20                              # 每帧物理子步
TRAIL_PER_FRAME = 4                   # 每帧绘制的拖影圈数
END_GAP = 6
SR = 44100
FONT = "WenQuanYi Zen Hei"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

BG = (0.027, 0.027, 0.05)

# 《欢乐颂》（贝多芬，公有领域），MIDI 音高
ODE = [64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62,
       64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 62, 60, 60,
       62, 62, 64, 60, 62, 64, 65, 64, 60, 62, 64, 65, 64, 62, 60, 62, 55,
       64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 62, 60, 60]
MELODY = [n + 12 for n in ODE]


# ---------------------------------------------------------------- 物理模拟
def simulate(seed=7):
    rnd = random.Random(seed)
    x, y, vx, vy, r = 0.0, -R * 0.55, 420.0, 0.0, float(R0)
    e0 = 0.5 * (vx * vx + vy * vy) - GRAVITY * y   # 保持总能量，球不会停下
    dt = 1 / (FPS * SUB)
    t = 0.0
    frames, bounces = [], []
    while True:
        trail = []
        for i in range(SUB):
            vy += GRAVITY * dt
            x += vx * dt
            y += vy * dt
            t += dt
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
                    bounces.append((t, nx, ny))
            if (i + 1) % (SUB // TRAIL_PER_FRAME) == 0:
                trail.append((x, y, r))
        frames.append({"x": x, "y": y, "r": r, "trail": trail, "t": t})
        if r >= R - END_GAP:
            return frames, bounces


# ---------------------------------------------------------------- 绘图工具
def hue(h, s=0.85, v=1.0):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def ease_out_back(p):
    p = min(max(p, 0.0), 1.0)
    c1 = 1.70158
    return 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2


def text(ctx, s, x, y, size, color, scale=1.0, alpha=1.0, stroke=10):
    if alpha <= 0 or scale <= 0:
        return
    ctx.save()
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    ext = ctx.text_extents(s)
    ctx.translate(x, y)
    ctx.scale(scale, scale)
    ctx.move_to(-ext.x_advance / 2, ext.height / 2 - (ext.height + ext.y_bearing))
    ctx.text_path(s)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_source_rgba(0, 0, 0, 0.85 * alpha)
    ctx.set_line_width(stroke)
    ctx.stroke_preserve()
    ctx.set_source_rgba(*color, alpha)
    ctx.fill()
    ctx.restore()


def make_base():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    g = cairo.RadialGradient(CX, CY, 100, CX, CY, 1300)
    g.add_color_stop_rgb(0, 0.10, 0.06, 0.20)
    g.add_color_stop_rgb(1, 0.01, 0.01, 0.03)
    ctx.set_source(g)
    ctx.paint()
    for w, a in ((60, 0.05), (36, 0.08), (20, 0.12)):    # 圆环外发光
        ctx.arc(CX, CY, R + 5, 0, 2 * math.pi)
        ctx.set_source_rgba(0.6, 0.7, 1.0, a)
        ctx.set_line_width(w)
        ctx.stroke()
    ctx.arc(CX, CY, R, 0, 2 * math.pi)
    ctx.set_source_rgb(*BG)
    ctx.fill()
    return surf


# ---------------------------------------------------------------- 渲染
def render(frames, bounces):
    total_sim = len(frames)
    end_frames = int(3.9 * FPS)
    n_frames = total_sim + end_frames
    t_end = total_sim / FPS
    t_boom = t_end + 0.35
    n_bounce = len(bounces)
    bounce_frames = [int(b[0] * FPS) for b in bounces]

    trail = make_base()
    tctx = cairo.Context(trail)
    frame = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(frame)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
         os.path.join(OUT, "_video.mp4")], stdin=subprocess.PIPE)

    sparks, debris = [], []
    rnd = random.Random(3)
    bi = 0
    last_bounce_t = -10.0
    cover_written = False

    for f in range(n_frames):
        t = f / FPS
        sim = frames[min(f, total_sim - 1)]

        # 本帧发生的碰撞
        while bi < n_bounce and bounce_frames[bi] <= f and f < total_sim:
            _, nx, ny = bounces[bi]
            px, py = CX + nx * R, CY + ny * R
            col = hue(t * 0.12)
            for _ in range(14):
                a = math.atan2(-ny, -nx) + rnd.uniform(-1.2, 1.2)
                sp = rnd.uniform(250, 900)
                sparks.append([px, py, math.cos(a) * sp, math.sin(a) * sp, 0.45, col])
            last_bounce_t = t
            bi += 1

        # 拖影：在持久画布上叠加圆圈
        if f < total_sim:
            for (x, y, r) in sim["trail"]:
                tctx.arc(CX + x, CY + y, r, 0, 2 * math.pi)
                tctx.set_source_rgb(*BG)
                tctx.fill_preserve()
                tctx.set_source_rgb(*hue(t * 0.12))
                tctx.set_line_width(3)
                tctx.stroke()

        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(trail, 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        boomed = t >= t_boom
        if boomed and not debris:
            col0 = t * 0.12
            for _ in range(420):                 # 球炸开
                a = rnd.uniform(0, 2 * math.pi)
                rr = math.sqrt(rnd.random()) * sim["r"]
                sp = rnd.uniform(300, 1700) * (0.4 + rr / R)
                debris.append([CX + sim["x"] + math.cos(a) * rr, CY + sim["y"] + math.sin(a) * rr,
                               math.cos(a) * sp, math.sin(a) * sp, rnd.uniform(1.0, 2.2),
                               hue(col0 + rnd.uniform(-0.2, 0.2)), rnd.uniform(4, 11)])
            for _ in range(160):                 # 圆环碎片
                a = rnd.uniform(0, 2 * math.pi)
                sp = rnd.uniform(500, 1400)
                debris.append([CX + math.cos(a) * R, CY + math.sin(a) * R,
                               math.cos(a) * sp, math.sin(a) * sp, rnd.uniform(0.8, 1.6),
                               (1, 1, 1), rnd.uniform(3, 7)])

        if boomed:   # 爆炸后压暗图案
            k = min((t - t_boom) / 0.6, 1.0)
            ctx.set_source_rgba(0, 0, 0, 0.55 * k)
            ctx.paint()

        # 圆环
        if not boomed:
            flash = max(0.0, 1 - (t - last_bounce_t) / 0.18)
            tension = max(0.0, (t - t_end) / 0.35) if t > t_end else 0.0
            ring_col = [1 + (c - 1) * flash for c in hue(t * 0.12)]
            ctx.arc(CX, CY, R + 5, 0, 2 * math.pi)
            ctx.set_source_rgb(*ring_col)
            ctx.set_line_width(10 + 8 * flash + 14 * tension)
            ctx.stroke()

            # 当前的球
            shake = 7 * tension
            bx = CX + sim["x"] + rnd.uniform(-shake, shake)
            by = CY + sim["y"] + rnd.uniform(-shake, shake)
            br = sim["r"] * (1 + 0.015 * tension * math.sin(t * 90))
            c = hue(t * 0.12)
            g = cairo.RadialGradient(bx - br * 0.35, by - br * 0.35, br * 0.05, bx, by, br)
            g.add_color_stop_rgb(0, *[min(1, v + 0.45) for v in c])
            g.add_color_stop_rgb(1, *[v * 0.75 for v in c])
            ctx.arc(bx, by, br, 0, 2 * math.pi)
            ctx.set_source(g)
            ctx.fill_preserve()
            ctx.set_source_rgb(1, 1, 1)
            ctx.set_line_width(5)
            ctx.stroke()

        # 火花
        dt = 1 / FPS
        alive = []
        for p in sparks:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 1800 * dt
            p[4] -= dt
            if p[4] > 0:
                ctx.arc(p[0], p[1], 2 + 6 * p[4], 0, 2 * math.pi)
                ctx.set_source_rgba(*p[5], min(1, p[4] * 3))
                ctx.fill()
                alive.append(p)
        sparks = alive
        alive = []
        for p in debris:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[2] *= 0.985
            p[3] = p[3] * 0.985 + 900 * dt
            p[4] -= dt
            if p[4] > 0:
                ctx.arc(p[0], p[1], p[6] * min(1, p[4]), 0, 2 * math.pi)
                ctx.set_source_rgba(*p[5], min(1, p[4] * 1.5))
                ctx.fill()
                alive.append(p)
        debris = alive

        # 爆炸白闪
        if boomed and t - t_boom < 0.5:
            ctx.set_source_rgba(1, 1, 1, (1 - (t - t_boom) / 0.5) ** 2 * 0.9)
            ctx.paint()

        # 顶部钩子文案
        intro = ease_out_back(t / 0.35)
        text(ctx, "球每弹一次 就变大一点", CX, 250, 70, (1, 1, 1), scale=intro)
        intro2 = ease_out_back((t - 0.25) / 0.35)
        text(ctx, "猜猜它要弹几次才能填满？", CX, 350, 58, (1, 0.84, 0.29), scale=intro2)

        # 计数器
        hud_alpha = 1.0 if not boomed else max(0.0, 1 - (t - t_boom) / 0.3)
        pop = 1 + 0.3 * max(0.0, 1 - (t - last_bounce_t) / 0.15)
        fill_pct = int(round((sim["r"] / R) ** 2 * 100)) if f < total_sim - 1 else 100
        text(ctx, "弹跳次数", CX - 230, 1545, 44, (0.75, 0.78, 0.9), alpha=hud_alpha, stroke=8)
        text(ctx, "填充", CX + 230, 1545, 44, (0.75, 0.78, 0.9), alpha=hud_alpha, stroke=8)
        text(ctx, str(min(bi, n_bounce)), CX - 230, 1650, 120, hue(t * 0.12, 0.5),
             scale=pop, alpha=hud_alpha, stroke=12)
        text(ctx, f"{fill_pct}%", CX + 230, 1650, 120, (1, 1, 1),
             scale=pop, alpha=hud_alpha, stroke=12)

        # 结尾揭晓
        if boomed:
            te = t - t_boom
            text(ctx, "填满了！", CX, 880, 150, (1, 1, 1), scale=ease_out_back((te - 0.45) / 0.4))
            text(ctx, f"一共弹了 {n_bounce} 次", CX, 1060, 88, hue(0.55, 0.45),
                 scale=ease_out_back((te - 0.95) / 0.35))
            text(ctx, "你猜对了吗？", CX, 1230, 80, (1, 0.84, 0.29),
                 scale=ease_out_back((te - 1.55) / 0.35))
            text(ctx, "评论区报出你的答案", CX, 1340, 48, (0.85, 0.85, 0.95),
                 alpha=min(1, max(0, (te - 1.9) / 0.3)), stroke=8)

        frame.flush()
        ff.stdin.write(bytes(frame.get_data()))

        # 封面：视频中段最缤纷的图案 + 钩子文案，计数器换成问号避免剧透
        if not cover_written and f == int(total_sim * 0.55):
            cover = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
            cc = cairo.Context(cover)
            cc.set_source_surface(trail, 0, 0)
            cc.paint()
            cc.arc(CX, CY, R + 5, 0, 2 * math.pi)
            cc.set_source_rgb(1, 1, 1)
            cc.set_line_width(10)
            cc.stroke()
            cc.arc(CX + sim["x"], CY + sim["y"], sim["r"], 0, 2 * math.pi)
            cc.set_source_rgb(*hue(t * 0.12))
            cc.fill_preserve()
            cc.set_source_rgb(1, 1, 1)
            cc.set_line_width(5)
            cc.stroke()
            text(cc, "球每弹一次 就变大一点", CX, 250, 70, (1, 1, 1))
            text(cc, "猜猜它要弹几次才能填满？", CX, 350, 58, (1, 0.84, 0.29))
            text(cc, "???", CX, 1620, 150, (1, 1, 1), stroke=14)
            cover.write_to_png(os.path.join(OUT, "cover.png"))
            cover_written = True

    ff.stdin.close()
    ff.wait()
    return n_frames, t_boom


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


def add(buf, sig, start):
    i = int(start * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[:j - i]


def make_audio(bounces, n_frames, t_boom, path):
    dur = n_frames / FPS
    buf = np.zeros(int(dur * SR) + SR)
    prev = -1.0
    for i, (t, _, _) in enumerate(bounces):
        gap = t - prev
        prev = t
        amp = 0.32 * min(1.0, 0.35 + gap * 2.5)        # 越密集越轻，避免糊成一片
        note = MELODY[i % len(MELODY)]
        add(buf, tone(midi_hz(note), 1.2, amp), t)
        add(buf, tone(midi_hz(note - 12), 0.6, amp * 0.35), t)

    # 炸开前的上升音
    rise = np.arange(int(0.35 * SR)) / SR
    freq = 200 + 1400 * (rise / 0.35) ** 2
    add(buf, 0.12 * np.sin(2 * np.pi * np.cumsum(freq) / SR) * (rise / 0.35), t_boom - 0.35)

    # 爆炸：低频轰鸣 + 噪声 + 大三和弦
    tb = np.arange(int(1.6 * SR)) / SR
    boom = np.sin(2 * np.pi * (35 * tb + 10 * (1 - np.exp(-tb * 6)))) * np.exp(-tb * 3.5)   # 95Hz→35Hz
    noise = np.random.default_rng(1).standard_normal(len(tb)) * np.exp(-tb * 7)
    add(buf, 0.7 * boom + 0.25 * noise, t_boom)
    for k, n in enumerate((72, 76, 79, 84, 88)):
        add(buf, tone(midi_hz(n), 3.0, 0.2), t_boom + k * 0.035)
    for k, at in enumerate((0.45, 0.95, 1.55)):         # 文字出现时的叮
        add(buf, tone(midi_hz((84, 88, 91)[k]), 1.2, 0.18), t_boom + at)

    # 简单混响
    ir_t = np.arange(int(1.3 * SR)) / SR
    ir = np.random.default_rng(2).standard_normal(len(ir_t)) * np.exp(-ir_t * 4.5)
    ir[0] = 0
    n = len(buf) + len(ir)
    wet = np.fft.irfft(np.fft.rfft(buf, n) * np.fft.rfft(ir, n), n)[:len(buf)]
    wet *= 0.25 * np.max(np.abs(buf)) / (np.max(np.abs(wet)) + 1e-9)
    mix = np.tanh((buf + wet) * 1.4) * 0.9
    mix = mix[:int(dur * SR)]
    fade = int(0.3 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    frames, bounces = simulate()
    print(f"模拟：{len(frames) / FPS:.1f}s，{len(bounces)} 次碰撞")
    n_frames, t_boom = render(frames, bounces)
    make_audio(bounces, n_frames, t_boom, os.path.join(OUT, "_audio.wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(OUT, "_video.mp4"),
         "-i", os.path.join(OUT, "_audio.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", os.path.join(OUT, "bounce.mp4")], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(OUT, "cover.png"),
         "-q:v", "2", os.path.join(OUT, "cover.jpg")], check=True)
    for tmp in ("_video.mp4", "_audio.wav", "cover.png"):
        os.remove(os.path.join(OUT, tmp))
    print("完成：output/bounce.mp4, output/cover.jpg")


if __name__ == "__main__":
    main()
