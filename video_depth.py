#!/usr/bin/env python
"""Turn any video into a FaceDepth depth-map video.

Decodes anything ffmpeg can read (mov, mp4, mkv, and the rest, up to 4K), runs every
frame through FaceDepth, and re-encodes. A progress bar reports frames per second and
ETA. Output is either the depth map alone or a side-by-side with the source on the left.

Frames stream through an ffmpeg pipe, so a long 4K clip never lands on disk as a frame
dump and memory stays flat.

Stability
---------
FaceDepth predicts each frame independently, which can make the colour mapping pulse
between frames. Two inference-time controls damp that:

  --range-ema     smooths the near/far normalisation range across frames. On by default.
                  Removes brightness pulsing with no ghosting. Leave it on.
  --smooth-depth  blends each depth frame with the previous one. Off by default.
                  Cuts residual per-pixel jitter, but ghosts behind fast motion.
                  0.3 is a reasonable starting point for handheld footage.

Setup
-----
    pip install torch torchvision opencv-python numpy
    git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/DepthAnythingV2

Download FaceDepth_step15792.pt from https://huggingface.co/a-ml/FaceDepth and pass it
with --ckpt. ffmpeg must be on PATH.

Examples
--------
    python video_depth.py --input clip.mov --output depth.mp4
    python video_depth.py --input clip.mkv --output sbs.mp4 --side-by-side --colormap turbo
    python video_depth.py --input 4k.mp4 --output out.mp4 --smooth-depth 0.3 --bf16

Runs on Apple silicon (mps), CUDA, or CPU, picked automatically.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
for cand in (ROOT / "third_party" / "DepthAnythingV2", ROOT.parent / "third_party" / "DepthAnythingV2"):
    if cand.exists():
        sys.path.insert(0, str(cand))
        break
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    sys.exit("Could not import depth_anything_v2. Clone Depth-Anything-V2 into "
             "third_party/DepthAnythingV2 (see the setup notes at the top of this file).")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
COLORMAPS = {"inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA,
             "turbo": cv2.COLORMAP_TURBO, "viridis": cv2.COLORMAP_VIRIDIS,
             "plasma": cv2.COLORMAP_PLASMA, "bone": cv2.COLORMAP_BONE, "gray": None}


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    n = int(s["nb_frames"]) if s.get("nb_frames", "N/A") not in ("N/A", None) else \
        int(round(float(s.get("duration", 0)) * fps)) or 0
    return w, h, fps, n


def read_exact(pipe, n):
    buf = b""
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def fit14(w, h, res):
    """Scale so the longer side is about `res`, with both sides multiples of 14."""
    scale = res / max(w, h)
    return (max(14, int(round(w * scale / 14)) * 14),
            max(14, int(round(h * scale / 14)) * 14))


def bar(i, n, t0):
    frac = i / n if n else 0
    filled = int(40 * frac)
    el = time.time() - t0
    fps = i / max(el, 1e-6)
    eta = (n - i) / max(fps, 1e-6) if n else 0
    print(f"\r[{'#' * filled}{'-' * (40 - filled)}] {i}/{n} {100 * frac:5.1f}%  "
          f"{fps:4.1f} fps  ETA {eta:5.0f}s", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt", default="FaceDepth_step15792.pt")
    ap.add_argument("--res", type=int, default=910,
                    help="model longer-side resolution, rounded to a multiple of 14. "
                         "Higher is sharper and slower.")
    ap.add_argument("--side-by-side", action="store_true", help="source left, depth right")
    ap.add_argument("--colormap", choices=list(COLORMAPS), default="inferno")
    ap.add_argument("--invert", action="store_true", help="flip so near reads dark")
    ap.add_argument("--near-pct", type=float, default=2.0)
    ap.add_argument("--far-pct", type=float, default=98.0)
    ap.add_argument("--range-ema", type=float, default=0.85,
                    help="temporal smoothing of the near/far range, 0 disables")
    ap.add_argument("--smooth-depth", type=float, default=0.0,
                    help="temporal smoothing of depth itself, 0 disables, ghosts on motion")
    ap.add_argument("--bf16", action="store_true", help="roughly 2x faster, negligible quality cost")
    ap.add_argument("--keep-audio", action="store_true", default=True)
    ap.add_argument("--no-audio", dest="keep_audio", action="store_false")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality, lower is better")
    args = ap.parse_args()

    inp = str(Path(args.input).expanduser())
    W, H, fps, N = probe(inp)
    mw, mh = fit14(W, H, args.res)
    outW, outH = (W * 2, H) if args.side_by_side else (W, H)
    print(f"input {W}x{H} @ {fps:.3f}fps, {N or '?'} frames  ->  model {mw}x{mh}  ->  "
          f"output {outW}x{outH} ({'side-by-side' if args.side_by_side else 'depth'})", flush=True)

    dev = pick_device()
    m = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(ck.get("ema_model") or ck.get("model") or ck)
    m = m.to(dev).eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    print(f"loaded {Path(args.ckpt).name} on {dev}", flush=True)

    @torch.no_grad()
    def infer(rgb):
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
        if args.bf16 and dev != "cpu":
            with torch.autocast(dev, dtype=torch.bfloat16):
                d = m((x - mean) / std)
        else:
            d = m((x - mean) / std)
        return d.float().cpu().numpy()[0]

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", inp, "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    enc_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{outW}x{outH}", "-r", f"{fps}", "-i", "-"]
    if args.keep_audio:
        enc_cmd += ["-i", inp, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
    enc_cmd += ["-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p", args.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=10 ** 8)

    frame_bytes = W * H * 3
    ema_lo = ema_hi = ema_depth = None
    t0 = time.time()
    i = 0
    try:
        while True:
            raw = read_exact(dec.stdout, frame_bytes)
            if raw is None:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
            small = cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            disp = infer(rgb)  # inverse depth, larger is nearer

            if args.smooth_depth > 0:
                ema_depth = disp if ema_depth is None else \
                    (1 - args.smooth_depth) * disp + args.smooth_depth * ema_depth
                disp = ema_depth

            lo = np.percentile(disp, args.near_pct)
            hi = np.percentile(disp, args.far_pct)
            a = args.range_ema
            if a > 0:
                if ema_lo is None:
                    ema_lo, ema_hi = lo, hi
                else:
                    ema_lo = (1 - a) * lo + a * ema_lo
                    ema_hi = (1 - a) * hi + a * ema_hi
                lo, hi = ema_lo, ema_hi

            norm = np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1)
            if args.invert:
                norm = 1 - norm
            gray = (norm * 255).astype(np.uint8)
            cmap = COLORMAPS[args.colormap]
            dvis = cv2.applyColorMap(gray, cmap) if cmap is not None else \
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            dvis = cv2.resize(dvis, (W, H), interpolation=cv2.INTER_CUBIC)

            out = np.hstack([frame, dvis]) if args.side_by_side else dvis
            enc.stdin.write(out.tobytes())

            i += 1
            if i % 5 == 0 or i == N:
                bar(i, N, t0)
    finally:
        print()
        if dec.stdout:
            dec.stdout.close()
        dec.wait()
        if enc.stdin:
            enc.stdin.close()
        enc.wait()
    print(f"done: {i} frames -> {args.output}  "
          f"({i / max(time.time() - t0, 1e-6):.1f} fps avg)", flush=True)


if __name__ == "__main__":
    main()
