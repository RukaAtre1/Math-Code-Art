"""Run from the project root: py -3.14 render_exhibition_v3.py --help."""
import argparse
from datetime import datetime
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from music_spirograph.exhibition_v3 import (Features, Physics, Renderer,
    TRAIL_SECONDS, analyze_track_profile, make_bodies, place_harmony_body)
from music_spirograph.art_direction import (config_for_audio, derive_palette, prepare_background,
    OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_DPI)
from music_spirograph.curve_families import FAMILIES, NAMES, recommend_family

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "02_Source_Materials" / "Audio"


def tracks():
    return sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in (".ogg", ".mp3", ".wav", ".flac")
                  and p.stem.lower() != "luzhouyue")


def choose_family(requested, art, track_profile):
    """Curated family by default; 'auto' borrows the most similar piece's."""
    if requested == "track":
        return (art.family, "curated") if art else (recommend_family(track_profile), "auto")
    if requested == "auto":
        return recommend_family(track_profile), "auto"
    return requested, "override"


def render(path, args, output):
    print(f"Analyzing: {path.name}", flush=True)
    track_profile = analyze_track_profile(path)
    family, family_source = choose_family(getattr(args, "family", "track"), art=config_for_audio(path),
                                          track_profile=track_profile)
    print(f"Curve family: {family} ({family_source})", flush=True)
    features = Features(path, args.start, args.duration)
    height = args.width * 9 // 16
    height -= height % 2
    art = config_for_audio(path)
    background, background_credit, palette = None, None, None
    if art is not None:
        background, background_credit = prepare_background(art, args.width, height)
        palette = derive_palette(background, art.palette)
    bodies = make_bodies(args.count, args.seed, track_profile, palette=palette,
                         color_profile=art.color_profile if art else None, family=family)
    harmony_body = bodies[3] if args.count == 4 else None
    if harmony_body is not None:
        harmony_body.presence = 0.0
    physics = Physics()
    height = args.width * 9 // 16
    height -= height % 2
    renderer = Renderer(bodies, args.width, height, args.guides, args.labels, background=background)
    output.mkdir(parents=True, exist_ok=False)
    movie = output / f"{path.stem}.mp4"
    log_path = output / "ffmpeg.log"
    # Stream raw frames: no thousands of PNGs and no deletion of old exports.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "rawvideo",
           "-pixel_format", "rgb24", "-video_size", f"{args.width}x{height}",
           "-framerate", str(args.fps), "-i", "pipe:0", "-ss", str(args.start),
           "-i", str(path), "-map", "0:v:0", "-map", "1:a:0", "-t", str(features.duration),
           "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(movie)]
    frame_count = math.ceil(features.duration * args.fps)
    captures = {int((frame_count - 1) * fraction) for fraction in (0.25, 0.5, 0.75)}
    min_gap = float("inf")
    harmony_seconds = 0.0
    harmony_entries = 0
    color_samples = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in range(frame_count):
                t = frame / args.fps
                f = features.sample(t)
                active_bodies = bodies[:3]
                if harmony_body is not None:
                    wants_harmony = f["harmony_active"] >= 0.5
                    if wants_harmony and harmony_body.presence == 0:
                        wants_harmony = place_harmony_body(harmony_body, bodies[:3])
                        if wants_harmony:
                            harmony_entries += 1
                    harmony_body.update_presence(wants_harmony, 1 / args.fps)
                    if harmony_body.presence > 0:
                        active_bodies = bodies
                        harmony_seconds += 1 / args.fps
                for b in active_bodies:
                    b.update_visual(f, 1 / args.fps)
                    b.update_motion(f, 1 / args.fps)
                if frame:
                    physics.step(active_bodies, 1 / args.fps)
                for i, a in enumerate(active_bodies):
                    for b in active_bodies[i + 1:]:
                        gap = math.hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius
                        min_gap = min(min_gap, gap)
                        if gap < -1e-6:
                            raise RuntimeError(f"Collider overlap at {t:.3f}s: {gap}")
                process.stdin.write(renderer.draw(bodies, frame).tobytes())
                if frame % args.fps == 0:
                    color_samples.append(dict(time_seconds=t,
                        features={k: f[k] for k in ("energy", "onset", "timbre", "flux", "bass", "high",
                            "phrase_energy", "phrase_anchor", "onset_density", "phrase_change", "crescendo")},
                        voices=[dict(role=b.role, rgb=b.paint_rgb.tolist(), width_px_720p=b.linewidth,
                                     mode=b.drawing_mode) for b in active_bodies]))
                if frame in captures:
                    renderer.save(output / f"{path.stem}_{t:06.2f}s.png", dpi=100)
                if frame % (args.fps * 5) == 0:
                    print(f"  {t:.0f}/{features.duration:.0f}s", flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            renderer.close()
    info = dict(audio=str(path), start_seconds=args.start, duration_seconds=features.duration,
                artwork_title=art.title if art else path.stem,
                palette=art.palette if art else None, background=background_credit,
                stroke_style=dict(version="layered_strokes_v2", profile=art.color_profile if art else None,
                                  recomputed_width_pixels_at_720p="0.8 + 0.35*energy + 0.15*onset",
                                  ink_width_pixels_at_720p="1.4 + 0.55*energy + 0.25*onset",
                                  color_history_seconds=1.5,
                                  phrase_layer="slow energy, spectral brightness, onset density and chroma-change proxies; not emotion or phrase-boundary recognition",
                                  color_mapping="per-voice instantaneous features blended with slow phrase anchor; recent style history mapped tail-to-head on regenerated geometry"),
                drawing_model="hybrid_recomputed_curve_and_rotating_ink_v3",
                curve_family=family, curve_family_source=family_source,
                curve_family_description=FAMILIES[family].description if family in FAMILIES
                    else "Original V3 double-rotor formula",
                motif_switches={b.role: b.switches for b in bodies},
                drawing_modes={b.role: b.drawing_mode for b in bodies},
                track_profile=asdict(track_profile),
                trail_seconds=TRAIL_SECONDS,
                core_count=3, maximum_count=args.count,
                harmony_mode="automatic" if harmony_body is not None else "disabled",
                harmony_visible_seconds=round(harmony_seconds, 3),
                harmony_entries=harmony_entries,
                seed=args.seed, fps=args.fps, width=args.width, height=height,
                guides=args.guides, labels=args.labels, collision_contacts=physics.contacts,
                minimum_collider_gap=min_gap,
                normalization="5th/95th percentiles across full recording; attack/release smoothing",
                rights="Not verified by renderer; retain the source recording license separately.")
    (output / "settings.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "color_timeline.json").write_text(json.dumps(color_samples, indent=2), encoding="utf-8")
    print(f"Saved: {movie}", flush=True)


def main():
    # Windows consoles may default to GBK; filenames include Chinese and ö.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Three music-driven spirographs, with an optional automatically entering harmonic voice.")
    parser.add_argument("--audio", type=Path, help="One local audio file; default Sugar Plum Fairy")
    parser.add_argument("--all", action="store_true", help="Render all local tracks except luzhouyue")
    parser.add_argument("--list", action="store_true", help="List batch tracks without rendering")
    parser.add_argument("--exhibition", action="store_true", help="Export five titled print artworks, credits and contact sheet")
    parser.add_argument("--backgrounds-only", action="store_true", help="Export five clean graded backgrounds without spirographs")
    parser.add_argument("--background-output", type=Path, default=ROOT / "01_Deliverables" / "Clean_Backgrounds")
    parser.add_argument("--print-width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--print-height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI)
    parser.add_argument("--print-output", type=Path, default=ROOT / "01_Deliverables" / "Print_Artworks")
    parser.add_argument("--count", type=int, choices=(3, 4), default=4,
                        help="Maximum count: 4 enables an automatic temporary harmony voice; 3 disables it")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=None, help="Seconds; omitted means entire remaining track")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=3840,
                        help="Video width in pixels; 3840 (4K) is the 12.8 x 7.2 inch frame at 300 dpi")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--family", choices=("track", "auto", *NAMES), default="track",
                        help="Curve family: 'track' uses the curated family (auto for unconfigured audio); "
                             "'auto' uses the most similar exhibition piece's; double_rotor is the original V3 look")
    parser.add_argument("--guides", action="store_true", help="Show otherwise invisible circle colliders")
    parser.add_argument("--labels", action="store_true", help="Show feature labels for comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "03_Generation_Tools" / "Generated_Runs" / "Previews")
    args = parser.parse_args()
    if args.backgrounds_only:
        if args.exhibition or args.audio or args.all or args.list:
            parser.error("--backgrounds-only exports scenery only; omit --exhibition, --audio, --all and --list")
        from music_spirograph.exhibition_export import export_backgrounds
        export_backgrounds(args.background_output, args.print_width, args.print_height, args.dpi)
        return
    if args.exhibition:
        if args.audio or args.all or args.list:
            parser.error("--exhibition exports the five configured tracks; omit --audio, --all and --list")
        from music_spirograph.exhibition_export import export_series
        export_series(args.print_output, args.print_width, args.print_height, args.dpi, args.seed)
        return
    if args.list:
        for p in tracks():
            print(p.name)
        return
    if args.all and args.audio:
        parser.error("Choose --all or --audio, not both")
    if args.start < 0 or (args.duration is not None and args.duration <= 0):
        parser.error("Start must be nonnegative and duration positive")
    if args.width < 320 or args.width % 2 or args.fps < 1:
        parser.error("Width must be even and at least 320; FPS must be positive")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required on PATH")
    selected = tracks() if args.all else [args.audio or AUDIO_DIR / "Dance_of_the_Sugar_Plum_Fairy.ogg"]
    for path in selected:
        if not path.is_file():
            parser.error(f"Audio file missing: {path}")
    run = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for path in selected:
        suffix = "3spiros_auto_harmony" if args.count == 4 else "3spiros"
        art = config_for_audio(path)
        # Long recording names repeated in directory and capture filenames can
        # exceed Windows' legacy path limit. Use the short exhibition key.
        folder = art.key if art else path.stem[:48]
        render(path.resolve(), args, run / f"{folder}_{suffix}")


if __name__ == "__main__":
    main()
