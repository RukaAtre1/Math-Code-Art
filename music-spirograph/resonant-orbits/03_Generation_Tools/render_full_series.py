"""Render full recordings concurrently using the exhibition V3 renderer."""
import os

# Keep each independent render from reserving the whole machine for BLAS.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import html
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.parse import quote

from render_exhibition_v3 import ROOT, render
from music_spirograph.art_direction import TRACK_CONFIG, resolve_audio


def render_one(key, run, width, fps):
    config = TRACK_CONFIG[key]
    output = run / key
    args = SimpleNamespace(start=0, duration=None, width=width, fps=fps,
                           count=4, seed=17, guides=False, labels=False, family="track")
    with (run / f"{key}.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            render(resolve_audio(config), args, output)
    return verify_one(key, run, width, fps)


def verify_one(key, run, width, fps):
    config = TRACK_CONFIG[key]
    output = run / key
    record_dir = output / "Production_Records" if (output / "Production_Records").exists() else output
    settings = json.loads((record_dir / "settings.json").read_text(encoding="utf-8"))
    movie = next(output.glob("*.mp4"))
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of", "json", str(movie)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    # -t/-shortest align the final video frame to the source audio endpoint.
    # Ceil produces the streamed frame count, but the muxed last frame may be
    # trimmed. Require at most one frame difference and independently check
    # source duration; never accept a truncated song.
    expected_frames = math.ceil(settings["duration_seconds"] * fps)
    assert abs(int(video["nb_frames"])-expected_frames) <= 1, (video, expected_frames)
    assert (video["width"], video["height"]) == (width, settings["height"])
    assert video["r_frame_rate"] == f"{fps}/1"
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    assert abs(float(probe["format"]["duration"])-settings["duration_seconds"]) < 0.1
    assert settings["minimum_collider_gap"] >= -1e-6
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(movie),
                    "-f", "null", "-"], check=True)
    record = dict(key=key, title=config.title, video=movie.relative_to(run).as_posix(),
                  duration_seconds=settings["duration_seconds"], frames=int(video["nb_frames"]),
                  expected_streamed_frames=expected_frames,
                  muxed_duration_seconds=float(probe["format"]["duration"]),
                  width=width, height=settings["height"], fps=fps, size_bytes=int(probe["format"]["size"]),
                  audio_present=True, full_decode="passed", collision_check="passed",
                  stroke_style=settings["stroke_style"], background=settings["background"])
    (record_dir / "quality_report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def write_gallery(run, records):
    cards = []
    for key in TRACK_CONFIG:
        if key not in records:
            continue
        item = records[key]
        folder = run / key
        poster_dir = folder / "Production_Records" if (folder / "Production_Records").exists() else folder
        poster = sorted(poster_dir.glob("*.png"))[1]
        seconds = round(item["duration_seconds"])
        cards.append(f'<article><h2>{html.escape(item["title"])}</h2>'
                     f'<p>{seconds//60}:{seconds%60:02d} · {item["width"]}×{item["height"]} · {item["fps"]} fps</p>'
                     f'<video controls preload="none" poster="{quote(poster.relative_to(run).as_posix())}" '
                     f'src="{quote(item["video"])}"></video>'
                     f'<p><a href="{quote(item["video"])}" download>Download full video</a></p></article>')
    (run / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Resonant Orbits · Full Videos</title>
<style>body{background:#080d17;color:#d5dde7;font:16px system-ui;margin:32px auto;max-width:1300px;padding:0 24px}video{width:100%;background:black;border-radius:10px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:28px}h2{font-size:22px}p{color:#99a7bb}a{color:#aad9ed}</style>
<h1>Resonant Orbits · Full Videos</h1><p>Exhibition V3 · Dim photographic backgrounds · Music-driven stroke colors · Original audio</p><section>'''
        + "".join(cards) + "</section></html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--width", type=int, default=3840,
                        help="Video width in pixels; 3840 (4K) is the 12.8 x 7.2 inch frame at 300 dpi")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "03_Generation_Tools" / "Generated_Runs" / "Full_Series")
    parser.add_argument("--verify-run", type=Path, help="Verify an existing completed series without rendering again")
    args = parser.parse_args()
    if args.jobs < 1 or args.fps < 1 or args.width < 320 or args.width % 2:
        parser.error("Jobs/FPS must be positive; width must be even and at least 320")
    if args.verify_run:
        run = args.verify_run.resolve()
        records = {key: verify_one(key, run, args.width, args.fps) for key in TRACK_CONFIG}
        (run / "series_manifest.json").write_text(json.dumps(dict(videos=records, errors={}),
            ensure_ascii=False, indent=2), encoding="utf-8")
        write_gallery(run, records)
        print(f"All five full videos verified. Gallery: {run / 'index.html'}", flush=True)
        return
    run = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Output: {run}", flush=True)
    # Longest tracks first so short tracks can use workers as they finish.
    order = ("lunar_drift", "mountain_current", "celestial_waltz", "crystal_bloom", "dawn_of_an_orbit")
    records, errors = {}, {}
    with ProcessPoolExecutor(max_workers=min(args.jobs, 5)) as executor:
        pending = {executor.submit(render_one, key, run, args.width, args.fps): key for key in order}
        for future in as_completed(pending):
            key = pending[future]
            try:
                records[key] = future.result()
                print(f"Verified: {TRACK_CONFIG[key].title} ({records[key]['duration_seconds']:.1f}s)", flush=True)
            except Exception as exc:
                errors[key] = str(exc)
                print(f"Failed: {key}: {exc}", flush=True)
            (run / "series_manifest.json").write_text(json.dumps(dict(videos=records, errors=errors),
                ensure_ascii=False, indent=2), encoding="utf-8")
            write_gallery(run, records)
    if errors:
        raise RuntimeError(f"Incomplete renders: {errors}; completed videos retained in {run}")
    print(f"All five full videos verified. Gallery: {run / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
