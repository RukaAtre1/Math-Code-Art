"""Print export using the V3 features and each artwork's curated curve family."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .art_direction import (TRACK_CONFIG, SERIES_TITLE, OUTPUT_DPI, OUTPUT_WIDTH,
                            OUTPUT_HEIGHT, derive_palette, prepare_background, resolve_audio)
from .curve_families import LEGACY
from .exhibition_v3 import Features, Renderer, analyze_track_profile, make_bodies


def representative_time(features, config):
    """Pick a sustained musical passage, not an isolated noisy transient."""
    energy = features.values["energy"]
    score = 0.6 * energy + 0.25 * features.values["high"] + 0.15 * features.values["onset"]
    if config.key == "dawn_of_an_orbit":
        score = energy
    window = min(len(score), 151)
    score = np.convolve(score, np.ones(window) / window, mode="same")
    valid = (features.times >= 8) & (features.times <= features.duration - 3)
    if not valid.any():
        return features.duration * 0.7
    score[~valid] = -1
    return float(features.times[int(np.argmax(score))])


def render_artwork(config, output, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, dpi=OUTPUT_DPI, seed=17):
    audio = resolve_audio(config)
    print(f"Analyzing {config.title}: {audio.name}", flush=True)
    profile = analyze_track_profile(audio)
    features = Features(audio)
    fps = 30
    moment = representative_time(features, config)
    # Warm up the pen/rotor integration with real preceding audio: eight
    # seconds for the double rotor, twelve for families so every voice shows
    # its complete latest figure even at slow pen speeds. Static composition
    # changes screen placement, not geometry.
    begin = max(0, moment - (8 if config.family == LEGACY else 12))
    background, credits = prepare_background(config, width, height)
    palette = derive_palette(background, config.palette)
    bodies = make_bodies(3, seed, profile, palette=palette, color_profile=config.color_profile,
                         family=config.family)
    for t in np.arange(begin, moment, 1/fps):
        f = features.sample(t)
        for body in bodies:
            body.update_visual(f, 1/fps)
    f = features.sample(moment)
    for body in bodies:
        body.update_visual(f, 1/fps)
    renderer = Renderer(bodies, width, height, background=background, composition=config.layout)
    number = list(TRACK_CONFIG).index(config.key) + 1
    filename = f"{number:02d}_{config.title.replace(' ', '_')}.png"
    path = output / filename
    try:
        renderer.draw(bodies, round(moment * fps))
        renderer.save(path, dpi=dpi)
    finally:
        renderer.close()
    row = dict(artwork_title=config.title, series_title=SERIES_TITLE,
               music_title=config.music, composer=config.composer,
               performer_or_recording_artist=config.performer, audio_filename=audio.name,
               # Audio and Backgrounds are sibling folders of 02_Source_Materials.
               background_filename=str(Path(credits["filename"]).relative_to(audio.parent.parent)),
               background_creator=credits["creator"], background_source_url=credits["source_url"],
               background_license=credits["license"], music_source_url=config.source_url,
               music_license=config.license, artwork_width_px=width, artwork_height_px=height, dpi=dpi)
    evidence = dict(artwork=filename, audio=str(audio),
                    audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                    sample_time_seconds=moment, warmup_begin_seconds=begin,
                    geometry=(f"Curve family {config.family}; three pen-drawn voices, each showing "
                              "its latest complete figure at an active moment" if config.family != LEGACY else
                              "Double rotor; 2 recomputed curves and 1 continuous-ink voice"),
                    curve_family=config.family,
                    static_composition=config.layout, track_profile=asdict(profile),
                    palette=config.palette, effective_palette=palette.tolist(),
                    sampled_features={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in f.items()},
                    background=credits, music_provenance_note=config.note,
                    normalization="5th–95th percentiles over complete recording; attack/release smoothing plus color EMA")
    print(f"Saved {config.title} | {audio.name} | {Path(credits['filename']).name} | "
          f"{width} x {height} | {dpi} DPI | {config.license} | {credits['license']} | {path}", flush=True)
    return row, evidence


def write_documentation(output, rows, evidence):
    with (output / "artwork_metadata.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [f"# {SERIES_TITLE}\n"]
    credits = [f"# {SERIES_TITLE} — Music and image credits\n",
               "Music identities and license names below are supplied by the project owner, supplemented by local OGG tags where available. "
               "Some exact recording source URLs are absent from the local project; missing fields are intentionally empty. "
               "This file is an attribution draft, not independent verification of recording rights. "
               "Before public display/distribution, attach the recording source pages and any required license evidence.\n",
               "No replacement audio was downloaded. No third-party background is used by the procedural fallback. "
               "The artworks contain no title or credit typography.\n"]
    for row, record in zip(rows, evidence):
        c = next(c for c in TRACK_CONFIG.values() if c.title == row["artwork_title"])
        labels.append(f"## {c.title}\nSeries: {SERIES_TITLE}\n\nMusic:\n{c.music}\n\n"
                      f"Composer:\n{c.composer}\n\nPerformance / Recording:\n{c.performer}\n\n"
                      f"Music License:\n{c.license}" + (" (source URL recorded in credits)\n\n" if c.source_url else " (source record pending)\n\n") +
                      f"Visual background:\n{row['background_creator']} / {row['background_license']}\n\n"
                      "Generated through a music-driven mathematical spirograph system.\n")
        credits.append(f"## {c.title}\n- Music: **{c.music}**, composed by {c.composer}.\n"
                       f"- Performance / recording: {c.performer}.\n- Local audio: `{row['audio_filename']}`.\n"
                       f"- License: {c.license}.\n- Recording source URL: {c.source_url or 'Not documented; needs source record'}.\n"
                       f"- Provenance note: {c.note}\n- Background: `{row['background_filename']}`.\n"
                       f"- Background creator: {row['background_creator']}.\n"
                       f"- Background license: {row['background_license']}.\n"
                       f"- Background source URL: {row['background_source_url'] or 'Not applicable — procedural original'}.\n"
                       f"- Selected musical moment: {record['sample_time_seconds']:.2f} s.\n")
        if c.license == "CC BY 3.0":
            credits.append("- License text: [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/). "
                           "The local audio file is unchanged; the visual is generated from analyzed features. "
                           "The draft credit must accompany publication and be completed with the recording source.\n")
        elif c.key == "mountain_current":
            credits.append("- License reference: [Pixabay Content License](https://pixabay.com/service/license-summary/).\n")
        else:
            credits.append("- License reference: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).\n")
    (output / "labels.md").write_text("\n".join(labels), encoding="utf-8")
    (output / "MUSIC_AND_IMAGE_CREDITS.md").write_text("\n".join(credits), encoding="utf-8")
    (output / "render_manifest.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_export(output, rows, evidence, width, height, dpi):
    checks = []
    for row, record in zip(rows, evidence):
        with Image.open(output / record["artwork"]) as image:
            assert image.size == (width, height), image.size
            assert all(abs(value - dpi) < 0.05 for value in image.info.get("dpi", (0, 0)))
            image.verify()
        bg = record["background"]
        assert bg["creator"] and bg["license"] and bg["commercial_use"]
        if bg.get("kind") != "procedural":
            assert bg["source_url"]
        assert row["artwork_title"] and row["composer"] and row["performer_or_recording_artist"]
        assert Path(record["audio"]).is_file()
        assert hashlib.sha256(Path(record["audio"]).read_bytes()).hexdigest() == record["audio_sha256"]
        checks.append(dict(artwork=record["artwork"], pixels=[width, height], dpi=dpi,
                           audio_present=True, audio_unchanged=True, png_valid=True,
                           background_provenance_present=True))
    report = dict(technical_checks="passed", artwork_count=len(checks), checks=checks,
                  music_rights="Owner-supplied license metadata; exact source URLs still need documentation",
                  visual_review="Composition and visual hierarchy require review of the contact sheet",
                  external_backgrounds=[r["background"]["source_url"] for r in evidence
                                        if r["background"].get("kind") != "procedural"],
                  no_external_image_downloads=all(r["background"].get("kind") == "procedural" for r in evidence))
    (output / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def contact_sheet(output, rows, evidence):
    tile_width, tile_height = 800, 450
    sheet = Image.new("RGB", (1648, 1664), "#080D17")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 23)
        small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((24, 15), "RESONANT ORBITS / FIVE MUSICAL WORLDS", font=font, fill="#D5DDE7")
    for i, (row, record) in enumerate(zip(rows, evidence)):
        x, y = 16 + (i % 2) * 816, 65 + (i // 2) * 530
        with Image.open(output / record["artwork"]) as image:
            thumb = image.convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.text((x + 8, y + 460), row["artwork_title"], font=font, fill="#D5DDE7")
        draw.text((x + 8, y + 492), row["composer"], font=small, fill="#8896A8")
    sheet.save(output / "contact_sheet.jpg", quality=94)


def export_backgrounds(output, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, dpi=OUTPUT_DPI):
    """Export the same graded scenery used by video, without drawing any bodies."""
    if min(width, height, dpi) <= 0 or width * 9 != height * 16:
        raise ValueError("Background dimensions must be positive and 16:9; DPI must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rows, evidence = [], []
    for number, config in enumerate(TRACK_CONFIG.values(), 1):
        background, credits = prepare_background(config, width, height)
        filename = f"{number:02d}_{config.title.replace(' ', '_')}_background.png"
        background.save(output / filename, dpi=(dpi, dpi))
        with Image.open(output / filename) as saved:
            assert saved.size == (width, height)
            saved.verify()
        rows.append(dict(artwork_title=config.title, composer=config.composer))
        evidence.append(dict(artwork=filename, background=credits,
                             pixels=[width, height], dpi=dpi, spirographs=False))
        print(f"Saved clean background: {output / filename}", flush=True)
    (output / "render_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(output, rows, evidence)


def export_series(output, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, dpi=OUTPUT_DPI, seed=17):
    if min(width, height, dpi) <= 0 or width * 9 != height * 16:
        raise ValueError("Print dimensions must be positive and have a 16:9 aspect ratio; DPI must be positive")
    # Validate all five before any expensive rendering.
    for config in TRACK_CONFIG.values():
        resolve_audio(config)
    output.mkdir(parents=True, exist_ok=True)
    rows, evidence = [], []
    for config in TRACK_CONFIG.values():
        row, record = render_artwork(config, output, width, height, dpi, seed)
        rows.append(row)
        evidence.append(record)
    write_documentation(output, rows, evidence)
    verify_export(output, rows, evidence, width, height, dpi)
    contact_sheet(output, rows, evidence)
    print(f"Verified all five artworks: {width} x {height}, {dpi} DPI. Contact sheet: {output / 'contact_sheet.jpg'}", flush=True)
