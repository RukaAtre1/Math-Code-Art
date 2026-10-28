"""Review and export curated excerpts from the completed five-video series."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '01_Deliverables/Full_Videos'
OUT = ROOT / '03_Generation_Tools/Generated_Runs/Selected_Clips'
CLIPS = [
    ('dawn_of_an_orbit', 52, '01_orbital_dawn', 'Orbital Dawn', 'Golden space · Majestic', 'Fuller golden lines echo the orbital sunrise.'),
    ('crystal_bloom', 16, '02_crystal_rose', 'Crystal Rose', 'Rose and violet · Playful', 'Rose and violet petals preserve a playful fairy-tale atmosphere.'),
    ('mountain_current', 176, '03_bamboo_current', 'Bamboo Current', 'Jade landscape · Flowing', 'Cyan strokes contrast with yellow-green curves against the bamboo forest.'),
    ('celestial_waltz', 134, '04_river_waltz', 'River Waltz', 'City at night · Radiant', 'Jade gradually turns to gold over the riverside night scene.'),
    ('lunar_drift', 78, '05_lunar_reverie', 'Lunar Reverie', 'Blue and violet moonlight · Quiet', 'Blue, violet and cyan curves evolve around the moon for a quiet closing excerpt.'),
]
# One excerpt per theme. Prismatic Bloom (a second Crystal Bloom excerpt with the
# same background) was dropped on 2026-10-17 at the professor's request.
CLIP_SECONDS = 28

def run(args):
    return subprocess.run(args, check=True, capture_output=True)

def frame(movie, time, width=480):
    import io
    return Image.open(io.BytesIO(run(['ffmpeg', '-v', 'error', '-ss', str(time), '-i', str(movie), '-frames:v', '1', '-vf', f'scale={width}:-1', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-threads', '1', '-']).stdout)).convert('RGB')

def review():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'series_manifest.json').read_text(encoding='utf-8'))
    for key, item in manifest['videos'].items():
        duration = item['duration_seconds']
        times = [round(10 + (duration - 20) * i / 11, 1) for i in range(12)]
        sheet = Image.new('RGB', (1440, 4 * 304), '#101521')
        draw = ImageDraw.Draw(sheet)
        for i, t in enumerate(times):
            im = frame(SOURCE / item['video'], t)
            x, y = (i % 3) * 480, (i // 3) * 304
            sheet.paste(im, (x, y))
            draw.text((x + 12, y + 276), f'{key} | {t:.1f}s', fill='white')
        sheet.save(OUT / f'review_{key}.jpg', quality=90)
        print(key, times, flush=True)

def source_size(movie):
    probe = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height', '-of', 'json', str(movie)]).stdout)
    return probe['streams'][0]['width'], probe['streams'][0]['height']

def export():
    manifest = json.loads((SOURCE / 'series_manifest.json').read_text(encoding='utf-8'))
    source_w, source_h = source_size(SOURCE / manifest['videos'][CLIPS[0][0]]['video'])
    # Native clips keep the render's own resolution. The 1080p projection copy
    # is a Lanczos downscale of high-resolution renders, an upscale otherwise.
    upscaled = source_h < 1080
    native = OUT / f'native_{source_h}p'
    projection = OUT / ('projection_1080p_upscaled' if upscaled else 'projection_1080p')
    for folder in (native, projection):
        folder.mkdir(parents=True, exist_ok=True)
    frames = CLIP_SECONDS * 30

    def export_one(selection):
        key, start, stem, title, style, reason = selection
        source = SOURCE / manifest['videos'][key]['video']
        assert source_size(source) == (source_w, source_h), source
        result = dict(id=stem, title=title, theme=key, style=style, reason=reason,
                      source=str(source), start_seconds=start, end_seconds=start + CLIP_SECONDS,
                      duration_seconds=CLIP_SECONDS, fps=30, source_resolution=[source_w, source_h],
                      native_file=f'{native.name}/{stem}.mp4',
                      projection_file=f'{projection.name}/{stem}.mp4',
                      projection_is_upscaled=upscaled, poster=f'{stem}.jpg')
        frame(source, start + CLIP_SECONDS // 2, 1920).save(OUT / result['poster'], quality=94)
        end = CLIP_SECONDS - 0.5
        for width, folder in ((source_w, native), (1920, projection)):
            filters = f'scale={width}:{width*9//16}:flags=lanczos,setsar=1,fade=t=in:st=0:d=0.5,fade=t=out:st={end}:d=0.5'
            target = folder / f'{stem}.mp4'
            run(['ffmpeg', '-v', 'error', '-n', '-ss', str(start), '-i', str(source), '-t', str(CLIP_SECONDS),
                 '-map', '0:v:0', '-map', '0:a:0', '-vf', filters,
                 '-af', f'afade=t=in:st=0:d=0.5,afade=t=out:st={end}:d=0.5',
                 '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '17',
                 '-pix_fmt', 'yuv420p', '-r', '30', '-c:a', 'aac', '-b:a', '256k',
                 '-ar', '48000', '-ac', '2', '-movflags', '+faststart', str(target)])
            probe = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(target)]).stdout)
            video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            assert (video['width'],video['height']) == (width,width*9//16)
            assert int(video['nb_frames']) == frames
            assert abs(float(probe['format']['duration'])-CLIP_SECONDS) < 0.1
            assert any(s['codec_type']=='audio' for s in probe['streams'])
            run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(target),'-f','null','-'])
        result['verification'] = f'Dimensions, {frames} frames, duration, audio and full decode passed for both versions'
        print(f'Export verified: {stem}', flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(export_one, CLIPS))
    concat = OUT / 'playlist.ffconcat'
    total = CLIP_SECONDS * len(results)
    # The compilation keeps the renders' native resolution.
    concat.write_text('ffconcat version 1.0\n' + ''.join(f"file '{r['native_file']}'\n" for r in results), encoding='utf-8')
    compilation = OUT / f'Resonant_Orbits_{len(results)}_Clips_{source_h}p.mp4'
    run(['ffmpeg','-v','error','-n','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(compilation)])
    run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(compilation),'-f','null','-'])
    probe = json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(compilation)]).stdout)
    assert abs(float(probe['format']['duration']) - total) < 0.15
    assert int(next(s for s in probe['streams'] if s['codec_type']=='video')['nb_frames']) == frames * len(results)
    themes = len({r['theme'] for r in results})
    assert themes == len(results), 'one excerpt per theme'
    report = dict(clips=results, unique_themes=themes, total_duration_seconds=total,
                  loop_transition='0.5 second video/audio fades to black at each boundary; not a seamless motion loop',
                  resolution_note=(f'Native clips are {source_w}x{source_h} renders; 1080p files are Lanczos '
                                   + ('upscales of them.' if upscaled else 'downscales of them.')),
                  style_note='One excerpt per theme; each has its own background.',
                  compilation=compilation.name, compilation_full_decode='passed')
    (OUT / 'selection_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cards = ''.join(f'''<article><h2>{html.escape(r['title'])}</h2><p>{html.escape(r['style'])} · Source {r['start_seconds']//60:02d}:{r['start_seconds']%60:02d}–{r['end_seconds']//60:02d}:{r['end_seconds']%60:02d} · {CLIP_SECONDS} seconds</p><video controls loop preload="none" poster="{r['poster']}" src="{r['projection_file']}"></video><p>{html.escape(r['reason'])}</p><p><a href="{r['native_file']}" download>Native {source_h}p</a> · <a href="{r['projection_file']}" download>1080p</a></p></article>''' for r in results)
    (OUT / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resonant Orbits · Selected Clips</title><style>body{background:#090e18;color:#e6eaf0;font:16px system-ui;max-width:1240px;margin:36px auto;padding:0 24px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:28px}video{width:100%;background:black;border-radius:10px}p{color:#aab6c7;line-height:1.6}a{color:#9ed8ee}h2{font-size:23px}</style><h1>Resonant Orbits · Selected Clips</h1><p>''' + f'{len(results)} × {CLIP_SECONDS} seconds · 30 fps · {source_w}×{source_h} · Original audio · 0.5-second fades. One excerpt per theme.' + f'''</p><h2>Compilation · {total // 60} minutes {total % 60} seconds</h2><video controls loop preload="metadata" src="{compilation.name}"></video><p><a download href="{compilation.name}">Download compilation</a></p><section>''' + cards + '</section></html>',encoding='utf-8')
    notes = ['# Selected Clips\n',f'Each excerpt is {CLIP_SECONDS} seconds at 30 fps with original audio and 0.5-second fades. Loop boundaries pass through black; motion is not seamless.\n',
             report['resolution_note'] + '\n',
             'One excerpt per theme. Prismatic Bloom, a second Crystal Bloom excerpt with the same background, was dropped.\n',
             '| Clip | Source time | Selection reason |\n| --- | --- | --- |']
    for r in results:
        notes.append(f"| {r['title']} | {r['start_seconds']//60:02d}:{r['start_seconds']%60:02d}–{r['end_seconds']//60:02d}:{r['end_seconds']%60:02d} | {r['reason']} |")
    notes.append(f'\nExport verification: all {2*len(results)} clips and the compilation passed full decoding. Clips have {frames} frames each; the compilation lasts about {total} seconds. See the selection manifest for sources and timings.')
    (OUT / 'SELECTION.md').write_text('\n'.join(notes),encoding='utf-8')
    sheet = Image.new('RGB',(1440,304*((len(results)+2)//3)),'#101521')
    draw = ImageDraw.Draw(sheet)
    for i,r in enumerate(results):
        x,y=(i%3)*480,(i//3)*304
        sheet.paste(Image.open(OUT/r['poster']).resize((480,270)),(x,y))
        draw.text((x+12,y+276),f"{r['id']} | {r['start_seconds']}s - {r['end_seconds']}s",fill='white')
    sheet.save(OUT/'selected_contact_sheet.jpg',quality=94)
    print(f'Complete: {OUT / "index.html"}',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE,
                        help='Full-video folder with series_manifest.json, e.g. a Generated_Runs/Full_Series run')
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    export() if args.export else review()
