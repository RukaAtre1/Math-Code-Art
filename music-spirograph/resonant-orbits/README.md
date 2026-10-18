# Resonant Orbits — music-driven double-rotor spirographs

Resonant Orbits is the exhibition version of Music Spirograph. Five recordings are each drawn as an animated scene: three spirographs that respond to the music, plus a fourth that appears during passages with rich pitch-class activity. Videos render natively at 4K (3840 × 2160, a 12.8 × 7.2 inch frame at 300 dpi) and prints at 6000 × 3375, 300 DPI.

Rendered videos: Google Drive (link to be added).

## How the music drives the drawing

All roles read the same mixed recording; no instruments or voices are separated.

| Spirograph | Main musical input | What it changes |
| --- | --- | --- |
| Rhythm | onset strength and estimated beats | rotor proportions, pen arm, drawing, rotation and travel speed |
| Dynamics | RMS energy and bass | how far the figure opens, its size, drawing and rotation speed |
| Timbre | spectral centroid, bandwidth and flux | radii, rotor proportions, pen arm and sway |
| Pitch-class activity | the twelve-pitch-class (chroma) distribution | appears and fades with sustained multi-pitch activity; this approximates harmonic activity and does not identify chords |

Features are normalized per recording (5th–95th percentiles) and smoothed. A fixed seed and whole-recording statistics set each piece's starting shapes, and small periodic sways are added for artistic reasons, so not every visual change comes from the music alone.

Each piece has a curated palette; the music moves the color within it. Stronger sounds give brighter, more saturated lines. Palettes and mood words are artistic choices, not detected emotions.

## One double-rotor mechanism per piece

Every figure is a double rotor: a ring, a rotor rolling in it, a second rotor rolling in that, and a pen. The pieces differ in gear ratios and arm lengths, chosen by hand like the palettes (not by genre detection):

| Artwork | Music | Family | Look |
| --- | --- | --- | --- |
| Lunar Drift | Clair de Lune | `rotor_hub` | rotor 2 divides the ring a whole number of times and the small arms match, so every loop passes through a few bright knots |
| Crystal Bloom | Dance of the Sugar Plum Fairy | `rotor_crystal` | a small first rotor gives sharp six- and eight-point stars |
| Mountain Current | Mountain Spring | `rotor_eddy` | both rotors roll outside: a round rim around a ring of eddies |
| Dawn of an Orbit | Also sprach Zarathustra: Sunrise | `rotor_corona` | a slow second rotor leaves a ringed sun disc inside radiating points |
| Celestial Waltz | The Blue Danube | `rotor_waltz` | slow wide gear ratios with three-fold symmetry (an artistic nod to the waltz; meter is not detected) |

Gear ratios are whole numbers, as on a toy spirograph, so each figure is one closed curve. The music changes arm lengths and phases, never the gear ratios. When the slow chroma-change cue is strong (at most every 15 seconds), a figure cross-fades to a second gear set.

Each figure shows the complete figure its pen has just drawn, in thin lines; older strokes fade out. Keeping every stroke was tried, and within two minutes the figures filled into solid shapes. `--family double_rotor` reproduces the original V3 drawing.

Design notes: [`03_Generation_Tools/EXHIBITION_V3.md`](03_Generation_Tools/EXHIBITION_V3.md).

## Running it

Requirements: Python (developed and tested on 3.14), FFmpeg on `PATH`, and the packages in `03_Generation_Tools/requirements.txt`.

```bash
pip install -r 03_Generation_Tools/requirements.txt
```

Place the five recordings in `02_Source_Materials/Audio/` (see its README). Background photographs are optional; without them, the renderer generates procedural backgrounds on first run. Run from this folder:

```bash
python 03_Generation_Tools/render_exhibition_v3.py --audio 02_Source_Materials/Audio/Dance_of_the_Sugar_Plum_Fairy.ogg --duration 20 --output 03_Generation_Tools/Generated_Runs/Preview
```

```bash
python 03_Generation_Tools/render_full_series.py --jobs 5
```

```bash
python 03_Generation_Tools/render_exhibition_v3.py --exhibition --print-output 03_Generation_Tools/Generated_Runs/Print_Artworks
```

```bash
python 03_Generation_Tools/select_exhibition_clips.py --export --source 03_Generation_Tools/Generated_Runs/Full_Series/<run folder>
```

```bash
python -m unittest discover -s 03_Generation_Tools -p "test_*.py"
```

`--width 1280` renders quickly at 720p for drafts. Rendering at 4K takes about 0.5 s per frame per piece on a modern CPU; the renderer uses Matplotlib on the CPU, not the GPU.

## Credits

Music and background credits are listed in `02_Source_Materials/Audio/README.md` and `02_Source_Materials/Backgrounds/SOURCES.md`. Background photographs are not redistributed here; each `.jpg.json` sidecar records its source and download URL.

Tutorial lead: Junhao Jia. Faculty advisor: Dr. Jiayin Lu (UCLA Mathematics).
