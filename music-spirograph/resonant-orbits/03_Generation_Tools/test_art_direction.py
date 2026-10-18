"""Color constraints, geometry independence, and print/provenance regressions."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from music_spirograph import art_direction as art
from music_spirograph.exhibition_v3 import Renderer, make_bodies
from test_exhibition_v3 import features


class ArtDirectionTests(unittest.TestCase):
    def test_clean_background_export_preserves_pixels_without_constructing_renderer(self):
        from music_spirograph.exhibition_export import export_backgrounds
        background = Image.new("RGB", (320, 180), (21, 34, 55))
        with tempfile.TemporaryDirectory() as directory:
            with patch("music_spirograph.exhibition_export.prepare_background",
                       return_value=(background, {"filename": "original.jpg"})), \
                 patch("music_spirograph.exhibition_export.Renderer") as renderer:
                export_backgrounds(Path(directory), 320, 180, 300)
                renderer.assert_not_called()
            files = list(Path(directory).glob("*_background.png"))
            self.assertEqual(len(files), 5)
            for path in files:
                with Image.open(path) as saved:
                    np.testing.assert_array_equal(np.asarray(saved), np.asarray(background))
            manifest = json.loads((Path(directory) / "render_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(record["spirographs"] is False for record in manifest))

    def test_all_five_local_recordings_resolve_without_ambiguity(self):
        audio = art.ROOT / "02_Source_Materials" / "Audio"
        if not any(p.suffix.lower() in (".ogg", ".mp3", ".wav", ".flac") for p in audio.glob("*")):
            self.skipTest("Recordings are not distributed with the code; see 02_Source_Materials/Audio/README.md")
        paths =[art.resolve_audio(c) for c in art.TRACK_CONFIG.values()]
        self.assertEqual(len(set(paths)), 5)

    def test_musical_style_stays_in_display_bounds_and_responds_to_audio(self):
        import colorsys
        for config in art.TRACK_CONFIG.values():
            palette = np.array([art.rgb(c) for c in config.palette])
            for level in np.linspace(0, 1, 11):
                for role in range(4):
                    color, glow, width, opacity = art.musical_style(palette, features(level), role)
                    self.assertTrue(np.all((color >= 0) & (color <= 1)))
                    self.assertGreaterEqual(colorsys.rgb_to_hsv(*color)[1], 0.43-1e-9)
                    self.assertGreaterEqual(width, 0.8 if role < 2 else 1.4)
                    self.assertLessEqual(width, 1.3 if role < 2 else 2.2)
                    self.assertGreaterEqual(opacity, 0.85)
                    self.assertLessEqual(opacity, 0.97)
                    self.assertGreaterEqual(glow, 0)
                    self.assertLessEqual(glow, 1)
            quiet = dict(energy=0, onset=0, timbre=0, flux=0)
            loud = dict(energy=1, onset=1, timbre=0, flux=1)
            a, _, wa, aa = art.musical_style(palette, quiet, 0)
            b, _, wb, ab = art.musical_style(palette, loud, 0)
            self.assertGreater(wb, wa)
            self.assertGreater(ab, aa)
            self.assertGreater(colorsys.rgb_to_hsv(*b)[1], colorsys.rgb_to_hsv(*a)[1])
            self.assertGreater(max(b), max(a))
            c, _, _, _ = art.musical_style(palette, dict(quiet, timbre=1), 0)
            self.assertFalse(np.allclose(a, c))

    def test_palette_changes_color_but_preserves_all_geometry(self):
        configs = list(art.TRACK_CONFIG.values())
        a = make_bodies(palette=[art.rgb(c) for c in configs[0].palette])
        b = make_bodies(palette=[art.rgb(c) for c in configs[3].palette])
        for frame in range(90):
            for body in a+b:
                body.update_visual(features(frame/90), 1/30)
        for left, right in zip(a, b):
            np.testing.assert_equal(left.geometry(), right.geometry())
            np.testing.assert_equal(left.strokes(90)[0], right.strokes(90)[0])
            self.assertFalse(np.allclose(left.strokes(90)[1][:, :3], right.strokes(90)[1][:, :3]))

    def test_colors_are_continuous_and_old_ink_keeps_its_paint(self):
        body = make_bodies()[2]
        body.update_visual(features(0), 1/30)
        old = list(body.ink)
        color = body.paint_rgb.copy()
        body.update_visual(features(1), 1/30)
        self.assertLess(float(np.max(abs(body.paint_rgb-color))), 1-np.exp(-(1/30)/0.18)+1e-9)
        self.assertEqual(list(body.ink)[:len(old)], old)
        self.assertGreater(body.glow, 0)

    def test_print_tessellation_preserves_original_curve_sample_positions(self):
        body = make_bodies()[0]
        body.trail_offset = 0
        body.update_visual(features(), 1/30)
        original = body.strokes(0)[0]
        detailed = body.strokes(0, detail=6)[0].reshape(-1, 6, 2, 2)
        np.testing.assert_allclose(original[:, 0], detailed[:, 0, 0], atol=1e-12)
        np.testing.assert_allclose(original[:, 1], detailed[:, -1, 1], atol=1e-12)

    def test_recomputed_segments_use_bounded_music_color_history_at_all_detail_levels(self):
        body = make_bodies()[0]
        for level in np.r_[np.zeros(30), np.ones(30)]:
            body.update_visual(features(level), 1/30)
        for detail in (1, 4, 6):
            segments, colors, widths = body.strokes(60, detail=detail)
            self.assertEqual(len(segments), len(colors))
            self.assertEqual(len(segments), len(widths))
            self.assertGreater(np.linalg.norm(colors[0, :3]-colors[-1, :3]), 0.1)
            self.assertLessEqual(widths.max(), 1.3+1e-9)
            np.testing.assert_allclose(colors[-1, :3], body.paint_rgb, atol=0.01)
        self.assertLessEqual(body.elapsed-body.style_history[1][0], 1.5+1e-9)

    def test_phrase_layer_changes_color_without_changing_instantaneous_audio(self):
        palette = [art.rgb(c) for c in art.TRACK_CONFIG["crystal_bloom"].palette]
        a = dict(features(), phrase_anchor=0, phrase_energy=0)
        b = dict(features(), phrase_anchor=1, phrase_energy=1)
        ca = art.musical_style(palette, a, 0)[0]
        cb = art.musical_style(palette, b, 0)[0]
        self.assertGreater(np.linalg.norm(ca-cb), 0.1)

    def test_phrase_proxies_are_constant_for_unchanging_music(self):
        from music_spirograph.exhibition_v3 import phrase_features
        values = dict(energy=np.full(100, 0.3), timbre=np.full(100, 0.4), activity=np.ones(100))
        chroma = np.full((12, 100), 1/12)
        result = phrase_features(values, chroma, 0.01, [])
        np.testing.assert_allclose(result["phrase_energy"], 0.3)
        np.testing.assert_allclose(result["crescendo"], 0)
        np.testing.assert_allclose(result["phrase_change"], 0)
        self.assertAlmostEqual(float(np.ptp(result["phrase_anchor"])), 0)

    def test_print_metadata_does_not_rescale_canvas(self):
        bodies = make_bodies()
        for body in bodies:
            body.update_visual(features(), 1/30)
        with tempfile.TemporaryDirectory() as directory:
            renderer = Renderer(bodies, 640, 360)
            try:
                renderer.draw(bodies)
                path = Path(directory) / "print.png"
                renderer.save(path, dpi=300)
                with Image.open(path) as image:
                    self.assertEqual(image.size, (640, 360))
                    self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.02)
            finally:
                renderer.close()

    def test_background_provenance_and_repeatable_fallback(self):
        config = art.TRACK_CONFIG["lunar_drift"]
        with tempfile.TemporaryDirectory() as directory, patch.object(art, "BACKGROUND_DIR", Path(directory)):
            image, meta = art.prepare_background(config, 320, 180)
            again, other = art.prepare_background(config, 320, 180)
            np.testing.assert_equal(np.asarray(image), np.asarray(again))
            self.assertEqual(meta["kind"], "procedural")
            self.assertIn("no third-party", other["license"])
            self.assertTrue(config.background_path.with_suffix(".json").is_file())
        with tempfile.TemporaryDirectory() as directory, patch.object(art, "BACKGROUND_DIR", Path(directory)):
            Image.new("RGB", (20, 20)).save(config.background_path)
            with self.assertRaisesRegex(ValueError, "provenance required"):
                art.prepare_background(config, 320, 180)

    def test_still_layouts_fit_within_canvas_and_keep_hero_dominant(self):
        for config in art.TRACK_CONFIG.values():
            radii = []
            for x, y, scale in config.layout:
                radius = (1.38 - 0.18) * 0.98 * scale
                radii.append(radius)
                self.assertTrue(radius < x < 16-radius)
                self.assertTrue(radius < y < 9-radius)
            self.assertEqual(int(np.argmax(radii)), config.hero)

    def test_photo_sidecar_does_not_inherit_same_named_procedural_credits(self):
        config = art.TRACK_CONFIG["crystal_bloom"]
        with tempfile.TemporaryDirectory() as directory, patch.object(art, "BACKGROUND_DIR", Path(directory)):
            art.prepare_background(config, 320, 180)
            photo = Path(directory) / (config.key + ".jpg")
            Image.new("RGB", (640, 480), (150, 90, 50)).save(photo)
            meta = dict(kind="photograph", creator="Test fixture", source_url="https://example.org/fixture",
                        license="Test fixture only", commercial_use=True,
                        render=dict(brightness=0.8, saturation=1, focus_dim=0))
            photo.with_name(photo.name + ".json").write_text(json.dumps(meta), encoding="utf-8")
            image, credit = art.prepare_background(config, 320, 180)
            self.assertEqual(image.size, (320, 180))
            self.assertEqual(credit["kind"], "photograph")
            self.assertEqual(credit["creator"], "Test fixture")
            self.assertEqual(Path(credit["filename"]), photo)
            self.assertGreater(float(np.asarray(image).mean()), 40)
            fallback = json.loads(config.background_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(fallback["kind"], "procedural")


if __name__ == "__main__":
    unittest.main()
