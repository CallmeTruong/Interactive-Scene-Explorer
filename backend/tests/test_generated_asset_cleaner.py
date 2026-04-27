import tempfile
import unittest
from pathlib import Path

from backend.app.services.generated_asset_cleaner import GeneratedAssetCleaner


class GeneratedAssetCleanerTest(unittest.TestCase):
    def test_cleans_only_generated_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated_dir = Path(directory)
            png_path = generated_dir / "scene_old.png"
            webp_path = generated_dir / "scene_old.webp"
            svg_path = generated_dir / "mock_asset.svg"

            png_path.write_bytes(b"png")
            webp_path.write_bytes(b"webp")
            svg_path.write_text("<svg />")

            removed = GeneratedAssetCleaner(generated_dir).clean()

            self.assertEqual(removed, 2)
            self.assertFalse(png_path.exists())
            self.assertFalse(webp_path.exists())
            self.assertTrue(svg_path.exists())


if __name__ == "__main__":
    unittest.main()
