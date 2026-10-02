"""CPU-only regression checks; no weights or model inference."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
import trimesh

from scripts.inference_partcrafter import failed_part_indices, meshes_for_export
from src.models.attention_processor import FlashTripo2AttnProcessor2_0, TripoSGAttnProcessor2_0
from src.models.autoencoders.autoencoder_kl_triposg import TripoSGVAEModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils import inference_utils as geometry


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkValidityTests(unittest.TestCase):
    def test_supplied_latents_are_used_without_drawing_noise(self):
        supplied = torch.arange(24).reshape(2, 3, 4).double()
        with patch("src.pipelines.pipeline_partcrafter.randn_tensor") as draw:
            actual = PartCrafterPipeline.prepare_latents(
                None, 2, 3, 4, torch.float32, "cpu", None, supplied
            )
        draw.assert_not_called()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, supplied.float())
        with self.assertRaisesRegex(ValueError, "Expected latents shape"):
            PartCrafterPipeline.prepare_latents(None, 1, 3, 4, torch.float32, "cpu", None, supplied)

    def test_exact_processor_restored_after_flash(self):
        # Construct only tiny CPU modules; never execute a model forward pass.
        vae = TripoSGVAEModel(width_encoder=16, width_decoder=16, latent_channels=4,
                             num_attention_heads=2, num_layers_encoder=1, num_layers_decoder=1)
        vae.set_flash_decoder()
        self.assertIsInstance(vae.decoder.blocks[-1].attn2.processor, FlashTripo2AttnProcessor2_0)
        vae.set_exact_decoder()
        self.assertTrue(all(isinstance(p, TripoSGAttnProcessor2_0)
                            for name, p in vae.attn_processors.items() if name.startswith("decoder.")))

    def test_expansion_limit_precedes_eightfold_allocation(self):
        stats = {}
        with patch.object(torch, "stack", side_effect=AssertionError("expanded allocation")):
            with self.assertRaisesRegex(ValueError, "512 > 1"):
                geometry.expand_edge_region_fast(torch.tensor([[1, 1, 1]]), 4, torch.float32, 1, stats)
        self.assertEqual(stats["expanded_coords"], 512)

    def test_extraction_stats_and_legacy_default(self):
        def sphere(points):
            return (points.square().sum(dim=-1, keepdim=True) - 0.5)

        kwargs = dict(device="cpu", dtype=torch.float32, bounds=1.0,
                      dense_octree_depth=2, hierarchical_octree_depth=3)
        with contextlib.redirect_stdout(io.StringIO()):
            vertices, faces, stats = geometry.hierarchical_extract_geometry(sphere, verbose=True, **kwargs)
        self.assertGreater(len(faces), 3)
        self.assertEqual(stats, geometry.LAST_EXTRACTION_STATS)
        self.assertEqual([s["raw_candidates"] for s in stats], [64, 8])
        self.assertEqual([s["queried_coords"] for s in stats], [64, 512])
        default = geometry.hierarchical_extract_geometry(sphere, **kwargs)
        self.assertEqual(len(default), 2)
        self.assertEqual(len(geometry.LAST_EXTRACTION_STATS), 2)
        self.assertEqual(len(vertices), len(default[0]))

    def test_failure_detection_preserves_raw_outputs(self):
        valid = trimesh.creation.box()
        degenerate = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        outputs = [None, degenerate, valid]
        self.assertEqual(failed_part_indices(outputs), [0, 1])
        exports = meshes_for_export(outputs)
        self.assertIsNone(outputs[0])
        self.assertIsNotNone(exports[0])
        self.assertIs(exports[2], valid)

    def test_cli_failure_exports_and_exits_two(self):
        outputs = [None, trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]]), trimesh.creation.box()]
        pipe = Mock(device="cpu")
        pipe.return_value = SimpleNamespace(meshes=outputs)
        pipe.to.return_value = pipe
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            image_path = Path(directory) / "input.png"
            Image.new("RGB", (2, 2)).save(image_path)
            argv = ["inference_partcrafter", "--image_path", str(image_path), "--num_parts", "3",
                    "--output_dir", directory, "--tag", "check", "--device", "cpu",
                    "--dense_octree_depth", "2", "--hierarchical_octree_depth", "3",
                    "--decode_chunk_size", "7", "--num_inference_steps", "1",
                    "--band_mode", "logit", "--band_threshold", "0.25"]
            with patch("sys.argv", argv), patch("huggingface_hub.snapshot_download") as download, \
                    patch.object(PartCrafterPipeline, "from_pretrained", return_value=pipe), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runpy.run_path(str(ROOT / "scripts/inference_partcrafter.py"), run_name="__main__")
            self.assertEqual(raised.exception.code, 2)
            download.assert_called_once()
            export_dir = Path(directory) / "check"
            manifest = json.loads((export_dir / "manifest.json").read_text())
            self.assertEqual(manifest["failed_parts"], [0, 1])
            self.assertIsNone(manifest["part_stats"][0])
            self.assertTrue((export_dir / "part_02.glb").exists())
            self.assertTrue((export_dir / "object.glb").exists())
            self.assertEqual(pipe.call_args.kwargs["decode_chunk_size"], 7)
            self.assertEqual(pipe.call_args.kwargs["band_mode"], "logit")
            self.assertEqual(pipe.call_args.kwargs["band_threshold"], 0.25)
            self.assertEqual(manifest["band_mode"], "logit")
            self.assertIsNone(outputs[0])


class SelectiveRefinementTests(unittest.TestCase):
    def test_legacy_keeps_dtype_dependent_saturation(self):
        logits = torch.full((5, 5, 5), 9.0)
        self.assertEqual(len(geometry.find_candidates_band(logits, 1.0)), 27)
        self.assertEqual(len(geometry.find_candidates_band(logits.half(), 1.0)), 0)

    def test_logit_band_is_compared_in_float32(self):
        logits = torch.full((5, 5, 5), 3.0, dtype=torch.float16)
        logits[2, 2, 2] = 1.0
        candidates = geometry.find_candidates_band(logits, 1.0001, band_mode="logit")
        self.assertEqual(candidates.tolist(), [[2, 2, 2]])
        self.assertEqual(len(geometry.find_candidates_band(logits, 1.0, band_mode="logit")), 0)

    def test_sign_changes_cover_both_endpoints_on_all_axes_and_boundaries(self):
        for axis in range(3):
            logits = torch.full((5, 5, 5), 10.0)
            lower_half = [slice(None)] * 3
            lower_half[axis] = slice(None, 2)
            logits[tuple(lower_half)] = -10.0
            candidates = geometry.find_candidates_band(logits, 0.95, band_mode="logit")
            self.assertEqual(len(candidates), 50)
            self.assertEqual(set(candidates[:, axis].tolist()), {1, 2})

    def test_six_neighbours_exclude_diagonals(self):
        logits = torch.full((5, 5, 5), 10.0)
        logits[2, 2, 2] = -10.0
        candidates = geometry.find_candidates_band(logits, 0.95, band_mode="logit")
        self.assertEqual(len(candidates), 7)
        self.assertNotIn([1, 1, 1], candidates.tolist())

    def test_logit_refinement_reduces_queries_and_marches_raw_logits(self):
        def sphere(points):
            return points.square().sum(dim=-1, keepdim=True) - 0.1

        kwargs = dict(device="cpu", dtype=torch.float32, bounds=1.0,
                      dense_octree_depth=4, hierarchical_octree_depth=5)
        geometry.hierarchical_extract_geometry(sphere, **kwargs)
        legacy = [dict(s) for s in geometry.LAST_EXTRACTION_STATS]
        with patch.object(geometry.measure, "marching_cubes", wraps=geometry.measure.marching_cubes) as march:
            vertices, faces = geometry.hierarchical_extract_geometry(
                sphere, band_mode="logit", band_threshold=0.01, **kwargs
            )
        selective = geometry.LAST_EXTRACTION_STATS
        self.assertLess(selective[1]["raw_candidates"], legacy[1]["raw_candidates"])
        self.assertLess(selective[1]["queried_coords"], legacy[1]["queried_coords"])
        self.assertEqual(march.call_args.args[1], 0)
        self.assertGreater(march.call_args.args[0].max(), 1)
        self.assertGreater(len(faces), 3)
        self.assertGreater(len(vertices), 3)


if __name__ == "__main__":
    unittest.main()
