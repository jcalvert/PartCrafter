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
                    "--band_mode", "logit", "--band_threshold", "0.25",
                    "--dtype", "float32", "--dit_dtype", "float16"]
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
            self.assertEqual(manifest["dtype"], "torch.float32")
            self.assertEqual(manifest["dit_dtype"], "torch.float16")
            pipe.to.assert_called_once_with("cpu", torch.float32)
            pipe.transformer.to.assert_called_once_with(dtype=torch.float16)
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


class TransformerPrecisionTests(unittest.TestCase):
    def test_cli_dit_dtype_defaults_to_pipeline_dtype(self):
        pipe = Mock(device="cpu")
        pipe.return_value = SimpleNamespace(meshes=[trimesh.creation.box()])
        pipe.to.return_value = pipe
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            image_path = Path(directory) / "input.png"
            Image.new("RGB", (2, 2)).save(image_path)
            argv = ["inference_partcrafter", "--image_path", str(image_path), "--num_parts", "1",
                    "--output_dir", directory, "--tag", "check", "--device", "cpu",
                    "--dtype", "bfloat16"]
            with patch("sys.argv", argv), patch("huggingface_hub.snapshot_download"), \
                    patch.object(PartCrafterPipeline, "from_pretrained", return_value=pipe), \
                    contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / "scripts/inference_partcrafter.py"), run_name="__main__")
            pipe.to.assert_called_once_with("cpu", torch.bfloat16)
            pipe.transformer.to.assert_called_once_with(dtype=torch.bfloat16)
            manifest = json.loads((Path(directory) / "check/manifest.json").read_text())
            self.assertEqual(manifest["dit_dtype"], manifest["dtype"])
            self.assertNotIn("failed_parts", manifest)

    def test_pipeline_precision_boundaries_with_mocked_components(self):
        # Exercise pipeline orchestration only; all neural components are mocks.
        for pipeline_dtype in (torch.float32, torch.float16):
            for dit_dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(pipeline_dtype=pipeline_dtype, dit_dtype=dit_dtype):
                    self.check_precision_boundaries(pipeline_dtype, dit_dtype)

    def check_precision_boundaries(self, pipeline_dtype, dit_dtype):
        latent_history, query_history = [], []

        def transformer_call(latents, timestep, encoder_hidden_states, **kwargs):
            self.assertEqual(latents.dtype, dit_dtype)
            self.assertEqual(encoder_hidden_states.dtype, dit_dtype)
            # Distinct branches prove guidance is performed after promoting to fp32.
            noise = torch.empty_like(latents)
            noise[:2] = 0.1
            noise[2:] = 0.3
            return (noise,)

        transformer = Mock(dtype=dit_dtype, config=SimpleNamespace(in_channels=2),
                           side_effect=transformer_call)

        def scheduler_step(noise, timestep, latents, **kwargs):
            self.assertEqual(noise.dtype, torch.float32)
            self.assertEqual(latents.dtype, pipeline_dtype)
            expected = torch.tensor(0.1, dtype=dit_dtype).float() + 7 * (
                torch.tensor(0.3, dtype=dit_dtype).float() - torch.tensor(0.1, dtype=dit_dtype).float()
            )
            torch.testing.assert_close(noise, torch.full_like(noise, expected))
            latent_history.append(latents.clone())
            # Deliberately promote output to check that the pipeline restores latent dtype.
            return ((latents.float() - 0.01 * noise).double(),)

        scheduler = SimpleNamespace(order=1, timesteps=torch.tensor([1.0, 0.5]),
                                    set_timesteps=Mock(), step=Mock(side_effect=scheduler_step))

        def decode(latents, sampled_points, num_chunks):
            self.assertEqual(latents.dtype, pipeline_dtype)
            self.assertEqual(sampled_points.dtype, pipeline_dtype)
            self.assertEqual(num_chunks, 7)
            query_history.append(sampled_points.clone())
            return SimpleNamespace(sample=torch.zeros(1, len(sampled_points[0]), 1, dtype=pipeline_dtype))

        vae = Mock(decode=Mock(side_effect=decode))
        embeds = torch.ones(2, 3, 4, dtype=pipeline_dtype)
        supplied = torch.arange(12).reshape(2, 3, 2).to(pipeline_dtype)
        pipe = SimpleNamespace(
            image_encoder_dinov2=SimpleNamespace(dtype=pipeline_dtype),
            transformer=transformer, vae=vae, scheduler=scheduler,
            _execution_device=torch.device("cpu"), dtype=pipeline_dtype,
            encode_image=Mock(return_value=(embeds, torch.zeros_like(embeds))),
            prepare_latents=lambda *args: PartCrafterPipeline.prepare_latents(None, *args),
            do_classifier_free_guidance=True, guidance_scale=7, interrupt=False,
            set_progress_bar_config=Mock(),
            progress_bar=lambda **kwargs: contextlib.nullcontext(Mock()),
            maybe_free_model_hooks=Mock(),
        )
        self.assertEqual(PartCrafterPipeline.dtype.fget(pipe), pipeline_dtype)

        def extract(geometric_func, device, dtype, **kwargs):
            self.assertEqual(dtype, pipeline_dtype)
            self.assertEqual(kwargs["band_mode"], "logit")
            self.assertEqual(kwargs["band_threshold"], 0.25)
            # Multiple chunks per part, all through the exact decoder path.
            for _ in range(2):
                geometric_func(torch.zeros(1, 2, 3, dtype=dtype))
            box = trimesh.creation.box()
            return box.vertices, box.faces

        with patch("src.pipelines.pipeline_partcrafter.hierarchical_extract_geometry", side_effect=extract), \
                contextlib.redirect_stdout(io.StringIO()):
            result = PartCrafterPipeline.__call__(
                pipe, torch.zeros(2, 3, 2, 2), num_inference_steps=2, num_tokens=3,
                latents=supplied, decode_chunk_size=7, band_mode="logit", band_threshold=0.25,
            )
        torch.testing.assert_close(latent_history[0], supplied)
        self.assertEqual(transformer.call_count, 2)
        self.assertEqual(len(query_history), 4)
        self.assertEqual(len(result.meshes), 2)
        vae.set_exact_decoder.assert_called_once()
        vae.set_flash_decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
