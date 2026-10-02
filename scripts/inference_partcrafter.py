import argparse
import json
import os
import sys
from glob import glob
import time
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed

from src.utils.data_utils import get_colored_mesh_composition
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.utils.device_utils import get_device
import threading
import resource


class PeakMemory:
    """Samples accelerator memory in a thread; MPS has no peak counter of its own."""
    def __init__(self, device, interval=0.25):
        self.device, self.interval, self.peak, self._stop = device, interval, 0, False
    def _read(self):
        if self.device == "mps":
            return torch.mps.driver_allocated_memory()
        if self.device == "cuda":
            return torch.cuda.memory_allocated()
        return 0
    def _loop(self):
        while not self._stop:
            self.peak = max(self.peak, self._read())
            time.sleep(self.interval)
    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop = True; self._t.join(); self.peak = max(self.peak, self._read())


def masked_input(image_path, mask_path, export_dir):
    """Combine an RGB image with a supplied mask (or keep its own alpha) and write RGBA,
    so prepare_image takes its alpha path and never needs a background-removal net."""
    img = Image.open(image_path)
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L").resize(img.size)
        img = img.convert("RGB"); img.putalpha(mask)
    out = os.path.join(export_dir, "input_rgba.png")
    img.convert("RGBA").save(out)
    return out

@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = None,
    use_alpha: bool = False,
    dense_octree_depth: int = 8,
    hierarchical_octree_depth: int = 9,
    decode_chunk_size: int = 50000,
    band_mode: str = "legacy",
) -> trimesh.Scene:

    if rmbg or use_alpha:
        img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net, device=device)
    else:
        img_pil = Image.open(image_input)
    start_time = time.time()
    print(f"[partcrafter] start at {start_time:.1f}", flush=True)
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
        dense_octree_depth=dense_octree_depth,
        hierarchical_octree_depth=hierarchical_octree_depth,
        decode_chunk_size=decode_chunk_size,
        band_mode=band_mode,
    ).meshes
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    return outputs, img_pil

def failed_part_indices(meshes):
    return [i for i, mesh in enumerate(meshes) if mesh is None or len(mesh.faces) < 4]


def meshes_for_export(meshes):
    """Substitute missing parts only in the export copy, retaining failure evidence."""
    return [mesh if mesh is not None else trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
            for mesh in meshes]


MAX_NUM_PARTS = 16

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--num_parts", type=int, default=None, help="number of parts to generate (optional if --part_suggest is used)")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9)
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--dense_octree_depth", type=int, default=8)
    parser.add_argument("--hierarchical_octree_depth", type=int, default=9)
    parser.add_argument("--decode_chunk_size", type=int, default=50000)
    parser.add_argument("--band_mode", choices=["legacy"], default="legacy")
    parser.add_argument("--rmbg", action="store_true", help="run RMBG-1.4 (non-commercial licence); prefer --mask or an RGBA input")
    parser.add_argument("--mask", type=str, default=None, help="foreground mask to use instead of background removal")
    parser.add_argument("--use_alpha", action="store_true", help="input is RGBA (or --mask given): crop/pad by its alpha, no RMBG")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps or cpu (default: auto)")
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "float32", "bfloat16"], help="default: float16 on cuda, float32 elsewhere")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--part_suggest", action="store_true", help="use VLM to suggest num_parts automatically")
    parser.add_argument("--style_transfer", action="store_true", help="apply Objaverse-style transfer to input image")
    parser.add_argument("--part_provider", type=str, default="gemini", help="provider for part suggestion (default: gemini)")
    parser.add_argument("--part_model", type=str, default=None, help="model name for part suggestion (default: gemini-3-flash-preview)")
    parser.add_argument("--style_provider", type=str, default="gemini", help="provider for style transfer (default: gemini)")
    parser.add_argument("--style_model", type=str, default=None, help="model name for style transfer (default: gemini-3.1-flash-image-preview)")
    args = parser.parse_args()
    device = get_device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float16 if device == "cuda" else torch.float32)
    print(f"device={device} dtype={dtype}")

    if args.num_parts is not None:
        assert 1 <= args.num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"
    elif not args.part_suggest:
        parser.error("Either --num_parts or --part_suggest must be specified.")

    # download pretrained weights
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
    rmbg_net = None
    if args.rmbg:
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        # init rmbg model for background removal
        rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        rmbg_net.eval()
    t_load = time.time()

    # init tripoSG pipeline
    pipe: PartCrafterPipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)

    set_seed(args.seed)

    # create export directory early (style transfer saves here)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(export_dir, exist_ok=True)

    image_path = args.image_path
    use_alpha = args.use_alpha or args.mask is not None
    if use_alpha:
        image_path = masked_input(image_path, args.mask, export_dir)

    # style transfer: convert real-world photo to Objaverse-style rendering
    if args.style_transfer:
        from src.utils.style_transfer_utils import stylize_for_objaverse
        styled_path = os.path.join(export_dir, "styled_input.png")
        try:
            stylize_for_objaverse(image_path, styled_path, provider=args.style_provider, model_name=args.style_model)
            image_path = styled_path
            print(f"Style transfer complete: {styled_path}")
        except Exception as e:
            print(f"Warning: Style transfer failed ({e}), using original image.")

    # VLM part suggestion
    if args.part_suggest:
        from src.utils.vlm_utils import suggest_num_parts
        num_parts = suggest_num_parts(
            image_path, MAX_NUM_PARTS,
            mode="object",
            provider=args.part_provider,
            model_name=args.part_model,
        )
        print(f"VLM suggested {num_parts} parts")
    else:
        num_parts = args.num_parts

    # run inference
    if device == "mps":
        torch.mps.empty_cache()
    t0 = time.time()
    peak = PeakMemory(device)
    peak.__enter__()
    outputs, processed_image = run_triposg(
        pipe,
        image_input=image_path,
        num_parts=num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        rmbg=args.rmbg,
        dtype=dtype,
        device=device,
        use_alpha=use_alpha,
        dense_octree_depth=args.dense_octree_depth,
        hierarchical_octree_depth=args.hierarchical_octree_depth,
        decode_chunk_size=args.decode_chunk_size,
        band_mode=args.band_mode,
    )
    peak.__exit__()
    run_seconds = time.time() - t0
    processed_image.save(os.path.join(export_dir, "processed_input.png"))

    failed_parts = failed_part_indices(outputs)
    export_meshes = meshes_for_export(outputs)
    for i, mesh in enumerate(export_meshes):
        mesh.export(os.path.join(export_dir, f"part_{i:02}.glb"))

    merged_mesh = get_colored_mesh_composition(export_meshes)
    merged_mesh.export(os.path.join(export_dir, "object.glb"))

    # write manifest
    manifest = {
        "image_path": args.image_path,
        "num_parts": num_parts,
        "style_transferred": args.style_transfer,
        "vlm_suggested": args.part_suggest,
        "parts": [
            {"index": i, "file": f"part_{i:02}.glb"}
            for i in range(num_parts)
        ],
        "composite_file": "object.glb",
        "device": device,
        "dtype": str(dtype),
        "num_inference_steps": args.num_inference_steps,
        "dense_octree_depth": args.dense_octree_depth,
        "hierarchical_octree_depth": args.hierarchical_octree_depth,
        "decode_chunk_size": args.decode_chunk_size,
        "band_mode": args.band_mode,
        "use_flash_decoder": args.use_flash_decoder,
        "seed": args.seed,
        "run_seconds": round(run_seconds, 1),
        "peak_accelerator_gb": round(peak.peak / 2**30, 2),
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
        "part_stats": [
            {"vertices": len(m.vertices), "faces": len(m.faces), "watertight": bool(m.is_watertight),
             "bounds": np.asarray(m.bounds).round(3).tolist() if len(m.vertices) > 1 else None}
            if m is not None else None
            for m in outputs
        ],
    }
    if failed_parts:
        manifest["failed_parts"] = failed_parts
    with open(os.path.join(export_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if failed_parts:
        print(f"Failed parts: {failed_parts}; exported available meshes to {export_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Generated {len(outputs)} parts and saved to {export_dir}")

    if args.render:
        print("Start rendering...")
        from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
        num_views = 36
        radius = 4
        fps = 18
        rendered_images = render_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_normals = render_normal_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_grids = make_grid_for_images_or_videos(
            [
                [processed_image] * num_views,
                rendered_images,
                rendered_normals,
            ], 
            nrow=3
        )
        export_renderings(
            rendered_images,
            os.path.join(export_dir, "rendering.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_normals,
            os.path.join(export_dir, "rendering_normal.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_grids,
            os.path.join(export_dir, "rendering_grid.gif"),
            fps=fps,
        )

        rendered_image, rendered_normal, rendered_grid = rendered_images[0], rendered_normals[0], rendered_grids[0]
        rendered_image.save(os.path.join(export_dir, "rendering.png"))
        rendered_normal.save(os.path.join(export_dir, "rendering_normal.png"))
        rendered_grid.save(os.path.join(export_dir, "rendering_grid.png"))
        print("Rendering done.")
