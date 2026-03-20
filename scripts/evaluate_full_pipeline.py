#!/usr/bin/env python3

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import json
import cv2

from skimage.metrics import structural_similarity as ssim

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class SketchyEvaluator:
    def __init__(self, stage2_ckpt, stage1_ckpt, device='cuda'):
        self.device = device
        self.stage2_ckpt = stage2_ckpt
        self.stage1_ckpt = stage1_ckpt

        print("🚀 Initializing FULL pipeline evaluator...\n")

        self.load_models()
        self.load_dataset()

    def load_models(self):
        print("📦 Loading models...")

        model_name = "runwayml/stable-diffusion-v1-5"

        # ---- VAE ----
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device).eval()

        # ---- TEXT ----
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device).eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

        # ---- SCHEDULER ----
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # =====================================================
        # 🔹 STAGE 1
        # =====================================================
        from models.stage1_diffusion import Stage1SketchGuidedDiffusion

        self.stage1 = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=[320, 640, 1280, 1280],
            freeze_base_unet=False,
            use_lora=True,
            lora_rank=8
        ).to(self.device).eval()

        stage1_ckpt = torch.load(self.stage1_ckpt,map_location="cpu",weights_only=False)
        self.stage1.load_state_dict(stage1_ckpt["model_state_dict"], strict=False)

        print("   ✅ Stage 1 loaded")

        # =====================================================
        # 🔹 STAGE 2
        # =====================================================
        from models.stage2_refinement import Stage2SemanticRefinement

        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)

        self.stage2 = Stage2SemanticRefinement(unet=unet)

        stage2_ckpt = torch.load(self.stage2_ckpt,map_location="cpu",weights_only=False)
        self.stage2.load_state_dict(stage2_ckpt["model_state_dict"], strict=False)

        self.stage2.to(self.device).eval()

        print("   ✅ Stage 2 loaded\n")

    def load_dataset(self):
        print("📁 Loading Sketchy dataset...")

        from datasets.sketchy_dataset import SketchyDataset

        self.dataset = SketchyDataset(
            root_dir="/workspace/sketchy",
            split="test",
            image_size=256,
            augment=False
        )

        print(f"   Total samples: {len(self.dataset)}\n")

    def tensor_to_numpy(self, tensor):
        tensor = tensor[0].cpu().numpy().transpose(1, 2, 0)
        tensor = ((tensor + 1) / 2 * 255).astype(np.uint8)

        if tensor.shape[-1] != 3:
            tensor = np.repeat(tensor[..., None], 3, axis=-1)

        return tensor

    @torch.inference_mode()
    def run(self):
        print("🚀 Running Stage1 + Stage2 evaluation...\n")

        results = []
        output_dir = Path("outputs_stage1_stage2")
        output_dir.mkdir(exist_ok=True)

        self.scheduler.set_timesteps(20)

        for idx in tqdm(range(len(self.dataset))):
            sample = self.dataset[idx]

            sketch = sample['sketch'].unsqueeze(0).to(self.device)
            photo = sample['photo'].unsqueeze(0).to(self.device)

            # Category extraction
            photo_path = sample.get('photo_path', '')
            category = photo_path.split('/')[-2] if photo_path else "object"

            prompt = f"a photo of a {category}"

            try:
                # ---- TEXT ----
                text_inputs = self.tokenizer(
                    [prompt],
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                ).to(self.device)

                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

                # =====================================================
                # 🔹 STAGE 1 GENERATION
                # =====================================================
                latent = torch.randn(1, 4, 32, 32, device=self.device)

                sketch_features = self.stage1.encode_sketch(sketch)
                text_emb_stage1 = self.stage1.encode_text([prompt])

                for t in self.scheduler.timesteps:
                    t_tensor = torch.tensor([t], device=self.device)

                    noise_pred = self.stage1(
                        latent,
                        t_tensor,
                        sketch_features,
                        text_emb_stage1
                    )

                    latent = self.scheduler.step(noise_pred, t, latent).prev_sample

                stage1_img = self.vae.decode(latent / 0.18215).sample
                stage1_img = stage1_img.clamp(-1, 1)

                # =====================================================
                # 🔹 STAGE 2 REFINEMENT
                # =====================================================
                stage1_latent = self.vae.encode(stage1_img).latent_dist.sample()

                noise = torch.randn_like(stage1_latent)
                t_refine = torch.randint(0, 1000, (1,), device=self.device)

                noisy_latent = self.scheduler.add_noise(stage1_latent, noise, t_refine)

                region_graph = sample.get("region_graph", None)

                noise_pred = self.stage2(
                    noisy_latent,
                    t_refine,
                    region_graph,
                    text_embeddings,
                    return_dict=False
                )

                refined_latent = noisy_latent - noise_pred

                final_img = self.vae.decode(refined_latent / 0.18215).sample
                final_img = final_img.clamp(-1, 1)

                # =====================================================
                # 🔹 METRICS
                # =====================================================
                gen_np = self.tensor_to_numpy(final_img)
                gt_np = self.tensor_to_numpy(photo)

                ssim_val = ssim(gen_np, gt_np, channel_axis=2)

                # =====================================================
                # 🔹 SAVE IMAGE
                # =====================================================
                save_dir = output_dir / category
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f"{idx:05d}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR))

                results.append({
                    "index": idx,
                    "category": category,
                    "ssim": float(ssim_val)
                })

            except Exception as e:
                print(f"\n⚠️ Error at {idx}: {e}")
                continue

        # SAVE RESULTS
        with open("ssim_results_stage1_stage2.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n✅ COMPLETE!")
        print(f"Saved {len(results)} results")


if __name__ == "__main__":
    evaluator = SketchyEvaluator(
        stage2_ckpt="/root/checkpoints/stage2/epoch_10.pt",
        stage1_ckpt="/root/checkpoints/stage1_with_ssim/epoch_18.pt",
        device="cuda"
    )

    evaluator.run()