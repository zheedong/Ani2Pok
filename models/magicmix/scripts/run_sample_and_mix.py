import argparse
import os

import torch
from dotenv import load_dotenv
from PIL import Image

from magic_mix import magic_mix_from_scratch

torch.set_grad_enabled(False)

if __name__ == "__main__":

    load_dotenv(verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="contents")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--k_min_ratio", type=float, default=0.3)
    parser.add_argument("--k_max_ratio", type=float, default=0.6)
    parser.add_argument("--nu", type=float, default=0.9)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--guidance_scale_at_mix", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--layout_prompts",
        type=str,
        nargs="+",
        default=["a realistic photo of a cat", "a realistic photo of a rabbit"],
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--content_semantics_prompts",
        type=str,
        nargs="+",
        default=["coffee machine", "tiger"],
    )

    args = parser.parse_args()

    assert len(args.layout_prompts) == len(
        args.content_semantics_prompts
    ), "Number of layout prompts must match number of content semantics prompts"

    mixed_sementics, originals = magic_mix_from_scratch(
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        layout_prompts=args.layout_prompts,
        content_semantics_prompts=args.content_semantics_prompts,
        k_min=int(args.k_min_ratio * args.num_inference_steps),
        k_max=int(args.k_max_ratio * args.num_inference_steps),
        nu=args.nu,
        guidance_scale_at_mix=args.guidance_scale_at_mix,
        return_unmixed_sampled=True,
        seed=args.seed,
        device=args.device,
    )

    for i, pil_image in enumerate(originals):
        pil_image.save(
            os.path.join(
                args.output_dir,
                f"lp_{args.layout_prompts[i]}_csp_{args.content_semantics_prompts[i]}_original_{i}.png",
            )
        )

    for i, pil_image in enumerate(mixed_sementics):
        pil_image.save(
            os.path.join(
                args.output_dir,
                f"lp_{args.layout_prompts[i]}_csp_{args.content_semantics_prompts[i]}_mixed_nu:{args.nu}_{i}.png",
            )
        )
