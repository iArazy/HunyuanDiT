import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from PIL import Image
from torchvision import transforms as T
import dashscope

from hydit.config import get_args
from hydit.inference_controlnet import End2End
from mllm.dialoggen_demo import eval_model, init_dialoggen_model, DialogGen

# ================= Configuration =================
CUDA_DEVICES = "0,1"
API_KEY = ""
MODEL_ROOT = "ckpts"
DATA_ROOT = "data_test"
RESULT_DIR = "results"
CSV_PATH = "translate.csv"

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICES


# ================= Helper Functions =================

def get_transform():
    """Standard image normalization for the model."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


def refine_instruction_llm(prompt, api_key=API_KEY):
    """
    Calls the LLM to extract the core object name from the editing instruction.
    """
    if not api_key:
        logger.warning("API Key missing, skipping LLM refinement.")
        return prompt

    messages = [
        {
            "role": "system",
            "content": [{
                'type': 'text',
                'text': '我将会给你一段图像的修改介绍，请输出修改后的图像应该存在的新内容。'
                        '例如：”把白猫修改为黑狗“ -> 输出“黑狗”。'
                        '强调：只输出新内容，不要输出其它内容'
            }]
        },
        {"role": "user", "content": [{'type': 'text', 'text': prompt}]}
    ]

    try:
        response = dashscope.Generation.call(
            'qwen2-72b-instruct',
            api_key=api_key,
            messages=messages,
            seed=1234,
            result_format='message',
        )
        if response.status_code == 200:
            return response.output.choices[0]['message']['content']
        else:
            logger.error(f"LLM Error Code: {response.code}")
            return prompt
    except Exception as e:
        logger.error(f"LLM Exception: {e}")
        return prompt


def load_mask_and_process(mask_path, category):
    """
    Reads and processes the mask. Handles the specific logic for 'Background' inversion.
    Returns: mask_float (numpy array, 0.0 to 1.0)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Normalize to [0.0, 1.0]
    mask_float = mask.astype(np.float32) / 255.0

    # Special logic for 'Background' category:
    # Determines if the mask needs to be inverted based on pixel distribution.
    if category == "Background":
        # Heuristic check based on original code boundaries
        center_roi = mask_float[94:547, 94:546]
        center_sum = center_roi.sum()
        total_sum = mask_float.sum()

        # Check if borders have active pixels
        borders_active = (mask_float[0, :].sum() > 0 and mask_float[-1, :].sum() > 0 and
                          mask_float[:, 0].sum() > 0 and mask_float[:, -1].sum() > 0)

        # Check if inner cross area has active pixels
        inner_active = (mask_float[1, :].sum() > 0 and mask_float[-2, :].sum() > 0 and
                        mask_float[:, 1].sum() > 0 and mask_float[:, -2].sum() > 0)

        # Logic to decide inversion
        if not (center_sum < (total_sum - center_sum) and borders_active) and inner_active:
            mask_float = 1.0 - mask_float

    return mask_float


def prepare_masked_image_tensor(image_path, mask_float, target_size, transform, category="Local"):
    """
    Applies the mask to the image using vectorized NumPy operations.
    Converts to Tensor and moves to GPU.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Create a copy to avoid modifying original array
    masked_img = image_rgb.copy()

    if category == "Background":
        masked_img[mask_float == 0.0] = 0
    else:
        # Local, Remove, Addition
        masked_img[mask_float == 1.0] = 0

    # Resize and Transform
    h, w = target_size
    pil_img = Image.fromarray(masked_img).resize((w, h)).convert('RGB')

    # Normalize and move to GPU
    tensor = transform(pil_img).unsqueeze(0).cuda()

    return tensor


# ================= Main Execution =================

def main():
    # 1. Setup Directories
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. Load Inpainting Model (Gen1)
    logger.info("Loading Inpainting Model (Gen1)...")
    args = get_args()
    args.model_root = MODEL_ROOT
    gen_inpainting = End2End(args, Path(MODEL_ROOT))

    # Load Enhancer (DialogGen) if enabled
    enhancer = None
    if args.enhance:
        logger.info("Loading Prompt Enhancer...")
        enhancer = DialogGen(str(Path(MODEL_ROOT) / "dialoggen"), args.load_4bit)

    # 3. Load Global/ControlNet Model (Gen2 - Canny)
    logger.info("Loading Global Model (Gen2 - Canny)...")
    args2 = get_args()
    args2.model_root = MODEL_ROOT
    args2.control_type = 'canny'
    args2.load_key = 'distill'
    gen_global = End2End(args2, Path(MODEL_ROOT))

    # Load Captioner for Global intent understanding
    captioner = init_dialoggen_model(Path(MODEL_ROOT) / "dialoggen")

    # 4. Prepare Data
    df = pd.read_csv(CSV_PATH)
    norm_transform = get_transform()

    # Define processing range
    start_idx = 0
    end_idx = 8
    subset = df.iloc[start_idx:end_idx]

    logger.info(f"Processing {len(subset)} images from index {start_idx} to {end_idx}...")

    for i, row in subset.iterrows():
        current_idx = start_idx + i

        rel_path = row['instruction_target_image']
        full_image_path = str(Path(DATA_ROOT) / rel_path)

        mask_filename = f"image_{i + 1}.png"
        mask_path = str(Path(DATA_ROOT) / "mask_en" / mask_filename)

        category = row['category']
        raw_prompt = row['translate']
        obj_name = row['object']

        logger.info(f"[{current_idx}] Category: {category} | Instruction: {raw_prompt}")

        # ================= Branch A: Global Editing (ControlNet/Canny) =================
        if category == "Global":
            # 1. Understand Intent via VLM
            query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{raw_prompt}"
            model_instruction = eval_model(captioner, query=query, image_file=full_image_path)

            # 2. Canny Edge Detection
            img_cv = cv2.imread(full_image_path)
            if img_cv is None: continue

            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # 3. Prepare Control Condition
            h, w = args2.image_size
            edge_pil = Image.fromarray(edges_rgb).resize((w, h))
            edge_tensor = norm_transform(edge_pil).unsqueeze(0).cuda()

            # 4. Inference
            res = gen_global.predict(
                model_instruction,
                height=h, width=w,
                image=edge_tensor,
                seed=args2.seed,
                guidance_scale=args2.cfg_scale
            )

            # Save Result
            save_path = Path(RESULT_DIR) / f"{current_idx}.png"
            res['images'][0].save(save_path)
            logger.info(f"Saved Global result to {save_path}")
            continue

        # ================= Branch B: Inpainting (Local, BG, Remove, Add) =================

        # 1. Refine Prompt via LLM (Extract object name)
        refined_content = refine_instruction_llm(raw_prompt)

        # 2. Process Mask
        mask_float = load_mask_and_process(mask_path, category)
        if mask_float is None:
            logger.warning(f"Mask not found or empty: {mask_path}")
            continue

        # 3. Apply Mask to Image (Vectorized)
        h, w = args.image_size
        img_tensor = prepare_masked_image_tensor(full_image_path, mask_float, (h, w), norm_transform, category)

        # 4. Construct Prompts based on Category
        final_prompt = refined_content
        neg_prompt = args.negative

        if category == "Remove":
            final_prompt = ""
            neg_prompt += f", {obj_name}, 错误, 变形, 伪影, 模糊"
        elif category == "Addition":
            neg_prompt = ""

        # 5. Prompt Enhancement (Optional)
        enhanced_prompt = None
        if enhancer and final_prompt:
            _, enhanced_prompt = enhancer(final_prompt)

        # 6. Inference
        res = gen_inpainting.predict(
            final_prompt,
            height=h, width=w,
            image=img_tensor,
            seed=args.seed,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale
        )

        # Save Result
        save_path = Path(RESULT_DIR) / f"{current_idx}.png"
        res['images'][0].save(save_path)
        logger.info(f"Saved Inpainting result to {save_path}")


if __name__ == "__main__":
    main()