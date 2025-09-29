import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

import cv2
from PIL import Image
import torch
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to("cuda")
    model.eval()

    # img_path = "/workspace/datasets/NightReID_LIME/query/1196R2C100.jpg"
    # img_path = "/workspace/datasets/AY20/Night/IR/query/00123_c27s186_1161.png"
    # img_path = "/workspace/datasets/AY20/Night/IR/query/00133_c14s124_1841.png"
    # img_path = "/workspace/datasets/AY20/Night/IR/query/00164_c49s421_1161.png"
    # img_path = "/workspace/datasets/AY20/Night/IR/query/00216_c17s2539_641.png"
    # img_path = "/workspace/datasets/NightReID/query/0183R1C041.jpg"
    # img_path = "/workspace/datasets/NightReID/query/0380R2C124.jpg"
    img_path = "/workspace/datasets/NightReID/query/0523L3C044.jpg"

    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    from torchvision import transforms as T

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    pil_img = Image.fromarray(original_img)
    transformed_img = val_transforms(pil_img).unsqueeze(0).to("cuda")

    with torch.no_grad():
        feat, attentions = model(transformed_img)

    last_layer_attention = attentions[-1].squeeze(0).cpu()

    avg_attention = last_layer_attention.mean(dim=0)

    cls_token_attention = avg_attention[0, 1:]

     # ================== 디버깅 코드 추가 ==================
    print("\n--- Attention Map Debug Info ---")
    print(f"Min value: {cls_token_attention.min().item():.6f}")
    print(f"Max value: {cls_token_attention.max().item():.6f}")
    print(f"Mean value: {cls_token_attention.mean().item():.6f}")
    print("--------------------------------\n")
    # ======================================================

    patch_size = 16
    num_patches_h = cfg.INPUT.SIZE_TEST[0] // patch_size
    num_patches_w = cfg.INPUT.SIZE_TEST[1] // patch_size
    attention_map = cls_token_attention.reshape(num_patches_h, num_patches_w).numpy()

    # ================== [해결] 어텐션 맵 정규화 코드 추가 ==================
    # 어텐션 맵의 최소-최대 값을 기준으로 0~1 범위로 값을 재조정합니다.
    map_min, map_max = attention_map.min(), attention_map.max()
    attention_map_normalized = (attention_map - map_min) / (map_max - map_min)
    # ====================================================================

    # cv2.resize와 applyColorMap에는 정규화된 맵을 사용합니다.
    attention_map_resized = cv2.resize(attention_map_normalized, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    final_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./attentionmap.jpg", final_img_bgr)