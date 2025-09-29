# -*- coding: utf-8 -*-
"""
ReID Inference (single query) + Top-K rank strip + AP/CMC print

Usage:
python run_rank_strip.py --config_file path/to/config.yaml --query_path "D:\\ReID\\NightReID_LIME\\query\\0002R1C041.jpg" --topk 10 --size 128x256

Notes:
- cfg, make_dataloader, make_model, utils.metrics(euclidean_distance, re_ranking, eval_func) 필요
- cfg.TEST.FEAT_NORM (bool), cfg.TEST.RERANKING (bool, 선택) 사용 가능
- 저장 경로: cfg.OUTPUT_DIR/rank_strip_query_plus_topK.jpg
"""

import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

# 기존 프로젝트 유틸
from utils.metrics import euclidean_distance, re_ranking, eval_func
from utils.logger import setup_logger
from config import cfg
from datasets import make_dataloader
from model import make_model


# -----------------------------
# 0) 헬퍼들
# -----------------------------
def normpath_case(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))

def parse_wh(s: str, default=(128, 256)):
    # "128x256" -> (128,256)
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    except Exception:
        return default


# -----------------------------
# 1) 이미지 유틸 (왜곡 방지)
# -----------------------------
def fit_with_padding(img: Image.Image, target_size=(128, 256), fill=(255, 255, 255)):
    """원본 비율 유지해서 축소/확대 후, 남는 영역은 패딩으로 채움."""
    tw, th = target_size  # (W, H)
    w, h = img.size
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (tw, th), fill)
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(img_resized, offset)
    return canvas

def _draw_border(im, color=(0, 200, 0), thickness=6):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for t in range(thickness):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color)
    return im

def _draw_text_with_outline(draw, xy, text, fill=(255,255,255), outline=(0,0,0), font=None):
    x, y = xy
    # 간단한 외곽선 1px
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


# -----------------------------
# 2) Rank Strip(이미지) 생성
# -----------------------------
def make_rank_strip(
    save_path: str,
    query_path: str,
    top_paths: list,
    q_pid: int,
    g_pids_top: list,
    q_camid: int = None,
    g_camids_top: list = None,
    size=(128, 256),
    gap=6,
    draw_rank_text=True,
    scores_top: list=None,
    score_fmt: str="{:.3f}",
):
    """
    쿼리 + Rank-K 가로 스트립 저장
    - 초록 테두리: 정답(g_pid == q_pid [필요시 동일카메라 제외])
    - 빨강 테두리: 오답
    """
    # 정답 여부
    correct_flags = []
    for i, gpid in enumerate(g_pids_top):
        ok = (gpid == q_pid)
        if (q_camid is not None) and (g_camids_top is not None):
            ok = ok and (g_camids_top[i] != q_camid)  # 동일 카메라 제외 규칙 필요시 유지
        correct_flags.append(ok)

    # 캔버스 준비 (QUERY 1 + TOP K)
    N = 1 + len(top_paths)
    w, h = size
    canvas_w = N * w + (N - 1) * gap
    canvas_h = h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # 폰트 (환경에 따라 실패 가능)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        pass

    # QUERY (파란 테두리)
    x = 0
    # q_img = fit_with_padding(Image.open(query_path).convert("RGB"), target_size=size)
    q_img = Image.open(query_path).convert("RGB").resize(size)
    q_img = _draw_border(q_img, color=(0, 128, 255), thickness=6)
    if draw_rank_text:
        d = ImageDraw.Draw(q_img)
        d.text((6, 6), "QUERY", fill=(255, 255, 255), font=font)
        d.text((80, 6), f"PID: {q_pid}", fill=(255,255,255), font=font)
    canvas.paste(q_img, (x, 0))
    x += w + gap

    # RANK 1..K
    for r, (p, is_ok, pid) in enumerate(zip(top_paths, correct_flags, g_pids_top), start=1):
        im = Image.open(p).convert("RGB").resize(size)
        border_col = (0, 200, 0) if is_ok else (220, 0, 0)
        im = _draw_border(im, color=border_col, thickness=6)
        d = ImageDraw.Draw(im)
        if draw_rank_text:
            # 좌상단 Rank
            _draw_text_with_outline(d, (6, 6), f"R{r}", fill=(255,255,255), outline=(0,0,0), font=font)
            # 우상단 PID
            _draw_text_with_outline(d, (80, 6), f"PID: {pid}", fill=(255,255,255), outline=(0,0,0), font=font)
        # ✅ 하단 점수
        if scores_top is not None:
            try:
                s_val = float(scores_top[r-1])
                s = score_fmt.format(s_val) if np.isfinite(s_val) else "∞"
            except Exception:
                s = str(scores_top[r-1])
            w, h = im.size
            tw, th = d.textsize(s, font=font) if font else (len(s)*8, 16)
            _draw_text_with_outline(d, ((w - tw)//2, h - th - 8), f"d={s}",
                                    fill=(255,255,255), outline=(0,0,0), font=font)
        canvas.paste(im, (x, 0))
        x += w + gap

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)
    print(f"[saved] {save_path}")


# -----------------------------
# 3) Inference & Top-K 추출
# -----------------------------
@torch.no_grad()
def do_inference_single_query(cfg, model, val_loader, query_path, num_query, topk=20, root=''):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.test")
    logger.info('Enter inferencing')

    # 수집 버퍼: 쿼리 먼저, 갤러리 다음
    q_feats, q_pids, q_camids, q_paths = [], [], [], []
    g_feats, g_pids, g_camids, g_paths = [], [], [], []

    qpath_norm = normpath_case(query_path)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for inference')
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    img_cnt = 0
    for _, (img, pid, camid, camids, target_view, imgpaths) in enumerate(val_loader):
        img = img.to(device, non_blocking=True)
        camids = camids.to(device, non_blocking=True)
        target_view = target_view.to(device, non_blocking=True)

        feats = model(img, cam_label=camids, view_label=target_view)  # [B, D]

        # 배치 내 개별 샘플 단위로 분기
        for j, p in enumerate(imgpaths):
            img_cnt += 1
            p_norm = normpath_case(p)
            f = feats[j].detach().cpu()
            pid_j = int(pid[j])
            camid_j = int(camid[j])
            if p_norm == os.path.basename(qpath_norm):
                q_feats.append(f); q_pids.append(pid_j); q_camids.append(camid_j); q_paths.append(os.path.join(root, 'query', p))
            else:
                if img_cnt >= num_query:
                    g_feats.append(f); g_pids.append(pid_j); g_camids.append(camid_j); g_paths.append(os.path.join(root, 'bounding_box_test', p))

    # 쿼리 확인
    if len(q_feats) != 1:
        raise RuntimeError(f"쿼리 경로를 정확히 1개 찾지 못했습니다. found={len(q_feats)} path={query_path}")

    # 특징 정규화
    if getattr(cfg.TEST, "FEAT_NORM", True):
        qf = torch.nn.functional.normalize(torch.stack(q_feats, dim=0), dim=1, p=2)  # [1, D]
        gf = torch.nn.functional.normalize(torch.stack(g_feats, dim=0), dim=1, p=2)  # [N, D]
    else:
        qf = torch.stack(q_feats, dim=0)
        gf = torch.stack(g_feats, dim=0)

    # 거리행렬 or 재랭킹
    use_rerank = bool(getattr(cfg.TEST, "RERANKING", False))
    if use_rerank:
        print("=> Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)  # [1, N] (ndarray)
        if isinstance(distmat, torch.Tensor):
            dist = distmat[0].cpu().numpy()
        else:
            dist = distmat[0]
    else:
        print("=> Computing DistMat with euclidean_distance")
        distmat = euclidean_distance(qf, gf)  # [1, N] (ndarray)
        dist = distmat[0]

    # AP & CMC 계산용 (원본) -------------------------
    q_pids_np   = np.asarray([q_pids[0]])
    q_camids_np = np.asarray([q_camids[0]])
    g_pids_np   = np.asarray(g_pids)
    g_camids_np = np.asarray(g_camids)

    cmc, mAP = eval_func(distmat, q_pids_np, g_pids_np, q_camids_np, g_camids_np)
    print("==== Evaluation Results ====")
    print(f"mAP: {mAP:.2%}")
    for r in [1, 5, 10]:
        if r <= len(cmc):
            print(f"Rank-{r:<3}: {cmc[r-1]:.2%}")

    # Top-K (시각화용) ------------------------------
    filter_same_cam = True   # ✅ 질문 요구: 동일 camid는 뒤로 밀기
    dist_vis = dist.copy()

    if filter_same_cam:
        same_cam_mask = (g_pids_np == q_pids_np[0]) & (g_camids_np == q_camids_np[0])
        dist_vis[same_cam_mask] = np.inf  # 동일 카메라는 전부 뒤로 밀기

    # (참고) Market1501 규칙만 적용하려면 아래를 사용하세요:
    # junk_mask = (g_pids_np == q_pids_np[0]) & (g_camids_np == q_camids_np[0])
    # dist_vis[junk_mask] = np.inf

    order_vis = np.argsort(dist_vis)  # same-cam은 ∞로 밀려 뒤로 감
    topk = min(topk, len(order_vis))
    top_idx = order_vis[:topk]

    # 만약 same-cam 제외로 topk가 너무 적다면, 남는 자리는 원 dist에서 채우기(선택)
    if np.isinf(dist_vis[order_vis[:topk]]).any():
        # 유효(비-∞)만 먼저 채움
        valid = order_vis[np.isfinite(dist_vis[order_vis])]
        need  = topk - len(valid[:topk])
        if need > 0:
            # 뒤에 same-cam에서도 추가로 채움
            rest = order_vis[np.isinf(dist_vis[order_vis])]
            top_idx = np.concatenate([valid[:topk], rest[:need]])[:topk]

    top_paths = [g_paths[i] for i in top_idx]

    print("\n[DEBUG] Top-{} list (pid/camid/path)".format(topk))
    hits = 0
    for rank, gi in enumerate(top_idx, 1):
        is_hit = (g_pids[gi] == q_pids[0]) and (g_camids[gi] != q_camids[0])
        hits += int(is_hit)
        print(f" R{rank:>2}: pid={g_pids[gi]} cam={g_camids[gi]} "
            f"hit={is_hit}  path={g_paths[gi]}")
    print(f"[DEBUG] hits in top-{topk} (CMC 기준): {hits}\n")


    return {
        # "query_path": q_paths[0],
        "query_path": query_path,
        "q_pid": q_pids[0],
        "q_camid": q_camids[0],
        "g_pids": g_pids,
        "g_camids": g_camids,
        "g_paths": g_paths,
        "top_idx": top_idx,
        "top_paths": top_paths,
        "distmat": distmat,  # (1, N)
    }


# -----------------------------
# 3-1) 모든 피처 한 번에 추출하기
# -----------------------------
from torch.cuda.amp import autocast
@torch.no_grad()
def extract_all_features(cfg, model, val_loader, num_query, root=''):
    """데이터로더 전체를 순회하며 모든 이미지의 피처와 정보를 추출합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.test")
    logger.info('Extracting all features from val_loader...')

    # 모든 피처와 정보를 담을 리스트
    all_feats, all_pids, all_camids, all_paths = [], [], [], []

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    img_cnt = 0
    # 데이터로더를 한 번만 순회
    for _, (img, pid, camid, camids, target_view, imgpaths) in enumerate(val_loader):
        img = img.to(device, non_blocking=True)
        camids = camids.to(device, non_blocking=True)
        target_view = target_view.to(device, non_blocking=True)


        with autocast():
            feats, attn = model(img, cam_label=camids, view_label=target_view)

        # 피처와 정보 저장
        feats_cpu = feats.detach().to('cpu', copy=True)
        # all_feats.append(feats.detach().cpu())
        all_feats.append(feats_cpu)
        all_pids.extend(pid)
        all_camids.extend(camid)

        for p in imgpaths:
            img_cnt += 1
            if img_cnt < num_query: 
                full_path = os.path.join(root, 'query', p)
                # full_path = p
            else:
                full_path = os.path.join(root, 'bounding_box_test', p)
                # full_path = p
            all_paths.append(full_path)


    # 리스트들을 텐서와 numpy 배열로 변환
    all_feats = torch.cat(all_feats, dim=0)
    all_pids = np.array(all_pids, dtype=np.int32)
    all_camids = np.array(all_camids, dtype=np.int32)

    # 피처 정규화
    if getattr(cfg.TEST, "FEAT_NORM", True):
        logger.info('Normalizing features...')
        all_feats = torch.nn.functional.normalize(all_feats, dim=1, p=2)

    logger.info(f'Feature extraction complete. Total images: {len(all_paths)}')
    return all_feats, all_pids, all_camids, all_paths

# -----------------------------
# 3-2) 추출된 피처에서 검색 수행
# -----------------------------
def search_from_features(cfg, query_path, all_feats, all_pids, all_camids, all_paths, topk=10):
    """미리 추출된 전체 피처에서 특정 쿼리를 찾아 갤러리와 비교합니다."""
    
    # 쿼리 경로를 이용해 전체 데이터에서 쿼리 인덱스 찾기
    qpath_norm = normpath_case(query_path)
    all_paths_norm = [normpath_case(p) for p in all_paths]
    print(all_paths_norm[0])
    
    try:
        q_idx = all_paths_norm.index(qpath_norm)
    except ValueError:
        raise RuntimeError(f"쿼리 경로를 찾을 수 없습니다: {query_path}")

    # 쿼리와 갤러리 분리
    qf = all_feats[q_idx:q_idx+1]  # [1, D]
    q_pid = all_pids[q_idx]
    q_camid = all_camids[q_idx]
    
    # 갤러리 인덱스 (쿼리 자신을 제외)
    g_indices = np.arange(len(all_paths))
    g_indices = np.delete(g_indices, q_idx)

    gf = all_feats[g_indices]
    g_pids = all_pids[g_indices]
    g_camids = all_camids[g_indices]
    g_paths = [all_paths[i] for i in g_indices]

    # 거리 계산
    use_rerank = bool(getattr(cfg.TEST, "RERANKING", False))
    if use_rerank:
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        dist = distmat[0]
    else:
        distmat = euclidean_distance(qf, gf)
        dist = distmat[0]

    # AP & CMC 계산
    cmc, mAP = eval_func(distmat, np.array([q_pid]), g_pids, np.array([q_camid]), g_camids)
    
    # 시각화를 위한 Top-K 추출 (기존 코드와 동일)
    dist_vis = dist.copy()
    same_cam_mask = (g_pids == q_pid) & (g_camids == q_camid)
    dist_vis[same_cam_mask] = np.inf
    
    order_vis = np.argsort(dist_vis)
    topk = min(topk, len(order_vis))
    top_idx = order_vis[:topk]

    top_paths = [g_paths[i] for i in top_idx]

    # 결과 출력
    print(f"\n--- Query: {os.path.basename(query_path)} ---")
    print(f"mAP: {mAP:.2%}")
    for r in [1, 5, 10]:
        if r <= len(cmc):
            print(f"Rank-{r:<3}: {cmc[r-1]:.2%}")

    SHOW_TRUE_L2 = True and (not use_rerank)
    if SHOW_TRUE_L2:
        top_scores = [float(np.sqrt(dist[i])) for i in top_idx]
    else:
        top_scores = [float(dist[i]) for i in top_idx]
    
    return {
        "query_path": query_path,
        "q_pid": q_pid,
        "q_camid": q_camid,
        "g_pids": g_pids,
        "g_camids": g_camids,
        "top_idx": top_idx,
        "top_paths": top_paths,
        "top_scores": top_scores
    }

# -----------------------------
# 4) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ReID Inference + TopK Rank Strip + AP/CMC")
    parser.add_argument("--config_file", default="", type=str, help="path to config file")
    parser.add_argument("--query_path", default="", type=str, help="absolute path to the query image")
    parser.add_argument("--topk", default=10, type=int, help="K for rank visualization")
    parser.add_argument("--size", default="128x256", type=str, help="tile size WxH (e.g., 128x256)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options from CLI")
    args = parser.parse_args()

    # cfg 로드
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 출력 디렉토리 및 로거
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)
    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r', encoding="utf-8") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # GPU 선택
    if hasattr(cfg.MODEL, "DEVICE_ID"):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)

    # 데이터/모델
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    #########################################################
    NAMES = 'NightReID'
    root = os.path.join(cfg.DATASETS.ROOT_DIR, NAMES)

    # 피쳐 미리계산
    all_feats, all_pids, all_camids, all_paths = extract_all_features(cfg, model, val_loader, num_query, root=root)
    
    # query_path = args.query_path.strip() if args.query_path else "/workspace/datasets/NightReID/query/0137R3C008.jpg" # "D:\\ReID\\NightReID_LIME\\query\\0002R1C041.jpg"
    query_paths = ['/workspace/datasets/NightReID/query/1325L2C069.jpg', '/workspace/datasets/NightReID/query/0198R2C041.jpg', '/workspace/datasets/NightReID/query/0049L2C004.jpg', '/workspace/datasets/NightReID/query/0581L2C032.jpg', '/workspace/datasets/NightReID/query/0535R3C017.jpg', '/workspace/datasets/NightReID/query/0516L2C010.jpg', '/workspace/datasets/NightReID/query/0345R3C127.jpg', '/workspace/datasets/NightReID/query/0183R1C041.jpg', '/workspace/datasets/NightReID/query/1377L1C041.jpg', '/workspace/datasets/NightReID/query/1129R1C041.jpg', '/workspace/datasets/NightReID/query/0152R1C081.jpg', '/workspace/datasets/NightReID/query/1224R1C041.jpg', '/workspace/datasets/NightReID/query/0802L1C041.jpg', '/workspace/datasets/NightReID/query/0060R2C003.jpg', '/workspace/datasets/NightReID/query/0059R1C012.jpg', '/workspace/datasets/NightReID/query/0165R2C004.jpg', '/workspace/datasets/NightReID/query/0508L1C081.jpg', '/workspace/datasets/NightReID/query/0527L3C041.jpg', '/workspace/datasets/NightReID/query/1058R2C010.jpg', '/workspace/datasets/NightReID/query/1241R1C041.jpg', '/workspace/datasets/NightReID/query/0049R3C053.jpg', '/workspace/datasets/NightReID/query/1160R2C020.jpg', '/workspace/datasets/NightReID/query/0446R2C064.jpg', '/workspace/datasets/NightReID/query/1343R2C008.jpg', '/workspace/datasets/NightReID/query/0797R3C024.jpg', '/workspace/datasets/NightReID/query/0511L1C081.jpg', '/workspace/datasets/NightReID/query/0826R3C015.jpg', '/workspace/datasets/NightReID/query/1220R3C075.jpg', '/workspace/datasets/NightReID/query/0586L3C034.jpg', '/workspace/datasets/NightReID/query/0018R2C074.jpg', '/workspace/datasets/NightReID/query/0376R2C038.jpg', '/workspace/datasets/NightReID/query/0802L2C004.jpg', '/workspace/datasets/NightReID/query/0677L1C081.jpg', '/workspace/datasets/NightReID/query/0367R2C026.jpg', '/workspace/datasets/NightReID/query/0503L3C015.jpg', '/workspace/datasets/NightReID/query/0672L1C081.jpg', '/workspace/datasets/NightReID/query/0162R3C017.jpg', '/workspace/datasets/NightReID/query/0740L1C121.jpg', '/workspace/datasets/NightReID/query/0168L1C161.jpg', '/workspace/datasets/NightReID/query/0708R2C064.jpg', '/workspace/datasets/NightReID/query/0681R1C041.jpg', '/workspace/datasets/NightReID/query/1244R1C041.jpg', '/workspace/datasets/NightReID/query/0568L3C011.jpg', '/workspace/datasets/NightReID/query/0078R1C041.jpg', '/workspace/datasets/NightReID/query/0838L2C014.jpg', '/workspace/datasets/NightReID/query/1115R1C041.jpg', '/workspace/datasets/NightReID/query/0305R2C115.jpg', '/workspace/datasets/NightReID/query/0737R1C041.jpg', '/workspace/datasets/NightReID/query/0143R1C041.jpg', '/workspace/datasets/NightReID/query/1143R2C060.jpg', '/workspace/datasets/NightReID/query/0613L1C041.jpg', '/workspace/datasets/NightReID/query/1307L2C048.jpg', '/workspace/datasets/NightReID/query/1276R1C041.jpg', '/workspace/datasets/NightReID/query/0712R1C041.jpg', '/workspace/datasets/NightReID/query/1196R2C100.jpg', '/workspace/datasets/NightReID/query/0435R3C046.jpg', '/workspace/datasets/NightReID/query/0134L1C041.jpg', '/workspace/datasets/NightReID/query/0082R2C080.jpg', '/workspace/datasets/NightReID/query/1357L3C022.jpg', '/workspace/datasets/NightReID/query/0523L3C044.jpg', '/workspace/datasets/NightReID/query/0603L2C260.jpg', '/workspace/datasets/NightReID/query/0143R3C084.jpg', '/workspace/datasets/NightReID/query/0179R3C070.jpg', '/workspace/datasets/NightReID/query/0830R3C024.jpg', '/workspace/datasets/NightReID/query/1316L3C051.jpg', '/workspace/datasets/NightReID/query/0714R3C088.jpg', '/workspace/datasets/NightReID/query/0380R1C041.jpg', '/workspace/datasets/NightReID/query/0724L3C019.jpg', '/workspace/datasets/NightReID/query/0701R1C039.jpg', '/workspace/datasets/NightReID/query/0468R2C081.jpg', '/workspace/datasets/NightReID/query/1366L2C014.jpg', '/workspace/datasets/NightReID/query/0569R3C043.jpg', '/workspace/datasets/NightReID/query/1343L1C041.jpg', '/workspace/datasets/NightReID/query/0134R2C022.jpg', '/workspace/datasets/NightReID/query/1254R1C041.jpg', '/workspace/datasets/NightReID/query/1316L2C023.jpg', '/workspace/datasets/NightReID/query/0388R3C028.jpg', '/workspace/datasets/NightReID/query/1109R3C156.jpg', '/workspace/datasets/NightReID/query/0380R2C124.jpg', '/workspace/datasets/NightReID/query/0842L2C081.jpg', '/workspace/datasets/NightReID/query/0737R3C021.jpg', '/workspace/datasets/NightReID/query/0574L2C005.jpg', '/workspace/datasets/NightReID/query/1325L3C037.jpg', '/workspace/datasets/NightReID/query/1155R1C041.jpg', '/workspace/datasets/NightReID/query/0509L1C081.jpg', '/workspace/datasets/NightReID/query/0657L1C201.jpg', '/workspace/datasets/NightReID/query/0105R3C085.jpg', '/workspace/datasets/NightReID/query/0524L3C027.jpg', '/workspace/datasets/NightReID/query/0647L1C081.jpg', '/workspace/datasets/NightReID/query/0773L3C046.jpg', '/workspace/datasets/NightReID/query/0572L1C041.jpg', '/workspace/datasets/NightReID/query/0131R3C085.jpg', '/workspace/datasets/NightReID/query/0469R3C006.jpg', '/workspace/datasets/NightReID/query/1167R2C041.jpg', '/workspace/datasets/NightReID/query/0646L1C121.jpg', '/workspace/datasets/NightReID/query/0501L1C161.jpg', '/workspace/datasets/NightReID/query/1350L2C121.jpg', '/workspace/datasets/NightReID/query/1053R2C020.jpg', '/workspace/datasets/NightReID/query/0764R3C071.jpg', '/workspace/datasets/NightReID/query/1328L3C012.jpg']

    # query_paths = [os.path.join(root, 'query', os.path.basename(b).split(".")[0] + ".png") for b in query_paths]

    # query_paths = [os.path.join(root, 'query', os.path.basename(b)) for b in query_paths]
    
    # query_path1 = ['/workspace/datasets/AY21/query/00375_c18s3274_121.png', '/workspace/datasets/AY21/query/00346_c07s2795_521.png', '/workspace/datasets/AY21/query/00441_c54s3851_1801.png', '/workspace/datasets/AY21/query/00429_c57s3862_321.png', '/workspace/datasets/AY21/query/00421_c42s3794_461.png', '/workspace/datasets/AY21/query/00385_c34s3376_261.png', '/workspace/datasets/AY21/query/00371_c27s3334_141.png', '/workspace/datasets/AY21/query/00549_c10s4470_301.png', '/workspace/datasets/AY21/query/00366_c02s2887_1021.png', '/workspace/datasets/AY21/query/00564_c18s4500_881.png', '/workspace/datasets/AY21/query/00500_c12s4167_121.png', '/workspace/datasets/AY21/query/00348_c08s2802_421.png', '/workspace/datasets/AY21/query/00347_c18s2851_1461.png', '/workspace/datasets/AY21/query/00367_c23s2831_661.png', '/workspace/datasets/AY21/query/00419_c44s3807_581.png', '/workspace/datasets/AY21/query/00425_c48s3824_241.png', '/workspace/datasets/AY21/query/00533_c47s5412_641.png', '/workspace/datasets/AY21/query/00346_c18s2851_1161.png', '/workspace/datasets/AY21/query/00553_c20s4507_501.png', '/workspace/datasets/AY21/query/00411_c29s3737_1421.png', '/workspace/datasets/AY21/query/00549_c09s4468_1781.png', '/workspace/datasets/AY21/query/00499_c10s4163_81.png', '/workspace/datasets/AY21/query/00420_c40s3785_301.png', '/workspace/datasets/AY21/query/00511_c47s5418_161.png', '/workspace/datasets/AY21/query/00563_c26s4541_1621.png', '/workspace/datasets/AY21/query/00443_c30s3746_141.png', '/workspace/datasets/AY21/query/00340_c15s2906_1281.png', '/workspace/datasets/AY21/query/00393_c34s3374_501.png', '/workspace/datasets/AY21/query/00500_c42s4226_3181.png', '/workspace/datasets/AY21/query/00466_c63s4978_61.png']
    # query_path2 = ['/workspace/datasets/AY20/Night/IR/query/00164_c49s421_1161.png', '/workspace/datasets/AY20/Night/IR/query/00131_c21s172_1481.png', '/workspace/datasets/AY20/Night/IR/query/00244_c14s2527_281.png', '/workspace/datasets/AY20/Night/IR/query/00231_c47s2556_921.png', '/workspace/datasets/AY20/Night/IR/query/00216_c17s2539_641.png', '/workspace/datasets/AY20/Night/IR/query/00181_c25s1954_261.png', '/workspace/datasets/AY20/Night/IR/query/00160_c39s400_841.png', '/workspace/datasets/AY20/Night/IR/query/10126_c11s153_1401.png', '/workspace/datasets/AY20/Night/IR/query/00152_c40s406_141.png', '/workspace/datasets/AY20/Night/IR/query/00290_c22s2215_241.png', '/workspace/datasets/AY20/Night/IR/query/00133_c14s124_1841.png', '/workspace/datasets/AY20/Night/IR/query/00132_c27s186_1541.png', '/workspace/datasets/AY20/Night/IR/query/00156_c36s394_601.png', '/workspace/datasets/AY20/Night/IR/query/00214_c20s2505_141.png', '/workspace/datasets/AY20/Night/IR/query/00226_c44s2572_1341.png', '/workspace/datasets/AY20/Night/IR/query/00323_c02s1636_381.png', '/workspace/datasets/AY20/Night/IR/query/00131_c29s188_3041.png', '/workspace/datasets/AY20/Night/IR/query/10213_c20s2502_141.png', '/workspace/datasets/AY20/Night/IR/query/00205_c30s1975_661.png', '/workspace/datasets/AY20/Night/IR/query/10126_c11s153_1301.png', '/workspace/datasets/AY20/Night/IR/query/00289_c01s2185_581.png', '/workspace/datasets/AY20/Night/IR/query/00214_c48s2507_1861.png', '/workspace/datasets/AY20/Night/IR/query/00302_c02s1634_341.png', '/workspace/datasets/AY20/Night/IR/query/00245_c14s1074_21.png', '/workspace/datasets/AY20/Night/IR/query/00123_c27s186_1161.png', '/workspace/datasets/AY20/Night/IR/query/00190_c22s1992_1321.png', '/workspace/datasets/AY20/Night/IR/query/00290_c42s2238_881.png', '/workspace/datasets/AY20/Night/IR/query/00264_c37s1182_181.png', '/workspace/datasets/AY20/Night/IR/query/00244_c51s2560_1781.png', '/workspace/datasets/AY20/Night/IR/query/00188_c30s1981_161.png']
    # query_path3 = ['/workspace/datasets/AY20/Night/RGB/query/00148_c17s131_2861.png', '/workspace/datasets/AY20/Night/RGB/query/00127_c12s119_981.png', '/workspace/datasets/AY20/Night/RGB/query/00122_c25s141_201.png', '/workspace/datasets/AY20/Night/RGB/query/00133_c21s136_2081.png', '/workspace/datasets/AY20/Night/RGB/query/00132_c27s147_2141.png', '/workspace/datasets/AY20/Night/RGB/query/00132_c14s124_1741.png', '/workspace/datasets/AY20/Night/RGB/query/00128_c29s150_1661.png', '/workspace/datasets/AY20/Night/RGB/query/00126_c29s152_981.png', '/workspace/datasets/AY20/Night/RGB/query/00149_c22s137_241.png', '/workspace/datasets/AY20/Night/RGB/query/00144_c21s136_661.png', '/workspace/datasets/AY20/Night/RGB/query/00125_c14s125_421.png', '/workspace/datasets/AY20/Night/RGB/query/00146_c14s125_2781.png', '/workspace/datasets/AY20/Night/RGB/query/00138_c15s127_1201.png', '/workspace/datasets/AY20/Night/RGB/query/00122_c27s146_341.png', '/workspace/datasets/AY20/Night/RGB/query/00122_c26s144_901.png', '/workspace/datasets/AY20/Night/RGB/query/00125_c15s128_1181.png', '/workspace/datasets/AY20/Night/RGB/query/00132_c13s122_1861.png', '/workspace/datasets/AY20/Night/RGB/query/00132_c21s136_1861.png', '/workspace/datasets/AY20/Night/RGB/query/00142_c22s139_1121.png', '/workspace/datasets/AY20/Night/RGB/query/00146_c22s139_901.png', '/workspace/datasets/AY20/Night/RGB/query/00122_c25s141_461.png', '/workspace/datasets/AY20/Night/RGB/query/00145_c13s122_1541.png', '/workspace/datasets/AY20/Night/RGB/query/00131_c22s139_2401.png', '/workspace/datasets/AY20/Night/RGB/query/10126_c17s131_2441.png', '/workspace/datasets/AY20/Night/RGB/query/00148_c26s142_521.png', '/workspace/datasets/AY20/Night/RGB/query/00150_c26s145_1621.png', '/workspace/datasets/AY20/Night/RGB/query/00138_c14s124_2021.png', '/workspace/datasets/AY20/Night/RGB/query/00132_c13s122_2141.png', '/workspace/datasets/AY20/Night/RGB/query/00139_c15s129_2101.png', '/workspace/datasets/AY20/Night/RGB/query/00145_c29s149_1141.png']
    # query_paths = query_path1 + query_path2 + query_path3
    for query_path in query_paths:
        # 실행
        size = parse_wh(args.size, default=(128, 256))
        # result = do_inference_single_query(cfg, model, val_loader, query_path, num_query, topk=args.topk, root='/workspace/datasets/NightReID/')
        # 미리 추출된 피처를 이용해 검색
        result = search_from_features(cfg, query_path, all_feats, all_pids, all_camids, all_paths, topk=args.topk)

        # 스트립 저장
        top_paths = result["top_paths"]
        top_idx   = result["top_idx"]
        q_pid     = result["q_pid"]
        q_camid   = result["q_camid"]
        g_pids    = result["g_pids"]
        g_camids  = result["g_camids"]
        scores_top = result["top_scores"]

        g_pids_top   = [g_pids[i] for i in top_idx]
        g_camids_top = [g_camids[i] for i in top_idx]

        save_path = os.path.join(cfg.OUTPUT_DIR, 'rank10_vis', f"rank_10_vis_{os.path.basename(query_path).split('.')[0]}.jpg")
        make_rank_strip(
            save_path=save_path,
            query_path=result["query_path"],
            top_paths=top_paths,
            q_pid=q_pid,
            g_pids_top=g_pids_top,
            # 동일 카메라 제외 규칙 쓰려면 아래 두 줄 활성화
            q_camid=q_camid,
            g_camids_top=g_camids_top,
            size=size,
            gap=6,
            draw_rank_text=True,
            scores_top=scores_top,
            score_fmt="{:.3f}"
        )
        print("[DONE] Saved:", save_path)


if __name__ == "__main__":
    main()
