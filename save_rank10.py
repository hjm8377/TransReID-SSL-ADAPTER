import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont

from utils.metrics import euclidean_distance, re_ranking, eval_func
from utils.logger import setup_logger
from config import cfg
from datasets import make_dataloader
from model import make_model

def normpath_case(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))

def parse_wh(s: str, default=(128, 256)):
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    
    except Exception:
        return default
    

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
        # 하단 점수 (유클리드 거리)
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

def main():
    pass
# pth 선택.
# 테스트셋 선택
