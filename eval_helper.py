import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import precision_recall_fscore_support
import torch
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jiwer

def compute_velocity_magnitude(joints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=1)

def compute_motion_peaks(joints: np.ndarray, threshold_percentile: float = 70.0) -> np.ndarray:
    velocity = compute_velocity_magnitude(joints)
    peaks, _ = find_peaks(velocity, height=np.percentile(velocity, threshold_percentile))
    return peaks

def beat_tracking_f1(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    pred_peaks = compute_motion_peaks(pred_joints)
    gt_peaks = compute_motion_peaks(gt_joints)

    pred_mask = np.zeros(len(pred_joints), dtype=int)
    gt_mask = np.zeros(len(gt_joints), dtype=int)
    pred_mask[pred_peaks] = 1
    gt_mask[gt_peaks] = 1

    min_len = min(len(pred_mask), len(gt_mask))
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_mask[:min_len], pred_mask[:min_len], average='binary', zero_division=0)
    return f1

def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    assert pred.shape == gt.shape
    return np.linalg.norm(pred - gt, axis=-1).mean()

def calculate_bt_and_mpjpe(references, hypotheses):
    bt_f1_scores = []
    mpjpe_scores = []

    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        #_, ref_max_idx = torch.max(ref[:, -1], 0)
        #_, hyp_max_idx = torch.max(hyp[:, -1], 0)
        ref_np = ref.cpu().numpy().reshape(-1, 75, 3)
        hyp_np = hyp.cpu().numpy().reshape(-1, 75, 3)

        min_len = min(ref_np.shape[0], hyp_np.shape[0])
        ref_np = ref_np[:min_len]
        hyp_np = hyp_np[:min_len]

        bt_f1_scores.append(beat_tracking_f1(hyp_np, ref_np))
        mpjpe_scores.append(mpjpe(hyp_np, ref_np))

    return bt_f1_scores, mpjpe_scores

def compute_bleu_scores(references: List[str], hypotheses: List[str]) -> (float, float):
    smoothie = SmoothingFunction().method4
    bleu_1_scores = []
    bleu_4_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_1_scores.append(bleu_1)
        bleu_4_scores.append(bleu_4)
    return np.mean(bleu_1_scores), np.mean(bleu_4_scores)


def compute_rouge(references: List[str], hypotheses: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_score = 0.0
    valid_samples = 0

    for ref, hyp in zip(references, hypotheses):
        try:
            if ref.strip() == "" or hyp.strip() == "":
                continue
            score = scorer.score(ref.strip(), hyp.strip())
            total_score += score["rougeL"].fmeasure
            valid_samples += 1
        except Exception as e:
            continue  # Skip problematic lines

    return total_score / valid_samples if valid_samples > 0 else 0.0



def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    return jiwer.wer(references, hypotheses)


from scipy.linalg import sqrtm

def compute_fid(fake_feats: np.ndarray, real_feats: np.ndarray) -> float:
    if fake_feats.shape[0] < 10 or real_feats.shape[0] < 10:
        print("Too few samples for reliable FID. Returning dummy value.")
        return 0.0

    mu1, sigma1 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    mu2, sigma2 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)

    diff = mu1 - mu2

    # Matrix square root
    try:
        covmean = sqrtm(sigma1.dot(sigma2))

        # Check for numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            print("Warning: Complex sqrtm result detected. Taking real part.")

        fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(np.real(fid))

    except Exception as e:
        print(f"FID computation failed due to: {e}")
        return 0.0


def mpjae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    joint_error = torch.abs(predictions - targets)
    return torch.mean(joint_error).item()


