#!/usr/bin/env python3
"""
CLI metrics reporter for the wakeword model.
Evaluates the best saved model (best_wakeword_model.pth) on the validation split
and prints metrics to stdout while saving an HTML report with ROC and Precision-Recall curves.

Usage (PowerShell):
  python scripts/report_metrics.py `
    --positive_dir ./positive_dataset `
    --negative_dir ./negative_dataset `
    --background_dir ./background_noise `
    --batch_size 32 `
    --val_split 0.2 `
    --test_split 0.1 `
    --output evaluation_report.html

Requires CUDA (enforced by gradio_app import).
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    print("Plotly import error:", e, file=sys.stderr)
    raise

# Import app (enforces CUDA-only policy)
import gradio_app as ga
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)


def evaluate(app: ga.WakewordTrainingApp):
    if not os.path.exists('best_wakeword_model.pth'):
        raise FileNotFoundError("best_wakeword_model.pth bulunamadı. Önce eğitimi tamamlayın ve modeli kaydedin.")

    ckpt = torch.load('best_wakeword_model.pth', map_location=app.device)
    app.model.load_state_dict(ckpt['model_state_dict'])
    app.model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for data, target in app.val_loader:
            data = data.to(app.device)
            target = target.to(app.device).squeeze()
            logits = app.model(data)
            probs = torch.softmax(logits, dim=1)[:, 1]
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(target.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = [0, 1], [0, 1], float('nan')

    try:
        pr_prec, pr_rec, pr_th = precision_recall_curve(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    except Exception:
        pr_prec, pr_rec, pr_th, ap = [1, 0], [0, 1], [0.5], float('nan')

    try:
        brier = brier_score_loss(all_labels, all_probs)
    except Exception:
        brier = float('nan')

    # Best threshold by F1
    best_thr, best_f1 = 0.5, f1
    try:
        for thr in pr_th:
            preds_thr = [1 if p >= thr else 0 for p in all_probs]
            f1_thr = f1_score(all_labels, preds_thr, zero_division=0)
            if f1_thr > best_f1:
                best_f1, best_thr = f1_thr, float(thr)
    except Exception:
        pass

    # Figure
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Confusion Matrix', 'Metrics', 'ROC Curve', 'Precision-Recall'
    ))

    fig.add_trace(go.Heatmap(z=cm, colorscale='Blues',
                             x=['Negative', 'Wakeword'], y=['Negative', 'Wakeword'],
                             showscale=True), row=1, col=1)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'AP', 'Brier']
    values = [acc, prec, rec, f1, roc_auc, ap, brier]
    fig.add_trace(go.Bar(x=metrics, y=values, name='Metrics', marker_color='lightblue'), row=1, col=2)

    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=pr_rec, y=pr_prec, name=f'PR (AP={ap:.3f})', line=dict(color='green')), row=2, col=2)

    fig.update_layout(height=800, showlegend=True, title_text='Model Evaluation Report')

    summary = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'average_precision': ap,
        'brier_score': brier,
        'best_threshold_f1': best_thr,
        'best_threshold_f1_value': best_f1,
        'confusion_matrix': cm.tolist(),
    }

    return summary, fig


def main():
    parser = argparse.ArgumentParser(description='Wakeword model metrics reporter (CUDA-only).')
    parser.add_argument('--positive_dir', type=str, default='./positive_dataset')
    parser.add_argument('--negative_dir', type=str, default='./negative_dataset')
    parser.add_argument('--background_dir', type=str, default='./background_noise')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='evaluation_report.html')
    parser.add_argument('--json', type=str, default='evaluation_metrics.json', help='JSON output file for metrics summary')
    args = parser.parse_args()

    app = ga.WakewordTrainingApp()
    status, train_len, val_len = app.load_data(
        args.positive_dir, args.negative_dir, args.background_dir,
        args.batch_size, args.val_split, args.test_split
    )
    print(status)

    summary, fig = evaluate(app)

    # Save report
    try:
        fig.write_html(args.output)
        print(f"HTML raporu kaydedildi: {args.output}")
    except Exception as e:
        print("HTML kayıt hatası:", e, file=sys.stderr)

    try:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"JSON metrik özeti kaydedildi: {args.json}")
    except Exception as e:
        print("JSON kayıt hatası:", e, file=sys.stderr)

    # Print metrics
    print("\n📊 MODEL TEST SONUÇLARI")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall: {summary['recall']:.4f}")
    print(f"F1-Score: {summary['f1']:.4f}")
    print(f"ROC AUC: {summary['roc_auc']:.4f}")
    print(f"Average Precision (AP): {summary['average_precision']:.4f}")
    print(f"Brier Score: {summary['brier_score']:.4f}")
    print(f"En İyi Eşik (F1): {summary['best_threshold_f1']:.3f} (F1={summary['best_threshold_f1_value']:.4f})")


if __name__ == '__main__':
    main()
