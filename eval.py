#!/usr/bin/env python3
"""评估脚本 - 支持单文件和批量评估 submission

使用 private 数据集中的 gold_submission.csv 作为真实标签
"""

import pandas as pd
import glob
import argparse
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def evaluate(submission_path: str, gold_path: str, verbose: bool = True):
    """评估 submission 文件，返回指标字典

    Args:
        submission_path: submission.csv 路径
        gold_path: gold_submission.csv 路径
        verbose: 是否打印详细信息
    """
    submission = pd.read_csv(submission_path)
    gold = pd.read_csv(gold_path)

    if 'Insult' not in gold.columns or 'Insult' not in submission.columns:
        if verbose:
            print("错误: 缺少 'Insult' 列")
        return None

    min_len = min(len(submission), len(gold))
    y_pred_proba = submission['Insult'].values[:min_len]
    y_true = gold['Insult'].values[:min_len]

    auc = roc_auc_score(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'auc': auc,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'n_samples': min_len
    }

    if verbose:
        print(f"文件: {Path(submission_path).name}")
        print(f"  样本数: {min_len}, 预测范围: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
        print(f"  AUC: {auc:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    return metrics


def evaluate_batch(submission_dir: str, gold_path: str, output_json: bool = False):
    """批量评估目录下的所有 submission_*.csv 文件"""
    submission_files = sorted(Path(submission_dir).glob("submission_*.csv"))

    if not submission_files:
        print(f"错误: {submission_dir} 中没有找到 submission_*.csv 文件")
        return

    print(f"找到 {len(submission_files)} 个 submission 文件\n")

    results = []
    for i, file in enumerate(submission_files, 1):
        print(f"[{i}/{len(submission_files)}]", end=" ")
        metrics = evaluate(str(file), gold_path, verbose=True)
        if metrics:
            metrics['filename'] = file.name
            results.append(metrics)

    # 统计
    if results:
        print("\n" + "=" * 60)
        print("统计结果")
        print("=" * 60)

        for metric in ['auc', 'f1', 'accuracy']:
            values = [r[metric] for r in results]
            series = pd.Series(values)
            print(f"\n{metric.upper()}: max={max(values):.4f}, min={min(values):.4f}, "
                  f"mean={series.mean():.4f}, std={series.std():.4f}")

        best = max(results, key=lambda x: x['auc'])
        print(f"\n最佳提交: {best['filename']} (AUC: {best['auc']:.4f})")
        print("=" * 60)

        if output_json:
            output_path = Path(submission_dir) / "evaluation_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="评估 submission 文件")
    parser.add_argument('--exp_name', type=str, default='ml_master',
                        help='实验名称 (默认: ml_master)')
    parser.add_argument('--submission_dir', type=str,
                        help='submission 目录路径')
    parser.add_argument('--batch', action='store_true',
                        help='批量模式：评估目录下所有 submission_*.csv')
    parser.add_argument('--save', action='store_true',
                        help='批量模式下保存结果到 JSON')

    args = parser.parse_args()

    gold_path = Path(f"playground/{args.exp_name}/data/private/gold_submission.csv")

    if args.batch:
        if not args.submission_dir:
            # 查找最新的 submission 目录
            runs = glob.glob(f"runs/{args.exp_name}_*/workspaces/task_0/submission")
            args.submission_dir = sorted(runs)[-1] if runs else None

        if not args.submission_dir:
            print("错误: 未找到 submission 目录")
            return

        evaluate_batch(args.submission_dir, str(gold_path), args.save)

    else:
        # 单文件模式
        if args.submission_dir:
            submission_path = Path(args.submission_dir) / "workspaces/task_0/best_submission/submission.csv"
        else:
            runs = glob.glob(f"runs/{args.exp_name}_*/workspaces/task_0/best_submission/submission.csv")
            submission_path = sorted(runs)[-1] if runs else None

        if not submission_path:
            print("错误: 未找到 submission 文件")
            return

        print(f"评估文件: {submission_path}")
        print(f"标准标签: {gold_path}\n")
        evaluate(str(submission_path), str(gold_path), verbose=True)


if __name__ == "__main__":
    main()