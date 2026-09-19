"""Object-only inference and within-class AUROC ranking diagnostics."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import math
from pathlib import Path


def diagnose(rows):
    """Count discordant opposite-label pairs; ties contribute half a loss."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['checkpoint'], row['dataset'], row['class_name'])].append(row)
    summaries = []
    for (checkpoint, dataset, class_name), samples in groups.items():
        scores = {
            label: sorted(float(r['object_score']) for r in samples if int(r['object_label']) == label)
            for label in (0, 1)
        }
        n0, n1 = len(scores[0]), len(scores[1])
        pair_count = n0 * n1
        total_loss = 0.0
        for row in samples:
            label, score = int(row['object_label']), float(row['object_score'])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score: {row['sample_path']}")
            opposite = scores[1 - label]
            left, right = bisect_left(opposite, score), bisect_right(opposite, score)
            wrong = len(opposite) - right if label == 1 else left
            ties = right - left
            loss = wrong + 0.5 * ties
            row.update(
                misordered_pairs=wrong, tied_pairs=ties,
                pair_loss=loss,
                opposite_count=len(opposite),
                error_rate=loss / len(opposite) if opposite else float('nan'),
                auroc_loss_contribution=loss / pair_count if pair_count else float('nan'),
            )
            if label == 1:
                total_loss += loss
        auroc = 1 - total_loss / pair_count if pair_count else float('nan')
        for row in samples:
            remaining_pairs = pair_count - row['opposite_count']
            without = 1 - (total_loss - row['pair_loss']) / remaining_pairs if remaining_pairs else float('nan')
            row['class_auroc'] = auroc
            row['auroc_without_sample'] = without
            row['auroc_delta_if_removed'] = without - auroc
        summaries.append(dict(checkpoint=checkpoint, dataset=dataset, class_name=class_name,
                              normal_count=n0, anomaly_count=n1, object_auroc=auroc))
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def infer(args):
    import torch
    from tqdm import tqdm
    from data.anomaly_datasets import PointCloudDataset
    from test import build_models, find_checkpoints
    from utils.utils import compute_global_anomaly_score, compute_patch_scores

    if not args.ckpt_root:
        raise ValueError('--ckpt_root is required for inference')
    checkpoints = find_checkpoints(args.ckpt_root)
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found: {args.ckpt_root}')
    dataset = PointCloudDataset(args.data_root, split='test', classes=args.classes,
                                dataset_name=args.dataset_name)
    if not len(dataset):
        raise FileNotFoundError(f'No test samples: {args.data_root}')
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    encoder, embedder, geo_encoder, prompt_learner = build_models(args, device)
    rows = []
    with torch.no_grad():
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            for module, key in ((embedder, 'patch_feature_embedder'),
                                (geo_encoder, 'patch_geo_encoder'),
                                (prompt_learner, 'prompt_learner')):
                module.load_state_dict(checkpoint[key])
                module.eval()
            _, tokens = prompt_learner()
            normal, anomaly = encoder.encode_text_from_tokens(tokens)
            for idx in tqdm(range(len(dataset)), desc=f'Object eval {checkpoint_path}'):
                sample = dataset[idx]
                points = sample['points'].unsqueeze(0).to(device)
                features = encoder.encode_pointcloud(points, return_intermediate=True)
                layers = [features['layer_feats'][layer][:, 1:, :].contiguous()
                          for layer in args.return_layers]
                geometry = geo_encoder(points, features['patch_idx'])
                patches = embedder(layers, geometry, features['global'])
                _, patch_probs = compute_patch_scores(patches, normal, anomaly)
                _, global_probs = compute_global_anomaly_score(features['concat'], normal, anomaly)
                k = max(1, int(patch_probs.shape[1] * args.topk_ratio))
                local_probs = torch.topk(patch_probs, k=k, dim=1).values.mean(dim=1)
                combined = args.global_alpha * global_probs + (1 - args.global_alpha) * local_probs
                rows.append(dict(
                    checkpoint=str(Path(checkpoint_path).resolve()), dataset=args.dataset_name,
                    class_name=dataset.PRESETS[args.dataset_name][sample['category']],
                    sample_path=str(sample['path']), object_label=int(sample['labels'].sum() > 0),
                    object_score=combined.item(), global_score=global_probs.item(),
                    local_topk_score=local_probs.item(), local_max_score=patch_probs.max().item(),
                    global_alpha=args.global_alpha, topk_ratio=args.topk_ratio,
                ))
            # Keep completed checkpoint predictions even if subsequent inference fails.
            write_csv(Path(args.output_dir) / 'test_object_scores.csv', rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--ckpt_root', help='Checkpoint file or directory containing best.pth files')
    source.add_argument('--scores_csv', help='Analyze existing test_object_scores.csv without inference')
    parser.add_argument('--output_dir', default='./outputs/object_auroc')
    parser.add_argument('--train_class', help='Exclude this class from the printed macro average')
    parser.add_argument('--data_root', default='./data/Real3D-AD-2048-npz')
    parser.add_argument('--model_path', default='./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt')
    parser.add_argument('--dataset_name', choices=['Real3D', 'AnomalyShapeNet'], default='Real3D')
    parser.add_argument('--classes', nargs='+')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--return_layers', type=int, nargs='+', default=[2, 5, 8, 11])
    parser.add_argument('--depth', type=int, default=9)
    parser.add_argument('--n_ctx', type=int, default=8)
    parser.add_argument('--t_n_ctx', type=int, default=4)
    parser.add_argument('--geo_dim', type=int, default=33)
    parser.add_argument('--global_alpha', type=float, default=0.5)
    parser.add_argument('--topk_ratio', type=float, default=0.2)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if not 0 <= args.global_alpha <= 1 or not 0 < args.topk_ratio <= 1:
        parser.error('Require 0 <= global_alpha <= 1 and 0 < topk_ratio <= 1')
    return args


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.scores_csv:
        with open(args.scores_csv, newline='') as stream:
            rows = list(csv.DictReader(stream))
        identities = [(r['checkpoint'], r['dataset'], r['class_name'], r['sample_path']) for r in rows]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate samples in CSV; select one evaluation run before analysis.')
    else:
        rows = infer(args)
    if not rows:
        raise ValueError('No sample scores to analyze')
    summaries = diagnose(rows)
    ranked = sorted(rows, key=lambda r: (r['checkpoint'], r['dataset'], r['class_name'], -r['pair_loss']))
    write_csv(output / 'object_sample_diagnostics.csv', ranked)
    write_csv(output / 'object_auroc_per_class.csv', summaries)
    averages = defaultdict(list)
    for row in summaries:
        print(f"{row['checkpoint']} | {row['class_name']}: AUROC={row['object_auroc']:.6f}")
        train_class = args.train_class or Path(row['checkpoint']).parent.parent.name
        if row['class_name'] != train_class and math.isfinite(row['object_auroc']):
            averages[(row['checkpoint'], row['dataset'])].append(row['object_auroc'])
    for (checkpoint, dataset), values in averages.items():
        print(f'{checkpoint} | {dataset} macro AUROC (excluding train class): {sum(values) / len(values):.6f}')
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
