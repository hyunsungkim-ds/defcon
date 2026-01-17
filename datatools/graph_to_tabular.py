import argparse
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Data


def graph_to_tabular(
    graphs: List[Optional[Data]],
    labels: torch.Tensor,
    concat_intent: bool = True,
    padding_value: float = -1.0,
) -> torch.Tensor:
    """
    Build tabular features by concatenating possessor node features and intent node features.

    Possessor node is detected via graph.x[:, 13] == 1.
    Intent node index is stored in labels[:, 5].
    """
    if len(graphs) != len(labels):
        raise ValueError(f"Graph/list length mismatch: {len(graphs)} graphs vs {len(labels)} labels.")

    feature_dim = graphs[0].x.shape[1]
    tabular_rows: List[torch.Tensor] = []

    for graph, label in zip(graphs, labels):
        if graph is None or graph.x is None:
            padding_dim = feature_dim * 2 if concat_intent else feature_dim
            tabular_rows.append(torch.full((padding_dim,), padding_value, dtype=torch.float32))
            continue

        node_features = graph.x
        poss_mask = node_features[:, 13] == 1
        poss_indices = torch.nonzero(poss_mask, as_tuple=False).view(-1)

        if poss_indices.numel() == 0:
            poss_feature = torch.full((feature_dim,), padding_value, dtype=node_features.dtype)
        else:
            poss_feature = node_features[poss_indices[0]]

        if concat_intent:
            intent_idx = int(label[5].item())
            if intent_idx < 0 or intent_idx >= node_features.shape[0]:
                intent_feature = torch.full((feature_dim,), padding_value, dtype=node_features.dtype)
            else:
                intent_feature = node_features[intent_idx]
            tabular_rows.append(torch.cat([poss_feature, intent_feature], dim=0))
        else:
            tabular_rows.append(poss_feature)

    return torch.stack(tabular_rows, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert saved graph features to tabular features.")
    parser.add_argument("--action_type", type=str, default="action", choices=["action", "augmented_shot"])
    parser.add_argument("--return_type", type=str, default="disc_0.9", help="Way of defining future xG.")
    args = parser.parse_args()

    feature_dir = Path(f"data/ajax/features/{args.action_type}_graphs")
    output_dir = Path(f"data/ajax/features/{args.action_type}_tabular")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.action_type == "all":
        label_dir = Path(f"data/ajax/features/action_labels_{args.return_type}")
    else:
        label_dir = Path(f"data/ajax/features/{args.action_type}_labels")

    label_paths = sorted(label_dir.glob("*.pt"))
    if not label_paths:
        raise FileNotFoundError(f"No label files found in {label_dir.resolve()}")

    concat_intent = "shot" not in args.action_type

    for label_path in label_paths:
        match_id = label_path.stem
        graph_path = feature_dir / f"{match_id}.pt"
        if not graph_path.exists():
            print(f"[Skip] Graph file missing for {match_id} ({graph_path})")
            continue

        labels = torch.load(label_path, map_location="cpu")
        graphs = torch.load(graph_path, map_location="cpu")

        tabular_features = graph_to_tabular(graphs, labels, concat_intent)
        out_path = output_dir / f"{match_id}.pt"
        torch.save(tabular_features, out_path)
        print(f"[Saved] {match_id}: {tabular_features.shape} -> {out_path}")
