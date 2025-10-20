import argparse

import numpy as np
import scanpy as sc

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN, refine
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import FilterGenesMatch
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed
from dance.utils.matrix import pairwise_distance
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
from dance.typing import Sequence


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "spatial", overwrite=True)
class SpaGCNGraph(BaseTransform):

    _DISPLAY_ATTRS = ("alpha", "beta")

    def __init__(self, alpha, beta, *, channels: Sequence[str] = ("spatial", "spatial_pixel", "image"),
                 channel_types: Sequence[str] = ("obsm", "obsm", "uns"), **kwargs):
        """Initialize SpaGCNGraph.

        Parameters
        ----------
        alpha
            Controls the color scale.
        beta
            Controls the range of the neighborhood when calculating grey values for one spot.

        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.channels = channels
        self.channel_types = channel_types

    def __call__(self, data):
        xy = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])
        img = data.get_feature(return_type="numpy", channel=self.channels[2], channel_type=self.channel_types[2])
        self.logger.info("Start calculating the adjacency matrix using the histology image")
        g = np.zeros((xy.shape[0], 3))
        beta_half = round(self.beta / 2)
        x_lim, y_lim = img.shape[:2]
        for i, (x_pixel, y_pixel) in enumerate(xy_pixel):
            top = max(0, x_pixel - beta_half)
            left = max(0, y_pixel - beta_half)
            bottom = min(x_lim, x_pixel + beta_half + 1)
            right = min(y_lim, y_pixel + beta_half + 1)
            local_view = img[top:bottom, left:right]
            g[i] = np.mean(local_view, axis=(0, 1))
        g_var = g.var(0)
        self.logger.info(f"Variances of c0, c1, c2 = {g_var}")

        z = (g * g_var).sum(1, keepdims=True) / g_var.sum()
        z = (z - z.mean()) / z.std()
        z *= xy.std(0).max() * self.alpha

        xyz = np.hstack((xy, z)).astype(np.float32)
        self.logger.info(f"Varirances of x, y, z = {xyz.var(0)}")
        data.data.obsp[self.out] = pairwise_distance(xyz, dist_func_id=0)

        return data


@register_preprocessor("graph", "spatial", overwrite=True)
class SpaGCNGraph2D(BaseTransform):

    def __init__(self, *, channel: str = "spatial_pixel", **kwargs):
        super().__init__(**kwargs)

        self.channel = channel

    def __call__(self, data):
        x = data.get_feature(channel=self.channel, channel_type="obsm", return_type="numpy")
        data.data.obsp[self.out] = pairwise_distance(x.astype(np.float32), dist_func_id=0)
        return data

# EVOLVE-BLOCK-END

def get_preprocessing_pipeline(alpha: float = 1, beta: int = 49, dim: int = 50, log_level: LogLevel = "INFO"):
    return Compose(
        FilterGenesMatch(prefixes=["ERCC", "MT-"]),
        AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
        AnnDataTransform(sc.pp.log1p),
        SpaGCNGraph(alpha=alpha, beta=beta),
        SpaGCNGraph2D(),
        CellPCA(n_components=dim),
        SetConfig({
            "feature_channel": ["CellPCA", "SpaGCNGraph", "SpaGCNGraph2D"],
            "feature_channel_type": ["obsm", "obsp", "obsp"],
            "label_channel": "label",
            "label_channel_type": "obs"
        }),
        log_level=log_level,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--beta", type=int, default=49, help="")
    parser.add_argument("--alpha", type=int, default=1, help="")
    parser.add_argument("--p", type=float, default=0.05,
                        help="percentage of total expression contributed by neighborhoods.")
    parser.add_argument("--l", type=float, default=0.5, help="the parameter to control percentage p.")
    parser.add_argument("--start", type=float, default=0.01, help="starting value for searching l.")
    parser.add_argument("--end", type=float, default=1000, help="ending value for searching l.")
    parser.add_argument("--tol", type=float, default=5e-3, help="tolerant value for searching l.")
    parser.add_argument("--max_run", type=int, default=200, help="max runs.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--n_clusters", type=int, default=7, help="the number of clusters")
    parser.add_argument("--step", type=float, default=0.1, help="")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--device", default="cpu", help="Computation device.")
    parser.add_argument("--seed", type=int, default=100, help="")
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGCN(device=args.device)
        preprocessing_pipeline = get_preprocessing_pipeline(alpha=args.alpha, beta=args.beta)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
        print(data)
        raise Exception("Stop here")
        (x, adj, adj_2d), y = data.get_train_data()

        # Train and evaluate model
        l = model.search_l(args.p, adj, start=args.start, end=args.end, tol=args.tol, max_run=args.max_run)
        model.set_l(l)
        res = model.search_set_res((x, adj), l=l, target_num=args.n_clusters, start=0.4, step=args.step, tol=args.tol,
                                   lr=args.lr, epochs=args.epochs, max_run=args.max_run)

        pred = model.fit_predict((x, adj), init_spa=True, init="louvain", tol=args.tol, lr=args.lr, epochs=args.epochs,
                                 res=res)
        score = model.default_score_func(y, pred)
        
        refined_pred = refine(sample_id=data.data.obs_names.tolist(), pred=pred.tolist(), dis=adj_2d, shape="hexagon")
        score_refined = model.default_score_func(y, refined_pred)
        
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(calculate_unified_scores({
            "silhouette": silhouette_score(x, refined_pred),
            "calinski_harabasz": calinski_harabasz_score(x, refined_pred),
            "davies_bouldin": davies_bouldin_score(x, refined_pred)
        }))
        print(f"ARI: {score:.4f}")

        scores.append(score_refined)
        print(f"ARI (refined): {score_refined:.4f}")
    print(f"SpaGCN {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
""" To reproduce SpaGCN on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spagcn.py --sample_number 151673 --lr 0.1

human dorsolateral prefrontal cortex sample 151676:
$ python spagcn.py --sample_number 151676 --lr 0.02

human dorsolateral prefrontal cortex sample 151507:
$ python spagcn.py --sample_number 151507 --lr 0.009
"""
