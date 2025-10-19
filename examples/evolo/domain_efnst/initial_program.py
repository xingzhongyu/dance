import argparse
import math
import os
from pathlib import Path
import random

from PIL import Image
from efficientnet_pytorch import EfficientNet
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import BatchNorm, Sequential
from torchvision import transforms
import torchvision.transforms as transforms
from tqdm import tqdm

from dance import logger
from dance.data.base import Data
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.EfNST import (
    EfNSTAugmentTransform,
    EfNSTConcatgTransform,
    EfNSTGraphTransform,
    EfNSTImageTransform,
    EfNsSTRunner,
)
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import (
    FilterGenesPercentile,
    HighlyVariableGenesLogarithmizedByTopGenes,
)
from dance.transforms.misc import Compose, SetConfig
from dance.utils import set_seed
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
from torch_sparse import SparseTensor
"""Created on Tue Jan 23 18:54:08 2024.

@author: lenovo

"""
# -*- coding: utf-8 -*-

# Standard library imports


# typing.Literal for compatibility
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):

            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values, )
                return type("Literal_", (Literal, ), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass




# EVOLVE-BLOCK-START
class SpatialImageDataset(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.spot_names = paths.index.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spot_name = self.spot_names[idx]
        img_path = self.paths.iloc[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels

        if self.transform:
            image = self.transform(image)

        return image, spot_name
def extract_features_batch(adata, model, device, batch_size=64, num_workers=4):
    """Extracts features from image slices in an efficient, batched manner.

    Args:
        adata (anndata.AnnData): AnnData object with 'slices_path' in obs.
        model (torch.nn.Module): The pre-trained PyTorch model.
        device (torch.device): The device to run the model on (e.g., torch.device('cuda')).
        batch_size (int): Number of images to process in one batch.
        num_workers (int): Number of CPU workers for loading data in parallel.

    Returns:
        pd.DataFrame: A DataFrame where rows are spots and columns are features.

    """
    # 1. Set model to evaluation mode and disable gradient computation
    model.eval()

    # 2. Define image preprocessing pipeline
    # Consistent with your original steps: resize -> to tensor -> convert to float
    # torchvision.transforms can efficiently complete these operations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # This automatically converts [0, 255] PIL Image to [0.0, 1.0] FloatTensor
        # If your model requires specific normalization, add it here
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Create Dataset and DataLoader
    # shuffle=False is CRITICAL to keep the order of spots
    dataset = SpatialImageDataset(paths=adata.obs['slices_path'], transform=preprocess)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Speeds up CPU to GPU transfer
    )

    all_features = []
    all_spot_names = []

    # 4. Efficient batch processing loop
    with torch.no_grad():  # Very important! Disable gradient computation to save memory and computation
        for image_batch, spot_name_batch in tqdm(data_loader, desc="Extracting features"):
            # DataLoader has already packed a batch of images into a (B, C, H, W) tensor
            # B = batch_size, C=3, H=224, W=224

            # Move the entire batch of data to GPU at once
            image_batch = image_batch.to(device)

            # Perform inference on the entire batch
            result_batch = model(image_batch)

            # Move results back to CPU and store
            # .cpu() returns a tensor on CPU, .numpy() converts it to numpy array
            all_features.append(result_batch.cpu().numpy())
            all_spot_names.extend(list(spot_name_batch))

    # 5. Merge all batch results and create DataFrame (execute only once!)
    final_features = np.concatenate(all_features, axis=0)
    feat_df = pd.DataFrame(final_features, index=all_spot_names)

    return adata, feat_df

class Image_Feature:

    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='efficientnet-b0',
        verbose=False,
        seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def efficientNet_model(self):
        efficientnet_versions = {
            'efficientnet-b0': 'efficientnet-b0',
            'efficientnet-b1': 'efficientnet-b1',
            'efficientnet-b2': 'efficientnet-b2',
            'efficientnet-b3': 'efficientnet-b3',
            'efficientnet-b4': 'efficientnet-b4',
            'efficientnet-b5': 'efficientnet-b5',
            'efficientnet-b6': 'efficientnet-b6',
            'efficientnet-b7': 'efficientnet-b7',
        }
        if self.cnnType in efficientnet_versions:
            model_version = efficientnet_versions[self.cnnType]
            cnn_pretrained_model = EfficientNet.from_pretrained(model_version)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(f"{self.cnnType} is not a valid EfficientNet type.")
        return cnn_pretrained_model

    def Extract_Image_Feature(self, ):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAutocontrast(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
            transforms.RandomInvert(),
            transforms.RandomAdjustSharpness(random.uniform(0, 1)),
            transforms.RandomSolarize(random.uniform(0, 1)),
            transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            transforms.RandomErasing()
        ]
        img_to_tensor = transforms.Compose(transform_list)
        feat_df = pd.DataFrame()
        model = self.efficientNet_model()
        model.eval()
        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run the function image_crop first")
        # for spot, slice_path in self.adata.obs['slices_path'].items():
        #     spot_slice = Image.open(slice_path)
        #     spot_slice = spot_slice.resize((224, 224))
        #     spot_slice = np.asarray(spot_slice, dtype="int32")
        #     spot_slice = spot_slice.astype(np.float32)
        #     tensor = img_to_tensor(spot_slice)
        #     tensor = tensor.resize_(1, 3, 224, 224)
        #     tensor = tensor.to(self.device)
        #     result = model(Variable(tensor))
        #     result_npy = result.data.cpu().numpy().ravel()
        #     feat_df[spot] = result_npy
        #     feat_df = feat_df.copy()
        _, feat_df = extract_features_batch(self.adata, model, self.device)
        feat_df = feat_df.transpose()
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata


def image_crop(adata, save_path, crop_size=50, target_size=224, verbose=False, quality='hires'):
    image = adata.uns["image"]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(img_as_ubyte(image))
    tile_names = []
    with tqdm(total=len(adata), desc="Tiling image", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obsm["spatial_pixel"]['x_pixel'], adata.obsm["spatial_pixel"]['y_pixel']):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.LANCZOS)  #####
            tile.resize((target_size, target_size))  ######
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print("generate tile at location ({}, {})".format(str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)
    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata


class graph:

    def __init__(
        self,
        data,
        rad_cutoff,
        k,
        distType='euclidean',
    ):
        super().__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

    def graph_computing(self):
        dist_list = ["euclidean", "cosine"]
        graphList = []
        if self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(self.data.shape[0])
                         for j in range(indices.shape[1])]
        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList = [(node_idx, indices[j]) for node_idx in range(self.data.shape[0])
                         for j in np.where(A[node_idx] == 1)[0]]
        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(indices.shape[0])
                         for j in range(indices[node_idx].shape[0]) if distances[node_idx][j] > 0]
        return graphList

    def List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for end1, end2 in graphList:
            tdict[end1] = ""
            tdict[end2] = ""
            graphdict.setdefault(end1, []).append(end2)
        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []
        return graphdict

    def mx2SparseTensor(self, mx):
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)

    def main(self):
        adj_mtx = self.graph_computing()
        graph_dict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
        adj_pre = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
        adj_pre.eliminate_zeros()
        adj_norm = self.preprocess_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
        graph_dict = {"adj_norm": adj_norm, "adj_label": adj_label, "norm_value": norm}
        return graph_dict

    def combine_graph_dicts(self, dict_1, dict_2):
        tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
        graph_dict = {
            "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
            "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
            "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
        }
        return graph_dict



@register_preprocessor("misc",overwrite=True)
class EfNSTImageTransform(BaseTransform):

    def __init__(self, data_name, cnnType='efficientnet-b0', pca_n_comps=200, save_path="./", verbose=False,
                 crop_size=50, target_size=224, **kwargs):
        self.data_name = data_name
        self.verbose = verbose
        self.save_path = save_path
        self.pca_n_comps = pca_n_comps
        self.cnnType = cnnType
        self.crop_size = crop_size
        self.target_size = target_size
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        adata = data.data
        save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', f'{self.data_name}'))
        save_path_image_crop.mkdir(parents=True, exist_ok=True)
        adata = image_crop(adata, save_path=save_path_image_crop, quality='fulres', crop_size=self.crop_size,
                           target_size=self.target_size)
        adata = Image_Feature(adata, pca_components=self.pca_n_comps, cnnType=self.cnnType).Extract_Image_Feature()
        if self.verbose:
            save_data_path = Path(os.path.join(self.save_path, f'{self.data_name}'))
            save_data_path.mkdir(parents=True, exist_ok=True)
            adata.write(os.path.join(save_data_path, f'{self.data_name}.h5ad'), compression="gzip")
        return data
def cal_spatial_weight(
    data,
    spatial_k=50,
    spatial_type="KDTree",
):
    from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    elif spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    indices = indices[:, 1:]
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    for i in range(indices.shape[0]):
        ind = indices[i]
        for j in ind:
            spatial_weight[i][j] = 1
    return spatial_weight


def cal_gene_weight(data, n_components=50, gene_dist_type="cosine"):

    pca = PCA(n_components=n_components)
    if isinstance(data, np.ndarray):
        data_pca = pca.fit_transform(data)
    elif isinstance(data, csr_matrix):
        data = data.toarray()
        data_pca = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data_pca, metric=gene_dist_type)
    return gene_correlation
def cal_weight_matrix(adata, platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine",
                      gb_dist_type="correlation", n_components=50, no_morphological=True, spatial_k=30,
                      spatial_type="KDTree", verbose=False):
    if platform == "Visium":
        img_row = adata.obsm['spatial_pixel']['x_pixel']
        img_col = adata.obsm['spatial_pixel']['y_pixel']
        array_row = adata.obsm["spatial"]['x']
        array_col = adata.obsm["spatial"]['y']
        # img_row = adata.obs["imagerow"]
        # img_col = adata.obs["imagecol"]
        # array_row = adata.obs["array_row"]
        # array_col = adata.obs["array_col"]
        rate = 3
        reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
        reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)

        #   physical_distance = pairwise_distances(adata.obsm['spatial_pixel'][["y_pixel", "x_pixel"]], metric=pd_dist_type,n_jobs=-1)
        #   physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
        coords = adata.obsm['spatial_pixel'][["y_pixel", "x_pixel"]].values
        n_spots = coords.shape[0]
        radius = rate * unit
        nbrs = NearestNeighbors(radius=radius, metric=pd_dist_type, n_jobs=-1).fit(coords)
        distances, indices = nbrs.radius_neighbors(coords, return_distance=True)
        row_ind = []
        col_ind = []
        for i in range(n_spots):
            row_ind.extend([i] * len(indices[i]))
            col_ind.extend(indices[i])
        data = np.ones(len(row_ind), dtype=np.int8)
        physical_distance = csr_matrix((data, (row_ind, col_ind)), shape=(n_spots, n_spots))
    else:
        physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k=spatial_k, spatial_type=spatial_type)

    gene_counts = adata.X.copy()
    gene_correlation = cal_gene_weight(data=gene_counts, gene_dist_type=gb_dist_type, n_components=n_components)
    del gene_counts
    if verbose:
        adata.obsm["gene_correlation"] = gene_correlation
        adata.obsm["physical_distance"] = physical_distance

    if platform == 'Visium':
        morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
        morphological_similarity[morphological_similarity < 0] = 0
        if verbose:
            adata.obsm["morphological_similarity"] = morphological_similarity
        adata.obsm["weights_matrix_all"] = (physical_distance * gene_correlation * morphological_similarity)
        if no_morphological:
            adata.obsm["weights_matrix_nomd"] = (gene_correlation * physical_distance)
    else:
        adata.obsm["weights_matrix_nomd"] = (gene_correlation * physical_distance)
    return adata

def find_adjacent_spot(adata, use_data="raw", neighbour_k=4, weights='weights_matrix_all', verbose=False):
    if use_data == "raw":
        if isinstance(adata.X, (csr_matrix, np.ndarray)):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"""{type(adata.X)} is not a valid type.""")
    else:
        gene_matrix = adata.obsm[use_data]
    weights_matrix = adata.obsm[weights]
    weights_list = []
    final_coordinates = []
    for i in range(adata.shape[0]):
        if weights == "physical_distance":
            current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k + 3):][:(neighbour_k + 2)]
        else:
            current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:][:neighbour_k - 1]
        spot_weight = adata.obsm[weights][i][current_spot]
        spot_matrix = gene_matrix[current_spot]
        if spot_weight.sum() > 0:
            spot_weight_scaled = spot_weight / spot_weight.sum()
            weights_list.append(spot_weight_scaled)
            spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
            spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
        else:
            spot_matrix_final = np.zeros(gene_matrix.shape[1])
            weights_list.append(np.zeros(len(current_spot)))
        final_coordinates.append(spot_matrix_final)
    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)
    return adata


def augment_gene_data(adata, Adj_WT=0.2):
    adjacent_gene_matrix = adata.obsm["adjacent_data"].astype(float)
    if isinstance(adata.X, np.ndarray):
        augment_gene_matrix = adata.X + Adj_WT * adjacent_gene_matrix
    elif isinstance(adata.X, csr_matrix):
        augment_gene_matrix = adata.X.toarray() + Adj_WT * adjacent_gene_matrix
    adata.obsm["augment_gene_data"] = augment_gene_matrix
    del adjacent_gene_matrix
    return adata
def augment_adata(adata, platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine", gb_dist_type="correlation",
                  n_components=50, no_morphological=False, use_data="raw", neighbour_k=4, weights="weights_matrix_all",
                  Adj_WT=0.2, spatial_k=30, spatial_type="KDTree"):
    adata = cal_weight_matrix(
        adata,
        platform=platform,
        pd_dist_type=pd_dist_type,
        md_dist_type=md_dist_type,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        no_morphological=no_morphological,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    )
    adata = find_adjacent_spot(adata, use_data=use_data, neighbour_k=neighbour_k, weights=weights)
    adata = augment_gene_data(
        adata,
        Adj_WT=Adj_WT,
    )
    return adata

@register_preprocessor("misc",overwrite=True)
class EfNSTAugmentTransform(BaseTransform):

    def __init__(self, Adj_WT=0.2, neighbour_k=4, weights="weights_matrix_all", spatial_k=30, platform="Visium",
                 **kwargs):
        super().__init__(**kwargs)
        self.Adj_WT = Adj_WT
        self.neighbour_k = neighbour_k
        self.weights = weights
        self.spatial_k = spatial_k
        self.platform = platform

    def __call__(self, data: Data) -> Data:
        adata = data.data
        adata_augment = augment_adata(
            adata,
            Adj_WT=self.Adj_WT,
            neighbour_k=self.neighbour_k,
            platform=self.platform,
            weights=self.weights,
            spatial_k=self.spatial_k,
        )
        assert adata is adata_augment
        return adata_augment


@register_preprocessor("graph", "cell",overwrite=True)
class EfNSTGraphTransform(BaseTransform):

    def __init__(self, distType="Radius", k=12, rad_cutoff=150, **kwargs):
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        adata = data.data
        graph_dict = graph(adata.obsm['spatial'], distType=self.distType, k=self.k, rad_cutoff=self.rad_cutoff).main()
        adata.uns['EfNSTGraph'] = graph_dict
# EVOLVE-BLOCK-END
def get_preprocessing_pipeline(data_name, verbose=False, cnnType='efficientnet-b0', pca_n_comps=200, distType="KDTree",
                               k=12, dim_reduction=True, min_cells=3, platform="Visium"):
        return Compose(
            EfNSTImageTransform(data_name=data_name, verbose=verbose, cnnType=cnnType),
            EfNSTAugmentTransform(),
            EfNSTGraphTransform(distType=distType, k=k),
            EfNSTConcatgTransform(dim_reduction=dim_reduction, min_cells=min_cells, platform=platform,
                                  pca_n_comps=pca_n_comps),  #xenium can also be processed using Visium method
            SetConfig({
                "feature_channel": ["feature.cell", "EfNSTGraph"],
                "feature_channel_type": ["obsm", "uns"],
                "label_channel": "label",
                "label_channel_type": "obs"
            }))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151507",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--cnnType", type=str, default="efficientnet-b0")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain the model.")
    parser.add_argument("--pre_epochs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--Conv_type", type=str, default="ResGatedGraphConv")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information.")
    parser.add_argument("--pca_n_comps", type=int, default=200, help="Number of PCA components.")
    parser.add_argument("--distType", type=str, default="KDTree", help="Distance type.")
    parser.add_argument("--k", type=int, default=12, help="Number of neighbors.")
    parser.add_argument("--no_dim_reduction", action="store_true", help="Print detailed information.")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum number of cells.")
    parser.add_argument("--platform", type=str, default="Visium", help="Platform type.")
    args = parser.parse_args()

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(args.seed, extreme_mode=True)
        try:
            EfNST = EfNsSTRunner(
                platform=args.platform,
                pre_epochs=args.pre_epochs,  #### According to your own hardware, choose the number of training
                epochs=args.epochs,
                cnnType=args.cnnType,
                Conv_type=args.Conv_type,
                random_state=seed)
            dataloader = SpatialLIBDDataset(data_id=args.sample_number)
            data = dataloader.load_data(transform=None, cache=args.cache)
            preprocessing_pipeline = get_preprocessing_pipeline(
                data_name=args.sample_number, verbose=args.verbose, cnnType=args.cnnType, pca_n_comps=args.pca_n_comps,
                distType=args.distType, k=args.k, dim_reduction=not args.no_dim_reduction, min_cells=args.min_cells,
                platform=args.platform)
            preprocessing_pipeline(data)
            (x, adj), y = data.get_data()
            adata = data.data
            adata = EfNST.fit(adata, x, graph_dict=adj, pretrain=args.pretrain)
            n_domains = len(np.unique(y))
            adata = EfNST._get_cluster_data(adata, n_domains=n_domains, priori=True)
            y_pred = EfNST.predict(adata)
            silhouette_score = resolve_score_func("silhouette")
            calinski_harabasz_score = resolve_score_func("calinski_harabasz")
            davies_bouldin_score = resolve_score_func("davies_bouldin")
            inner_scores.append(calculate_unified_scores({
                "silhouette": silhouette_score(x, y_pred),
                "calinski_harabasz": calinski_harabasz_score(x, y_pred),
                "davies_bouldin": davies_bouldin_score(x, y_pred)
            }))
        finally:
            EfNST.delete_imgs(adata)
        score = adjusted_rand_score(y, y_pred)
        scores.append(score)
        print(f"ARI: {score:.4f}")
    print(f"EfNST {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
"""
python EfNST.py --sample_number 151507
python EfNST.py --sample_number 151673
python EfNST.py --sample_number 151676

"""
