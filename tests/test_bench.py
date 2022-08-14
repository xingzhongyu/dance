import gc
import logging
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

HOME_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = HOME_DIR / "examples"
logging.info(f"{HOME_DIR=}")

SKIP_LIST: List[str] = [
    "joint_embedding-dcca",  # OOM with 64GB mem and V100 GPU (succeed with 80GB mem)
]

light_options_dict: Dict[str, Tuple[str, str]] = {
    # {task}-{method}-{dataset}: {command_line_options}
    # Single modality
    "cell_type_annotation-actinn-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --lambd 0.0001 --device cuda --num_epochs 2 --runs 1",
    "cell_type_annotation-celltypist-spleen": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --label_conversion False",
    "cell_type_annotation-scdeepsort-spleen": "--data_type scdeepsort --tissue Spleen --test_data 1759 --gpu 0 --n_epochs 2",
    "cell_type_annotation-singlecellnet-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --dLevel Cell_type --dtype Cell_type",
    "cell_type_annotation-svm-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "clustering-graphsc-10X_PBMC": "--dataset 10X_PBMC --epochs 2",
    "clustering-scdcc-10X_PBMC": "--data_file 10X_PBMC --label_cells_files label_10X_PBMC.txt --gamma 1.5 --maxiter 2 --pretrain_epochs 2",
    "clustering-scdeepcluster-10X_PBMC": "--data_file 10X_PBMC --pretrain_epochs 2",
    "clustering-scdsc-10X_PBMC": "--name 10X_PBMC --method cos --topk 30 --v 7 --binary_crossentropy_loss 0.75 --ce_loss 0.5 --re_loss 0.1 --zinb_loss 2.5 --sigma 0.4 --n_epochs 2 --pretrain_epochs 2",
    "clustering-sctag-10X_PBMC": "--pretrain_epochs 2 --epochs 2 --data_file 10X_PBMC --W_a 0.01 --W_x 3 --W_c 0.1 --dropout 0.5",
    "imputation-deepimpute-brain": "--train_dataset mouse_brain_data --filetype h5 --hidden_dim 200 --dropout 0.4 --n_epochs 2 --gpu 0",
    "imputation-graphsci-brain": "--train_dataset mouse_brain_data --gpu 0 --n_epochs 2",
    "imputation-scgnn-brain": "--train_dataset mouse_brain_data --Regu_epochs 2 --EM_epochs 2 --cluster_epochs 2 --GAEepochs 2 --gpu 0",
    # Multi modality
    "predict_modality-babel-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --max_epochs 2 --earlystop 2",
    "predict_modality-cmae-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --max_epochs 2",
    "predict_modality-scmm-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --epochs 2",
    "predict_modality-scmogcn-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --epoch 2",
    "match_modality-cmae-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --max_epochs 2",
    "match_modality-scmm-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --epochs 2",
    "match_modality-scmogcn-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda --epochs 2",
    "joint_embedding-jae-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "joint_embedding-scmvae-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda --max_epoch 2 --anneal_epoch 2 --epoch_per_test 2 --max_iteration 10",
    "joint_embedding-dcca-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda --max_epoch 2 --max_iteration 10 --anneal_epoch 2 --epoch_per_test 2",
    "joint_embedding-scmogcn-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    # Spatial
    "cell_type_deconvo-card-card_synth": "--dataset CARD_synthetic --max_iter 2",
    "cell_type_deconvo-dstg": None,
    "cell_type_deconvo-spatialdecon": None,
    "cell_type_deconvo-spotlight": None,
    "spatial_domain-louvain-151507": "--sample_number 151507 --seed 10",
    "spatial_domain-spagcn-151507": "--sample_number 151507 --lr 0.009",
    "spatial_domain-stagate-151507": "--sample_number 151507 --seed 2021",
    "spatial_domain-stlearn-151507": "--n_clusters 20 --sample_number 151507 --seed 0",
}  # yapf: disable

full_options_dict: Dict[str, Tuple[str, str]] = {
    # Single modality
    "cell_type_annotation-actinn-brain": "--species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695 --lambd 0.1 --device cuda:0",
    "cell_type_annotation-actinn-kidney": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --lambd 0.01 --device cuda:0",
    "cell_type_annotation-actinn-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --lambd 0.0001 --device cuda:0",
    "cell_type_annotation-celltypist-brain": "--species mouse --tissue Brain --train_dataset 753 --test_dataset 2695 --label_conversion True",
    "cell_type_annotation-celltypist-kidney": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --label_conversion False",
    "cell_type_annotation-celltypist-spleen": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --label_conversion False",
    "cell_type_annotation-scdeepsort-brain": "--data_type scdeepsort --tissue Brain --test_data 2695 --gpu 0",
    "cell_type_annotation-scdeepsort-kidney": "--data_type scdeepsort --tissue Kidney --test_data 203 --gpu 0",
    "cell_type_annotation-scdeepsort-spleen": "--data_type scdeepsort --tissue Spleen --test_data 1759 --gpu 0",
    "cell_type_annotation-singlecellnet-brain": "--species mouse --tissue Brain --train_dataset 753 --test_dataset 2695 --dLevel Cell_type --dtype Cell_type",
    "cell_type_annotation-singlecellnet-kidney": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --dLevel Cell_type --dtype Cell_type",
    "cell_type_annotation-singlecellnet-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --dLevel Cell_type --dtype Cell_type",
    "cell_type_annotation-svm-brain": "--species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695",
    "cell_type_annotation-svm-kidney": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203",
    "cell_type_annotation-svm-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "clustering-graphsc-10X_PBMC": "--dataset 10X_PBMC",
    "clustering-graphsc-mouse_ES_cell": "--dataset mouse_ES_cell",
    "clustering-graphsc-mouse_bladder_cell": "--dataset mouse_bladder_cell",
    "clustering-graphsc-worm_neuron_cell": "--dataset worm_neuron_cell",
    "clustering-scdcc-10X_PBMC": "--data_file 10X_PBMC --label_cells_files label_10X_PBMC.txt --gamma 1.5",
    "clustering-scdcc-mouse_ES_cell": "--data_file mouse_ES_cell --label_cells_files label_mouse_ES_cell.txt --gamma 1 --ml_weight 0.8 --cl_weight 0.8",
    "clustering-scdcc-mouse_bladder_cell": "--data_file mouse_bladder_cell --label_cells_files label_mouse_bladder_cell.txt --gamma 1.5 --pretrain_epochs 100 --sigma 3",
    "clustering-scdcc-worm_neuron_cell": "--data_file worm_neuron_cell --label_cells_files label_worm_neuron_cell.txt --gamma 1 --pretrain_epochs 300",
    "clustering-scdeepcluster-10X_PBMC": "--data_file 10X_PBMC",
    "clustering-scdeepcluster-mouse_ES_cell": "--data_file mouse_ES_cell",
    "clustering-scdeepcluster-mouse_bladder_cell": "--data_file mouse_bladder_cell --pretrain_epochs 300 --sigma 2.75",
    "clustering-scdeepcluster-worm_neuron_cell": "--data_file worm_neuron_cell --pretrain_epochs 300",
    "clustering-scdsc-10X_PBMC": "--name 10X_PBMC --method cos --topk 30 --v 7 --binary_crossentropy_loss 0.75 --ce_loss 0.5 --re_loss 0.1 --zinb_loss 2.5 --sigma 0.4",
    "clustering-scdsc-mouse_ES_cell": "--name mouse_ES_cell --method heat --topk 50 --v 7 --binary_crossentropy_loss 0.1 --ce_loss 0.01 --re_loss 1.5 --zinb_loss 0.5 --sigma 0.1",
    "clustering-scdsc-mouse_bladder_cell": "--name mouse_bladder_cell --method p --topk 50 --v 7 --binary_crossentropy_loss 2.5 --ce_loss 0.1 --re_loss 0.5 --zinb_loss 1.5 --sigma 0.6",
    "clustering-scdsc-worm_neuron_cell": "--name worm_neuron_cell --method p --topk 20 --v 7 --binary_crossentropy_loss 2 --ce_loss 2 --re_loss 3 --zinb_loss 0.1 --sigma 0.4",
    "clustering-sctag-10X_PBMC": "--pretrain_epochs 100 --data_file 10X_PBMC --W_a 0.01 --W_x 3 --W_c 0.1 --dropout 0.5",
    "clustering-sctag-mouse_ES_cell": "--data_file mouse_ES_cell --W_a 0.01 --W_x 2 --W_c 0.25 --k 1",
    "clustering-sctag-mouse_bladder_cell": "--pretrain_epochs 100 --data_file mouse_bladder_cell --W_a 0.01 --W_x 0.75 --W_c 1",
    "clustering-sctag-worm_neuron_cell": "--pretrain_epochs 100 --data_file worm_neuron_cell --W_a 0.1 --W_x 2.5 --W_c 3",
    "imputation-deepimpute-brain": "--train_dataset mouse_brain_data --filetype h5 --hidden_dim 200 --dropout 0.4",
    "imputation-deepimpute-embryo": "--train_dataset mouse_embryo_data --filetype gz --hidden_dim 200 --dropout 0.4",
    "imputation-graphsci-brain": "--train_dataset mouse_brain_data --gpu 0",
    "imputation-graphsci-embryo": "--train_dataset mouse_embryo_data --gpu 0",
    "imputation-scgnn-brain": "--train_dataset mouse_brain_data --gpu 0",
    "imputation-scgnn-embryo": "--train_dataset mouse_embryo_data --gpu 0",
    # Multi modality
    "predict_modality-babel-cp2m2": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-babel-mp2m2": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-babel-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-cmae-cp2m2": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-cmae-mp2m2": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-cmae-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-scmm-cp2m2": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-scmm-mp2m2": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-scmm-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-scmogcn-cp2m2": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-scmogcn-mp2m2": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-scmogcn-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "match_modality-cmae-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "match_modality-scmm-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "match_modality-scmogcn-mp2r": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "joint_embedding-dcca-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "joint_embedding-jae-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda", # this is scDEC
    "joint_embedding-scmogcn-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "joint_embedding-scmvae-mp2": "--subtask openproblems_bmmc_multiome_phase2 --device cuda --max_epoch 100",
    # Spatial
    "cell_type_deconvo-card-card_synth": "--dataset CARD_synthetic",
    "cell_type_deconvo-card-gse174746": "--dataset GSE174746 --location_free",
    "cell_type_deconvo-card-spotlight_synth": "--dataset SPOTLight_synthetic --location_free",
    "cell_type_deconvo-dstg": None,  # TODO
    "cell_type_deconvo-dstg_mousebrain": None,  # TODO
    "cell_type_deconvo-spatialdecon": None,  # TODO
    "cell_type_deconvo-spotlight": None,  # TODO
    "spatial_domain-louvain-151507": "--sample_number 151507 --seed 10",
    "spatial_domain-louvain-151673": "--sample_number 151673 --seed 5",
    "spatial_domain-louvain-151676": "--sample_number 151676 --seed 203",
    "spatial_domain-spagcn-151507": "--sample_number 151507 --lr 0.009",
    "spatial_domain-spagcn-151673": "--sample_number 151673 --lr 0.1",
    "spatial_domain-spagcn-151676": "--sample_number 151676 --lr 0.02",
    "spatial_domain-stagate-151507": "--sample_number 151507 --seed 2021",
    "spatial_domain-stagate-151673": "--sample_number 151673 --seed 16",
    "spatial_domain-stagate-151676": "--sample_number 151676 --seed 2030",
    "spatial_domain-stlearn-151507": "--n_clusters 20 --sample_number 151507 --seed 0",
    "spatial_domain-stlearn-151673": "--n_clusters 20 --sample_number 151673 --seed 93",
    "spatial_domain-stlearn-151676": "--n_clusters 20 --sample_number 151676 --seed 11",
}  # yapf: disable


def find_script_path(script_name: str, task_name: str) -> Path:
    for dir_, subdirs, files in os.walk(SCRIPTS_DIR):
        if script_name in files and task_name in dir_:
            logging.info(f"Found {script_name} under {dir_}")
            return Path(dir_)
    raise FileNotFoundError(f"Failed to locate {script_name!r} for task {task_name!r} under {SCRIPTS_DIR!s}")


def run_benchmarks(name, options_dict):
    # Check to see if the test run name is contained in any element of the
    # SKIP_LIST and return immediatel if so
    if any(map(lambda to_skip: to_skip in name, SKIP_LIST)):
        logging.warning(f"Skipping run {name!r} as it is contained in one of the elements in {SKIP_LIST=}")
        return

    task_name = name.split("-")[0]
    script_name = name.split("-")[1] + ".py"
    os.chdir(find_script_path(script_name, task_name))

    # Overwrite sysargv and run test script
    args = options_dict[name]
    sys.argv = [None] + (args.split(" ") if args else [])
    logging.info(f"Start runing [{name}] with {args=!r}")
    t = time.perf_counter()
    runpy.run_path(script_name, run_name="__main__")
    t = time.perf_counter() - t
    logging.info(f"Finshed runing [{name}] - took {int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}")

    # Post run cleanup
    gc.collect()


@pytest.mark.parametrize("name", sorted(full_options_dict))
@pytest.mark.full_test
def test_bench_full(name):
    run_benchmarks(name, full_options_dict)


@pytest.mark.parametrize("name", sorted(light_options_dict))
@pytest.mark.light_test
def test_bench_light(name):
    run_benchmarks(name, light_options_dict)