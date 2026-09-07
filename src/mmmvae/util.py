# -*- coding: utf-8 -*-
"""
@author: TianMiaoMiao

util functions for MMMVAE model
"""
import torch
import importlib.resources
from importlib.resources.abc import Traversable
import mmmvae
import os

def get_data_path(filename: str) -> str:
    """
    Get the path to the resource file under src/mmmvae/data within the package
    Used to read built-in data such as module_gene_m168.csv, compatible with pip installation mode
    """
    data_dir: Traversable = importlib.resources.files(mmmvae).joinpath("data")
    target_path = data_dir.joinpath(filename)
    if hasattr(target_path, "__fspath__"):
        real_path = target_path.__fspath__()
    else:
        real_path = str(target_path)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Package data file not found: {real_path}")
    return real_path



def pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mimics `scipy.stats.pearsonr`, torch tensor version
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : torch.Tensor
        pearson correlation coefficient between x and y
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val