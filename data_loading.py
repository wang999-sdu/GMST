# data_loading.py
import scanpy as sc
import pandas as pd
import numpy as np
from config import data_path, image_path, scalefactors_path

def load_data():
    adata = sc.read_visium(
        path=data_path,
        count_file='151507_filtered_feature_bc_matrix.h5',
        library_id="0.DLPFC/151507",
        load_images=True,
        source_image_path=image_path
    )
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata