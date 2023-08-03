"""
Project configurations.
"""
import os


class Config(object):
    """ Datasets and experiments folders for the whole project. """
    #####################
    ## Dataset setting ##
    #####################
    default_dataroot = os.path.join(
        os.path.dirname(__file__),
        '..','..','..','data-ssl'
    )
    default_dataroot = os.path.abspath(default_dataroot)
    default_exproot = os.path.join(
        os.path.dirname(__file__),
        '..','..','..','exp-ssl'
    )
    default_exproot = os.path.abspath(default_exproot)

    DATASET_ROOT = os.getenv("DATASET_ROOT", default_dataroot)  # TODO: path to your datasets folder
    if not os.path.exists(DATASET_ROOT):
        os.makedirs(DATASET_ROOT)
    
    # Synthetic shape dataset
    synthetic_dataroot = os.path.join(DATASET_ROOT, "synthetic_shapes")
    synthetic_cache_path = os.path.join(DATASET_ROOT, "synthetic_shapes")
    if not os.path.exists(synthetic_dataroot):
        os.makedirs(synthetic_dataroot)
    
    # Exported predictions dataset
    export_dataroot = os.path.join(DATASET_ROOT, "export_datasets")
    export_cache_path = os.path.join(DATASET_ROOT, "export_datasets")
    if not os.path.exists(export_dataroot):
        os.makedirs(export_dataroot)
    
    # York Urban dataset
    yorkurban_dataroot = os.path.join(DATASET_ROOT, "YorkUrbanDB")
    yorkurban_cache_path = os.path.join(DATASET_ROOT, "YorkUrbanDB")

    # Wireframe dataset
    wireframe_dataroot = os.path.join(DATASET_ROOT, "wireframe")
    wireframe_cache_path = os.path.join(DATASET_ROOT, "wireframe")

    # Holicity dataset
    holicity_dataroot = os.path.join(DATASET_ROOT, "Holicity")
    holicity_cache_path = os.path.join(DATASET_ROOT, "Holicity")
    
    ########################
    ## Experiment Setting ##
    ########################
    EXP_PATH = os.getenv("EXP_PATH", default_exproot)  # TODO: path to your experiments folder

    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH)
