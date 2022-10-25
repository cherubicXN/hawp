from torch.utils.data import Dataset
import os
import math
from tqdm import tqdm
from skimage.io import imread
from skimage import color
import PIL
import numpy as np
import h5py
import cv2
import pickle
from .synthetic_util import get_line_heatmap
from torchvision import transforms
import torch
import torch.utils.data.dataloader as torch_loader
# Augmentation libs
from ..config.project_config import Config as cfg
from .transforms import photometric_transforms as photoaug
from .transforms import homographic_transforms as homoaug
# Some visualization tools
from ..misc.visualize_util import plot_junctions, plot_line_segments
# Some data parsing tools
from ..misc.train_utils import parse_h5_data
# Inherit from private dataset
# from dataset.private_dataset import PrivateDataset


# Implements the customized collate_fn for yorkurban dataset
def yorkurban_collate_fn(batch):
    batch_keys = ["image", "junction_map", "valid_mask", "heatmap",
                  "heatmap_pos", "heatmap_neg", "homography"]
    list_keys = ["junctions", "line_map", "line_map_pos", "line_map_neg", "file_key",
                 "aux_junctions", "aux_line_map"]

    outputs = {}
    for data_key in batch[0].keys():
        batch_match = sum([_ in data_key for _ in batch_keys])
        list_match = sum([_ in data_key for _ in list_keys])
        # print(batch_match, list_match)
        if batch_match > 0 and list_match == 0:
            outputs[data_key] = torch_loader.default_collate([b[data_key] for b in batch])
        elif batch_match == 0 and list_match > 0:
            outputs[data_key] = [b[data_key] for b in batch]
        elif batch_match == 0 and list_match == 0:
            continue
        else:
            raise ValueError("[Error] A key matches batch keys and list keys simultaneously.")

    return outputs

# The processed wireframe.
class YorkUrbanDataset(Dataset):
    # Initialize the dataset
    def __init__(self, mode="test", config=None):
        super(YorkUrbanDataset, self).__init__()
        # Check mode => "train", "val", "test
        if not mode in ["test"]:
            raise ValueError("[Error] Unknown mode for york urban dataset. Only 'test' mode is available.")
        self.mode = mode

        self.config = config

        # Get cache setting
        self.dataset_name = self.get_dataset_name()
        self.cache_name = self.get_cache_name()
        self.cache_path = cfg.yorkurban_cache_path

        # Get the filename dataset
        print("[Info] Initializing york urban dataset...")
        self.filename_dataset, self.datapoints = self.get_filename_dataset()
        # Get dataset length
        self.dataset_length = len(self.datapoints)
        
        # Get repeatability evaluation set
        if self.mode == "test" and self.config.get("evaluation", None) is not None:
            # Get the cache name
            tmp = self.cache_name.split(self.mode)
            self.rep_i_cache_name = tmp[0] + self.mode + "_rep_i" + tmp[1]
            self.rep_v_cache_name = tmp[0] + self.mode + "_rep_v" + tmp[1]

            # Get the repeatability config
            self.rep_config = self.config["evaluation"]["repeatability"]

            self.rep_eval_dataset = self.construct_rep_eval_dataset()
            self.rep_eval_datapoints = self.get_rep_eval_datapoints()

        # Print some info
        print("[Info] Successfully initialized dataset")
        print("\t Name: yorkurban")
        print("\t Mode: %s" %(self.mode))
        print("\t Gt: %s" %(self.config.get("gt_source_%s"%(self.mode), "official")))
        print("\t Counts: %d" %(self.dataset_length))
        print("----------------------------------------")
    
    def get_filename_dataset(self):
        # Get the path to the dataset
        if self.mode == "train":
            raise NotImplementedError
        elif self.mode == "test":
            dataset_path = os.path.join(cfg.yorkurban_dataroot)
        # Get paths to all image files
        folder_lst = sorted([os.path.join(dataset_path, _) for _ in os.listdir(dataset_path) \
            if os.path.isdir(os.path.join(dataset_path, _))])
        folder_lst = folder_lst[:-1]
        #folder_lst = [f for f in folder_lst if f.startswith('P')]
        image_paths = []
        for folder in folder_lst:
            image_path = [os.path.join(folder, _) for _ in os.listdir(folder) \
                if os.path.splitext(_)[-1] == ".jpg" or os.path.splitext(_)[-1] == ".png"]
            image_paths += image_path

        # Verify all the images and labels exist
        for idx in range(len(image_paths)):
            image_path = image_paths[idx]
            if not os.path.exists(image_path):
                raise ValueError("[Error] The image does not exist. %s"%(image_path))
        
        # Construct the filename dataset
        num_pad = int(math.ceil(math.log10(len(image_paths))) + 1)
        filename_dataset = {}
        for idx in range(len(image_paths)):
            # Get the file key
            key = self.get_padded_filename(num_pad, idx)

            filename_dataset[key] = {
                "image": image_paths[idx]
            }
        
        # Get the datapoints
        datapoints = list(sorted(filename_dataset.keys()))

        return filename_dataset, datapoints
    
    # Get the padded filename using adaptive padding
    @staticmethod
    def get_padded_filename(num_pad, idx):
        file_len = len("%d" % (idx))
        filename = "0" * (num_pad - file_len) + "%d" % (idx)

        return filename
    
    # Get dataset name from dataset config / default config
    def get_dataset_name(self):
        if self.config["dataset_name"] is None:
            dataset_name = "yorkurban_dataset" + f"_{self.mode}"
        else:
            dataset_name = self.config["dataset_name"] + f"_{self.mode}" 

        return dataset_name
    
    # Get cache name from dataset config / default config
    def get_cache_name(self):
        if self.config["dataset_name"] is None:
            dataset_name = "yorkurban_dataset" + f"_{self.mode}"
        else:
            dataset_name = self.config["dataset_name"] + f"_{self.mode}" 
        # Compose cache name
        cache_name = dataset_name + "_cache.pkl"

        return cache_name
    
    ###########################################
    ## Repeatability evaluation related APIs ##
    ###########################################
    # Construct repeatability evaluation dataset (from scratch or from cache)
    def construct_rep_eval_dataset(self):
        rep_eval_dataset = {}
        # Check if viewpoint and illumination cache exists
        if self.rep_config["photometric"]["enable"]:
            if self._check_rep_eval_dataset_cache(split="i"):
                print("\t Found repeatability illumination cache %s at %s"%(self.rep_i_cache_name, self.cache_path))
                print("\t Load repeatability illumination cache...")
                rep_i_keymap, rep_i_dataset_name = self.get_rep_eval_dataset_from_cache(split="i")
            else:
                print("\t Can't find repeatability illumination cache ...")
                print("\t Create repeatability illumination dataset from scratch...")
                rep_i_keymap, rep_i_dataset_name = self.get_rep_eval_dataset(split="i")
                print("\t Create filename dataset cache...")
                self.create_rep_eval_dataset_cache("i", rep_i_keymap, rep_i_dataset_name)
        else:
            rep_i_keymap = None
            rep_i_dataset_name = None
        
        rep_eval_dataset["illumination"] = {
            "keymap": rep_i_keymap,
            "dataset_name": rep_i_dataset_name
        }
        
        if self.rep_config["homographic"]["enable"]:
            if self._check_rep_eval_dataset_cache(split="v"):
                print("\t Found repeatability viewpoint cache %s at %s"%(self.rep_v_cache_name, self.cache_path))
                print("\t Load repeatability viewpoint cache...")
                rep_v_keymap, rep_v_dataset_name = self.get_rep_eval_dataset_from_cache(split="v")
            else:
                print("\t Can't find repeatability viewpoint cache ...")
                print("\t Create repeatability viewpoint dataset from scratch...")
                rep_v_keymap, rep_v_dataset_name = self.get_rep_eval_dataset(split="v")
                print("\t Create filename dataset cache...")
                self.create_rep_eval_dataset_cache("v", rep_v_keymap, rep_v_dataset_name)
        else:
            rep_v_keymap = None
            rep_v_dataset_name = None
        
        rep_eval_dataset["viewpoint"] = {
            "keymap": rep_v_keymap,
            "dataset_name": rep_v_dataset_name
        }

        return rep_eval_dataset
    
    # Create filename dataset cache for faster initialization
    def create_rep_eval_dataset_cache(self, split, keymap, dataset_name):
        # Check cache path exists
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if split == "i":
            cache_file_path = os.path.join(self.cache_path, self.rep_i_cache_name)
        elif split == "v":
            cache_file_path = os.path.join(self.cache_path, self.rep_v_cache_name)
        else:
            raise ValueError("[Error] Unknown split for repeatability evaluation.")

        data = {
            "keymap": keymap,
            "dataset_name": dataset_name
        }
        with open(cache_file_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    # Get filename dataset from cache
    def get_rep_eval_dataset_from_cache(self, split):
        # Load from pkl cache
        if split == "i":
            cache_file_path = os.path.join(self.cache_path, self.rep_i_cache_name)
        elif split == "v":
            cache_file_path = os.path.join(self.cache_path, self.rep_v_cache_name)
        else:
            raise ValueError("[Error] Unknown split for repeatability evaluation.")

        with open(cache_file_path, "rb") as f:
            data = pickle.load(f)
        
        return data["keymap"], data["dataset_name"]
    
    # Initialize the repeatability evaluation dataset from scratch
    def get_rep_eval_dataset(self, split):
        image_shape = self.config["preprocessing"]["resize"]

        # Initialize the illumination set
        if split == "i":
            # Set the random seed before continuing
            seed = self.rep_config["seed"]
            np.random.seed(seed)
            torch.manual_seed(seed)

            raise NotImplementedError
        
        # Initialize the viewpoint set
        elif split == "v":
            # Set the random seed before continuing
            seed = self.rep_config["seed"]
            np.random.seed(seed)
            torch.manual_seed(seed)

            v_keymap = {}
            # Get the name for the output h5 dataset
            v_dataset_name = self.rep_v_cache_name.split(".pkl")[0] + ".h5"
            v_dataset_path = os.path.join(self.cache_path, v_dataset_name)
            with h5py.File(v_dataset_path, "w") as f:
                # Iterate through all the file_key in test set
                for idx, key in enumerate(tqdm(list(self.filename_dataset.keys()), ascii=True)):
                    # Sample N random homography
                    file_key_lst = []
                    for i in range(self.rep_config["homographic"]["num_samples"]):
                        file_key = key + "_" + str(i)

                        # Sample a random homography
                        homo_mat, _ = homoaug.sample_homography(image_shape, 
                                **self.rep_config["homographic"]["params"])
                        
                        file_key_lst.append(file_key)
                        f.create_dataset(file_key, data=homo_mat, compression="gzip")
                    
                    v_keymap[key] = file_key_lst
            
            return v_keymap, v_dataset_name
        
        else:
            raise ValueError("[Error] Unknow split for repeatability evaluation.")
    
    # Convert ref image and warped images to list of evaluation pairs
    def get_rep_eval_datapoints(self):
        datapoints = {
            "illumination": [],
            "viewpoint": []
        }

        # Iterate through all the ref image
        if self.rep_eval_dataset["illumination"]["keymap"] is not None:
            for ref_key in sorted(self.rep_eval_dataset["illumination"]["keymap"].keys()):
                pair_lst = [[ref_key, _] for _ in self.rep_eval_dataset["illumination"]["keymap"][ref_key]]
                datapoints["illumination"] += pair_lst
        
        if self.rep_eval_dataset["viewpoint"]["keymap"] is not None:
            for ref_key in sorted(self.rep_eval_dataset["viewpoint"]["keymap"].keys()):
                pair_lst = [[ref_key, _] for _ in self.rep_eval_dataset["viewpoint"]["keymap"][ref_key]]
                datapoints["viewpoint"] += pair_lst
        
        return datapoints
    
    # Check if the repeatability cache dataset exists
    def _check_rep_eval_dataset_cache(self, split):
        if split == "i":
            cache_file_path = os.path.join(self.cache_path, self.rep_i_cache_name)
        else:
            cache_file_path = os.path.join(self.cache_path, self.rep_v_cache_name)
        
        return os.path.exists(cache_file_path)
    
    ###########################################
    ## Repeatability evaluation related APIs ##
    ###########################################
    # Get the corresponding data according to the "index in rep_eval_datapoints".
    def get_rep_eval_data(self, split, idx):
        assert split in ["viewpoint", "illumination"]
        datapoint = self.rep_eval_datapoints[split][idx]

        # Get reference image
        ref_key = datapoint[0]
        # Get the data paths
        data_path = self.filename_dataset[ref_key]
        # Read in the image and npz labels (but haven't applied any transform)
        image = imread(data_path["image"])

        # Resize the image before photometric and homographical augmentations
        image_size = image.shape[:2]
        if not(list(image_size) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)[:2] # Only H and W dimensions

            image = cv2.resize(image, tuple(self.config['preprocessing']['resize'][::-1]),
                               interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)
        
        # Optionally convert the image to grayscale
        if self.config["gray_scale"]:
            image = (color.rgb2gray(image) *255.).astype(np.uint8)
        
        image_transform = photoaug.normalize_image()
        image = image_transform(image)

        # Get target image
        if split == "viewpoint":
            target_key = datapoint[1]
            dataset_path = os.path.join(self.cache_path, self.rep_eval_dataset[split]["dataset_name"])
            with h5py.File(dataset_path, "r") as f:
                homo_mat = np.array(f[target_key])
            
            # Warp the image
            target_size = (image.shape[1], image.shape[0])
            target_image = cv2.warpPerspective(image, homo_mat, target_size,
                                            flags=cv2.INTER_LINEAR)

        else:
            raise NotImplementedError

        # Convert to tensor and return the results
        to_tensor = transforms.ToTensor()

        return {
            "ref_image": to_tensor(image),
            "ref_key": ref_key,
            "target_image": to_tensor(target_image),
            "target_key": target_key,
            "homo_mat": homo_mat
        }
    
    ############################################
    ## Pytorch and preprocessing related APIs ##
    ############################################
    # Get the length of the dataset
    def __len__(self):
        return self.dataset_length

    # Get data from the information from filename dataset
    @staticmethod
    def get_data_from_path(data_path):
        output = {}

        # Get image data
        image_path = data_path["image"]
        image = imread(image_path)
        output["image"] = image
        
        return output
    
    # The test preprocessing
    def test_preprocessing(self, data, numpy=False):
        # Fetch the corresponding entries
        image = data["image"]
        image_size = image.shape[:2]

        # Resize the image before photometric and homographical augmentations
        if not(list(image_size) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)[:2] # Only H and W dimensions

            image = cv2.resize(image, tuple(self.config['preprocessing']['resize'][::-1]),
                               interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)

        # Optionally convert the image to grayscale
        if self.config["gray_scale"]:
            image = (color.rgb2gray(image) *255.).astype(np.uint8)

        # Still need to normalize image
        image_transform = photoaug.normalize_image()
        image = image_transform(image)

        # Update image size
        image_size = image.shape[:2]
        valid_mask = np.ones(image_size)

        # Convert to tensor and return the results
        to_tensor = transforms.ToTensor()
        if not numpy:
            return {
                "image": to_tensor(image),
                "valid_mask": to_tensor(valid_mask).to(torch.int32)
            }
        else:
            return {
                "image": image,
                "valid_mask": valid_mask.astype(np.int32)
            }
    
    # Define the getitem method
    def __getitem__(self, idx):
        """Return data
        file_key: str, keys used to retrieve certain data from the filename dataset.
        image: torch.float, C*H*W range 0~1,
        valid_mask: torch.int32, 1*H*W range 0 or 1
        """
        # Get the corresponding datapoint and get contents from filename dataset
        file_key = self.datapoints[idx]
        data_path = self.filename_dataset[file_key]
        # Read in the image and npz labels (but haven't applied any transform)
        data = self.get_data_from_path(data_path)

        # Perform transform and augmentation
        if self.mode == "train" or self.config["add_augmentation_to_all_splits"]:
            raise NotImplementedError
        else:
            data = self.test_preprocessing(data)
        
        # Add file key to the output
        data["file_key"] = file_key
        
        return data
    
if __name__ == "__main__":
    import sys
    import yaml
    import matplotlib
    import matplotlib.pyplot as plt
    plt.switch_backend("TkAgg")
    from torch.utils.data import DataLoader
    sys.path.append("../")

    # Load configuration file
    with open("./config/yorkurban_dataset_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["add_augmentation_to_all_splits"] = False

    # Initialize the dataset
    test_dataset = YorkUrbanDataset(mode="test", config=config)
    import ipdb; ipdb.set_trace()