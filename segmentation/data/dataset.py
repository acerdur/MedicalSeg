"""This module holds the Dataset class."""
import json
import os
import glob
import typing
import random 
from tqdm import tqdm
from typing import Any, Type, Union, Optional
from PIL import Image
from copy import deepcopy

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF

from punkreas.data import Dataset
from punkreas.data import ScanImage, MaskImage
from punkreas.transform import Compose

def random_transform_image_and_mask(img, mask, transforms: [Optional[dict]]):
    """
    Apply identical transformations to the image and its mask.

    Arguments:
        img:            torch.Tensor
        mask:           torch.Tensor
        transforms:     dict or None - Name of the transformations to be applied
    """
    if not transforms:
        return img, mask
    else:
        if 'resize' in transforms.keys():
            new_size = transforms['resize']
            img = TF.resize(img, size=new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size=new_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        if ('hflip' in transforms.keys()) and (random.random() > 0.5):
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        
        if ('vflip' in transforms.keys()) and (random.random() > 0.5):
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if ('rotate' in transforms.keys()) and (random.random() > 0.5):
            angle = random.randint(-90,90)
            img = TF.rotate(img, angle=angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle=angle, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        return img, mask

    

class SegmentationDataset(Dataset):
    """Medical image segmentation dataset for PyTorch.

    This class is inherited from punkreas' base Dataset class. It overrides 
    methods and discards "labels" which are useful for classification, and 
    rather uses "masks".

    Attributes:
        scans: List of the paths to the medical scan images.
        labels: List of labels corresponding to the scans. Just fed to the class, but will never be used.
        masks: List of the paths to the medical mask images.
    """

    def __init__(
        self,
        scans: list[str],
        labels: list,
        transform: "Transform" = None,
        scan_dtype: Type[Any] = np.float32,
        label_dtype: Type[Any] = np.int64,
        masks: Union[list[str], None] = None,
    ) -> None:
        """Create a medical image classification dataset from scans and labels.

        Use a list of file paths for scans and corresponding labels to create a
        medical image dataset.

        Args:
            scans: List of paths to the medical scans.
            labels: List of the labels corresponding to the medical scans.
            transform: Transformation applied the scan images.
            scan_dtype: Data type of scan data.
            label_dtype: Data type of label data.
            masks: Paths to masks corresponding the medical scans.
        """

        if (len(masks) != len(scans)):
            raise ValueError(
                "Number of scans and masks does not match! {} != {}".format(
                    len(scans), len(masks)
                )
            )

        # Save scans, labels and masks as public attributes
        self.scans = scans
        self.masks = masks

        # Save transformation and dtypes as private attribute
        self._transform = transform
        self._scan_dtype = scan_dtype
        self._label_dtype = label_dtype

        # Filter transformations for mask images ang generate compose class
        mask_acceptable_transforms = ['Resize', 'Pad', 'Crop', 'ToReferencePosition'] # crop class does not exist yet - necessary?
        
        # Prepare transformation instances on the dataset
        if self._transform:
            self._mask_transform = Compose([ 
            deepcopy(transform) for transform in self._transform._transformations if str(transform) in mask_acceptable_transforms
            ])
            self._transform.prepare(self)
            self._mask_transform.prepare(self,order=0) # change the interpolation order on mask resizing

        #import pdb; pdb.set_trace()
    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            length: Size of the dataset.
        """
        return len(self.scans)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get scans and labels by index.

        Args:
            idx: Index of the sample to return.

        Returns:
            torch_scan: The scan data tensor.
            torch_label: The label tensor.
        """

        # Select the sample
        scan = ScanImage.from_path(self.scans[idx], dtype=self._scan_dtype)
        mask = MaskImage.from_path(self.masks[idx], dtype=self._label_dtype) 
        # segmentation masks are not merged, i.e., loaded as {void, pancreas, tumor} labels


        #scan_name = self.scans[idx].split('/')[-1]

        # Apply transformation to scan & mask images
        if self._transform:
            self._transform(scan, idx)
            self._mask_transform(mask, idx)

        # integrated into punkreas transforms
        #scan.array = SegmentationDataset2D.rotate_and_flip(scan.array)
        #mask.array = SegmentationDataset2D.rotate_and_flip(mask.array) 

        #import pdb; pdb.set_trace()
        # Tranform to pytorch tensor
        torch_scan = scan.to_torch() #.transpose(1,3) # change to DHW format
        torch_mask = mask.to_torch().to(torch.long) #.transpose(0,2) # change masks back to integer

        return torch_scan, torch_mask, self.scans[idx]

    

class SegmentationDataset2D(Dataset):
    def __init__(
        self,
        dataroot: str,
        creation_transform: "Transform" = None,
        loading_transform: "Transform" = None,
        mode: str = 'segmentation',
        merge_masks: bool = False,
        output_type: str = 'single',
        temporal: int = 4,
        is_train: bool = True,
        indices_3d: list = None,
        force_create: bool = False,
    ) -> None:
        """Create a medical image classification dataset from scans and labels.

        Args:
            scans: List of paths to the medical scans.
            creation_transform: Transformation applied to the 3D scans before building the 2D dataset.
            loading_transform: Transformation applied to the 2D images before loading to the model. Can be data augmentations.
            scan_dtype: Data type of scan data.
            label_dtype: Data type of label data.
            masks: Paths to masks corresponding the medical scans.
            mode: Pixel-wise ''segmentation' or slice 'classification'
            merge_masks: Gather pancreas and tumor labels together as pancreas
            output_type: 'Single' image output for 2D or 'sequnce' for 2.5D models
            temporal: Sequence length of input for 2.5D model
            is_train: For 2.5D models, whether the output should be a random sequence 
                of 'temporal' length, or the whole scan 
            indices_3d: If given, dataset is constructed only using the given index of 3D scans,
                useful to construct train & val datasets
        """
        self.dataroot = dataroot
        self.mode = mode
        self.output_type = output_type
        self.temporal = temporal
        self.is_train = is_train
        self.merge_masks = merge_masks

        if self.mode == 'classification':
            # remove redundant operation in __getitem__
            self.merge_masks = False

        # Save transformation and dtypes as private attribute
        self._transform3d = creation_transform
        self._mask_transform3d = None
        self._load_transfrom = loading_transform
        
        # Filter transformations for mask images ang generate compose class
        mask_acceptable_transforms = ['Resize', 'Pad', 'Crop', 'ToReferencePosition'] # crop class does not exist yet - necessary?
        
        # Prepare transformation instances on the 3D dataset
        if self._transform3d:
            self._mask_transform3d = Compose([ 
            deepcopy(transform) for transform in self._transform3d._transformations if str(transform) in mask_acceptable_transforms
            ])
            self._transform3d.prepare(self)
            self._mask_transform3d.prepare(self,order=0) # change the interpolation order on mask resizing

        if self._load_transfrom:
            self._load_transfrom = {k.lower():v for k,v in self._load_transfrom.items()}
            for t in self._load_transfrom.keys():
                assert t in ['rotate', 'hflip', 'vflip', 'resize']

        self.scan_paths, self.mask_paths, self.scan_names = SegmentationDataset2D.create_2d_dataset(self.dataroot, [self._transform3d, self._mask_transform3d], force=force_create, indices=indices_3d)
        
        ## regroup paths into nested lists for sequence case
        if self.output_type == 'sequence':
            sequenced_scan_paths = []
            sequenced_mask_paths = []
            for scan_name in self.scan_names:
                scan_slices_paths = [pth for pth in self.scan_paths if scan_name in pth]
                # sort the filenames by X in <scan>_sliceX.png for correct ordering of scans
                scan_slices_paths.sort(key=lambda pth: int(''.join(filter(str.isdigit, pth.split('/')[-1].split('_')[-1]))))
                mask_slices_paths = [pth.replace('scans', 'masks') for pth in scan_slices_paths]
                if self.is_train:
                    ## [[seq1],[seq2],...[seqN]] with seqi having <self.temporal> length
                    times = len(scan_slices_paths) // self.temporal
                    for t in range(times):
                        seq_start = t * self.temporal
                        slice_sequences = scan_slices_paths[seq_start:seq_start + self.temporal]
                        mask_sequences = mask_slices_paths[seq_start:seq_start + self.temporal]
                        sequenced_scan_paths.append(slice_sequences)
                        sequenced_mask_paths.append(mask_sequences)
                else:
                    ## [[seq1],[seq2],...[seqN]] with seqi having length of the whole scan
                    sequenced_scan_paths.append(scan_slices_paths)
                    sequenced_mask_paths.append(mask_slices_paths)
                
            self.scan_paths = sequenced_scan_paths
            self.mask_paths = sequenced_mask_paths 

        elif self.output_type != 'single':
            raise KeyError('Unknown output type')

        assert len(self.scan_paths) == len(self.mask_paths)

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            length: Size of the dataset.
        """
        return len(self.scan_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get scans and labels by index.

        Args:
            idx: Index of the sample to return.

        Returns:
            torch_slice: The scan data tensor.
            torch_label: The label tensor.
        """
        # for single outputs this is a sinlge path-string, for sequences it is a list
        slice_path = self.scan_paths[idx]
        mask_path = self.mask_paths[idx] 

        if self.output_type == 'single':
            img, mask = Image.open(slice_path), Image.open(mask_path)
            torch_slice = torchvision.transforms.ToTensor()(img) # 1 x H x W
            torch_mask = torchvision.transforms.ToTensor()(mask) # # 1 x H x W
            scan_name = slice_path.split('/')[-1].split('_')[0]
        else:
            imgs = [torchvision.transforms.ToTensor()(Image.open(pth)) for pth in slice_path]
            masks = [torchvision.transforms.ToTensor()(Image.open(pth)) for pth in mask_path]
            torch_slice = torch.cat(imgs,dim=0).unsqueeze(1) # temporal x 1 x H x W
            torch_mask = torch.cat(masks,dim=0) # temporal x H x W
            scan_name = slice_path[0].split('/')[-1].split('_')[0]
        
        torch_mask = torch.round(torch_mask*2).to(int)

        torch_slice, torch_mask = random_transform_image_and_mask(torch_slice, torch_mask, self._load_transfrom)

        if self.merge_masks:
            torch_mask = torch.where(torch_mask != 0, 1, torch_mask)

        if self.mode == 'classification':
            torch_mask = SegmentationDataset2D.mask_to_label(torch_mask) # temporal x 1 
        
        return torch_slice, torch_mask, scan_name

    @staticmethod
    def get_slice_images(scan, mask):
        """
        Divides the scan into 1-channel images of all slices 
        Works with a single scan 

        Arguments: 
            scan:       np.array,
            mask:       np.array
        Returns:
            slices:     list(np.array)
            masks:      list(np.array)
        """
        
        depth = scan.shape[-1]
        slices = np.split(scan, depth, axis=2)  # list of all HxW slices
        masks = np.split(mask, depth, axis=2)
            
        return slices, masks
    
    @staticmethod
    def rotate_and_flip(array) -> np.array:
        """
        Perform rotations and flipping to move patient into reference position

        Arguments:  
            array:  np.array
        Return:
            np.array
        """
        array = np.rot90(array)
        array = np.fliplr(array)

        return array

    @staticmethod
    def mask_to_label(mask) -> torch.Tensor:
        """
        Gather the segmentation mask of the slice to a binary scalar label. 
        Means, if there is at least one pixel equal to 1, whole slice is labeled as 1 
        (pancreas in slice)
        
        Works for both for whole 3D masks or a single 2D slice.

        Arguments:
            slice: torch.Tensor, if 3D must be DHW format
        Return:
            torch.Tensor
        """
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0) #np.expand_dims(mask, axis=2) 

        label = torch.any(torch.any(mask,dim=1),dim=1).to(int) #np.any(mask, axis=(1,2))

        return label

    @classmethod
    def create_2d_dataset(cls, dataroot, transforms=None, force=False, max_number_scans=float("inf"), indices=None):
        """
        Take the 3D scans and save each slice as a 1-channel .jpg with the folder structure:
        <scan_id>_slice<slice_id>.jpg

        Arguments: 
            dataroot:   str     parent path that contains 3D scan and 2D directories both
            transforms: list    composed transforms to be applied on scans, should have order [scan_transforms, mask_transforms]
            force:      bool    flag to create dataset again even if it exists
            indices:    list    index of 3D scans to be used only 
        """
        scan_paths_3d = glob.glob(os.path.join(dataroot, "scans", "*"))
        mask_paths_3d = glob.glob(os.path.join(dataroot, "masks", "*"))
        if indices:
            scan_paths_3d = [scan_paths_3d[i] for i in indices]
            mask_paths_3d = [mask_paths_3d[i] for i in indices]

        images_2d_root = os.path.join(dataroot, "2D", "scans")
        masks_2d_root = os.path.join(dataroot, "2D", "masks")

        if (len(mask_paths_3d) != len(scan_paths_3d)):
            raise ValueError(
                "Number of scans and masks does not match! {} != {}".format(
                    len(scan_paths_3d), len(mask_paths_3d)
                )
            )
        scan_names = [pth.split('/')[-1].split('.')[0] for pth in scan_paths_3d]
        dataset_exists = True
        try:
            img_paths = glob.glob(os.path.join(images_2d_root, "*"))
            mask_paths = glob.glob(os.path.join(masks_2d_root, "*"))
            img_names = [pth.split('/')[-1].split('.')[0].split('_')[0] for pth in img_paths]
            mask_names = [pth.split('/')[-1].split('.')[0].split('_')[0] for pth in mask_paths]

            if not (set(scan_names).issubset(set(img_names)) or set(scan_names).issubset(set(mask_names))):
                # flag that dataset has missing scans and should be created again
                dataset_exists = False
        except:
            # flag that the dataset does not exist and should be created
            dataset_exists = False

        
        if force or not dataset_exists:
            #create folders
            os.makedirs(images_2d_root, exist_ok=True)
            os.makedirs(masks_2d_root, exist_ok=True)

            img_paths = []
            mask_paths = []

            print("*"*40 + " Preparing dataset " + "*"*40)
            #start iterating over 3D scans to create dataset
            for i, pth in enumerate(tqdm(scan_paths_3d)):
                if i == max_number_scans:
                    break

                scan_name = pth.split('/')[-1].split('.')[0]

                scan =  ScanImage.from_path(pth, dtype=np.float32)
                mask = MaskImage.from_path(mask_paths_3d[i], dtype=np.int64) 
                
                if transforms:
                    transforms[0](scan, i)
                    transforms[1](mask, i) 
                
                #integrated into punkreas transforms
                # scan.array = cls.rotate_and_flip(scan.array)
                # mask.array = cls.rotate_and_flip(mask.array) 

                # only contain regions where the scan is nonzero
                h,w,d = scan.shape
                zero_vol = np.zeros((h,w,1))
                no_zero_start = 0 
                for d_i in range(d):
                    if not (scan.array[:,:,d_i] == zero_vol).all():
                        no_zero_start = d_i
                        break
                no_zero_end = 0
                for d_i in range(no_zero_start, d):
                    if not (scan.array[:,:,d_i] == zero_vol).all():
                        no_zero_end += 1
                scan.array = scan.array[:,:,no_zero_start:no_zero_end]
                mask.array = mask.array[:,:,no_zero_start:no_zero_end]

                mask.array = (mask.array - mask.min()) / (mask.max() + mask.min()) #normalize into [0,1] range

                slices, masks = cls.get_slice_images(scan.array, mask.array)
                for j, slc in enumerate(slices):
                    slc = (slc * 255).astype("uint8")
                    slc = Image.fromarray(slc.squeeze(2),mode='L')
                    slc.save(os.path.join(images_2d_root, f"{scan_name}_slice{j + 1}.png"),'PNG')
                    img_paths.append(os.path.join(images_2d_root, f"{scan_name}_slice{j + 1}.png"))
                    msk = (masks[j] * 255).astype("uint8")
                    msk = Image.fromarray(msk.squeeze(2),mode='L')
                    msk.save(os.path.join(masks_2d_root, f"{scan_name}_slice{j + 1}.png"),'PNG')
                    mask_paths.append(os.path.join(masks_2d_root, f"{scan_name}_slice{j + 1}.png"))


        else:
            print("*"*10 + f"  Dataset exists. Loading from {os.path.join(dataroot, '2D')}  " + "*"*10)
            filtered_img_paths = []
            filtered_mask_paths = []
            for scan_name in scan_names:
                # only take the .png files that contains the selected 3D scan names, 
                # convenient for train, val split
                img_paths_contains_scan = glob.glob(os.path.join(images_2d_root, f"{scan_name}_*"))
                mask_paths_contains_scan = glob.glob(os.path.join(masks_2d_root, f"{scan_name}_*"))
                filtered_img_paths.extend(img_paths_contains_scan)
                filtered_mask_paths.extend(mask_paths_contains_scan)
            img_paths = filtered_img_paths
            mask_paths = filtered_mask_paths

        return img_paths, mask_paths, scan_names




if __name__ == '__main__':
    import punkreas.transform as transform
    dataroot = '/home/erdurc/punkreas/segmentation/datasets/MSD'
    
    transforms = {'Clip': {'amin': -150, 'amax': 250},
                'Normalize': {'bounds': [-150, 250]}
                  }
    # transformations = [
    #     Compose([getattr(transform,name)(**options) for name,options in transforms.items()])
    #     ]
    transformations = [None]
    dataset = SegmentationDataset2D(dataroot=dataroot, creation_transform=transformations[0], mode='classification', output_type='single', is_train=False)
    # import pdb; pdb.set_trace()
    dataset1 = SegmentationDataset2D(dataroot=dataroot, creation_transform=transformations[0], mode='segmentation', output_type='sequence', is_train=False)
    import pdb; pdb.set_trace()
    # dataset = SegmentationDataset2D(dataroot=dataroot, transform=transformations[0], mode='classification', output_type='sequence', is_train=False)
    # import pdb; pdb.set_trace() 