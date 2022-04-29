import os
import torch
from numpy import clip


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)



def mask_to_bbox(mask:torch.Tensor):
    """
    Get the bounding box indices for the mask.
    Determines the smallest possible bounding box around non-background
    voxels in the mask (assuming background voxels are labeled 0).

    Supports 1 mask at a time.

    Arguments:
        mask: torch.Tensor (predicted) segmentation mask of a scan

    Returns:
        x_lim: Start and end index of bounding box in x-direction.
        y_lim: Start and end index of bounding boy in y-direction.
        z_lim: Start and end index of bounding boy in z-direction.
    """
    assert len(mask.shape) == 4, "Mask should be in CHWD format"

    # reduce to HWD 
    if mask.shape[0] == 3:
        mask = mask.argmax(dim=0)
    elif mask.shape[0] == 1:
        mask = mask.squeeze(0)

    # using keepdim=True for later readibility of the reduced dims 
    xmin, xmax = torch.where(torch.any(torch.any(mask,dim=1,keepdim=True),dim=2,keepdim=True))[0][[0,-1]]
    ymin, ymax = torch.where(torch.any(torch.any(mask,dim=0,keepdim=True),dim=2,keepdim=True))[1][[0,-1]]
    zmin, zmax = torch.where(torch.any(torch.any(mask,dim=0,keepdim=True),dim=1,keepdim=True))[2][[0,-1]]

    return (xmin.item(), xmax.item()), (ymin.item(), ymax.item()), (zmin.item(),zmax.item())

def crop_to_bbox(scan:torch.Tensor, mask:torch.Tensor, margin=None):

    assert len(scan.shape) == 5, "Scan should be BHCWD format"
    assert scan.shape[0] == 1, "Only batch_size=1 is supported."
     
    if type(margin) == int:
        margin = [margin] * 3
    elif type(margin) == list:
        assert len(margin) == 3, "If given as a list, margin should be specified for each dimension"

    shape = scan.shape[2:]
    bbox = mask_to_bbox(mask)
    import pdb; pdb.set_trace()
    if margin:
        new_bbox = []
        for i, (minv, maxv) in enumerate(bbox):
            new_bbox.append(tuple(
                clip((minv - margin[i], maxv + margin[i]), 0, shape[i]-1)
            ))
        bbox = tuple(new_bbox)
    import pdb; pdb.set_trace()
    scan = scan[:,:,bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]]

    return scan