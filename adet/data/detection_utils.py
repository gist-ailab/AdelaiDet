import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno

import math

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import pycocotools.mask as mask_utils



# def transform_instance_annotations(
#     annotation, transforms, image_size, *, keypoint_hflip_indices=None
# ):

#     annotation = d2_transform_inst_anno(
#         annotation,
#         transforms,
#         image_size,
#         keypoint_hflip_indices=keypoint_hflip_indices,
#     )

#     if "beziers" in annotation:
#         beziers = transform_beziers_annotations(annotation["beziers"], transforms)
#         annotation["beziers"] = beziers
#     return annotation

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        annotation = transform_segm_in_anno(annotation, transforms, "segmentation")

    if "visible_mask" in annotation:
        annotation = transform_segm_in_anno(annotation, transforms, "visible_mask")
        
    if "occluded_mask" in annotation:
        annotation = transform_segm_in_anno(annotation, transforms, "occluded_mask")

        
    return annotation

def mask_to_rle(mask):
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def transform_segm_in_anno(annotation, transforms, key):
    segm = annotation[key]
    if isinstance(segm, list):
        # polygons
        polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
        annotation[key] = [
            p.reshape(-1) for p in transforms.apply_polygons(polygons)
        ]
    elif isinstance(segm, dict):
        # RLE
        mask = mask_utils.decode(segm)
        mask = transforms.apply_segmentation(mask)
        annotation[key] = mask_to_rle(np.array(mask, dtype=np.uint8, order='F'))
    else:
        raise ValueError(
            "Cannot transform segmentation of type '{}'!"
            "Supported types are: polygons as list[list[float] or ndarray],"
            " COCO-style RLE as a dict.".format(type(segm))
        )
    return annotation



def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


# def annotations_to_instances(annos, image_size, mask_format="polygon"):
#     instance = d2_anno_to_inst(annos, image_size, mask_format)

#     if not annos:
#         return instance

#     # add attributes
#     if "beziers" in annos[0]:
#         beziers = [obj.get("beziers", []) for obj in annos]
#         instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

#     if "rec" in annos[0]:
#         text = [obj.get("rec", []) for obj in annos]
#         instance.text = torch.as_tensor(text, dtype=torch.int32)

#     return instance


def convert_to_mask(segms):
    masks = []
    for segm in segms:
        if isinstance(segm, list):
            # polygon
            masks.append(polygons_to_bitmask(segm, *image_size))
        elif isinstance(segm, dict):
            # COCO RLE
            masks.append(mask_utils.decode(segm))
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        else:
            raise ValueError(
                "Cannot convert segmentation of type '{}' to BitMasks!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict, or a binary segmentation mask "
                " in a 2D numpy array of shape HxW.".format(type(segm))
            )
    return masks

def merge_bitmask(masks):
    return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

class_id_map = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7, 
    8: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    47: 18,
    48: 19,
    49: 20,
    50: 21,
    51: 22,
    52: 23,
    53: 24,
    54: 25,
    56: 26,
    58: 27,
    59: 28,
    60: 29,
    61: 30,
    62: 31,
    75: 32,
    76: 33,
    77: 34,
    78: 35,
    79: 36,
    80: 37,
    81: 38,
    82: 39,
    83: 40
}

def annotations_to_instances(annos, image_size, mask_format="polygon", amodal=False):
    
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    if amodal:
        occ_classes = [int(obj["occluded_rate"] >= 0.05) for obj in annos]
        target.gt_occludeds = torch.tensor(occ_classes, dtype=torch.int64)

    classes = [int(obj["category_id"]) for obj in annos]
    ## HSE class remap
    # classes = [class_id_map[clas]-1 for clas in classes]
    # print('classes', classes)
    # print('classes_remap', classes_remap)
    
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    # print('prev', classes, 'remap', classes_remap)

    if len(annos) and "segmentation" in annos[0]:

        if amodal:
            amodal_masks = convert_to_mask([obj["segmentation"] for obj in annos])
            visible_masks = convert_to_mask([obj["visible_mask"] for obj in annos])
            occluded_masks = convert_to_mask([obj["occluded_mask"] for obj in annos])
        else:
            visible_masks = convert_to_mask([obj["segmentation"] for obj in annos])

        if amodal:
            target.gt_masks = merge_bitmask(amodal_masks)
            target.gt_visible_masks = merge_bitmask(visible_masks)
            target.gt_occluded_masks = merge_bitmask(occluded_masks)
            target.gt_occluded_rate = torch.Tensor([obj["occluded_rate"] for obj in annos])
        else:
            target.gt_masks = merge_bitmask(visible_masks)
            target.gt_boxes = target.gt_masks.get_bounding_boxes()

    if not annos:
        return target

    return target


def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""



class HeatmapGenerator():
    def __init__(self, num_joints, sigma, head_sigma):
        self.num_joints = num_joints
        self.sigma = sigma
        self.head_sigma = head_sigma

        self.p3_sigma = sigma / 2

        size = 2*np.round(3 * sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        size = 2*np.round(3 * self.p3_sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.p3_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.p3_sigma ** 2))

        size = 2*np.round(3 * head_sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.head_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * head_sigma ** 2))

    def __call__(self, gt_instance, gt_heatmap_stride):
        heatmap_size = gt_instance.image_size
        heatmap_size = [math.ceil(heatmap_size[0]/ 32)*(32/gt_heatmap_stride),
                    math.ceil(heatmap_size[1]/ 32)*(32/gt_heatmap_stride)]

        h,w = heatmap_size
        h,w = int(h),int(w) 
        joints = gt_instance.gt_keypoints.tensor.numpy().copy()
        joints[:,:,[0,1]] = joints[:,:,[0,1]] / gt_heatmap_stride
        sigma = self.sigma
        head_sigma = self.head_sigma
        p3_sigma = self.p3_sigma

        output_list = []
        head_output_list = []
        for p in joints:
            hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            head_hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= w or y >= h:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

                    ul = int(np.round(x - 3 * head_sigma - 1)), int(np.round(y - 3 * head_sigma - 1))
                    br = int(np.round(x + 3 * head_sigma + 2)), int(np.round(y + 3 * head_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    head_hms[idx, aa:bb, cc:dd] = np.maximum(
                        head_hms[idx, aa:bb, cc:dd], self.head_g[a:b, c:d])
                    
            hms = torch.from_numpy(hms)
            head_hms = torch.from_numpy(head_hms)
            output_list.append(hms)
            head_output_list.append(head_hms)

        h,w = h//4, w//4
        p3_output_list = []
        joints = gt_instance.gt_keypoints.tensor.numpy().copy()
        joints[:,:,[0,1]] = joints[:,:,[0,1]] / 8
        for p in joints:
            p3_hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= w or y >= h:
                        continue

                    ul = int(np.round(x - 3 * p3_sigma - 1)), int(np.round(y - 3 * p3_sigma - 1))
                    br = int(np.round(x + 3 * p3_sigma + 2)), int(np.round(y + 3 * p3_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    p3_hms[idx, aa:bb, cc:dd] = np.maximum(
                        p3_hms[idx, aa:bb, cc:dd], self.p3_g[a:b, c:d])
                    
            p3_hms = torch.from_numpy(p3_hms)
            p3_output_list.append(p3_hms)
        output_list = torch.stack(output_list,dim=0)
        p3_output_list = torch.stack(p3_output_list,dim=0)
        head_output_list = torch.stack(head_output_list,dim=0)
        gt_instance.keypoint_heatmap = output_list
        gt_instance.head_heatmap = head_output_list
        gt_instance.p3_output_list = p3_output_list
        return gt_instance