

import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import json
from pycocotools import mask as m
import datetime

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



def mask_to_rle(mask):
    rle = m.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) ==0:
        return None, None, None, None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)


def create_image_info(image_id, color_path, depth_path, W, H):
    return {
        "id": image_id,
        "file_name": "/".join(color_path.split("/")[-5:]), 
        "depth_file_name": "/".join(depth_path.split("/")[-5:]),
        "width": W,
        "height": H,
        "date_captured": datetime.datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }        


def get_file_paths_from_syn(data_root, split, file_paths):

    file_path_name =  "{}_file_paths.json".format(split)
    file_path_json = os.path.join(data_root, file_path_name)
    
    split_info_syn_path = "./assets/split_info_syn.json"
    with open(split_info_syn_path, "r") as f:
        split_info_syn = json.load(f)

    scene_folder_paths = sorted(glob.glob(os.path.join(data_root, "data2_syn_source", "train_pbr/*")))
    for scene_folder_path in tqdm(scene_folder_paths):
        scene_id = int(scene_folder_path.split("/")[-1])
        # !TODO: change this
        if str(scene_id) not in split_info_syn.keys():
            continue
        if split != split_info_syn[str(scene_id)]:
            continue
    
        scene_gt_path = os.path.join(scene_folder_path, "scene_gt.json")
        with open(scene_gt_path, "r") as f:
            scene_gts = json.load(f)

        rgbs = sorted(glob.glob(os.path.join(scene_folder_path, "rgb/*")))
        for rgb in rgbs:
            image_id = rgb.split("/")[-1].split(".")[0]
            if int(image_id) < 1:
                continue
            scene_gt = scene_gts[str(int(image_id))]
 
            visible_masks = glob.glob(os.path.join(scene_folder_path, f'mask_visib/{image_id}_*'))
            amodal_masks = []
            category_ids = []
            for visible_mask in visible_masks:
                inst_id = visible_mask.split("/")[-1].split("_")[-1].split(".")[0]
                amodal_mask = os.path.join(scene_folder_path, f'mask/{image_id}_{inst_id}.png')
                amodal_masks.append(amodal_mask)
                category_ids.append(scene_gt[int(inst_id)]["obj_id"])

            file_paths["color"].append(rgb)
            file_paths["depth"].append(os.path.join(scene_folder_path, "depth", image_id + ".png"))
            file_paths["visible_mask"].append(visible_masks)
            file_paths["amodal_mask"].append(amodal_masks)
            file_paths['category_id'].append(category_ids)

    with open(file_path_json, "w") as f:
        json.dump(file_paths, f)
    return file_paths


def get_file_paths_from_real(data_root, split, file_paths):

    file_path_name =  "{}_file_paths.json".format(split)
    file_path_json = os.path.join(data_root, file_path_name)

    split_info_real_path = "./assets/split_info_syn.json"
    with open(split_info_real_path, "r") as f:
        split_info_real = json.load(f)

    object_set_info_path = "./assets/object_set_info.json"
    with open(object_set_info_path, "r") as f:
        object_set_info = json.load(f)

    scene_folder_paths = sorted(glob.glob(os.path.join(data_root, "data2_real_source", "all/*")))
    for scene_folder_path in tqdm(scene_folder_paths):
        scene_id = int(scene_folder_path.split("/")[-1])
        # !TODO: change this
        if str(scene_id) not in object_set_info.keys():
            continue
        if split != split_info_real[str(scene_id)]:
            continue
        if "ycb" not in object_set_info[str(scene_id)]:
            continue

        scene_gt_path = os.path.join(scene_folder_path, "scene_gt_{:06d}.json".format(scene_id))
        with open(scene_gt_path, "r") as f:
            scene_gts = json.load(f)

        rgbs = sorted(glob.glob(os.path.join(scene_folder_path, "rgb/*")))
        for rgb in rgbs:
            image_id = rgb.split("/")[-1].split(".")[0]
            if int(image_id) < 1:
                continue
            scene_gt = scene_gts[str(int(image_id))]
 
            visible_masks = glob.glob(os.path.join(scene_folder_path, f'mask_visib/{image_id}_*'))
            amodal_masks = []
            category_ids = []
            for visible_mask in visible_masks:
                inst_id = visible_mask.split("/")[-1].split("_")[-1].split(".")[0]
                amodal_mask = os.path.join(scene_folder_path, f'mask/{image_id}_{inst_id}.png')
                amodal_masks.append(amodal_mask)
                category_ids.append(scene_gt[int(inst_id)]["obj_id"])

        
            file_paths["color"].append(rgb)
            file_paths["depth"].append(os.path.join(scene_folder_path, "depth", image_id + ".png"))
            file_paths["visible_mask"].append(visible_masks)
            file_paths["amodal_mask"].append(amodal_masks)
            file_paths['category_id'].append(category_ids)

    with open(file_path_json, "w") as f:
        json.dump(file_paths, f)
    return file_paths

def create_coco_annotation(data_root, split, task):
    
    file_paths = {"color": [], "depth": [], "visible_mask": [], "amodal_mask": [], "category_id": []}
    file_paths = get_file_paths_from_syn(data_root, split, file_paths)
    file_paths = get_file_paths_from_real(data_root, split, file_paths)

    coco_json_path = "coco_anns_clora_{}_{}.json".format(task, split)

    print("Loading dataset for {}, {} Num_IMG: {}".format(task, split, len(file_paths["color"])))
    annotations = []
    image_infos = []
    annotation_id = 1

    for img_id in tqdm(range(len(file_paths["color"]))):

        color_path = file_paths["color"][img_id] 
        depth_path = file_paths["depth"][img_id]
        visible_mask_paths = file_paths["visible_mask"][img_id]
        visible_masks = {}
        category_ids = {}
        if task == "amodal":
            amodal_mask_paths = file_paths["amodal_mask"][img_id]
            amodal_masks = {}
            occluded_masks = {}
            occluded_rates = {}

        inst_id = 0
        for idx, visible_mask_path in enumerate(visible_mask_paths):
            visible_mask = cv2.imread(visible_mask_path)
            visible_mask = np.array(visible_mask[:, :, 0], dtype=bool, order='F')

            if task == "amodal":
                amodal_mask_path = amodal_mask_paths[idx]
                amodal_mask = cv2.imread(amodal_mask_path)
                amodal_mask = np.array(amodal_mask[:, :, 0], dtype=bool, order='F')
                
                # get only occluded mask with overlapping ratio > 0.05
                occluded_mask_all = np.uint8(np.logical_and(amodal_mask, np.logical_not(visible_mask)))
                valid_contours = []
                occluded_rate = 0
                contours, _ = cv2.findContours(np.uint8(occluded_mask_all*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                valid_contours = []
                for contour in contours:
                    overlapping_ratio = cv2.contourArea(contour) / np.sum(amodal_mask)
                    if overlapping_ratio >= 0.05:
                        valid_contours.append(contour)
                
                occluded_mask = np.uint8(np.zeros((480, 640)))
                if len(valid_contours) != 0: 
                    # if occluded -> amodal = visible + occluded
                    occluded_mask = cv2.drawContours(occluded_mask, valid_contours, -1, 255, -1)
                    occluded_mask = np.array(occluded_mask, dtype=bool, order='F')
                    amodal_mask = np.bitwise_or(occluded_mask, visible_mask)
                else: # if no occluded -> amodal = visible
                    amodal_mask = visible_mask
                    occluded_mask = np.zeros_like(visible_mask, dtype=bool, order='F')
            
            visible_masks[inst_id] = visible_mask
            category_ids[inst_id] = file_paths["category_id"][img_id][idx]

            if task == "amodal":
                amodal_masks[inst_id] = amodal_mask
                occluded_masks[inst_id] = occluded_mask
                occluded_rate = np.sum(occluded_mask) / np.sum(amodal_mask)
                occluded_rates[inst_id] = occluded_rate
            inst_id += 1

        for idx in visible_masks.keys():

            visible_mask = visible_masks[idx]
            visible_bbox = get_bbox(visible_mask)


            if visible_bbox[0] is None: 
                print("Filtering none bbox")
                continue
            if visible_bbox[2] <= 1 or visible_bbox[3] <= 1:
                print("Filtering too small mask", color_path)
                continue
            H, W = visible_mask.shape
            annotation = {}
            annotation["id"] = annotation_id
            annotation_id += 1
            annotation["image_id"] = img_id
            annotation["category_id"] = category_ids[idx]
            annotation["height"] = W
            annotation["width"] = W
            annotation["iscrowd"] = 0
            if task == "amodal":
                amodal_mask = amodal_masks[idx]
                occluded_mask = occluded_masks[idx]
                amodal_bbox = get_bbox(amodal_mask)
                annotation["bbox"] = amodal_bbox
                annotation["segmentation"] = mask_to_rle(amodal_mask)
                annotation["area"] = int(np.sum(amodal_mask))
                annotation["visible_mask"] = mask_to_rle(visible_mask)
                annotation["visible_bbox"] = visible_bbox
                annotation["occluded_mask"] = mask_to_rle(occluded_mask)
                annotation["occluded_rate"] = occluded_rates[idx]
                annotation["occluding_rate"] = np.sum(occluded_masks[idx]) / np.sum(amodal_masks[idx])
            elif task == "visible":
                annotation["bbox"] = visible_bbox
                annotation["segmentation"] = mask_to_rle(visible_mask)
                annotation["area"] = int(np.sum(visible_mask))

            annotations.append(annotation)
        image_infos.append(create_image_info(img_id, color_path, depth_path, W, H))

    coco_json["annotations"] = annotations
    coco_json["images"] = image_infos
    with open(coco_json_path, "w") as f:
        print("Saving annotation as COCO format to", coco_json_path)
        json.dump(coco_json, f)
    return coco_json_path

if __name__ == "__main__":
    
    data_root = "/OccludedObjectDataset/ours/data2"
    create_coco_annotation(data_root, split="train", task="visible")
    # create_coco_annotation(data_root, split="val")
    # create_coco_annotation(data_root, split="test")