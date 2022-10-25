# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

from re import L

LMO_CATEGORIES = [
    {"color": [0, 0, 230], "isthing": 1, "id": 1, "name": "obj_000001"}, # ape
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "obj_000005"}, # can
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "obj_000006"}, # cat
    {"color": [119, 11, 32], "isthing": 1, "id": 8, "name": "obj_000008"}, # drill
    {"color": [0, 0, 70], "isthing": 1, "id": 9, "name": "obj_000009"}, # duck
    {"color": [0, 80, 100], "isthing": 1, "id": 10, "name": "obj_000010"}, # egg_box
    {"color": [220, 20, 60], "isthing": 1, "id": 11, "name": "obj_000011"}, # glue
    {"color": [0, 0, 142], "isthing": 1, "id": 12, "name": "obj_000012"}, # hole_punch
]


YCBV_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "obj_000001"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "obj_000002"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "obj_000003"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "obj_000004"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "obj_000005"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "obj_000006"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "obj_000007"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "obj_000008"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "obj_000009"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "obj_000010"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "obj_000011"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "obj_000012"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "obj_0000013"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "obj_000014"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "obj_000015"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "obj_000016"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "obj_000017"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "obj_000018"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "obj_000019"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "obj_000020"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "obj_000021"}
]


CLORA_DATA2_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "banana"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "apple"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "pear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "strawberry"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "orange"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "peach"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "plum"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "lemon"},
    {"color": [250, 170, 30], "isthing": 1, "id": 9, "name": "master_chef_can"},
    {"color": [100, 170, 30], "isthing": 1, "id": 10, "name": "cracker_box"},
    {"color": [220, 220, 0], "isthing": 1, "id": 11, "name": "sugar_box"},
    {"color": [175, 116, 175], "isthing": 1, "id": 12, "name": "mustard_bottle"},
    {"color": [250, 0, 30], "isthing": 1, "id": 13, "name": "tomato_soup_can"},
    {"color": [165, 42, 42], "isthing": 1, "id": 14, "name": "tuna_fish_can"},
    {"color": [109, 63, 54], "isthing": 1, "id": 15, "name": "pudding_box"},
    {"color": [207, 138, 255], "isthing": 1, "id": 16, "name": "gelatin_box"},
    {"color": [151, 0, 95], "isthing": 1, "id": 17, "name": "potted_meat_milk"},
    {"color": [9, 80, 61], "isthing": 1, "id": 18, "name": "sponge"},
    {"color": [84, 105, 51], "isthing": 1, "id": 19, "name": "fork"},
    {"color": [74, 65, 105], "isthing": 1, "id": 20, "name": "knife"},
    {"color": [166, 196, 102], "isthing": 1, "id": 21, "name": "spoon"},
    {"color": [199, 100, 0], "isthing": 1, "id": 22, "name": "spatula"},
    {"color": [72, 0, 118], "isthing": 1, "id": 23, "name": "bleach_cleanser"},
    {"color": [255, 179, 240], "isthing": 1, "id": 24, "name": "scissors"},
    {"color": [0, 125, 92], "isthing": 1, "id": 25, "name": "large_marker"},
    {"color": [209, 0, 151], "isthing": 1, "id": 26, "name": "foam_brick"},
    {"color": [188, 208, 182], "isthing": 1, "id": 27, "name": "spanner"},
    {"color": [0, 220, 176], "isthing": 1, "id": 28, "name": "flat_screwdriver"},
    {"color": [255, 99, 164], "isthing": 1, "id": 29, "name": "philips_screwdriver"},
    {"color": [92, 0, 73], "isthing": 1, "id": 30, "name": "extra_large_clamp"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "large_clamp"},
    {"color": [78, 180, 255], "isthing": 1, "id": 32, "name": "cooking_skillet_with_glass_lid"},
    {"color": [0, 228, 0], "isthing": 1, "id": 33, "name": "mug"},
    {"color": [174, 255, 243], "isthing": 1, "id": 34, "name": "plate"},
    {"color": [45, 89, 255], "isthing": 1, "id": 35, "name": "bowl"},
    {"color": [134, 134, 103], "isthing": 1, "id": 36, "name": "wood_block"},
    {"color": [145, 148, 174], "isthing": 1, "id": 37, "name": "pitcher_base"},
    {"color": [255, 208, 186], "isthing": 1, "id": 38, "name": "hammer"},
    {"color": [197, 226, 255], "isthing": 1, "id": 39, "name": "power_drill"},
    {"color": [171, 134, 1], "isthing": 1, "id": 40, "name": "padlock"},
]  


CLORA_DATA2_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "banana"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "apple"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "pear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "strawberry"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "orange"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "peach"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "plum"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "lemon"},
    {"color": [250, 170, 30], "isthing": 1, "id": 9, "name": "master_chef_can"},
    {"color": [100, 170, 30], "isthing": 1, "id": 10, "name": "cracker_box"},
    {"color": [220, 220, 0], "isthing": 1, "id": 11, "name": "sugar_box"},
    {"color": [175, 116, 175], "isthing": 1, "id": 12, "name": "mustard_bottle"},
    {"color": [250, 0, 30], "isthing": 1, "id": 13, "name": "tomato_soup_can"},
    {"color": [165, 42, 42], "isthing": 1, "id": 14, "name": "tuna_fish_can"},
    {"color": [109, 63, 54], "isthing": 1, "id": 15, "name": "pudding_box"},
    {"color": [207, 138, 255], "isthing": 1, "id": 16, "name": "gelatin_box"},
    {"color": [151, 0, 95], "isthing": 1, "id": 17, "name": "potted_meat_can"},
    {"color": [9, 80, 61], "isthing": 1, "id": 18, "name": "sponge"},
    {"color": [84, 105, 51], "isthing": 1, "id": 19, "name": "fork"},
    {"color": [74, 65, 105], "isthing": 1, "id": 20, "name": "knife"},
    {"color": [166, 196, 102], "isthing": 1, "id": 21, "name": "spoon"},
    {"color": [199, 100, 0], "isthing": 1, "id": 22, "name": "spatula"},
    {"color": [72, 0, 118], "isthing": 1, "id": 23, "name": "bleach_cleanser"},
    {"color": [255, 179, 240], "isthing": 1, "id": 24, "name": "scissors"},
    {"color": [0, 125, 92], "isthing": 1, "id": 25, "name": "large_marker"},
    {"color": [209, 0, 151], "isthing": 1, "id": 26, "name": "foam_brick"},
    {"color": [188, 208, 182], "isthing": 1, "id": 27, "name": "spanner"},
    {"color": [0, 220, 176], "isthing": 1, "id": 28, "name": "flat_screwdriver"},
    {"color": [255, 99, 164], "isthing": 1, "id": 29, "name": "phillips_screwdriver"},
    {"color": [92, 0, 73], "isthing": 1, "id": 30, "name": "extra_large_clamp"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "large_clamp"},
    {"color": [78, 180, 255], "isthing": 1, "id": 32, "name": "cooking_skillet_with_glass_lid"},
    {"color": [0, 228, 0], "isthing": 1, "id": 33, "name": "mug"},
    {"color": [174, 255, 243], "isthing": 1, "id": 34, "name": "plate"},
    {"color": [45, 89, 255], "isthing": 1, "id": 35, "name": "bowl"},
    {"color": [134, 134, 103], "isthing": 1, "id": 36, "name": "wood_block"},
    {"color": [145, 148, 174], "isthing": 1, "id": 37, "name": "pitcher_base"},
    {"color": [255, 208, 186], "isthing": 1, "id": 38, "name": "hammer"},
    {"color": [197, 226, 255], "isthing": 1, "id": 39, "name": "power_drill"},
    {"color": [171, 134, 1], "isthing": 1, "id": 40, "name": "padlock"},
]  


def _get_uoais_instances_meta():

    # !TODO: modify this for uoais dataset
    thing_ids = [1]
    thing_colors = [[92, 85, 25]]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = ['object']
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_wisdom_instances_meta():

    # !TODO: modify this for uoais dataset
    thing_ids = [1]
    thing_colors = [[92, 85, 25]]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = ['object']
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_lmo_instances_meta():

    thing_ids = [11, 8, 12, 1, 5, 6, 10, 9]
    thing_colors = [k["color"] for k in LMO_CATEGORIES if k["isthing"] == 1]
    # thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_dataset_id_to_contiguous_id = {1: 0, 5: 1, 6: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7}
    thing_classes = [k["name"] for k in LMO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_ycbv_instances_meta():

    thing_ids = [k["id"] for k in YCBV_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YCBV_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [str(k["id"]) for k in YCBV_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_clora_data2_instances_meta():

    thing_ids = [k["id"] for k in CLORA_DATA2_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CLORA_DATA2_CATEGORIES if k["isthing"] == 1]

    # Mapping from the incontiguous category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [str(k["name"]) for k in CLORA_DATA2_CATEGORIES if k["isthing"] == 1]

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_builtin_metadata(dataset_name):
    if dataset_name == "uoais":
        return _get_uoais_instances_meta()
    elif dataset_name == "wisdom":
        return _get_wisdom_instances_meta()
    elif dataset_name == "lmo":
        return _get_lmo_instances_meta()
    elif dataset_name == "ycbv":
        return _get_ycbv_instances_meta()
    elif dataset_name == "clora_data2":
        return _get_clora_data2_instances_meta()
    


    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
