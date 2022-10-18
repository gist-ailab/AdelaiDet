import os
import json

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import cm
from pycocotools.coco import COCO


root_path = '/ailab_mat/dataset/AIHUB_data/data2/'
with open(root_path + 'annotations/coco_anns_clora_visible_val.json', 'r') as f:
    visible_val_json = json.load(f)

# print(visible_val_json.keys())
# print(len(visible_val_json['annotations']))
# print(len(visible_val_json['images']))
# print('annotations ', visible_val_json['annotations'][0])
# print('images ', visible_val_json['images'][0])

# img = Image.open(root_path + visible_val_json['images'][0]['file_name'])
# annot = visible_val_json['annotations'][0]
# print(annot)
# img.save('img_0.png')

imgId = 408
coco = COCO(root_path + 'annotations/coco_anns_clora_visible_val.json')



cat_ids = sorted(coco.getCatIds())
cats = coco.loadCats(cat_ids)
# print(cat_ids)
# print(cats)

# appleIds = coco.getCatIds(catNms=['apple'])
annIds = coco.getAnnIds(imgIds=imgId, catIds=cat_ids)
anns = coco.loadAnns(annIds)
print(annIds)

imgInfo = coco.loadImgs(imgId)
# print(imgInfo)
img = Image.open(root_path + imgInfo[0]['file_name']).convert('RGB')
# coco.showAnns(anns, draw_bbox=True)
img.save(f'img_{imgId}.png')

print(len(anns))
mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i]) * int(i/3)
mask_img = Image.fromarray(np.uint8(cm.Accent(mask)*255))
mask_img.save(f'mask_{imgId}.png')

anns2 = coco.imgsToAnns[imgId]
print(anns == anns2)