import json
import os
from pycocotools.coco import COCO

def create_people_annotations(annotations_file, output_path, object_area_threshold, foreground_class_name):

	print("Processing {}...".format(annotations_file))
	coco = COCO(annotations_file)

	foreground_class_id = 1
	dataset = coco.dataset

	images = dataset['images']

	foreground_category = None
	background_category = {'supercategory': 'background', 'id':0, 'name':'background'}

	for category in dataset['categories']:
		if category['name'] == foreground_class_name:
			foreground_class_id = category['id']
			foreground_category = category

	foreground_category['id'] = 1
	background_category['name'] =  "not-{}".format(foreground_category['name'])
	categories = [background_category, foreground_category]

    if not 'annotations' in dataset:
        raise KeyError('Need annotations in json file to build the dataset.')

    new_ann_id = 0
    annotations = []
    positive_img_ids = set()
    foreground_imgs_ids = coco.getImgIds(catIds=foreground_class_id)

    for img_id in foreground_img_ids:
    	img = coco.imgs[img_id]
    	img_area = img['height'] * img['width']

    	for ann_id in coco.getAnnIds(imgIds=img_id, catIds=foreground_class_id):
    		ann = coco.anns[ann_id]

