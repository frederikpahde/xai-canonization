import torchvision
import torch
import numpy as np
import os, logging
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, resize

logger = logging.getLogger(__name__)


class CocoEvaluation(torchvision.datasets.CocoDetection):
    def get_denormalizer(self):
        return self.denormalizer
    def get_class_name_by_index_dict(self):
        return {i: self.all_classes[i] for i in range(len(self.all_classes))}

    def __init__(self, annotation_path, image_path, classes=None, img_transform=(lambda x: x),
                 mask_type='segm', mask_shape=None, denormalizer=(lambda x:x)):
        super().__init__(root=image_path, annFile=annotation_path)

        assert mask_type == 'segm' or mask_type == 'bbox'
        self.all_classes = [self.coco.cats[cat]['name'] for cat in self.coco.cats.keys()]
        self.cat_id_to_output_index = {cat_id: i for i, cat_id in enumerate(self.coco.cats.keys())}
        if classes is not None and len(classes) > 0:
            if isinstance(classes[0], int):
                classes=[self.all_classes[i] for i in classes]
            self.classes = classes
        else:
            self.classes = self.all_classes

        self.mask_shape = mask_shape
        self.mask_type = mask_type
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.img_transform = img_transform
        self.denormalizer=denormalizer
        # Validates the arguments, by checking if the input path and the label map file exist
        assert os.path.isdir(self.image_path), "The image path does not exist or is not a directory."
        assert os.path.isfile(self.annotation_path), "The annotation file does not exist."

        for cl in self.classes:
            assert not len(self.coco.getCatIds(catNms=[cl])) == 0, f"The class {cl} is invalid."

        self.samples = []
        self.cat_ids = self.coco.getCatIds(catNms=self.classes)
        self.ann_ids = self.coco.getAnnIds(catIds=self.cat_ids)
        self.img_ids = []
        for category in self.cat_ids:
            self.img_ids = self.img_ids + self.coco.getImgIds(catIds=[category])
        self.img_ids = list(set(self.img_ids))

        for i, id in enumerate(self.ids):
            if id in self.img_ids:
                addedClasses = []
                for annot in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[id])):
                    c_id = annot["category_id"]
                    if c_id in self.cat_ids and c_id not in addedClasses:
                        self.samples.append((i, annot["category_id"]))
                        addedClasses.append(c_id)
        logger.info(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def combine_masks(self, m):
        m = np.array(m)
        m = np.sum(m, axis=0)
        m = np.clip(m, 0, 1)
        return np.array(m, dtype=np.int32)

    def __getitem__(self, item):
        id, cls = self.samples[item]
        img, ann = super().__getitem__(id)
        bbox_mask = np.zeros((img.size[1], img.size[0]), dtype=np.int32)
        segment_masks = []
        filtered_ann = []
        for a in ann:
            if a["category_id"] == cls:
                filtered_ann.append(a)
        for a in filtered_ann:
            if self.mask_type == "bbox":
                bbox = a["bbox"]
                for i, b in enumerate(bbox):
                    bbox[i] = int(b)
                bbox_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 1
            else:
                segment_masks.append(np.array(self.coco.annToMask(a), dtype=np.int32))
        if self.mask_type == 'bbox':
            mask = bbox_mask
        else:
            mask = self.combine_masks(segment_masks)
        if self.mask_shape is not None:
            mask = pil_to_tensor(resize(to_pil_image(mask * 255), self.mask_shape))
            mask[mask > 0] = 1
            return self.img_transform(img), mask, self.cat_id_to_output_index[cls]
        else:
            return self.img_transform(img), np.expand_dims(mask, axis=0), self.cat_id_to_output_index[cls]
