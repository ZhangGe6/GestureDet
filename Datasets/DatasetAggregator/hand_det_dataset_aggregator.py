import random
import json
import os
import cv2
from tqdm import tqdm
from dataset_parsers import GANeratedParser, MHPParser, EGOHandParser, SelfCollectedParser

class COCOAggregator():
    def __init__(self, datasets, dst_json_dir):
        self.datasets = datasets
        self.dst_json_dir = dst_json_dir

        self.res_dict = dict()
        self.res_dict['info'] = self.gen_datasets_info()
        self.res_dict['licenses'] = []
        self.res_dict['categories'] = self.set_categories()
        self.res_dict['train_images'] = []
        self.res_dict['train_annotations'] = []
        self.res_dict['val_images'] = []
        self.res_dict['val_annotations'] = []

        self.img_id_count = 0
        self.anno_id_count = 0
        self.val_sample_freq = 100
        
    def gen_datasets_info(self):
        info = dict()
        description = 'This dataset is aggreated from'
        for dataset in self.datasets:
            description += ' ' + dataset['name']

        info['description'] = description
        return info

    def set_categories(self):
        return [
            {
                "id" : 0,
                "name" : "person",
                "supercategory" : "None"
            },
            {
                "id" : 1,
                "name" : "hand",
                "supercategory" : "None"
            },
            {
                "id" : 2,
                "name" : "others",
                "supercategory" : "None"
            },
        ]

    def aggragate_image_anno(self):
        for dataset in self.datasets:
            dataset_name, parser_cls, data_root = dataset['name'], dataset['parser'], dataset['data_root']
            print("aggragating dataset {} ...".format(dataset_name))

            parser = parser_cls(data_root=data_root)
            # self.aggregated_samples += parser.samples
            for sample in tqdm(parser.samples):
                img = dict()
                img['id'] = self.img_id_count
                img['height'], img['width'] = sample['img_size']
                img['image_path'] = sample['image_path']
                if self.img_id_count % self.val_sample_freq == 0:
                    self.res_dict['val_images'].append(img)
                    # img_with_bbox = cv2.imread(img['image_path'])
                else:
                    self.res_dict['train_images'].append(img)
                    
                bboxes = sample['object_bbox']
                for bbox in bboxes:
                    anno = dict()
                    x, y, w, h, conf, class_ = bbox
                    anno['id'] = self.anno_id_count
                    anno['image_id'] = self.img_id_count
                    anno['category_id'] = int(class_)
                    anno['bbox'] = [x, y, w, h]
                    anno['iscrowd'] = 0
                    if self.img_id_count % self.val_sample_freq == 0:
                        self.res_dict['val_annotations'].append(anno)
                        # img_with_bbox = cv2.rectangle(img_with_bbox, (int(x), int(y)), (int(x+w), int(y+h)), color=(0, 0, 255), thickness=2)
                        # img_with_bbox = cv2.putText(img_with_bbox, str(class_), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
                        # img_save_path = '/home/zg/wdir/zg/moyu/GestureDet/nanodet/workspace/dummy_samples/' + str(self.img_id_count) + '.jpg'
                        # cv2.imwrite(img_save_path, img_with_bbox)
                        # print(img_save_path)
                        # input()

                    else:
                        self.res_dict['train_annotations'].append(anno)

                    self.anno_id_count += 1
                self.img_id_count += 1

    def dump_json(self):
        if not os.path.exists(self.dst_json_dir):
            os.mkdir(self.dst_json_dir)

        train_json_path = os.path.join(self.dst_json_dir, 'train_od.json')
        val_json_path = os.path.join(self.dst_json_dir, 'val_od.json')

        train_info = {
            "info" : self.res_dict['info'],
            "licenses" : self.res_dict['licenses'],
            "images" : self.res_dict['train_images'],
            "annotations" : self.res_dict['train_annotations'],
            "categories" : self.res_dict['categories']
        }
        val_info = {
            "info" : self.res_dict['info'],
            "licenses" : self.res_dict['licenses'],
            "images" : self.res_dict['val_images'],
            "annotations" : self.res_dict['val_annotations'],
            "categories" : self.res_dict['categories']
        }

        with open(train_json_path, 'w') as f:
            json.dump(train_info, f)
        with open(val_json_path, 'w') as f:
            json.dump(val_info, f)

        print("Done! Generate {} train samples and {} validate samples, json files save at {}".format(
            len(self.res_dict['train_images']), len(self.res_dict['val_images']),
            os.path.abspath(self.dst_json_dir)
        ))

if __name__ == '__main__':
    datasets = [
        # {
        # 'name' : 'GANerated',
        #  'parser': GANeratedParser,
        #  'data_root' : "/home/zg/wdir/Datasets/GestureDet_dataset/RealHands/data"
        # },

        # {
        #  'name' : 'MHP',
        #  'parser': MHPParser,
        #  'data_root' : "/home/zg/wdir/zg/moyu/GestureDet/Datasets/MHP_dataset"
        # },

        {
         'name' : 'EGO',
         'parser': EGOHandParser,
         'data_root' : "/home/zg/wdir/zg/moyu/GestureDet/Datasets/EGOHand"
        },

        {
         'name' : 'SelfCollected',
         'parser': SelfCollectedParser,
         'data_root' : "/home/zg/wdir/zg/moyu/GestureDet/Datasets/SelfCollected/object_anno.json"
        },

    ]

    dst_json_dir = '../train_val_jsons'

    aggregator = COCOAggregator(datasets=datasets, dst_json_dir=dst_json_dir)
    aggregator.aggragate_image_anno()
    # aggregator.shuffle_samples()
    # aggregator.split_train_val()
    aggregator.dump_json()


