import random
import json
import os
from dataset_parsers import GANeratedParser, MHPParser, SelfCollectedParser

class HandposeDatasetAggregator():
    def __init__(self, datasets, dst_json_dir):
        self.datasets = datasets
        self.aggregated_samples = []
        self.dst_json_dir = dst_json_dir

    def aggragate_samples(self):
        for dataset in self.datasets:
            dataset_name, parser_cls, data_root = dataset['name'], dataset['parser'], dataset['data_root']
            print("aggragating dataset {} ...".format(dataset_name))
            
            parser = parser_cls(data_root=data_root)
            self.aggregated_samples += parser.samples
    
    def filter_no_hand_sample(self):
        full_sample_num = len(self.aggregated_samples)
        self.aggregated_samples = [sample for sample in self.aggregated_samples if len(sample['hand_bbox']) > 0]
        filtered_sample_num = len(self.aggregated_samples)
        print("filtered {} image samples which have no hands on them".format(
            full_sample_num - filtered_sample_num
        ))

    def shuffle_samples(self):
        print("shuffling...")
        random.shuffle(self.aggregated_samples)

    def split_train_val(self, val_portion=0.01):
        val_sample_num = int(len(self.aggregated_samples) * val_portion)
        self.val_samples = self.aggregated_samples[:val_sample_num]
        self.train_samples = self.aggregated_samples[val_sample_num:]
        # print("done! generate {} train samples and {} validate samples.".format(
        #     len(self.train_samples), len(self.val_samples)
        #     )
        # )
            

    def dump_json(self):
        train_json_path = os.path.join(self.dst_json_dir, 'train_pose.json')
        val_json_path = os.path.join(self.dst_json_dir, 'val_pose.json')

        train_info = {
            'split' : 'train',
            'samples' : self.train_samples
        }
        val_info = {
            'split' : 'validate',
            'samples' : self.val_samples
        }
        with open(train_json_path, 'w') as f:
            json.dump(train_info, f)
        with open(val_json_path, 'w') as f:
            json.dump(val_info, f)
        
        print("Done! Generated {} train samples and {} validate samples. json files save at {}".format(
                len(self.train_samples), len(self.val_samples), self.dst_json_dir
            ) 
        )

if __name__ == '__main__':
    datasets = [
        {'name' : 'GANerated',
         'parser': GANeratedParser,
         'data_root' : "/home/zg/wdir/zg/moyu/GestureDet/Datasets/GANerated/data"
        },

        {'name' : 'MHP',
         'parser': MHPParser,
         'data_root' : "/home/zg/wdir/zg/moyu/GestureDet/Datasets/MHP_dataset"
        },

        {
         'name' : 'self_collected',
         'parser': SelfCollectedParser,
         'data_root' : "/home/zg/wdir/zg/moyu/GestureDet/Datasets/SelfCollected/object_anno.json"
        },

    ]

    dst_json_dir = '../train_val_jsons'

    aggregator = HandposeDatasetAggregator(datasets=datasets, dst_json_dir=dst_json_dir)
    aggregator.aggragate_samples()
    aggregator.filter_no_hand_sample()
    aggregator.shuffle_samples()
    aggregator.split_train_val()
    aggregator.dump_json()


