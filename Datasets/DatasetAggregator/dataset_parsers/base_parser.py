
class BaseParser():
    def __init__(self):
        pass

    def collect_samples(self):
        raise NotImplementedError
    
    # only for object detection now
    def convert2coco(self):
        raise NotImplementedError


