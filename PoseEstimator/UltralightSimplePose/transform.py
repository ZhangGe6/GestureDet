# This script is largely refered to https://github.com/MVIG-SJTU/AlphaPose/blob/master/alphapose/utils/transforms.py
import cv2
import numpy as np
import random
import copy
from pose_utils.transforms import flip_joints, get_bbox_from_joints, _box_to_center_scale, \
                  get_affine_transform, affine_transform
import imgaug.augmenters as iaa
#https://github.com/aleju/imgaug

class SimpleTransform(object):
    def __init__(self, model_input_size, model_output_size, data_aug):
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size
        self.feat_stride = np.array(self.model_input_size) / np.array(self.model_output_size)
        self.data_aug = data_aug
        # print("aug mode", self.data_aug)
        # print(self.feat_stride)
        self.aug_type_list = ['Add', 'AdditiveGaussianNoise', 'CoarseDropout', 'SaltAndPepper', 'JpegCompression', 'MotionBlur', 'ChangeColorTemperature']

        # see https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/coco/resnet/256x192_res50_lr1e-3_1x-concat.yaml
        self.sigma = 2
        self.scale_factor = 0.3
        self.rot = 40
        self.keypoints_num = 21
    
    def __call__(self, img, bbox, joints):
        xmin, ymin, xmax, ymax = bbox
        # print(joints)
        # print(xmin, ymin, xmax, ymax)
        center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin)

        # rescale
        if self.data_aug:
            sf = self.scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self.data_aug:
            rf = self.rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        else:
            r = 0
        # print(scale)
        # print(r)

        # horizontal flip
        if random.random() > 0.5 and self.data_aug:
            assert img.shape[2] == 3   # essure in HWC format
            img_width = img.shape[1]
            img = img[:, ::-1, :]
            joints = flip_joints(joints, img_width)
            center[0] = img_width - center[0] - 1   

        input_height, input_width = self.model_input_size
        # Note that the `input` here means `input` to the network
        # and `output` here means `output` from the affine transform. So these two have the same meaning
        trans = get_affine_transform(center, scale, rot=r, output_size=[input_width, input_height])
        # print(bbox)
        # print(trans)
        affined_img = cv2.warpAffine(img, trans, (input_width, input_height), flags=cv2.INTER_LINEAR)
        if self.data_aug:
            aug_tool = self.img_augmenter(self.aug_type_list[np.random.randint(0,len(self.aug_type_list))])  
            affined_img = aug_tool(image=affined_img)
        # print(affined_img.shape)
        affined_joints = np.zeros((len(joints), 2))
        for i in range(len(joints)):
            affined_joints[i][0:2] = affine_transform(joints[i][0:2], trans)
        # print(joints)

        target_mask, target_weight = self.target_generator(affined_joints)

        # return affined_img, target_mask, target_weight, bbox, joints
        return affined_img, target_mask, target_weight, affined_joints

    def target_generator(self, joints):
        r_size = self.sigma * 3
        # TODO: more to do for target_weight, refer to https://github1s.com/MVIG-SJTU/AlphaPose/blob/master/alphapose/utils/presets/simple_transform.py#L127
        target_weight = np.ones((self.keypoints_num, 1), dtype=np.float32)
        output_height, output_width = self.model_output_size
        target_mask_padded = np.zeros((self.keypoints_num, output_height+2*r_size, output_width+2*r_size), dtype=np.float32)
        for i, pt in enumerate(joints):
            x, y = pt
            # print(x, y)
            mu_y = int(y / self.feat_stride[0] + 0.5)
            mu_x = int(x / self.feat_stride[1] + 0.5)
            # print(mu_x, mu_y) 

            # the second `+ r_size` means pad by r_rize
            patch_ymin, patch_xmin = mu_y - r_size + r_size, mu_x - r_size + r_size
            patch_ymax, patch_xmax = mu_y + r_size + 1 + r_size, mu_x + r_size + 1 + r_size
            if patch_ymin < 0 or patch_xmin < 0 or patch_ymax >= output_height or patch_xmax >= output_width:
                target_weight[i] = 0   # this can handle some wrong annotations like `Ricki_unit_8.flv_000002_l`

            size = 2 * r_size + 1
            mesh_x = np.arange(0, size, 1, np.float32)
            mesh_y = mesh_x[:, np.newaxis]
            mesh_x0 = mesh_y0 = size // 2
            patch_guassian = np.exp(-((mesh_x - mesh_x0) ** 2 + (mesh_y - mesh_y0) ** 2) / (2 * (self.sigma ** 2)))
            # print(patch_guassian.shape)  # (7, 7)
            # print(target_mask_padded[i].shape) # (262, 198)
            # print(patch_ymin, patch_ymax, patch_xmin, patch_xmax)  # 271 278 75 82
            # print((target_mask_padded[i][patch_ymin:patch_ymax, patch_xmin:patch_xmax].shape))  # (0, 7)
            if target_weight[i] > 0.5:
                target_mask_padded[i][patch_ymin:patch_ymax, patch_xmin:patch_xmax] = patch_guassian
            target_mask = target_mask_padded[:, r_size:-r_size, r_size:-r_size]
            # print(target_mask.shape)
            # print()

        return target_mask, np.expand_dims(target_weight, -1)

    def img_augmenter(self, aug_type):
        # To apply a constant random param for one clip augmentation
        rand_Add = np.random.randint(-40, 40)
        rand_AdditiveGaussianNoise = np.random.uniform(0, 0.2*255)
        #rand_CoarseDropout = 
        rand_SaltAndPepper = np.random.uniform(0.05, 0.2)
        rand_JpegCompression = np.random.randint(90, 100)
        rand_MotionBlur = np.random.randint(5, 15)
        rand_ChangeColorTemperature =np.random.randint(1000, 40000)
        #rand_fliplr = 
        # rand_PerspectiveTransform = np.random.uniform(0.05, 0.15)

        aug_dict =  {
            'Add' : iaa.Add(rand_Add),
            #https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#add
            
            'AdditiveGaussianNoise' : iaa.AdditiveGaussianNoise(scale=rand_AdditiveGaussianNoise),
            #https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#additivegaussiannoise

            'CoarseDropout' : iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            #https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#coarsedropout

            'SaltAndPepper' : iaa.SaltAndPepper(rand_SaltAndPepper),
            #https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#saltandpepper

            'JpegCompression' : iaa.JpegCompression(compression=rand_JpegCompression),
            #https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#jpegcompression

            'MotionBlur' : iaa.MotionBlur(k=rand_MotionBlur),
            #https://imgaug.readthedocs.io/en/latest/source/overview/blur.html#motionblur

            'ChangeColorTemperature' : iaa.ChangeColorTemperature(rand_ChangeColorTemperature),
            #https://imgaug.readthedocs.io/en/latest/source/overview/color.html#changecolortemperature

            # 'Fliplr' : iaa.Fliplr(1),
            #https://imgaug.readthedocs.io/en/latest/source/overview/flip.html#fliplr

            # 'PerspectiveTransform' : iaa.PerspectiveTransform(scale=rand_PerspectiveTransform)
            #https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#perspectivetransform
            #Similar to crop?
        }

        return iaa.Sequential([aug_dict[aug_type]])