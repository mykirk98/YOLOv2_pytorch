"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from config import config as cfg
from util.augmentation import augment_img
import os
import pdb


class Custom_yolo_dataset(Dataset):
    def __init__(self, data, train=True, scale_Crop=False):
        self.train       =  train
        self.images      =  self._load_data(data)
        self.scale_Crop  =  scale_Crop
        # print('done')

    def _load_data(self,file):
        images = []
        with open(file, 'r') as f:
            for line in f:
                label = line.split('.')[0] + '.txt'
                if os.path.isfile(label):
                    with open(label, 'r') as f1:
                        try:
                            x = f1.readlines() 
                            if len(x):
                                images.append(line.split()[0])
                        except:
                            NotImplementedError      
        return images

    def xywh2xyxy(self, x, w, h):
        
        x1 = ((float(x[0])) - (float(x[2])/2)) if ((float(x[0])) - (float(x[2])/2)) >= 0.0 else 0.0
        y1 = ((float(x[1])) - (float(x[3])/2)) if ((float(x[1])) - (float(x[3])/2)) >= 0.0 else 0.0
        x2 = ((float(x[0])) + (float(x[2])/2)) if ((float(x[0])) + (float(x[2])/2)) <= 1.0 else 1.0
        y2 = ((float(x[1])) + (float(x[3])/2)) if ((float(x[1])) + (float(x[3])/2)) <= 1.0 else 1.0
        return [round(x1*w,4), round(y1*h,4), round(x2*w,4), round(y2*h,4)]
    
    def scaleAndcrop(self, img, box, classes, sf=1.6):
        _img = img
        w, h = _img.size
        _boxes   = np.array(box)
        _classes = np.array(classes)
        _boxes[:,0::2] /= w
        _boxes[:,1::2] /= h
        img_scaled = _img.resize((int(w*sf),int(h*sf)))
        w, h = img_scaled.size
        _boxes[:,0::2] *= w
        _boxes[:,1::2] *= h

        x1 = int(0.20*w)
        x2 = int(0.80*w)
        y1 = int(0.25*h)
        y2 = int(0.90*h)
        img_cropped = img_scaled.crop((x1, y1, x2, y2))
        w_, h_ = img_cropped.size

        _boxes[:,0::2] -= x1
        _boxes[:,1::2] -= y1
        _boxes[:,0::2] =  np.clip(_boxes[:,0::2], 0, w_-1)
        _boxes[:,1::2] =  np.clip(_boxes[:,1::2], 0, h_-1)

        keep     = (_boxes[:, 0] != _boxes[:, 2]) & (_boxes[:, 1] != _boxes[:, 3])
        _boxes   = _boxes[keep, :]
        _classes = _classes[keep]
        keep =  ((_boxes[:, 2] - _boxes[:, 0]) > 8) & ((_boxes[:, 3] - _boxes[:, 1])>6)
        _boxes   = _boxes[keep, :]
        _classes = _classes[keep]

        # img_ = np.array(img_cropped)
        # img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
        # for i in range(_boxes.shape[0]):
        #     label = _boxes[i]
        #     x =  label[0]
        #     y =  label[1]
        #     p =  label[2]
        #     q =  label[3]
        #     cv2.rectangle(img_, (int(x),int(y)), (int(p),int(q)), (0,0,255), 2)
        # cv2.imshow('', img_)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return img_cropped, _boxes.tolist(), _classes.tolist()
    
    def nroi_at(self,i):
        im_path = self.images[i]
        label_path = im_path.split('.')[0] + '.txt'     # implement no label cases (no file or empty label file)
        
        # if (os.path.isfile(label_path) == True):
            # print('Printing current label path------','\n', f'{i}: ', label_path)   
        with open(label_path, 'r') as f:
            im    = Image.open(im_path)
            w, h  = im.size
            boxes = []
            gt_classes = []
            # image = im
            for line in f:
                label = line.split(None,1)
                gt_classes.append(int(label[0]))
                _box = label[1].split()
                box  = self.xywh2xyxy(_box, w, h)
                # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                # cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 1)
                # cv2.imshow('', image)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                # if any(x>1 for x in box) or any(x<0 for x in box):
                #     print('---//----Extra ordinary value found in current label-------')
                #     print('---//----Extra ordinary box: ', box)
                #     print('---//----Path of file containing extra ordinary box: ', label_path)
                boxes.append(box)
            if self.train and self.scale_Crop:
                im, boxes, gt_classes = self.scaleAndcrop(im,boxes,gt_classes)   
            # print('Printing boxes list for current label file-----', '\n', boxes)
            # print('Printing number of labels in current file-----', '\n', len(boxes))
        return im, np.array(boxes), np.array(gt_classes)
    
    def __getitem__(self, i):
        im_data, boxes, gt_classes = self.nroi_at(i)
        # w, h
        im_info = torch.FloatTensor([im_data.size[0], im_data.size[1]])
        if self.train:
            # print('Trace here----//')
            # pdb.set_trace()
            if boxes.shape[0] > 0:
                im_data, boxes, gt_classes = augment_img(im_data, boxes, gt_classes)

                w, h = im_data.size[0], im_data.size[1]

                # image = cv2.cvtColor(np.array(im_data), cv2.COLOR_RGB2BGR)
                # for m in range(boxes.size[0]):
                #     box = boxes[m]
                #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 1)
                #     cv2.imshow('', image)
                #     cv2.waitKey()
                #     cv2.destroyAllWindows()

                if np.any(boxes):
                    boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
                    boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

                # resize image
                input_h, input_w = cfg.input_size
                im_data = im_data.resize((input_w, input_h))
                im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
                im_data_resize = im_data_resize.permute(2, 0, 1)
                boxes = torch.from_numpy(boxes)
                gt_classes = torch.from_numpy(gt_classes)
                num_obj = torch.Tensor([boxes.size(0)]).long()
                return im_data_resize, boxes, gt_classes, num_obj, im_info
            return None, None, None, None, None
        else:
            input_h, input_w = cfg.test_input_size
            im_data = im_data.resize((input_w, input_h))
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
            im_data_resize = im_data_resize.permute(2, 0, 1)
            return im_data_resize, im_info, self.images[i]

    def __len__(self):
        return len(self.images)   

class RoiDataset(Dataset):
    def __init__(self, imdb, train=True):
        super(RoiDataset, self).__init__()
        self._imdb = imdb
        self._roidb = imdb.roidb
        self.train = train
        self._image_paths = [self._imdb.image_path_at(i) for i in range(len(self._roidb))]

    def roi_at(self, i):
        image_path = self._image_paths[i]
        im_data = Image.open(image_path)
        boxes = self._roidb[i]['boxes']
        gt_classes = self._roidb[i]['gt_classes']

        return im_data, boxes, gt_classes

    def __getitem__(self, i):
        im_data, boxes, gt_classes = self.roi_at(i)
        # w, h
        im_info = torch.FloatTensor([im_data.size[0], im_data.size[1]])

        if self.train:
            im_data, boxes, gt_classes = augment_img(im_data, boxes, gt_classes)

            w, h = im_data.size[0], im_data.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize image
            input_h, input_w = cfg.input_size
            im_data = im_data.resize((input_w, input_h))
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
            im_data_resize = im_data_resize.permute(2, 0, 1)
            boxes = torch.from_numpy(boxes)
            gt_classes = torch.from_numpy(gt_classes)
            num_obj = torch.Tensor([boxes.size(0)]).long()
            return im_data_resize, boxes, gt_classes, num_obj, im_info

        else:
            input_h, input_w = cfg.test_input_size
            im_data = im_data.resize((input_w, input_h))
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
            im_data_resize = im_data_resize.permute(2, 0, 1)
            return im_data_resize, im_info, []

    def __len__(self):
        return len(self._roidb)

    def __add__(self, other):
        self._roidb = self._roidb + other._roidb
        self._image_paths = self._image_paths + other._image_paths
        return self


def detection_collate(batch):
    """
    Collate data of different batch, it is because the boxes and gt_classes have changeable length.
    This function will pad the boxes and gt_classes with zero.

    Arguments:
    batch -- list of tuple (im, boxes, gt_classes)

    im_data -- tensor of shape (3, H, W)
    boxes -- tensor of shape (N, 4)
    gt_classes -- tensor of shape (N)
    num_obj -- tensor of shape (1)

    Returns:

    tuple
    1) tensor of shape (batch_size, 3, H, W)
    2) tensor of shape (batch_size, N, 4)
    3) tensor of shape (batch_size, N)
    4) tensor of shape (batch_size, 1)

    """

    # kind of hack, this will break down a list of tuple into
    # individual list
    bsize = len(batch)
    im_data, boxes, gt_classes, num_obj, im_info = zip(*batch)
    # check for None entries and remove them
    im_data     =  tuple(xi for xi in im_data    if xi is not None)
    boxes       =  tuple(xi for xi in boxes      if xi is not None)
    gt_classes  =  tuple(xi for xi in gt_classes if xi is not None)
    num_obj     =  tuple(xi for xi in num_obj    if xi is not None)
    im_info     =  tuple(xi for xi in im_info    if xi is not None)
    
    cur_bsize   =  len(im_data)
    if cur_bsize != bsize:
        bsize = cur_bsize
    
    max_num_obj    = max([x.item() for x in num_obj])
    padded_boxes   = torch.zeros((bsize, max_num_obj, 4))
    padded_classes = torch.zeros((bsize, max_num_obj,))

    for i in range(bsize):
        if len(boxes[i]) > 0:
            padded_boxes[i, :num_obj[i], :] = boxes[i]
            padded_classes[i, :num_obj[i]]  = gt_classes[i]
        else:
            pass    

    return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0), im_info


class TinyRoiDataset(RoiDataset):
    def __init__(self, imdb, num_roi):
        super(TinyRoiDataset, self).__init__(imdb)
        self._roidb = self._roidb[:num_roi]