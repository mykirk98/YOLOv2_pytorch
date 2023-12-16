from multiprocessing import Process
from tqdm import tqdm
import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
from yolov2_tiny import Yolov2
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, Custom_yolo_dataset
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
from torch.utils.data import DataLoader
from config import config as cfg
import pascalvoc
from util.data_util import check_dataset
import shutil
import warnings
warnings.filterwarnings('ignore')

def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07test', type=str)
    parser.add_argument('--data', type=str,
                        default=None, help='Give the path of custom data yaml file' )
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output_800', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2-pytorch/data/pretrained/yolov2-tiny-voc.pth',
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=1, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=2, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--vis', dest='vis',
                        default=False, type=bool)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    # parser.add_argument('weights', type=str,
    #                     default='yolov2-pytorch/data/pretrained/yolov2-tiny-voc.pth',
    #                     help='model .pth path')

    args = parser.parse_args()
    return args


def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info

# def test():
#     args = parse_args()
#     args.conf_thresh = 0.005
#     args.nms_thresh = 0.45
#     if args.vis:
#         args.conf_thresh = 0.5
#     print('Called with args:')
#     print(args)

#     # prepare dataset

#     if args.dataset == 'voc07trainval':
#         args.imdbval_name = 'voc_2007_trainval'

#     elif args.dataset == 'voc07test':
#         args.imdbval_name = 'voc_2007_test'

#     else:
#         raise NotImplementedError

#     val_imdb = get_imdb(args.imdbval_name)

#     val_dataset = RoiDataset(val_imdb, train=False)
#     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#     # load model
#     model = Yolov2()
#     # weight_loader = WeightLoader()
#     # weight_loader.load(model, 'yolo-voc.weights')
#     # print('loaded')

#     model_path = os.path.join(args.output_dir, args.model_name+'.pth')
#     print('loading model from {}'.format(model_path))
#     if torch.cuda.is_available():
#         checkpoint = torch.load(model_path)
#     else:
#         checkpoint = torch.load(model_path, map_location='cpu')
#     model.load_state_dict(checkpoint['model'])

#     if args.use_cuda:
#         model.cuda()

#     model.eval()
#     print('model loaded')

#     dataset_size = len(val_imdb.image_index)

#     all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

#     det_file = os.path.join(args.output_dir, 'detections.pkl')

#     img_id = -1
#     with torch.no_grad():
#         for batch, (im_data, im_infos) in enumerate(val_dataloader):
#             if args.use_cuda:
#                 im_data_variable = Variable(im_data).cuda()
#             else:
#                 im_data_variable = Variable(im_data)

#             yolo_outputs = model(im_data_variable)
#             for i in range(im_data.size(0)):
#                 img_id += 1
#                 output = [item[i].data for item in yolo_outputs]
#                 im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
#                 detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
#                                        nms_threshold=args.nms_thresh)
#                 print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
#                 if len(detections) > 0:
#                     for cls in range(val_imdb.num_classes):
#                         inds = torch.nonzero(detections[:, -1] == cls).view(-1)
#                         if inds.numel() > 0:
#                             cls_det = torch.zeros((inds.numel(), 5))
#                             cls_det[:, :4] = detections[inds, :4]
#                             cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
#                             all_boxes[cls][img_id] = cls_det.cpu().numpy()

#                 if args.vis:
#                     img = Image.open(val_imdb.image_path_at(img_id))
#                     if len(detections) == 0:
#                         continue
#                     det_boxes = detections[:, :5].cpu().numpy()
#                     det_classes = detections[:, -1].long().cpu().numpy()
#                     im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
#                     plt.figure()
#                     plt.imshow(im2show)
#                     plt.show()

#     with open(det_file, 'wb') as f:
#         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

#     val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)

def appendLists(a=[],b=[], im_info={}):
    w = im_info['width'].item()
    h = im_info['height'].item()
    for i in range(len(b)):
        # if round(b[i][1],2) >= 0.2:
        _smal_list = f'{int(b[i][0])} {round(b[i][1],2)} {round(b[i][2]/w,4)} {round(b[i][3]/h,4)} {round(b[i][4]/w,4)} {round(b[i][5]/h,4)} \n'
        a.append(_smal_list)
    if len(a)==0:
        a.append('0 0 0 0 0 \n')
    return a


def test(args):
    args.conf_thresh = 0.001
    args.nms_thresh = 0.45
    if args.vis:
        args.conf_thresh = 0.5
    print('Called with args:')
    print(args)
    
    if args.dataset == 'custom':
        data_dict = check_dataset(args.data)
        _, val_data, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
        nc = int(data_dict['nc'])  # number of classes
        names = data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {args.data}'  # check
    
    save_dir = f'{args.output_dir}/preds'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(f'{save_dir}', ignore_errors=True)
        print(f'{save_dir} was existing and removed...')
        os.mkdir(save_dir)

    try:
        val_data
    except:
        val_data =None    
    
    if val_data is not None:
        # if args.withTrain:
        #     args.conf_thresh = 0.18
        #     args.nms_thresh = 0.35
        args.scale = False
        val_dataset = Custom_yolo_dataset(data=val_data, train=False)
        dataset_size = len(val_dataset)
        num_classes = nc
        # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    else:
        args.scale = True
        args.dataset = "voc07test"
        # args.conf_thresh = 0.001
        # args.nms_thresh = 0.45
        # args.data_limit = 16
        # print(args)

        # prepare dataset

        if args.dataset == 'voc07trainval':
            args.imdbval_name = 'voc_2007_trainval'

        elif args.dataset == 'voc07test':
            args.imdbval_name = 'voc_2007_test'

        else:
            raise NotImplementedError

        val_imdb = get_imdb(args.imdbval_name)

        val_dataset = RoiDataset(val_imdb, train=False)
        dataset_size = len(val_imdb.image_index)
        num_classes = val_imdb.num_classes
        all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    
    if not args.data_limit == 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(0, args.data_limit))

    # args.output_dir = args.temp_path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # model = model    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    if args.dataset == 'custom':
        model = Yolov2(classes=names)
    else:
        model = Yolov2()    
    # weight_loader = WeightLoader()
    # weight_loader.load(model, 'yolo-voc.weights')
    # print('loaded')

    # model_path = os.path.join(args.output_dir, 'weights.pth')
    # torch.save({'model': model.state_dict(),} , model_path)
    if torch.cuda.is_available():
        checkpoint = torch.load(args.model_name)
    else:
        checkpoint = torch.load(args.model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded from {args.model_name}')

    if args.use_cuda:
        model.cuda()
        print(f"Validating using CUDA")

    model.eval()

    args.output_dir = os.path.join(args.output_dir, "Outputs")
    os.makedirs( args.output_dir, exist_ok=True )
    if args.dataset!='custom':
        det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos, paths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Performing validation.", ascii=' ~'):
        # for batch, (im_data, im_infos) in enumerate(val_dataloader):
        # for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)

            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                if args.dataset=='custom':
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh, scale=args.scale)
                # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    if args.dataset!='custom':
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_det = torch.zeros((inds.numel(), 5))
                                cls_det[:, :4] = detections[inds, :4]
                                cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                                all_boxes[cls][img_id] = cls_det.cpu().numpy()
                    elif args.dataset=='custom':
                        _detAllclass = []
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_ = torch.zeros(inds.numel())
                                cls_[:] = cls
                                cls_det = torch.zeros((inds.numel(), 6))
                                cls_det[:,0] = cls_
                                cls_det[:, 2:6] = detections[inds, :4]
                                cls_det[:, 1] = detections[inds, 4] * detections[inds, 5]
                                _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                                _detAllclass = appendLists(_detAllclass, _det1Class, im_info)
                        if not os.path.exists(f'{save_dir}/labels'):
                            os.mkdir(f'{save_dir}/labels')
                        with open(f'{save_dir}/labels/{name}', 'w') as f:
                            f.writelines(_detAllclass)                                    
    if args.data is not None:
        args.gtFolder = val_dir
        args.detFolder = save_dir
        args.iouThreshold = 0.5
        args.gtFormat = 'xywh'
        args.detFormat = 'xywh'
        args.gtCoordinates = 'rel'
        args.detCoordinates = 'rel'
        args.imgSize = '1920,1280'   # for bdd --> 1280, 720 and waymo --> 1920, 1280
        args.savePath = 'yolov2-pytorch/output/plots'
        args.call_with_train = True
        args.showPlot = False
        map, class_metrics = pascalvoc.main(args)
    # elif args.customData and not args.withTrain:
    #     map, class_metrics = pascalvoc.main(args)    
    else:
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # map = val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)
        map = val_imdb.evaluate_detections_with_train(all_boxes, output_dir=args.output_dir)
        class_metrics = []
    return map, class_metrics   

def test_for_train(temp_path, model, args, val_data=None, _num_classes=None):
    # args = parse_args()
    # make a directory to save predictions paths
    save_dir = f'{args.output_dir}/preds'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(f'{save_dir}', ignore_errors=True)
        print(f'{save_dir} was existing and removed...')
        os.mkdir(save_dir)

    if val_data is not None:
        args.conf_thresh = 0.2
        args.nms_thresh = 0.45
        args.scale = False
        val_dataset = Custom_yolo_dataset(data=val_data, train=False)
        dataset_size = len(val_dataset)
        num_classes = _num_classes
        # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    else:
        args.scale = True
        args.dataset = "voc07test"
        args.conf_thresh = 0.001
        args.nms_thresh = 0.45
        # args.data_limit = 16
        # print(args)

        # prepare dataset

        if args.dataset == 'voc07trainval':
            args.imdbval_name = 'voc_2007_trainval'

        elif args.dataset == 'voc07test':
            args.imdbval_name = 'voc_2007_test'

        else:
            raise NotImplementedError

        val_imdb = get_imdb(args.imdbval_name)

        val_dataset = RoiDataset(val_imdb, train=False)
        dataset_size = len(val_imdb.image_index)
        num_classes = val_imdb.num_classes
        all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    
    if not args.data_limit==0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(0, args.data_limit))

    args.output_dir = temp_path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    model = model    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    # model = Yolov2()
    # weight_loader = WeightLoader()
    # weight_loader.load(model, 'yolo-voc.weights')
    # print('loaded')

    # model_path = os.path.join(args.output_dir, 'weights.pth')
    # torch.save({'model': model.state_dict(),} , model_path)
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(model_path)
    # else:
    #     checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # print(f'Model loaded from {model_path}')

    if args.use_cuda:
        model.cuda()
        print(f"Validating using CUDA")

    model.eval()

    args.output_dir = os.path.join(args.output_dir, "Outputs")
    os.makedirs( args.output_dir, exist_ok=True )
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos, paths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Performing validation.", ascii=' ~'):
        # for batch, (im_data, im_infos) in enumerate(val_dataloader):
        # for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)

            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                if args.data is not None:
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh, scale=args.scale)
                # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    if args.data == None:
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_det = torch.zeros((inds.numel(), 5))
                                cls_det[:, :4] = detections[inds, :4]
                                cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                                all_boxes[cls][img_id] = cls_det.cpu().numpy()
                    else:
                        _detAllclass = []
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_ = torch.zeros(inds.numel())
                                cls_[:] = cls
                                cls_det = torch.zeros((inds.numel(), 6))
                                cls_det[:,0] = cls_
                                cls_det[:, 2:6] = detections[inds, :4]
                                cls_det[:, 1] = detections[inds, 4] * detections[inds, 5]
                                _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                                _detAllclass = appendLists(_detAllclass, _det1Class, im_info)
                        if not os.path.exists(f'{save_dir}/labels'):
                            os.mkdir(f'{save_dir}/labels')
                        with open(f'{save_dir}/labels/{name}', 'w') as f:
                            f.writelines(_detAllclass)                                    
    if args.data is not None:
        args.gtFolder = args.val_dir
        args.detFolder = save_dir
        args.iouThreshold = 0.5
        args.gtFormat = 'xywh'
        args.detFormat = 'xywh'
        args.gtCoordinates = 'rel'
        args.detCoordinates = 'rel'
        args.imgSize = '1920,1280'   # for bdd --> 1280, 720 and waymo --> 1920, 1280
        args.savePath = 'yolov2-pytorch/output/plots'
        args.call_with_train = True
        args.showPlot = False
        map, class_metrics = pascalvoc.main(args)    
    else:
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # map = val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)
        map = val_imdb.evaluate_detections_with_train(all_boxes, output_dir=args.output_dir)
        class_metrics = []
    return map, class_metrics


if __name__ == '__main__':
    args = parse_args()
    map, metrics = test(args)