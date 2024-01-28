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
from collections import OrderedDict
import cv2
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
                        default=8, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--vis', dest='vis',
                        default=False, type=bool)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    # parser.add_argument('weights', type=str,
    #                     default='yolov2-pytorch/data/pretrained/yolov2-tiny-voc.pth',
    #                     help='model .pth path')
    parser.add_argument('--device', default=0,
                        help='Choose a gpu device 0, 1, 2 etc.')

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

def appendLists(a=[],b=[], im_info={}, thres=0.25):
    w = im_info['width'].item()
    h = im_info['height'].item()
    for i in range(len(b)):
        # if round(b[i][1],2) >= 0.2:
        # _smal_list = f'{int(b[i][0])} {round(b[i][1],2)} {round(b[i][2]/w,4)} {round(b[i][3]/h,4)} {round(b[i][4]/w,4)} {round(b[i][5]/h,4)} \n'
        if b[i][-1] > thres:
            width  = b[i][3] - b[i][1]
            height = b[i][4] - b[i][2]
            x_center = (b[i][3] + b[i][1]) / 2
            y_center = (b[i][4] + b[i][2]) / 2
            # _smal_list = f'{int(b[i][0])} {round(b[i][-1],4)} {round(b[i][1],4)} {round(b[i][2],4)} {round(b[i][3],4)} {round(b[i][4],4)} \n'
            # _smal_list = f'{int(b[i][0])} {round(b[i][1]/w,4)} {round(b[i][2]/h,4)} {round(b[i][3]/w,4)} {round(b[i][4]/h,4)} \n'
            # _smal_list = f'{int(b[i][0])} {round(b[i][-1],4)} {round(b[i][1],4)} {round(b[i][2],4)} {round(b[i][3],4)} {round(b[i][4],4)} \n'
            _smal_list = f'{int(b[i][0])} {round(b[i][-1],4)} {round(x_center/w, 4)} {round(y_center/h, 4)} {round(width/w, 4)} {round(height/h, 4)} \n'
            a.append(_smal_list)    
    # if len(a)==0:
    #     a.append('0 0 0 0 0 \n')
    return a

def util(check_point):
    dum = []
    for i, (k,v) in enumerate(check_point.items()):
        if k == 'conv9.0.weight':    #con9: torch.Size([40, 1024, 1, 1]), bias9: torch.Size([40])
            v = torch.rand((40, 1024, 1, 1))
            v /= 1000
            append = (k,v)
            dum.append(append)
        elif k == 'conv9.0.bias':
            v = torch.rand(40)
            v /= 10000
            append = (k,v)
            dum.append(append)
        else:
            append = (k,v)
            dum.append(append)
            # print(v)            
    modified_check_point = {"model": OrderedDict(dum)}
    return modified_check_point

def drawBox(label:np.array, img:np.ndarray, rel=False):
    # for i in range(label.shape[0]):
    h, w, _ = img.shape
    if label.size == 6:
        box = [label[1], label[2], label[3], label[4]]
    elif label.size == 5:
        box = [label[1], label[2], label[3], label[4]]    
    elif label.size == 4:    
        box = [label[0], label[1], label[2], label[3]]
    else:
        raise ValueError("Invalid size array only accept arrays of size 4 or 5")    
    
    color = list(np.random.random(size=3) * 256)
    if rel:
        img = cv2.rectangle(img,(int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), color, 3)
    else:
        img = cv2.rectangle(img,(int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
    return img

def showImg(img, labels, meta, relative=False):
    # Convert the tensor to a numpy array
    _image = img
    image_np = _image.numpy().transpose((1, 2, 0))
    # image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)*255
    _img = Image.fromarray(image_np.astype('uint8'), 'RGB')
    _img = np.array(_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
    _img = cv2.resize(_img, (int(meta['width'].item()),  int(meta['height'].item())), interpolation= cv2.INTER_LINEAR)
    for i in range(labels.shape[0]):
        label = labels[i].numpy()
        conf = label[-1]
        if conf >= 0.1:
            if relative:
                _img = drawBox(label, _img, True)
            else:    
                _img = drawBox(label, _img)
    cv2.imshow('', _img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def test(args):
    args.conf_thresh = 0.1
    args.nms_thresh = 0.45
    args.thres = 0.3
    if args.vis:
        args.conf_thresh = 0.5
    device = int(args.device)
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
        val_data = None    
    
    if val_data is not None:
        # if args.withTrain:
        #     args.conf_thresh = 0.18
        #     args.nms_thresh = 0.35
        args.scale   = True
        val_dataset  = Custom_yolo_dataset(data=val_data, train=False)
        dataset_size = len(val_dataset)
        num_classes  = nc
        # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    else:
        args.scale   = True
        args.dataset = "voc07test"
        num_classes  = 20
        # args.conf_thresh = 0.001
        # args.nms_thresh = 0.45
        # args.data_limit = 16
        # print(args)

        # prepare dataset

        if args.dataset      == 'voc07trainval':
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
        # model = Yolov2()
    else:
        model = Yolov2()    

    # model_path = os.path.join(args.output_dir, 'weights.pth')
    # torch.save({'model': model.state_dict(),} , model_path)
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(args.model_name)
    # else:
    #     checkpoint = torch.load(args.model_name, map_location='cpu')
    checkpoint = torch.load(args.model_name, map_location='cpu')
    _model     = checkpoint['model']
    if _model['conv9.0.weight'].shape[0] != (5+num_classes)*5:
        checkpoint = util(_model)
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded from {args.model_name}')

    if args.use_cuda:
        model.to(device)
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
                im_data_variable = Variable(im_data).to(device)
            else:
                im_data_variable = Variable(im_data)

            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img = im_data
                img_id += 1
                if args.dataset=='custom':
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh)
                # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    if args.dataset!='custom':
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_det = torch.zeros((inds.numel(), 5))
                                cls_det[:, :4] = detections[inds, :4]
                                cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                                showImg(im_data[i], cls_det, im_info)
                                all_boxes[cls][img_id] = cls_det.cpu().numpy()
                    elif args.dataset=='custom':
                        _detAllclass = []
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                # cls_ = torch.zeros(inds.numel())
                                # cls_[:] = cls
                                cls_det = torch.zeros((inds.numel(), 6))
                                cls_det[:,0] = detections[inds, -1]
                                cls_det[:, 1:6] = detections[inds, :5]
                                # cls_det[:, 1] = detections[inds, 4] * detections[inds, 5]
                                # showImg(im_data[i], cls_det, im_info, False)
                                _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                                _detAllclass = appendLists(_detAllclass, _det1Class, im_info, args.thres)
                        if not os.path.exists(f'{save_dir}/labels'):
                            os.mkdir(f'{save_dir}/labels')
                        with open(f'{save_dir}/labels/{name}', 'w') as f:
                            f.writelines(_detAllclass)                                    
    if args.data is not None:
        args.gtFolder           =     val_dir
        args.detFolder          =     f'{save_dir}/labels'
        args.iouThreshold       =     0.5
        args.gtFormat           =     'xywh'
        args.detFormat          =     'xywh'
        args.gtCoordinates      =     'rel'
        args.detCoordinates     =     'rel'
        args.imgSize            =     '1280,720'   # for bdd --> 1280, 720 and waymo --> 1920, 1280
        args.savePath           =     '/home/zafar/yolov2_pytorch/output/plots'
        args.call_with_train    =     True
        args.showPlot           =     False
        args.names              =     names
        args.val                =     True
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

def test_for_train(temp_path, model, 
                   args, val_data=None, 
                   classes=None, 
                   afterTrain=False):
    # args = parse_args()
    # make a directory to save predictions paths
    save_dir = f'{temp_path}/preds'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(f'{save_dir}', ignore_errors=True)
        print(f'{save_dir} was existing and removed...')
        os.mkdir(save_dir)

    if val_data is not None:
        args.conf_thresh = 0.01
        args.nms_thresh = 0.5
        args.thres = 0.25
        args.scale = True
        val_dataset = Custom_yolo_dataset(data=val_data, train=False)
        dataset_size = len(val_dataset)
        num_classes = len(classes)
        # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    else:
        args.scale = True
        args.dataset = "voc07test"
        args.conf_thresh = 0.01
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

            yolo_outputs = model(im_data_variable, im_info=im_infos)
            for i in range(im_data.size(0)):
                img_id += 1
                if args.data is not None:
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh)
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
                                # cls_ = torch.zeros(inds.numel())
                                # cls_[:] = cls
                                cls_det = torch.zeros((inds.numel(), 6))
                                cls_det[:,0] = detections[inds, -1]
                                cls_det[:, 1:6] = detections[inds, :5]
                                # cls_det[:, 1] = detections[inds, 4] * detections[inds, 5]
                                # showImg(im_data[i], cls_det, im_info, False)
                                _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                                _detAllclass = appendLists(_detAllclass, _det1Class, im_info, args.thres)
                        if not os.path.exists(f'{save_dir}/labels'):
                            os.mkdir(f'{save_dir}/labels')
                        if len(_detAllclass)>0:
                            with open(f'{save_dir}/labels/{name}', 'w') as f:
                                f.writelines(_detAllclass)                                    
    if args.data is not None:
        args.gtFolder           =   args.val_dir
        args.detFolder          =   f'{save_dir}/labels'
        args.iouThreshold       =   args.nms_thresh
        args.gtFormat           =   'xywh'
        args.detFormat          =   'xywh'
        args.gtCoordinates      =   'rel'
        args.detCoordinates     =   'rel'
        args.imgSize            =   '1280,720'   # for bdd --> 1280, 720 and waymo --> 1920, 1280
        args.savePath           =   '/home/zafar/yolov2_pytorch/results/plots'
        args.call_with_train    =   True
        args.showPlot           =   False
        args.names              =   classes
        args.val                =   afterTrain
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