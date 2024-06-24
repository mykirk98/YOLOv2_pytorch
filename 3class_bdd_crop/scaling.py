import numpy as np
import cv2
from PIL import Image

def xywh2xyxy(xywh, w, h):
    
    x1 = float(xywh[0]) - float(xywh[2]) / 2 if float(xywh[0]) - float(xywh[2])/2 >= 0.0 else 0.0
    y1 = float(xywh[1]) - float(xywh[3]) / 2 if float(xywh[1]) - float(xywh[3])/2 >= 0.0 else 0.0
    x2 = float(xywh[0]) + float(xywh[2]) / 2 if float(xywh[0]) + float(xywh[2])/2 <= 1.0 else 1.0
    y2 = float(xywh[1]) + float(xywh[3]) / 2 if float(xywh[1]) + float(xywh[3])/2 <= 1.0 else 1.0
    
    return [round(number=x1, ndigits=6), round(number=y1, ndigits=6), round(number=x2, ndigits=6), round(number=y2, ndigits=6)]

def ReadLabels(label_file_path):
    with open(file=label_file_path, mode='r') as file:
        lines = file.readlines()
        boxes, GT_classes = [], []

        for line in lines:
            label = line.split(sep=None, maxsplit=1)        # label[0] : [class] / label[1] : [x, y, w, h]
            GT_classes.append(int(label[0]))
            box = label[1].split()      # box[0] : x / box[1] : y / box[2] : w / box[3] : h
            
            box  = xywh2xyxy(box, width, height)    # x, y, w, h --> x1, y1, x2, y2

            boxes.append(box)

    return boxes, GT_classes

def Normalization(boxes, image):
    width, height = image.size
    boxes[:,0::2] /= width  # box[0], box[2] /= width
    boxes[:,1::2] /= height # box[1], box[3] /= height

    return boxes

def UpSampling(image, boxes, sf=3):
    """
    Upsampling images & boxes by scale_factor
    """
    width, height = image.size
    image = image.resize(size=(int(round(number=width*sf, ndigits=0)), int(round(number=height*sf, ndigits=0))))
    width, height = image.size
    boxes[:,0::2] *= width  # box[0], box[2] *= sf X width(former)
    boxes[:,1::2] *= height # box[1], box[3] *= sf * height(former)

    return image, width, height, boxes

def OptimumArea(boxes, bestLeft, bestTop, bestRight, bestBottom, most_width, most_height):
    """
    Find optimum Left, Top Right, Bottom to crop the image"""
    for box in boxes:
        box_width, box_height = box[2] - box[0], box[3] - box[1]

        if box_width * box_height > (0.25 * most_width) * (0.25 * most_height):
            # if bestLeft > box[0]:
            #     bestLeft = box[0]
            
            # if bestTop > box[1]:
            #     bestTop = box[1]
            
            # if bestRight < box[2]:
            #     bestRight = box[2]
            
            # if bestBottom < box[3]:
            #     bestBottom = box[3]

            bestLeft = box[0] if bestLeft > box[0] else bestLeft
            bestTop = box[1] if bestTop > box[1] else bestTop
            bestRight = box[2] if bestRight < box[2] else bestRight
            bestBottom = box[3] if bestBottom < box[3] else bestBottom

    return bestLeft, bestTop, bestRight, bestBottom

def Padding(image, bestLeft, bestTop, bestRight, bestBottom, alpha=0.025):
    width, height = image.size

    if 0 < bestLeft < alpha*width:
        bestLeft = 0
    else:
        bestLeft -= alpha * width

    if 0 < bestTop < alpha*height:
        bestTop = 0
    else:
        bestTop -= alpha * height

    if (1-alpha)*width < bestRight < width:
        bestRight = width
    else:
        bestRight += alpha * width

    if (1-alpha)*height < bestBottom < height:
        bestBottom = height
    else:
        bestBottom += alpha * height
    
    bestLeft = 0 if 0 < bestLeft < alpha * width else bestLeft - alpha * width
    bestTop = 0 if 0 < bestTop < alpha * height else bestTop - alpha * height
    bestRight = width if (1 - alpha) * width < bestRight < width else bestRight + alpha * width
    bestBottom = height if (1 - alpha) * height < bestBottom < height else bestBottom + alpha * height

    bestLeft, bestTop, bestRight, bestBottom = int(round(number=bestLeft, ndigits=0)), int(round(number=bestTop, ndigits=0)), int(round(number=bestRight, ndigits=0)), int(round(number=bestBottom, ndigits=0))

    # bestLeft = int(round(number=0 if 0 < bestLeft < alpha * width else bestLeft - alpha * width, ndigits=0))
    # bestTop = int(round(number=0 if 0 < bestTop < alpha * height else bestTop - alpha * height, ndigits=0))
    # bestRight = int(round(number=width if (1 - alpha) * width < bestRight < width else bestRight + alpha * width, ndigits=0))
    # bestBottom = int(round(number=height if (1 - alpha) * height < bestBottom < height else bestBottom + alpha * height, ndigits=0))

    return bestLeft, bestTop, bestRight, bestBottom

def AdjustBBOX(boxes, x1, y1, image):
    """
    move (x1,y1) to (0, 0) and this will be the basis\n
    And then move all boxes based on basis
    """
    width, height = image.size
    boxes[:,0::2] -= x1
    boxes[:,1::2] -= y1
    boxes[:,0::2] = np.clip(boxes[:,0::2], 0, width-1)
    boxes[:,1::2] = np.clip(boxes[:,1::2], 0, height-1)

    return boxes

def show_image(boxes, image):
    image = np.array(object=image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)     # If you want to implement in the yolov2-pytorch, you must change the variable name different with parameter "image"
                                                                # Because cv2.cvtColor() cause the error in the PIL.Image.transpose(Image.FLIP_LEFT_RIGHT)
    # show image
    for i in range(boxes.shape[0]):
            label = boxes[i]
            x1, y1, x2, y2 =  label[0], label[1], label[2], label[3]
            cv2.rectangle(img=image, pt1=(int(x1),int(y1)), pt2=(int(x2),int(y2)), color=(0, 255, 0), thickness=2)

    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite(filename="./result/7. Final result.jpg", img=image)       # save image




################################################################################################################################################################




# image_file_path = "/home/msis/Work/yolov2-pytorch/3class_bdd_crop/original_images/train/0af07355-56e42b91.jpg"
image_file_path = "/home/msis/Work/yolov2-pytorch/3class_bdd_crop/original_images/val/c55da7df-537c27fd.jpg"
label_file_path = image_file_path.split(sep='.')[0] + '.txt'

image = Image.open(fp=image_file_path, mode='r')
width, height = image.size

# Raed label
boxes, classes = ReadLabels(label_file_path=label_file_path)
boxes, classes = np.array(boxes), np.array(classes)  # convert to numpy array

boxes_ = np.array(boxes)
boxes_[ : , 0] *= width;     boxes_[ : , 2] *= width
boxes_[ : , 1] *= height;    boxes_[ : , 3] *= height
show_image(boxes=boxes_, image=image)

# Calculate MOST top, left, bottom, right       0 <=    <= 1
mostLeft, mostTop, mostRight, mostBottom = 1, 1, 0, 0
for box in boxes:
    mostLeft, mostTop, mostRight, mostBottom = min(box[0], mostLeft), min(box[1], mostTop), max(box[2], mostRight), max(box[3], mostBottom)

# Upsampling image, boxes       0, 0 <=     <= 2048, 1152
image, width, height, boxes = UpSampling(image=image, boxes=boxes, sf=1.6)  # width : 2048  height : 1152

# Upsampling mostLeft, mostRight, mostTop, mostBottom
mostLeft, mostRight, mostTop, mostBottom = int(round(number=mostLeft*width, ndigits=0)), int(round(number=mostRight*width, ndigits=0)), int(round(number=mostTop*height, ndigits=0)), int(round(number=mostBottom*height, ndigits=0))

# Calculate AVERAGE TOP, LEFT, BOTTOM, RIGHT
# avgLeft, avgTop, avgRight, avgBottom = np.average(a=boxes, axis=0)
# Find the optimum cropping area
# bestLeft, bestTop, bestRight, bestBottom = avgLeft, avgTop, avgRight, avgBottom
bestLeft, bestTop, bestRight, bestBottom = np.average(a=boxes, axis=0)
most_width, most_height = mostRight - mostLeft, mostBottom - mostTop
bestLeft, bestTop, bestRight, bestBottom = OptimumArea(boxes=boxes, bestLeft=bestLeft, bestTop=bestTop, bestRight=bestRight, bestBottom=bestBottom, most_width=most_width, most_height=most_height)

# Padding
bestLeft, bestTop, bestRight, bestBottom = Padding(image=image, bestLeft=bestLeft, bestTop=bestTop, bestRight=bestRight, bestBottom=bestBottom)
# bestLeft, bestTop, bestRight, bestBottom = int(round(number=bestLeft, ndigits=0)), int(round(number=bestTop, ndigits=0)), int(round(number=bestRight, ndigits=0)), int(round(number=bestBottom, ndigits=0))

# # adijust the avg coordinate
# bestLeft -= mostLeft
# bestRight -= mostLeft
# bestTop -= mostTop
# bestBottom -= mostTop

# cropping by best Left, Top, Right, Bottom
image = image.crop(box=(bestLeft, bestTop, bestRight, bestBottom))
width, height = image.size      # width : 855   height : 329

# Adjust the bounding box coordinate
boxes = AdjustBBOX(boxes=boxes, x1=bestLeft, y1=bestTop, image=image)

# keep the box that is not width == 0 | height == 0
keep    = (boxes[:, 0] != boxes[:, 2]) | (boxes[:, 1] != boxes[:, 3])
boxes   = boxes[keep, :]        # if {width == 0 | height == 0} exist, 'boxes' will change
classes = classes[keep]         # if {width == 0 | height == 0} exist, 'classes' will change

# keep the box that width > 8 & height > 6
# keep =  ((boxes[:, 2] - boxes[:, 0]) > 8) & ((boxes[:, 3] - boxes[:, 1]) > 6)
# boxes   = boxes[keep, :]
# classes = classes[keep]


show_image(boxes=boxes, image=image)