# -*- coding: utf-8 -*-
import argparse
import os
import math
import cv2
import torch
from torch.autograd import Variable
from detection import imgproc, craft_utils
from detection.craft import CRAFT
from collections import OrderedDict
from skimage import io
from RecognizeText import recognize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
args = parser.parse_args()


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Word:
    def __init__(self, upperleft, lowerright, base=0):
        self.upperleft = upperleft
        self.lowerright = lowerright
        self.base = base
        self.word = None
        self.position = None

    def mycmp(self, other):
        if abs(self.lowerright.y - other.lowerright.y) < self.base:
            if self.lowerright.x < other.lowerright.x:
                return -1
            else:
                return 1
        elif other.lowerright.y <= self.lowerright.y + self.base:
            return 1
        elif self.lowerright.y <= other.lowerright.y + self.base:
            return -1

    def __lt__(self, other):
        return self.mycmp(other) < 0

    def __gt__(self, other):
        return self.mycmp(other) > 0

    def __eq__(self, other):
        return self.mycmp(other) == 0

    def __le__(self, other):
        return self.mycmp(other) <= 0

    def __ge__(self, other):
        return self.mycmp(other) >= 0

    def __ne__(self, other):
        return self.mycmp(other) != 0


class Paragraph:
    def __init__(self, upperleft, lowerright, words):
        self.upperleft = upperleft
        self.lowerright = lowerright
        self.words = words

    def isOnTheSameCol(self, other):
        if self.upperleft.x <= other.upperleft.x <= self.lowerright.x:
            return True
        if self.upperleft.x <= other.lowerright.x <= self.lowerright.x:
            return True

    def mycmp(self, other):
        if self.isOnTheSameCol(other):
            if self.upperleft.y < other.upperleft.y:
                return -1
            else:
                return 1
        elif other.upperleft.x <= self.upperleft.x:
            return 1
        elif self.upperleft.x <= other.upperleft.x:
            return -1

    def __lt__(self, other):
        return self.mycmp(other) < 0

    def __gt__(self, other):
        return self.mycmp(other) > 0

    def __eq__(self, other):
        return self.mycmp(other) == 0

    def __le__(self, other):
        return self.mycmp(other) <= 0

    def __ge__(self, other):
        return self.mycmp(other) >= 0

    def __ne__(self, other):
        return self.mycmp(other) != 0


def print_position(word):
    print(word.upperleft.x, word.lowerright.x, word.upperleft.y, word.lowerright.y)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_model(path):
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(path, map_location='cpu')))
    net.eval()

    return net


def detect(net, recognizeModel, image_path):
    result_folder = '/Users/zhangliyun/Developer/CXSJ3/temp/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    image = imgproc.loadImage(image_path)

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        score_text, score_link = net(x)

    # make score and link map
    score_text = score_text.cpu().data.numpy()
    score_link = score_link.cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold,
                                           args.low_text, args.poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    def getParameters(box):
        minX = int(min(box[0][0], box[1][0], box[2][0], box[3][0]))
        maxX = int(math.ceil(max(box[0][0], box[1][0], box[2][0], box[3][0])))
        minY = int(min(box[0][1], box[1][1], box[2][1], box[3][1]))
        maxY = int(math.ceil(max(box[0][1], box[1][1], box[2][1], box[3][1])))

        return max(0, minX), max(0, maxX), max(0, minY), max(0, maxY)

    words = []
    highBox = -1
    shortBox = 999999999
    totalWidth = 0
    totalCharacter = 0
    image = io.imread(image_path)

    for box in polys:
        minX, maxX, minY, maxY = getParameters(box)
        highBox = max(maxY - minY, highBox)
        shortBox = min(maxY - minY, shortBox)
        word = Word(Point(minX, minY), Point(maxX, maxY))
        img = image[word.upperleft.y:word.lowerright.y, word.upperleft.x:word.lowerright.x]
        io.imsave(result_folder + "1.png", img)
        word.word = recognize(recognizeModel, result_folder + "1.png", 1)
        totalWidth += maxX - minX
        totalCharacter += len(word.word)
        words.append(word)

    wordWidth = 4 * (totalWidth / totalCharacter)
    wordHeight = highBox

    for i in range(len(words)):
        words[i].base = highBox - shortBox + 1
    words.sort()

    paragraphs = []
    cnt_paragraphs = 0

    # for index in range(len(words)):
    #     if index == 0:
    #         words[index].position = 0
    #     elif index > 0 and words[index].upperleft.x - words[index - 1].lowerright.x > wordWidth:
    #         words[index].position = words[index - 1].position + 1
    #     elif index > 0 and words[index].upperleft.x - words[index - 1].lowerright.x < -wordWidth:
    #         words[index].position = 0
    #     else:
    #         words[index].position = words[index - 1].position
    #     if words[index].position >= cnt_paragraphs:
    #         paragraphs.append([])
    #         cnt_paragraphs += 1
    #     paragraphs[words[index].position].append(words[index])

    ind = 0
    for word in words:
        flag = False
        for index in range(cnt_paragraphs):
            if paragraphs[index].lowerright.x <= word.upperleft.x and word.upperleft.x - paragraphs[index].lowerright.x > wordWidth:
                pass
            elif word.lowerright.x <= paragraphs[index].upperleft.x and paragraphs[index].upperleft.x - word.lowerright.x > wordWidth:
                pass
            elif word.upperleft.y - paragraphs[index].lowerright.y > wordHeight:
                pass
            else:
                flag = True
                paragraphs[index].words.append(word)
                paragraphs[index].upperleft.x = min(paragraphs[index].upperleft.x, word.upperleft.x)
                paragraphs[index].upperleft.y = min(paragraphs[index].upperleft.y, word.upperleft.y)
                paragraphs[index].lowerright.x = max(paragraphs[index].lowerright.x, word.lowerright.x)
                paragraphs[index].lowerright.y = max(paragraphs[index].lowerright.y, word.lowerright.y)
                break
        if not flag:
            paragraphs.append(Paragraph(Point(word.upperleft.x, word.upperleft.y),
                                        Point(word.lowerright.x, word.lowerright.y),
                                        [word]))
            cnt_paragraphs += 1
        ind += 1

    paragraphs.sort()

    result = ''

    for paragraph in paragraphs:
        first = True
        # paragraph.sort()
        paragraph.words.sort()
        # for word in paragraph:
        for word in paragraph.words:
            if not first:
                result += ' '
            result += word.word
            first = False
        result += '\n\n'

    # first = True
    # for word in words:
    #     img = image[word.upperleft.y:word.lowerright.y, word.upperleft.x:word.lowerright.x]
    #     io.imsave(result_folder + "1.png", img)
    #     if not first:
    #         result += ' '
    #     result += recognize(recognizeModel, result_folder + "1.png", 1)
    #     first = False

    return result


# def test():
#     from RecognizeText import load_model_re
#     image_path = '/Users/zhangliyun/Developer/CRAFT-pytorch-master/data/345.png'
#     detectionPath = "/Users/zhangliyun/Developer/CXSJ3/detection/craft.pth"
#     recognizationPath = "/Users/zhangliyun/Developer/CXSJ3/recognization/crnn.pth"
#     detection = load_model(detectionPath)
#     re = load_model_re(recognizationPath)
#     print(detect(detection, re, image_path))
#
#
# test()
