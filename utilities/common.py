import queue
import time
from threading import Thread

from PIL import Image
import cv2
import numpy as np

from pyzbar.pyzbar import decode
from config import Config
from utilities.general import SRC
from utilities.ocr_helper import PDOCR_MODEL
from utilities.general import tokai_debug

img_no_barcode = cv2.imread(Config.NO_BARCODE)
img_no_tagname = cv2.imread(Config.NO_TAGNAME)

queue_result = queue.Queue()

pd_ocr_model = PDOCR_MODEL()


class OutputLayout:
    def __init__(self):
        self.yolo_out = None
        self.num_barcode = 3
        self.num_ocr = 1
        self.conf_list = []
        self.text_list = []
        self.barcode_list = []
        self.barcode_err_list = []
        self.ocr_list = []
        self.visible = [[], []]  # barcode, tagname

    def update_yolo_output_image(self, im):
        self.yolo_out = im

    def add_barcode_output(self, id_bar, im=None, txt='', conf=0, visible=True):
        # self.conf_list.append(conf)
        # self.text_list.append(txt)
        if txt != "Decode Error":
            self.barcode_list.append((id_bar, conf, im, txt))
        else:
            self.barcode_err_list.append((id_bar, conf, im, txt))

        self.visible[0].append(visible)

    def add_ocr_output(self, id_ocr, im=None, txt='', visible=True):
        self.ocr_list.append((id_ocr, im, txt))
        # self.ocr_list.append(txt)
        self.visible[1].append(visible)

    def sort_barcode(self):
        self.barcode_list.sort(key=lambda x: x[1])
        self.barcode_err_list.sort(key=lambda x: x[1])
        self.barcode_list += self.barcode_err_list

    def result(self):
        # return barcode and ocr (include text), visibility (4 objects)
        self.sort_barcode()
        mb = min(len(self.barcode_list), 3)
        return self.yolo_out, self.barcode_list[:mb], self.ocr_list[:1]


def decode_barcode(cropimg, id):
    start = time.time()
    bin_im = sauvola(cropimg)
    decoded_objects = decode(Image.fromarray(bin_im))
    name = ''
    for obj in decoded_objects:
        barcode_data = obj.data.decode('utf-8')
        barcode_type = obj.type
        name += f"Code: {barcode_data}, Type: {barcode_type}\n"

    barcode_text = (name if len(name) > 0 else "Decode Error")
    tend = time.time() - start
    queue_result.put((0, id, barcode_text, tend))


def make_img_ocr(im, bbox, place, text=''):
    textsizey = text.split("\n")
    text_w = ''
    for i in textsizey:
        text_w = i if len(i) > len(text_w) else text_w

    yellow_color = np.array([[[255, 255, 255]]], dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0)
    text_size = cv2.getTextSize(text_w, font, font_scale, font_thickness)[0]

    yellow_image = np.full(((text_size[1] + 20) * len(textsizey), text_size[0] + 30, 3), yellow_color, dtype=np.uint8)
    border_color = (0, 0, 255)
    border_width = 3
    yellow_image_with_border = cv2.copyMakeBorder(yellow_image, border_width, border_width, border_width, border_width,
                                                  cv2.BORDER_CONSTANT, value=border_color)
    text_x = 5
    # text_y = (yellow_image_with_border.shape[0] + text_size[1]) // 2
    text_y = int((text_size[1] + 30) / 2)
    for i in textsizey:
        cv2.putText(yellow_image_with_border, i, (text_x, text_y), font, font_scale, text_color, font_thickness)
        text_y += text_size[1] + 20
    y, x, _ = yellow_image_with_border.shape

    y_im, x_im, _ = im.shape

    if place == 0:
        y1 = max(int(bbox[1]) - y, 0)
        y2 = min(int(bbox[1]), y_im)

    else:
        y1 = int(bbox[3])
        y2 = min(int(bbox[3]) + y, y_im)

    x1 = max(int(bbox[0]), 0)
    x2 = min(int(bbox[0]) + x, x_im)
    try:
        im[y1:y2, x1:x2] = yellow_image_with_border[:y2 - y1, :x2 - x1]
    except:
        pass
    return im


def rec_ocr(cropimg):
    start = time.time()
    ocr_img, ocr_text = pd_ocr_model.run_paddle_ocr(cropimg)
    tend = time.time() - start
    return ocr_img, ocr_text


# def rec_ocr(cropimg, id):
#     start = time.time()
#     # rec_res = ocr_det.ocr(cropimg, cls=True)[0]
#     # ocr_img, ocr_text = decode_ocr(rec_res, cropimg)
#     ocr_img, ocr_text = pd_ocr_model.run_paddle_ocr(cropimg)
#     tend = time.time() - start
#     queue_result.put((1, id, ocr_img, ocr_text, tend))


def crop(im, bbox, thresholdx=0, thresholdy=0):
    h, w = im.shape[:2]
    x1 = bbox[0] - thresholdx
    x2 = bbox[2] + thresholdx
    y1 = bbox[1] - thresholdy
    y2 = bbox[3] + thresholdy
    x1 = x1 if x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0
    x2 = x2 if x2 < w else w
    y2 = y2 if y2 < h else h
    return im[y1:y2, x1:x2]


def sauvola(image):
    h, w = image.shape[:2]
    image = cv2.resize(image, (w * 10, h * 10), fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window_sizex = int(gray_image.shape[1] / 10)
    binary_image = np.zeros_like(gray_image)

    for i in range(0, gray_image.shape[1], window_sizex):
        window = gray_image[0: gray_image.shape[0], i:i + window_sizex]
        mean = np.mean(window)
        std = np.std(window)
        threshold = mean * (1 + 0.2 * ((std / 128) - 1)) + 10
        window_binary = (window > threshold) * 255
        binary_image[0: gray_image.shape[0], i:i + window_sizex] = window_binary
    return binary_image


def make_img_text(im, bbox, place, text=''):
    text = text.replace("Data:", "").replace(" ", "").replace("Code:", "")
    yellow_color = np.array([[[0, 255, 255]]], dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    yellow_image = np.full((50, text_size[0] + 10, 3), yellow_color, dtype=np.uint8)
    border_color = (0, 0, 255)
    border_width = 3
    yellow_image_with_border = cv2.copyMakeBorder(yellow_image, border_width, border_width, border_width, border_width,
                                                  cv2.BORDER_CONSTANT, value=border_color)
    text_x = 5
    text_y = (yellow_image_with_border.shape[0] + text_size[1]) // 2
    cv2.putText(yellow_image_with_border, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    y, x, _ = yellow_image_with_border.shape

    y_im, x_im, _ = im.shape

    if place == 0:
        y1 = max(int(bbox[1]) - y, 0)
        y2 = min(int(bbox[1]), y_im)

    else:
        y1 = int(bbox[3])
        y2 = min(int(bbox[3]) + y, y_im)

    x1 = max(int(bbox[0]), 0)
    x2 = min(int(bbox[0]) + x, x_im)
    try:
        im[y1:y2, x1:x2] = yellow_image_with_border[:y2 - y1, :x2 - x1]
    except:
        pass
    return im


def intersection_area(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
    ymin1 -= 50
    ymax1 += 50
    ymin2 -= 50
    ymax2 += 50

    inter_width = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    inter_height = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))

    intersection_area = inter_width * inter_height
    return intersection_area > 0 and ymax1 > ymax2


def decode_img(im, BarcodeList1, current_Barcodes):
    BarcodeList = BarcodeList1.copy()
    if im is None:
        print("No image")
        return None, None
    not_appeared_tires = [tid for tid in BarcodeList.keys() if tid not in current_Barcodes]
    for idx in not_appeared_tires:
        BarcodeList.pop(idx)

    im_copy = im.copy()
    detection_result = OutputLayout()

    barcode_cnt = 0
    orc_cnt = 0

    total_threads = []

    for id, barcode in BarcodeList.items():
        if not barcode.status:
            if barcode.isbarcode == 0:
                barcode.cropimg = crop(im_copy, barcode.bbox, 30, 30)

                # barcode.decoding = decode_barcode(barcode.cropimg, id)
                th1 = Thread(target=decode_barcode, args=(barcode.cropimg, id,))
                th1.start()

                total_threads.append(th1)

            elif barcode.isbarcode == 1:
                barcode.status = True
                barcode.cropimg = crop(im_copy, barcode.bbox, 10, 10)

                start = time.time()
                # not use Thread to prevent "Windows fatal exception: access violation" error
                ocr_img, ocr_text = pd_ocr_model.run_paddle_ocr(barcode.cropimg)
                tend = time.time() - start
                BarcodeList[id].tagname = ocr_text
                BarcodeList[id].ocr_img = ocr_img
                tokai_debug.add_tag_end(tend)

    for th in total_threads:
        th.join()

    while not queue_result.empty():
        q_item = queue_result.get()

        if q_item[0] == 0:
            id, txt, tend = q_item[1], q_item[2], q_item[3]
            BarcodeList[id].decoding = txt
            tokai_debug.add_bar_end(tend)
            # print("IDB:", id)
            if txt != "Decode Error":
                BarcodeList[id].status = True

    for id, barcode in BarcodeList.items():
        if barcode.isbarcode == 0:
            for id2, barcode2 in BarcodeList.items():
                if barcode2.isbarcode == 0 and barcode.ID != barcode2.ID and \
                        intersection_area(barcode.bbox[0], barcode.bbox[2], barcode.bbox[1], barcode.bbox[3],
                                          barcode2.bbox[0], barcode2.bbox[2], barcode2.bbox[1], barcode2.bbox[3]):
                    barcode.place = 1

    for id, barcode in BarcodeList.items():
        if barcode.isbarcode == 0:
            barcode_cnt += 1
            cv2.rectangle(im, (int(barcode.bbox[0]), int(barcode.bbox[1])),
                          (int(barcode.bbox[2]), int(barcode.bbox[3])), (255, 0, 0), 2)

            barcode_img = (np.array(barcode.cropimg))
            barcode_text = barcode.decoding.split("\n")
            # barcode_text= barcode_text
            if len(barcode_text) > 1:
                barcode_text = barcode_text[barcode.place]
            else:
                barcode_text = barcode_text[0]

            im = make_img_text(im, barcode.bbox, barcode.place, barcode_text)
            detection_result.add_barcode_output(id, barcode_img, barcode_text, int(barcode.conf * 100))

        # nt_bang: Tag-name processing
        elif barcode.isbarcode == 1:
            orc_cnt += 1
            im = make_img_ocr(im, barcode.bbox, barcode.place, barcode.tagname)
            cv2.rectangle(im, (int(barcode.bbox[0]), int(barcode.bbox[1])),
                          (int(barcode.bbox[2]), int(barcode.bbox[3])), (0, 255, 0), 2)
            detection_result.add_ocr_output(id, barcode.ocr_img, barcode.tagname)

    detection_result.update_yolo_output_image(im)

    # nt_bang: If no barcode detected
    if barcode_cnt == 0:
        detection_result.add_barcode_output(-1, img_no_barcode, 'NO BARCODE DETECTED')

    # nt_bang: If no tag-name detected
    if orc_cnt == 0:
        detection_result.add_ocr_output(-1, img_no_tagname, 'NO TAGNAME DETECTED')

    return detection_result.result()


if __name__ == "__main__":
    pass
