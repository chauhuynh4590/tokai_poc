import time
from enum import Enum, auto

import numpy as np
import torch


def _auto_enumerate():
    # no inspection PyArgumentList
    return auto()


# for debug
class SRC(Enum):
    GUI = _auto_enumerate()
    DATASET = _auto_enumerate()
    GENERAL = _auto_enumerate()
    COMMON = _auto_enumerate()
    POPUP_WIN = _auto_enumerate()
    TIRE_TRACKING = _auto_enumerate()
    TIRES = _auto_enumerate()
    CONFIG = _auto_enumerate()
    TIRE_CHECK = _auto_enumerate()


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class ShowBiz:
    def __init__(self):
        self.bar_list = [
            {
                "will": False,
                "smash": ()
            },
            {
                "will": False,
                "smash": ()
            },
            {
                "will": False,
                "smash": ()
            }
        ]
        self.show = 0
        self.id_bars = []
        self.txt_bars = []
        self.ocr_list = []

    def is_exist_bar(self, id_bar):
        return id_bar in self.bar_list.keys()

    def is_txt_bar_true(self, id_bar, txt):
        try:
            return self.bar_list[id_bar][1] == txt
        except:
            return False

    def update_bar(self, id_bar, img, txt):
        self.bar_list[id_bar] = (img, txt)

    def is_exist_ocr(self, id_ocr):
        return id_ocr in self.ocr_list

    def add_bar(self, bars: [()]):

        for bar in bars:
            if bar[0] == -1:
                # print("Nohope")
                if self.show == 0:
                    item = {
                        "will": True,
                        "smash": (bar[2], bar[3])
                    }
                    self.bar_list[0] = item
                    self.show = -1
            else:
                if bar[0] not in self.id_bars:
                    if bar[3] in self.txt_bars and bar[3] != "Decode Error":
                        index = self.txt_bars.index(bar[3])
                        self.id_bars[index] = bar[3]
                    else:
                        # print("Not you")
                        self.id_bars.append(bar[0])
                        self.txt_bars.append(bar[3])
                        item = {
                            "will": True,
                            "smash": (bar[2], bar[3])
                        }
                        if self.show == -1:  # last no barcode
                            self.bar_list[0] = item
                            self.show = 1
                        else:
                            if self.show < 3:
                                self.bar_list[self.show] = item
                                self.show += 1
                            else:
                                self.bar_list.append(item)
                                self.id_bars.pop(0)
                                self.txt_bars.pop(0)
                                self.bar_list.pop(0)
                else:
                    # print("Never mind")
                    index = self.id_bars.index(bar[0])

                    d_old = self.bar_list[index]["smash"]
                    if bar[3] == "Decode Error":
                        img, txt = d_old[0], d_old[1]
                    else:
                        img, txt = bar[2], bar[3]

                    item = {
                        "will": True,
                        "smash": (img, txt)
                    }
                    self.bar_list[index] = item

    def add_ocr(self, id_ocr):
        self.ocr_list.append(id_ocr)
        if len(self.ocr_list) > 1:
            self.ocr_list.pop(0)


class TokaiDebug:
    def __init__(self):
        self.bar_time = 0.
        self.tag_time = 0.
        self.yolo_time = 0.
        self.total_time = 0.
        self.max_time = 0.
        self.bar_cnt = 0.
        self.tag_cnt = 0.
        self.yolo_cnt = 0.
        # self.bar_time_start = 0.
        # self.tag_time_start = 0.

    def add_bar_end(self, tend):
        self.bar_time += tend
        self.bar_cnt += 1

    def get_bar_time(self):
        return "{:.5f}".format(self.bar_time / self.bar_cnt) if self.bar_cnt > 0 else "NO BAR"

    def add_tag_end(self, tend):
        self.tag_time += tend
        self.tag_cnt += 1

    def get_tag_time(self):
        return "{:.5f}".format(self.tag_time / self.tag_cnt) if self.tag_cnt > 0 else "NO TAG"

    def yolo_start(self):
        self.yolo_time_start = time.time()
        self.total_time_start = time.time()
        self.yolo_cnt += 1

    def yolo_end(self):
        self.yolo_time += time.time() - self.yolo_time_start

    def get_yolo_time(self):
        return "{:.5f}".format(self.yolo_time / self.yolo_cnt) if self.yolo_cnt > 0 else "NO YOLO"

    def total_end(self):
        dur = time.time() - self.total_time_start
        self.max_time = max(self.max_time, dur)
        self.total_time += dur

    def get_total_time(self):
        # return "{:.5f}".format(self.total_time)
        return "{:.5f}".format(self.total_time / self.yolo_cnt) if self.yolo_cnt > 0 else "NO TOTAL"


tokai_debug = TokaiDebug()
