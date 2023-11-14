import cv2

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utilities.general import xyxy2xywh
from utilities.barcode import Barcode
from config import Config

# initialize deepsort
cfg = get_config()
# cfg.merge_from_file(opt.config_deepsort)
cfg.merge_from_file(Config.DEEP_SORT_CNF_FILE)


class Tracker:
    def __init__(self):

        self.deepsort = DeepSort(Config.DEEP_SORT_MODEL,
                                 "cpu",
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=100,  # cfg.DEEPSORT.MAX_AGE,
                                 n_init=0,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 )

    def deepsort_tracking_barcode(self, img0, barcode_list, torch_barcode, barcode_cnt):
        # print(torch_barcode)
        xywhs = xyxy2xywh(torch_barcode[:, 0:4])
        confs = torch_barcode[:, 4]
        clss = torch_barcode[:, 5]
        # print(clss)
        outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img0)
        im = img0.copy()
        current_barcode = []

        # use for check missing
        h, w, _ = img0.shape
        buff = 10
        w -= buff
        h -= buff

        # print(outputs)
        for output in outputs:
            # print(output)
            bbox = tuple((int(ele) for ele in output[0:4]))

            # nt_bang: make sure the object is not missing any part
            if bbox[0] < buff or bbox[1] < buff or bbox[2] > w or bbox[3] > h:
                continue
            # print(bbox)

            obj_id = str(output[4])
            c = int(output[5])
            conf = int(output[6])
            # cv2.putText(im, f"{obj_id}", (bbox[0] - 20, bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # if c == 0:
            current_barcode.append(obj_id)
            if obj_id not in barcode_list.keys():  # not detected yet
                # print(f"New {name}: - id = {obj_id}")
                barcode_list[obj_id] = Barcode(id=barcode_cnt, bbox=bbox, conf=conf, isbarcode=c)
                barcode_cnt += 1
            else:
                barcode_list[obj_id].update_self_bbox(bbox)

        return barcode_list, barcode_cnt, current_barcode, im

    def increment_ages(self):
        self.deepsort.increment_ages()
