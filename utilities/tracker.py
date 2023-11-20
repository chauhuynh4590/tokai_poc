from utilities.general import xyxy2xywh, tokai_debug
from utilities.barcode import Barcode
from deep_sort import DeepSort
from config import Config


class Tracker:
    def __init__(self):

        self.deepsort = DeepSort(Config.REID_MODEL,
                                 max_dist=0.2,
                                 max_iou_distance=0.7,
                                 max_age=100,
                                 n_init=0,
                                 nn_budget=100,
                                 )

    def deepsort_tracking_barcode(self, img0, barcode_list, torch_barcode, barcode_cnt):
        # print(torch_barcode)
        xywhs = xyxy2xywh(torch_barcode[:, 0:4])
        confs = torch_barcode[:, 4]
        clss = torch_barcode[:, 5]

        tokai_debug.ds_start()
        outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img0)
        tokai_debug.ds_end()

        im = img0.copy()
        current_barcode = []

        # use for check missing
        h, w, _ = img0.shape
        buff = 20
        a = h * w // 16  # 1/16 of image's area
        w -= buff
        h -= buff

        for output in outputs:
            bbox = tuple((int(ele) for ele in output[0:4]))

            # nt_bang: make sure the object is not missing any part
            if bbox[0] < buff or bbox[1] < buff or bbox[2] > w or bbox[3] > h or \
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > a:  # the area should not too large
                continue
            # print(w, h, bbox, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), a)

            obj_id = str(output[4])
            c = int(output[5])
            conf = int(output[6])
            # cv2.putText(im, f"{obj_id}", (bbox[0] - 20, bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            current_barcode.append(obj_id)
            if obj_id not in barcode_list.keys():  # not detected yet
                # print(f"New {name}: - id = {obj_id}")
                barcode_list[obj_id] = Barcode(id=barcode_cnt, bbox=bbox, conf=conf, isbarcode=c)
                barcode_cnt += 1
            else:
                barcode_list[obj_id].update_self_bbox(bbox)

        return barcode_list, barcode_cnt, current_barcode, im
