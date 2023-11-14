class Barcode:
    def __init__(self, id, bbox, conf, isbarcode=0):
        self.bbox = bbox
        self.ID = id
        self.conf = conf
        self.cropimg = None
        self.status = False
        self.isbarcode = isbarcode
        self.decoding = ''
        self.tagname = ''
        self.orc_img = None
        self.place = 0

    def update_self_bbox(self, bbox):
        self.bbox = bbox
