import faulthandler
import os
import traceback
import cv2
import numpy as np

from utilities.tracker import Tracker

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from config import Config

from utilities.dataset import CVFreshestFrame
from utilities.general import SRC, tokai_debug, ShowBiz
from utilities.common import decode_img
from utilities.yolo_helper import yolo_model
from utilities.popup_windows import ConfidencePopup, center, MessageBox

from tkinter import Tk, Button, Label, Menu, messagebox, ttk
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Notebook

from PIL import ImageTk, Image

faulthandler.enable()

def get_data_askfile(title: str):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    dirPath = str(askopenfilename(title=title)).strip()
    if not len(dirPath):
        return None
    return dirPath


class App:
    def __init__(self, window):
        self.window = window
        self.window.maxsize(1920, 1080)
        # self.window.resizable(0, 0)
        self.window.title("TokaiRika")
        self.window.iconbitmap(Config.APP_ICON)
        self.window.bind("<Control-o>", self.open_video)

        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

        self.defaultBackground = window.cget("background")

        self.link = ""

        self.runUpdate = False
        self.videoCapture = None

        self.currentHeight = 504
        self.currentWidth = 896
        # is in error box
        self.isErrorOpening = False
        self.conf = [Config.VERSION, Config.CONF_THRES, Config.IOU_THRES, Config.INFER_SIZE]

        self.create_tool_bar()

        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Consolas", "14", "bold"), padding=[10, 5])

        self.tab_control = Notebook(self.window)
        self.tab_control.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.create_tab_control()

        center(self.window)
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.mainloop()

    def quit(self):
        try:
            print(f"[{SRC.GUI}] - [DEBUG]\n"
                  f"-----> Avg time: {tokai_debug.yolo_cnt} - {tokai_debug.get_total_time()}\n"
                  f"-----> Max: {tokai_debug.max_time}\n"
                  f"-----> YOLO: {tokai_debug.yolo_cnt} - {tokai_debug.get_yolo_time()}\n"
                  f"-----> DeepSORT: {tokai_debug.ds_cnt} - {tokai_debug.get_ds_time()}\n"
                  f"-----> Barcode: {tokai_debug.bar_cnt} - {tokai_debug.get_bar_time()}\n"
                  f"-----> OCR: {tokai_debug.tag_cnt} - {tokai_debug.get_tag_time()}")

            self.runUpdate = False
            if self.videoCapture:
                self.videoCapture.release()
                cv2.destroyAllWindows()

            self.window.destroy()
            self.window.quit()

        except:
            # the window has been destroyed
            traceback.print_exc()

    def create_tool_bar(self):
        menu_bar = Menu(self.window)
        self.window.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Video", command=self.open_video)
        file_menu.add_command(label="Setting", command=self.popup_set_conf)
        file_menu.add_command(label="Version",
                              command=lambda: MessageBox(self.window, f"TokaiRika Version {Config.APP_VERSION}"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

    def create_tab_control(self):
        self.tab_video = ttk.Frame(self.tab_control)
        # self.tab_video.grid(row=0, column=0, sticky="nsew")
        self.tab_video.bind("<Configure>", self._resize_image)

        self.tab_video.rowconfigure(0, weight=1)
        self.tab_video.columnconfigure(0, weight=1)

        self.hypl_connect = Button(self.tab_video, text="Open Video", fg="blue", cursor="hand2", font=("Consolas", 32),
                                   bd=0, highlightthickness=0, width=39, height=10,
                                   command=self.open_video)
        self.hypl_connect.grid(row=0, column=0, sticky="nsew")
        # ----------------------------------------------------------------------------------------------- Tab barcode --
        self.tab_barcode = ttk.Frame(self.tab_control)
        self.tab_barcode.grid_columnconfigure((0, 1), weight=1)

        self.barcode_1_img = Label(self.tab_barcode, borderwidth=1, relief="solid", height=7)
        self.barcode_1_img.grid(row=0, column=0, sticky="nsew")

        self.barcode_1_txt = Label(self.tab_barcode, borderwidth=1, relief="solid", font=("Consolas", "14"), height=7)
        self.barcode_1_txt.grid(row=0, column=1, sticky="nsew")

        self.barcode_2_img = Label(self.tab_barcode, height=7)
        self.barcode_2_img.grid(row=1, column=0, sticky="nsew")

        self.barcode_2_txt = Label(self.tab_barcode, font=("Consolas", "14"), height=7)
        self.barcode_2_txt.grid(row=1, column=1, sticky="nsew")

        self.barcode_3_img = Label(self.tab_barcode, height=7)
        self.barcode_3_img.grid(row=2, column=0, sticky="nsew")

        self.barcode_3_txt = Label(self.tab_barcode, font=("Consolas", "14"), height=7)
        self.barcode_3_txt.grid(row=2, column=1, sticky="nsew")

        for wid in self.tab_barcode.winfo_children():
            wid.grid(padx=10, pady=10, )

        # --------------------------------------------------------------------------------------------------- Tab OCR --
        self.tab_ocr = ttk.Frame(self.tab_control)
        self.tab_ocr.grid_columnconfigure((0, 1), weight=1)

        self.ocr_1_img = Label(self.tab_ocr, borderwidth=1, relief="solid", height=20)
        self.ocr_1_img.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.ocr_1_txt = Label(self.tab_ocr, font=("Consolas", "14"), borderwidth=1, relief="solid", height=20)
        self.ocr_1_txt.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # add frames to self.tab_control
        self.tab_control.add(self.tab_video, text="Video")
        # self.tab_control.add(self.tab_ouput, text="Object Detection")
        self.tab_control.add(self.tab_barcode, text="Barcode Detection")
        self.tab_control.add(self.tab_ocr, text="Tagname Recognition")

    def _resize_image(self, event):
        new_width = event.width
        new_height = event.height
        cal_height = int(((new_width * 9 / 16 + 32) // 32) * 32)
        cal_width = int(((new_height * 16 / 9 + 32) // 32) * 32)
        if 360 <= cal_height <= new_height:
            self.currentWidth = new_width - 25  # padding
            self.currentHeight = cal_height - 25  # padding
        elif 640 <= cal_width <= new_width:
            self.currentWidth = cal_width - 25  # padding
            self.currentHeight = new_height - 25  # padding

    def _pause_detection(self):
        self.runUpdate = False  # turn off video detection
        self.reset_display()

    def _unpause_detection(self):
        if self.link != "":  # video or rtsp
            self.runUpdate = True
            self.open_vid_source(self.link)
        else:
            self.reset_display()

    def popup_set_conf(self):
        self._pause_detection()
        self.w = ConfidencePopup(self.window)
        try:
            if self.conf == self.w.conf:
                # print(f"[{SRC.GUI}] - CONF NO CHANGE")
                pass
            else:
                # print(f"[{SRC.GUI}] - NEW CONF", self.conf)
                ver = self.w.conf[0]
                # print(f"[{SRC.GUI}] - Version check: old = {self.conf[0]} and new = {ver}")
                if self.conf[0] != ver:
                    yolo_model.load(ver)
                self.conf = self.w.conf

            self._unpause_detection()

        except AttributeError:
            traceback.print_exc()
            # print(f"[{SRC.GUI}] - Set confidences: CANCEL")
            return

    def reset_display(self):
        try:
            # destroy current display
            self.hypl_connect.destroy()
            self.displayImage.destroy()
            self.barcode_1_img.configure(image='')
            self.barcode_1_txt.configure(text='')
            self.barcode_2_img.configure(image='', borderwidth=0)
            self.barcode_2_txt.configure(text='', borderwidth=0)
            self.barcode_3_img.configure(image='', borderwidth=0)
            self.barcode_3_txt.configure(text='', borderwidth=0)
            self.ocr_1_img.configure(image='')
            self.ocr_1_txt.configure(text='')
            self.link = ""
        except AttributeError:  # no displayImage
            # traceback.print_exc()
            pass

        # add link: Please connect to camera
        self.hypl_connect = Button(self.tab_video, text="Open Video", fg="blue", cursor="hand2",
                                   font=("Consolas", 32), bd=0, highlightthickness=0, width=39, height=10,
                                   command=self.open_video)
        self.hypl_connect.grid(row=0, column=0, sticky="nsew")

    # ==================================================================================================================
    # ----------------------------------------- RUN process ------------------------------------------------------------
    # ==================================================================================================================

    def open_video(self, event=None):
        self._pause_detection()
        video_file = get_data_askfile("Open Video file")
        # if self.quit()
        if video_file is None:
            self._unpause_detection()
            return
        # unsupported video format
        if not video_file.lower().endswith((".mp4", ".mkv", ".flv", ".wmv", ".mov", ".avi", ".webm")):
            messagebox.showinfo("Info", "Unsupported video format.")
            return
        self.link = str(video_file)

        self.open_vid_source(video_file)

    def open_vid_source(self, source):
        try:
            if self.videoCapture:
                self.videoCapture.release()
                # cv2.destroyAllWindows()

            self.cnt = 0
            if not os.path.isfile(source):
                messagebox.showerror("Error", f"Source '{source}' is not found!")
                return
            else:
                capf = cv2.VideoCapture(source)
                fps = capf.get(cv2.CAP_PROP_FPS)
                capf.release()
                self.videoCapture = CVFreshestFrame(source, fps)
                # print(f"[{SRC.GUI}] - Open video: {source}")

            self.in_running()
        except:
            traceback.print_exc()
            raise Exception("Exception occurred in open_vid_source")

    def in_running(self):
        self.runUpdate = True

        try:
            self.hypl_connect.destroy()
            self.displayImage.destroy()
        except AttributeError:  # no displayImage
            # traceback.print_exc()
            pass

        self.displayImage = Label(self.tab_video)
        self.displayImage.grid(row=0, column=0, sticky="nsew")
        self.displayImage.bind("<Configure>", self._resize_image)
        self.tracker = Tracker()
        self.BacodeList = {}
        self.BarcodeCnt = 1

        self.show_biz = ShowBiz()

        self.update_detection()

    def update_detection(self):
        try:
            if self.runUpdate:

                # print("Begin detection")

                cnt, img0 = self.videoCapture.read()

                # print("- End detection")

                if not cnt:
                    # print(f"[{SRC.GUI}] - [INFO] - Video is over.")
                    self.runUpdate = False
                    messagebox.showinfo("Info", "Video is over!")
                    self.videoCapture.release()
                    self.reset_display()
                    self.BacodeList = {}

                    return

                if not (img0 is None or img0.size == 0):
                    input_image = np.array(img0)

                    tokai_debug.yolo_start()
                    pred = yolo_model.detect(input_image[:, :, ::-1])
                    tokai_debug.yolo_end()

                    current_Barcodes = []
                    if len(pred) > 0:
                        self.BacodeList, self.BarcodeCnt, current_Barcodes, img0 = \
                            self.tracker.deepsort_tracking_barcode(img0, self.BacodeList, pred[0], self.BarcodeCnt)

                    img0, bar_list, ocr_list = decode_img(img0, self.BacodeList, current_Barcodes)

                    tokai_debug.total_end()

                    if img0.shape[0] != self.currentWidth:
                        # print(f"[{SRC.GUI}] - RESIZE", self.currentWidth, self.currentHeight)
                        img0 = cv2.resize(img0, (self.currentWidth, self.currentHeight))

                    imgrz = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(imgrz))

                    if self.displayImage:
                        # self.displayImage.create_image(0, 0, image=self.photo, anchor="nw")
                        self.displayImage.configure(image=self.photo, background=self.defaultBackground, text="")

                        self.show_biz.add_bar(bar_list)

                        if self.show_biz.bar_list[0]["will"]:
                            img, txt = self.show_biz.bar_list[0]["smash"]
                            img_bar1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.photo_bar1 = ImageTk.PhotoImage(image=Image.fromarray(img_bar1))
                            self.barcode_1_img.configure(image=self.photo_bar1, background=self.defaultBackground,
                                                         text="")
                            self.barcode_1_txt.configure(text=txt)

                        if self.show_biz.bar_list[1]["will"]:
                            img, txt = self.show_biz.bar_list[1]["smash"]
                            img_bar2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.photo_bar2 = ImageTk.PhotoImage(image=Image.fromarray(img_bar2))
                            self.barcode_2_img.configure(image=self.photo_bar2, background=self.defaultBackground,
                                                         borderwidth=1, relief="solid")
                            self.barcode_2_txt.configure(text=txt, borderwidth=1, relief="solid")

                        if self.show_biz.bar_list[2]["will"]:
                            img, txt = self.show_biz.bar_list[2]["smash"]
                            img_bar3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.photo_bar3 = ImageTk.PhotoImage(image=Image.fromarray(img_bar3))
                            self.barcode_3_img.configure(image=self.photo_bar3, background=self.defaultBackground,
                                                         borderwidth=1, relief="solid")
                            self.barcode_3_txt.configure(text=txt, borderwidth=1, relief="solid")

                        if ocr_list[0][0] == -1:
                            # no image existed
                            if len(self.show_biz.ocr_list) == 0:
                                img_ocr = cv2.cvtColor(ocr_list[0][1], cv2.COLOR_BGR2RGB)
                                self.photo_ocr = ImageTk.PhotoImage(image=Image.fromarray(img_ocr))
                                self.ocr_1_img.configure(image=self.photo_ocr, background=self.defaultBackground,
                                                         text="")
                                self.ocr_1_txt.configure(text=ocr_list[0][2])
                        else:
                            if not self.show_biz.is_exist_ocr(ocr_list[0][0]):
                                # print("Have ocr")
                                self.show_biz.add_ocr(ocr_list[0][0])
                                img_ocr = cv2.cvtColor(ocr_list[0][1], cv2.COLOR_BGR2RGB)
                                self.photo_ocr = ImageTk.PhotoImage(image=Image.fromarray(img_ocr))
                                self.ocr_1_img.configure(image=self.photo_ocr, background=self.defaultBackground,
                                                         text="")
                                self.ocr_1_txt.configure(text=ocr_list[0][2])

                self.window.after(10, self.update_detection)
        except:
            traceback.print_exc()
            self.runUpdate = False
            self.quit()
            return


def main():
    # Create a window and pass it to the Application object
    App(Tk())


if __name__ == "__main__":
    main()
