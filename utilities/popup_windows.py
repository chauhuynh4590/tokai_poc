import sys
import traceback
from tkinter import *
from tkinter import messagebox
from tkinter.font import Font

from PIL import ImageTk, Image

from config import Config, ConfigFile, MODEL_VERSIONS
from utilities.general import SRC


def center(win):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry("{}x{}+{}+{}".format(width, height, x, y))
    win.deiconify()


class MessageBox:
    """Prevent open multiple times when using default tkinter.messagebox"""
    def __init__(self, master, msg):
        self.top = Toplevel(master)
        self.top.geometry("0x0")
        self.top.resizable(False, False)
        self.top.overrideredirect(True)
        self.top.after(10, self.show, msg)
        # wait for popup complete
        self.top.transient(master)
        self.top.grab_set()
        master.wait_window(self.top)

    def show(self, msg):
        messagebox.showinfo("Information", msg)
        self.top.destroy()


class ConfirmBox:
    def __init__(self, master, msg):
        self.top = Toplevel(master)
        self.top.title("Confirm")
        self.top.resizable(False, False)

        text_font = Font(family="Consolas", size=14)

        self.isOK = False

        img = ImageTk.PhotoImage(Image.open(Config.ICON_ASK))

        Label(self.top, image=img, width=50, font=text_font).grid(row=0, column=0, padx=15, pady=15)
        Label(self.top, text=msg, width=25, height=2, font=text_font,
              wraplength=300, justify="left").grid(row=0, column=1, padx=5, pady=15)

        group_btn = LabelFrame(self.top, width=50, borderwidth=0)
        group_btn.grid(row=1, column=1, sticky="e")
        # Button that lets the user open a video
        self.btn_cancel = Button(group_btn, text="Cancel", width=8, height=1, font=text_font,
                                 command=self.top.destroy)
        self.btn_cancel.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # Button that lets the user open a video
        self.btn_ok = Button(group_btn, text="OK", width=8, height=1, font=text_font,
                             command=self.ok)
        self.btn_ok.grid(row=0, column=1, padx=20, pady=15, sticky="e")

        center(self.top)
        # wait for popup complete
        self.top.transient(master)
        self.top.grab_set()
        master.wait_window(self.top)

    def ok(self):
        self.isOK = True
        self.top.destroy()


class ConfidencePopup:
    def __init__(self, master):
        self.top = Toplevel(master, height=400, width=100)

        self.top.title("Setting")
        self.top.resizable(False, False)

        text_font = Font(family="Consolas", size=14)

        group_setConf = LabelFrame(self.top, borderwidth=0)
        group_setConf.grid(row=1, column=0, )

        Label(group_setConf, text="Model Version:", font=text_font).grid(row=0, column=0, padx=20, pady=15, sticky="w")

        self.entry_version = StringVar(group_setConf)
        self.entry_version.set(MODEL_VERSIONS[0])  # default value

        version = OptionMenu(group_setConf, self.entry_version, *MODEL_VERSIONS)
        version.config(width=11, font=text_font)
        version.grid(row=0, column=1, pady=15, sticky="w")

        Label(group_setConf, text="Confidence:", font=text_font).grid(row=1, column=0, padx=20, pady=15, sticky="w")
        self.entry_conf_thres = Entry(group_setConf, justify="right", width=15, font=text_font)
        self.entry_conf_thres.grid(row=1, column=1, pady=15, sticky="w")
        Label(group_setConf, text="(0 - 100)", width=10, font=text_font).grid(row=1, column=2, padx=15, pady=15, sticky="w")

        Label(group_setConf, text="IoU Threshold:", font=text_font).grid(row=2, column=0, padx=20, pady=15, sticky="w")
        self.entry_iou_thres = Entry(group_setConf, justify="right", width=15, font=text_font)
        self.entry_iou_thres.grid(row=2, column=1, pady=15, sticky="w")
        Label(group_setConf, text="(0 - 100)", width=10, font=text_font).grid(row=2, column=2, padx=15, pady=15, sticky="w")

        Label(group_setConf, text="Inference Size:", font=text_font).grid(row=3, column=0, padx=20, pady=15, sticky="w")
        self.entry_infer_size = Entry(group_setConf, justify="right", width=15, font=text_font)
        self.entry_infer_size.grid(row=3, column=1, pady=15, sticky="w")
        Label(group_setConf, text="(384 - 2048)", width=15, font=text_font).grid(row=3, column=2, padx=5, pady=15, sticky="e")

        self.conf = []
        self._load_conf_info()

        # Button that lets the user open a video
        self.btn_cancel = Button(group_setConf, text="Cancel", width=10, height=1, font=text_font,
                                 command=self._quit)
        self.btn_cancel.grid(row=4, column=0, padx=20, pady=15, sticky="w")

        # Button that lets the user open a video
        self.btn_save = Button(group_setConf, text="Save", width=10, height=1, font=text_font,
                               command=self._on_save_press)
        self.btn_save.grid(row=4, column=2, padx=20, pady=15, sticky="e")

        # wait for popup complete
        center(self.top)
        self.top.transient(master)
        self.top.grab_set()
        master.wait_window(self.top)

    def _load_conf_info(self):
        try:
            conf = list(ConfigFile.get_all())
            # print(conf)
            self.entry_version.set(str(conf[0]))

            self.entry_conf_thres.delete(0, "end")
            self.entry_conf_thres.insert("end", str(int(conf[1] * 100)))

            self.entry_iou_thres.delete(0, "end")
            self.entry_iou_thres.insert("end", str(int(conf[2] * 100)))

            self.entry_infer_size.delete(0, "end")
            self.entry_infer_size.insert("end", str(int(conf[3])))

            # print(f"[{SRC.POPUP_WIN}] - Last Conf: {conf}")
            self.conf = conf

        except:
            traceback.print_exc()
            pass

        return

    def _on_save_press(self):
        try:
            conf_str = [self.entry_version.get(), self.entry_conf_thres.get(), self.entry_iou_thres.get(),
                        self.entry_infer_size.get()]
            if any(x == "" for x in conf_str):
                MessageBox(self.top, "Fields cannot be blank!")
                return
            if any(int(x) != float(x) for x in conf_str[1:]):
                raise ValueError

            conf_percent = float(conf_str[1]) / 100.
            iou_percent = float(conf_str[2]) / 100.

            if conf_percent < 0. or conf_percent > 1.:
                MessageBox(self.top, "Confidence Threshold must be between 0 and 100!")
                return

            if iou_percent < 0. or iou_percent > 1.:
                MessageBox(self.top, "IoU Threshold must be between 0 and 100!")
                return

            infer_size = int(conf_str[3])
            if infer_size < 384 or infer_size > 2048:
                MessageBox(self.top, "Inference Size must be between 384 and 2048")
                return

            ok = ConfirmBox(self.top, "Do you want to save?")
            if not ok.isOK:
                # print(f"[{SRC.POPUP_WIN}] - not save")
                return

            cur_config = [conf_str[0], conf_percent, iou_percent, infer_size]

            # print(f"[{SRC.POPUP_WIN}] - Test conf: {self.conf} vs {cur_config}")

            # ok
            if self.conf != cur_config:
                self.conf = cur_config
                Config.set_configs(conf_str[0], conf_percent, iou_percent, infer_size)
                ConfigFile.update_conf(conf_str[0], conf_percent, iou_percent, infer_size)
            self._quit()
            pass

        except ValueError:
            MessageBox(self.top, "Confidence Threshold, IoU Threshold and Inference Size must be integers!")
            return

    def _quit(self):
        self.top.destroy()


if __name__ == "__main__":

    # -----------------    test
    class MainWindow(object):
        def __init__(self, master):
            self.master = master
            self.ff = LabelFrame(self.master, borderwidth=1)
            self.ff.pack(side="top", fill="both")

            self.master.bind("<Control-Alt-F9>", self.popup_connect)
            self.b = Button(self.ff, text="connect", command=self.popup_connect)
            self.b.pack()

            self.b1 = Button(self.ff, text="set conf", command=self.popup_setConf)
            self.b1.pack()

            self.b3 = Button(self.ff, text="confirm", command=self.popup_confirm)
            self.b3.pack()

            self.b4 = Button(self.ff, text="error", command=self.popup_error)
            self.b4.pack()

            self.b5 = Button(self.ff, text="test", command=self.popup_test)
            self.b5.pack()

            self.bload = Button(self.ff, text="loading", command=self.popup_load)
            self.bload.pack()

            self.b2 = Button(self.ff, text="print value", command=lambda: sys.stdout.write(self.entryValue() + "\n"))
            self.b2.pack()

        def popup_connect(self, event=None):
            pass
            # self.w = CameraPopup(self.master)
            # self.b["state"] = "disabled"
            # self.master.wait_window(self.w.top)
            # self.b["state"] = "normal"

        def popup_load(self, event=None):
            self.w = MessageBox(self.master, "123456789 123456789 123456789 ")


            # self.b["state"] = "disabled"
            # self.master.wait_window(self.w.top)
            # self.b["state"] = "normal"

        def popup_setConf(self):
            self.w = ConfidencePopup(self.master)
            # self.b["state"] = "disabled"
            # self.master.wait_window(self.w.top)
            # self.b["state"] = "normal"

        def popup_confirm(self):
            self.w = ConfirmBox(self.master, "Do you want to reset confidence to default?")
            # self.w = ConfirmBox(self.master, "Do you want to save?")
            # self.b["state"] = "disabled"
            # self.master.wait_window(self.w.top)
            # self.b["state"] = "normal"

        def popup_error(self):
            pass

        def popup_test(self):
            # from tkinter.filedialog import askopenfilename
            # from cv2 import cv2
            #
            # def get_data_askfile(title: str):
            #     Tk().withdraw()  # we don"t want a full GUI, so keep the root window from appearing
            #     dirPath = str(askopenfilename(title=title)).strip()
            #     if not len(dirPath):
            #         return 0
            #     return dirPath
            #
            # file = get_data_askfile("hola")
            # self.im = cv2.imread(file)
            # self.im = cv2.resize(self.im, (1280, 720))
            # imgrz = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            # self.w = CheckAreaPopup(self.master, imgrz)
            pass

        def entryValue(self):
            return self.w.value


    root = Tk()
    m = MainWindow(root)
    root.mainloop()
