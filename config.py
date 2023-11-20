import os
import traceback
from configparser import RawConfigParser, NoSectionError
from pathlib import Path
from typing import get_type_hints, Union


class AppConfigError(Exception):
    pass


def _parse_bool(val: Union[str, bool]) -> bool:  # pylint: disable=E1136
    return val if type(val) == bool else val.lower() in ["true", "yes", "1"]


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # base_path = sys._MEIPASS
        base_path = os.path.abspath(".")
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


ROOT = Path(os.getcwd())


# Ensure methods to raise an AppConfigError Exception
# when something was wrong
def safe_cfg_func(func):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            # print('Ouuups: err =', err, ', func =', func, ', args =', args, ', kwargs =', kwargs)
            raise AppConfigError(err)

    return wrapped


MODEL_VERSIONS = ["v0.2"]


class RawConfig:
    """Handle the config.cfg file"""

    rawConfig = RawConfigParser()
    app_cnf_file: str = "config.cfg"

    def remove_cfg_file(self):
        """Remove the config.cfg file

        TODO:
            1. Reset rawConfig (the parser)
            2. Remove the old config.cfg file
        """
        self.rawConfig = RawConfigParser()
        try:
            os.remove(self.app_cnf_file)
        except FileNotFoundError:
            pass

    def update_conf(self, ver, conf, iou, infer):
        self.update_raw_conf("Setting", "version", rf"{ver}")
        self.update_raw_conf("Setting", "conf", rf"{conf}")
        self.update_raw_conf("Setting", "iou", rf"{iou}")
        self.update_raw_conf("Setting", "infer_size", rf"{infer}")

    def update_raw_conf(self, section, key, value):
        """Update a section-key-value, create new if not present"""
        # Update config using section key and the value to change
        # call this when you want to update a value in configuration file
        # with some changes you can save many values in many sections
        try:
            self.rawConfig.read(self.app_cnf_file)
            self.rawConfig.set(section, key, rf"{value}")
            with open(self.app_cnf_file, "w") as output:
                self.rawConfig.write(output)
        except NoSectionError:
            self.rawConfig.add_section(section)
            self.update_raw_conf(section, key, value)

    @safe_cfg_func
    def create_cfg_file(
            self,
            cnf_file: str,
            model_version: str = 'v0.2',
            conf_thres: float = 0.3,
            iou_thres: float = 0.4,
            infer_size: int = 640
    ):
        """Create configuration file"""

        self.app_cnf_file = cnf_file

        self.rawConfig.add_section("Setting")
        self.rawConfig.set("Setting", "version", rf"{model_version}")
        self.rawConfig.set("Setting", "conf", rf"{conf_thres}")
        self.rawConfig.set("Setting", "iou", rf"{iou_thres}")
        self.rawConfig.set("Setting", "infer_size", rf"{infer_size}")

        with open(self.app_cnf_file, "w") as output:
            self.rawConfig.write(output)

        print(f"[INFO] - Create {self.app_cnf_file}")

    @safe_cfg_func
    def get_version(self):
        self.rawConfig.read(self.app_cnf_file)
        ver = self.rawConfig.get("Setting", "version")
        return ver

    @safe_cfg_func
    def get_conf_threshold(self):
        self.rawConfig.read(self.app_cnf_file)
        conf = float(self.rawConfig.get("Setting", "conf"))
        return conf

    @safe_cfg_func
    def get_conf_threshold(self):
        self.rawConfig.read(self.app_cnf_file)
        iou = float(self.rawConfig.get("Setting", "iou"))
        return iou

    @safe_cfg_func
    def get_inference_size(self):
        self.rawConfig.read(self.app_cnf_file)
        size = int(self.rawConfig.get("Setting", "infer_size"))
        return size

    @safe_cfg_func
    def get_all(self):
        self.rawConfig.read(self.app_cnf_file)

        ver = self.rawConfig.get("Setting", "version")
        if ver not in MODEL_VERSIONS:
            raise ValueError(f"Model version is not found (expected {MODEL_VERSIONS}, got {ver})")

        conf = float(self.rawConfig.get("Setting", "conf"))
        if conf < 0 or conf > 1:
            raise ValueError(f"Confidence threshold is corrupted (expected [0, 1], got {conf})")

        iou = float(self.rawConfig.get("Setting", "iou"))
        if iou < 0 or iou > 1:
            raise ValueError(f"IoU threshold is corrupted (expected [0, 1], got {iou})")

        size = int(self.rawConfig.get("Setting", "infer_size"))
        if size < 384 or size > 2048:
            raise ValueError(f"confidence is corrupted (expected [384, 2048], got {size})")

        # print(f"Version: {ver}\n"
        #       f"Conf = {conf}\n"
        #       f"Iou = {iou}\n"
        #       f"Infer = {size}\n"
        #       f"At {self.app_cnf_file}")

        return ver, conf, iou, size


ConfigFile = RawConfig()


class AppConfig:
    APP_VERSION: str = "1.5"
    ENV: str = "production"
    APP_TITLE: str = "TOKAIRIKA DEMO"  # Name of Application
    APP_ICON: str = ROOT / "data/app_icon.ico"  # Icon for Application
    OCR_CHAR_FONT: str = ROOT / r"data/latin.ttf"
    ICON_ASK: str = ROOT / r"data/ask.png"
    ICON_INFO: str = ROOT / r"data/info.png"
    NO_BARCODE: str = ROOT / r"data/no_barcode.jpg"
    NO_TAGNAME: str = ROOT / r"data/no_tagname.jpg"

    VERSION_LIST: dict = {
        MODEL_VERSIONS[0]: ROOT / f"model/v02_openvino_model/v02.xml",
    }

    FRAME_WIDTH = 1280
    FRAME_HEIGHT = int(((FRAME_WIDTH * 9 / 16 + 32) // 32) * 32)

    # check area
    PAD_LEFT: int = 30  # pad PAD_LEFT -> % of width
    PAD_RIGHT: int = 2  # pad PAD_RIGHT -> % of width

    # openvino device
    DEVICE: str = "CPU"

    # openvino models
    PDOCR_DET_MODEL: str = ROOT / "model/en_PP-OCRv3_det_infer/inference.pdmodel"
    PDOCR_REC_MODEL: str = ROOT / "model/en_PP-OCRv3_rec_infer/inference.pdmodel"
    REID_MODEL: str = ROOT / "model/person-reidentification-retail-0287/person-reidentification-retail-0287.xml"

    APP_CNF_FILE: str = ROOT / "config.cfg"

    try:  # load current config.csv file
        VERSION, CONF_THRES, IOU_THRES, INFER_SIZE = ConfigFile.get_all()

    except AppConfigError as e:
        print(f"Reset the config file because of the following error:"
              f"\n{'-' * 50}\n{traceback.format_exc()}\n{'-' * 50}\n")

        ConfigFile.remove_cfg_file()

        if not os.path.exists(APP_CNF_FILE):
            ConfigFile.create_cfg_file(cnf_file=APP_CNF_FILE)

        VERSION, CONF_THRES, IOU_THRES, INFER_SIZE = ConfigFile.get_all()

    """
    Map environment variables to class fields according to these rules:
      - Field won't be parsed unless it has a type annotation
      - Field will be skipped if not in all caps
      - Class field and environment variable name are the same
    """

    def __init__(self, env):
        for field in self.__annotations__:
            if not field.isupper():
                continue
            # Raise AppConfigError if required field not supplied
            default_value = getattr(self, field, None)
            if default_value is None and env.get(field) is None:
                raise AppConfigError("The {} field is required".format(field))

            # Cast env var value to expected type and raise AppConfigError on failure
            try:
                var_type = get_type_hints(AppConfig)[field]
                if var_type == bool:
                    value = _parse_bool(env.get(field, default_value))
                else:
                    value = var_type(env.get(field, default_value))

                self.__setattr__(field, value)
            except ValueError:
                raise AppConfigError("Unable to cast value of '{}' to type '{}' for '{}' field".format(
                    env[field],
                    var_type,
                    field
                )
                )

    def __repr__(self):
        return str(self.__dict__)

    def set_configs(self, ver, conf, iou, infer):
        self.VERSION, self.CONF_THRES, self.IOU_THRES, self.INFER_SIZE = ver, conf, iou, infer


# Expose Config object for app to import
Config = AppConfig(os.environ)
"""
It's fine if a guy is named Guy but weird if a girl is named Girl.
What do you call a blind deer? No eyes deer!
"""
