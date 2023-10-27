import torch 
import cv2
import yaml
from meter_reading_lib.algo import meter_reading_recognition
from meter_reading_lib.model_infer import (
    load_ocr_model, load_yolov8_model
)
from pathlib import Path



if __name__ == "__main__":

    yaml_f = "weights/cfg.yaml"
    with open(yaml_f, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device(0)


    meter_digit_det_model = load_yolov8_model(cfg["meter_digit_det"]["weights"])
    meter_digit_det_model_params = cfg["meter_digit_det"]["params"]

    scale_seg_model = load_yolov8_model(cfg["scale_seg"]["weights"])
    scale_seg_model_params = cfg["scale_seg"]["params"]

    scale_kpt_model = load_yolov8_model(cfg["scale_kpt"]["weights"])
    scale_kpt_model_params = cfg["scale_kpt"]["params"]

    pointer_kpt_model = load_yolov8_model(cfg["pointer_kpt"]["weights"])
    pointer_kpt_model_params = cfg["pointer_kpt"]["params"]

    ocr_model = load_ocr_model(cfg["ocr_model"]["weights"], device)

    root = Path("test_dataset")
    normal_root = root / "normal"
    abnormal_root = root / "abnormal"
    normal_root.mkdir(exist_ok=True)
    abnormal_root.mkdir(exist_ok=True)

    # img = cv2.imread("test_dataset/abnormal/0_nrxt_sub_defect_dir2_51_230504_seg_biaoji.jpg")
    # res = meter_reading_recognition(
    #     img, device, meter_digit_det_model, meter_digit_det_model_params,
    #     scale_seg_model, scale_seg_model_params, scale_kpt_model, scale_kpt_model_params,
    #     pointer_kpt_model, pointer_kpt_model_params, ocr_model
    # )
    # is_abnormal, reading, msg = res.get_info()
    # print(is_abnormal, reading, msg)

    for jpg_f in root.iterdir():
        if jpg_f.suffix == ".jpg":

            img = cv2.imread(f"{jpg_f}")
        
            res = meter_reading_recognition(
                img, device, meter_digit_det_model, meter_digit_det_model_params,
                scale_seg_model, scale_seg_model_params, scale_kpt_model, scale_kpt_model_params,
                pointer_kpt_model, pointer_kpt_model_params, ocr_model
            )
            is_abnormal, reading, msg = res.get_info()
            
            txt_f = jpg_f.with_suffix(".txt")
            with open(txt_f, "w") as f:
                f.write(f"is_abnormal={is_abnormal}, reading={reading}\n{msg}")
            
            if is_abnormal:
                jpg_f.rename(abnormal_root/jpg_f.name)
                txt_f.rename(abnormal_root/txt_f.name)
            else:
                jpg_f.rename(normal_root/jpg_f.name)
                txt_f.rename(normal_root/txt_f.name)