
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


from ultralytics import YOLO

from parseq.strhub.models.utils import create_model
from parseq.strhub.data.module import SceneTextDataModule

from meter_reading_lib.dataStruct import DetOut, PoseOut, SegOut


def load_yolov8_model(weight):
    model = YOLO(weight)
    return model


def yolov8_pose_infer(model, img, iou_thres, conf_thres, device, **kwargs):
    results = model(img, device=device, iou=iou_thres, conf=conf_thres, **kwargs)[0]
    
    res = []
    if results.keypoints.data.shape[1]:
        kpts = results.keypoints.xy.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()
        xywhs = results.boxes.xywh.cpu().numpy()
        xywhns = results.boxes.xywhn.cpu().numpy()
        xyxyns = results.boxes.xyxyn.cpu().numpy()  
        for box, kpt, xywh, xywhn, xyxyn in zip(boxes, kpts, xywhs, xywhns, xyxyns):
            xyxy, conf, class_id = box[:4], box[4], int(box[-1])
            res.append(PoseOut(
                xyxy, xyxyn, xywh, xywhn, class_id, conf, results.names[class_id], kpt
            ))
    return res


def yolov8_seg_infer(model, img, conf_thres, iou_thres, device, **kwargs):
    results = model(img, save_conf=True, conf=conf_thres, iou=iou_thres, device=device, **kwargs)[0]

    res = []
    if results.masks is not None:
        boxes = results.boxes.data.cpu().numpy()
        xywhs = results.boxes.xywh.cpu().numpy()
        xywhns = results.boxes.xywhn.cpu().numpy()
        xyxyns = results.boxes.xyxyn.cpu().numpy()  
        segments_list = results.masks.xy
        mask_list = results.masks.data.cpu().numpy() 
        for box, segments, mask, xywh, xywhn, xyxyn in zip(boxes, segments_list, mask_list, xywhs, xywhns, xyxyns):
            xyxy, conf, class_id = box[:4], box[4], int(box[-1])
            res.append(SegOut(
                xyxy, xyxyn, xywh, xywhn, class_id, conf, 
                results.names[class_id], mask, segments
            ))
    return res


def yolov8_det_infer(model, img, iou_thres, conf_thres, device, **kwargs):
    results = model(img, save_conf=True, conf=conf_thres, iou=iou_thres, device=device, **kwargs)[0]

    res = []
    if len(results.boxes.data) > 0:
        boxes = results.boxes.data.cpu().numpy()
        xywhs = results.boxes.xywh.cpu().numpy()
        xywhns = results.boxes.xywhn.cpu().numpy()
        xyxyns = results.boxes.xyxyn.cpu().numpy()  
        for box, xywh, xywhn, xyxyn in zip(boxes, xywhs, xywhns, xyxyns):
            xyxy, conf, class_id = box[:4], box[4], int(box[-1])
            res.append(DetOut(
                xyxy, xyxyn, xywh, xywhn, class_id, conf, results.names[class_id]
            ))
    return res


def load_ocr_model(weights, device):
    weights = Path(weights)
    model_name = weights.stem.split("-")[0]
    model = create_model(model_name)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def ocr_model_infer(model, img, device, conf_thres=0.6):
    
    # Load model and image transforms
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img[:, :, ::-1]) # opencv BGR -> PIL
    
    img = img_transform(img).unsqueeze(0).to(device)

    logits = model(img)

    # Greedy decoding
    pred = logits.softmax(-1)
    label, conf = model.tokenizer.decode(pred)

    # predict digit
    label, conf = label[0], conf[0].cpu()
    if conf[0] < conf_thres:
        return None
    try:
        if label in ["-0+", "+0-"]:
            label = 0
        else:
            label = float(label)
    except:
        label = None

    return label
    
from paddleocr import PaddleOCR
def load_ppocr_model(**kwargs):
    model = PaddleOCR(lang="en", det=False, use_angle_cls=True, **kwargs)
    return model


def ppocr_infer(model: PaddleOCR, img: np.ndarray):
    res = model.ocr(img, det=False)[0][0][0]
    try:
        if res in ["-0+", "+0-", "-0", "+0", "0-", "0+"]:
            res = 0
        else:
            res = float(res)
    except:
        res = None
    return res


if __name__ == '__main__':
    weights = "/home/zhangqin/wangjl_data/meter_reading_recognition/weights/pose_scale_digit_v1.pt"
    device = torch.device(0)
    model = load_yolov8_model(weights)

    img = cv2.imread("/home/zhangqin/wangjl_data/meter_reading_recognition/test_dataset/0_nrxt_sub_defect_dir2_51_230504_seg_biaoji.jpg")
    res = yolov8_pose_infer(model, img, 0.1, 0.6, device)
    print(res)
    # img_root = Path("/home/zhangqin/wangjl_data/parseq-main/test_data")
    # for img_f in img_root.iterdir():
    #     img = cv2.imread(str(img_f))

    #     res = ocr_model_infer(model, img, device)
    #     print(res)