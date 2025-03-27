#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import time
import os
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from plate_recognition.color_rec import plate_color_rec, init_color_model
from car_recognition.car_rec import init_car_rec_model, get_color_and_score
import gradio as gr

# Initialize models and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detect_model = attempt_load('weights/detect.pt', map_location=device)
plate_rec_model = init_model(device, 'weights/plate_rec_color.pth')
car_rec_model = init_car_rec_model('weights/car_rec_color.pth', device)

# Color definitions
clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']
object_color = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]
class_type = ['单层车牌', '双层车牌', '汽车']


def detect_and_recognize(image_path, img_size=384):
    """Main function to detect and recognize plates and cars"""
    img = cv_imread(image_path)
    if img is None:
        return None, "Error: Could not read image"

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Perform detection and recognition
    dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, img_size, car_rec_model)

    # Draw results on the image
    result_img = draw_result(img.copy(), dict_list)

    # Prepare text results
    result_text = "\n".join([format_result(result) for result in dict_list])

    return result_img, result_text


def format_result(result):
    """Format a single detection result as text"""
    if result['object_no'] == 2:  # Car
        return f"车辆 - 颜色: {result['car_color']} (置信度: {result['color_conf']:.2f})"
    else:  # License plate
        plate_type = "单层" if result['object_no'] == 0 else "双层"
        return f"车牌: {result['plate_no']} - 颜色: {result['plate_color']} - 类型: {plate_type}"


def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, car_rec_model=None):
    """Detect and recognize plates and cars in an image"""
    conf_thres = 0.3
    iou_thres = 0.5
    dict_list = []

    img0 = orgimg.copy()
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)

    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()

                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks,
                                                     class_num, device, plate_rec_model, car_rec_model)
                dict_list.append(result_dict)

    return dict_list


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model, car_rec_model):
    """Process a single detection and return recognition results"""
    h, w, c = img.shape
    result_dict = {}
    x1, y1, x2, y2 = map(int, xyxy)
    landmarks_np = np.zeros((4, 2))

    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    if int(class_num) == 2:  # Car
        car_roi_img = img[y1:y2, x1:x2]
        car_color, color_conf = get_color_and_score(car_rec_model, car_roi_img, device)
        result_dict.update({
            'class_type': class_type[int(class_num)],
            'rect': [x1, y1, x2, y2],
            'score': conf,
            'object_no': int(class_num),
            'car_color': car_color,
            'color_conf': color_conf
        })
    else:  # License plate
        roi_img = four_point_transform(img, landmarks_np)
        if int(class_num):  # Double plate
            roi_img = get_split_merge(roi_img)

        plate_number, plate_color = get_plate_result(roi_img, device, plate_rec_model)

        for dan in danger:
            if dan in plate_number:
                plate_number = '危险品'

        result_dict.update({
            'class_type': class_type[int(class_num)],
            'rect': [x1, y1, x2, y2],
            'landmarks': landmarks_np.tolist(),
            'plate_no': plate_number,
            'roi_height': roi_img.shape[0],
            'plate_color': plate_color,
            'object_no': int(class_num),
            'score': conf
        })

    return result_dict


def draw_result(orgimg, dict_list):
    """Draw detection and recognition results on the image"""
    for result in dict_list:
        rect_area = result['rect']
        object_no = result['object_no']
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]

        if not object_no == 2:  # License plate
            padding_w = 0.05 * w
            padding_h = 0.11 * h
            rect_area[0] = max(0, int(x - padding_w))
            rect_area[1] = max(0, int(y - padding_h))
            rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
            rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

            height_area = int(result['roi_height'] / 2)
            landmarks = result['landmarks']
            result_p = result['plate_no']

            if result['object_no'] == 0:  # Single plate
                result_p += " " + result['plate_color']
            else:  # Double plate
                result_p += " " + result['plate_color'] + "双层"

            for i in range(4):  # Draw landmarks
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)

            if "危险品" in result_p:
                orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], rect_area[3], (0, 255, 0), height_area)
            else:
                orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], rect_area[1] - height_area - 10, (0, 255, 0),
                                       height_area)
        else:  # Car
            height_area = int((rect_area[3] - rect_area[1]) / 20)
            car_color_str = "车辆颜色: " + result['car_color']
            orgimg = cv2ImgAddText(orgimg, car_color_str, rect_area[0], rect_area[1], (0, 255, 0), height_area)

        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), object_color[object_no], 2)

    return orgimg


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coordinates from img1_shape to img0_shape"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])

    return coords


def four_point_transform(image, pts):
    """Perform perspective transform to get license plate image"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def order_points(pts):
    """Order points for perspective transform"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# Create Gradio interface
demo = gr.Interface(
    fn=detect_and_recognize,
    inputs=[
        gr.Image(type="filepath", label="输入图像"),
        gr.Slider(128, 1024, value=384, step=32, label="图像尺寸")
    ],
    outputs=[
        gr.Image(label="检测结果"),
        gr.Textbox(label="识别结果")
    ],
    title="车辆与车牌识别系统",
    description="上传图像进行车辆和车牌检测与识别",
    examples=[
        ["imgs/example1.jpg", 384],
        ["imgs/example2.png", 384],
        ["imgs/example3.jpg", 384]
    ],
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        text_size="md",
        spacing_size="md",
        radius_size="lg"
    )
)

if __name__ == "__main__":
    demo.launch()