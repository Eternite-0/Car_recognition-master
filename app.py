#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import time
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import tempfile # For temporary video files

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, init_model, cv_imread # Removed allFilePath
# from plate_recognition.double_plate_split_merge import get_split_merge # Already in plate_rec
# from plate_recognition.color_rec import plate_color_rec, init_color_model # plate_rec_color handles this
from car_recognition.car_rec import init_car_rec_model, get_color_and_score
import gradio as gr

# --- Initialize models and device ---
print("Initializing models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    detect_model = attempt_load('weights/detect.pt', map_location=device)
    plate_rec_model = init_model(device, 'weights/plate_rec_color.pth') # This should handle color if it's plate_rec_color
    car_rec_model = init_car_rec_model('weights/car_rec_color.pth', device)
    print("Models initialized successfully.")
except Exception as e:
    print(f"Error initializing models: {e}")
    # Optionally, raise the exception or exit if models are critical
    # raise e 

# Color definitions
clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']
object_color = [(0, 255, 255), (0, 255, 0), (255, 255, 0)] # Assuming 0: single, 1: double, 2: car
class_type = ['单层车牌', '双层车牌', '汽车']


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coordinates (xyxy) from img1_shape to img0_shape"""
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

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
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
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model_instance, car_rec_model_instance):
    h,w,c = img.shape
    result_dict = {}
    x1,y1,x2,y2 = xyxy
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2) # Ensure int
    
    landmarks_np = np.zeros((4,2))
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x,point_y])

    # roi_img is only calculated if it's a plate
    
    class_num_int = int(class_num) # Ensure class_num is int for indexing

    if class_num_int == 2 : # Car
        car_roi_img = img[y1:y2, x1:x2]
        if car_roi_img.size == 0: # Check if ROI is empty
            # Handle empty ROI, perhaps skip or return error indication
            print(f"Warning: Car ROI is empty for bbox {xyxy}. Skipping car recognition for this instance.")
            return None # Or a dict indicating failure
            
        car_color, color_conf = get_color_and_score(car_rec_model_instance, car_roi_img, device)
        result_dict['class_type'] = class_type[class_num_int]
        result_dict['rect'] = [x1,y1,x2,y2]
        result_dict['score'] = conf 
        result_dict['object_no'] = class_num_int
        result_dict['car_color'] = car_color
        result_dict['color_conf'] = color_conf

    else: # Plate (single or double)
        roi_img = four_point_transform(img,landmarks_np) 
        if roi_img.size == 0:
            print(f"Warning: Plate ROI (after transform) is empty for landmarks. Skipping plate recognition.")
            return None

        from plate_recognition.double_plate_split_merge import get_split_merge 
        if class_num_int == 1: # Double plate
             roi_img=get_split_merge(roi_img)
             if roi_img.size == 0: # Check after split_merge as well
                 print(f"Warning: Double plate ROI (after split_merge) is empty. Skipping plate recognition.")
                 return None

        plate_number,plate_color = get_plate_result(roi_img, device, plate_rec_model_instance)
        for dan_char in danger: 
            if dan_char in plate_number:
                plate_number = "危险品" 
                break 

        result_dict['class_type'] = class_type[class_num_int]
        result_dict['rect'] = [x1,y1,x2,y2]
        result_dict['landmarks'] = landmarks_np.tolist()
        result_dict['plate_no'] = plate_number
        result_dict['roi_height'] = roi_img.shape[0]
        result_dict['plate_color'] = plate_color
        result_dict['object_no'] = class_num_int
        result_dict['score'] = conf
    return result_dict

def detect_recognition_plate_core(model, orgimg, device, plate_rec_model_instance, img_size, car_rec_model_instance):
    conf_thres = 0.3 
    iou_thres = 0.5 
    dict_list=[]

    img0 = orgimg.copy()
    h0, w0 = orgimg.shape[:2]  
    r = img_size / max(h0, w0) 
    if r != 1:  
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max()) 
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1) 
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() 
    img /= 255.0  
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
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model_instance, car_rec_model_instance)
                if result_dict: # Only append if result_dict is not None
                    dict_list.append(result_dict)
    return dict_list

def draw_result(orgimg, dict_list):
    for result in dict_list:
        rect_area = result['rect'] # This is a list, make a copy if modifying
        rect_area_copy = list(rect_area) # Work with a copy for modifications

        x, y, w_rect, h_rect = rect_area_copy[0], rect_area_copy[1], rect_area_copy[2] - rect_area_copy[0], rect_area_copy[3] - rect_area_copy[1]

        if not result['object_no'] == 2: # License Plate
            padding_w = 0.05 * w_rect 
            padding_h = 0.11 * h_rect
            rect_area_copy[0] = max(0, int(x - padding_w))
            rect_area_copy[1] = max(0, int(y - padding_h))
            rect_area_copy[2] = min(orgimg.shape[1], int(rect_area_copy[2] + padding_w))
            rect_area_copy[3] = min(orgimg.shape[0], int(rect_area_copy[3] + padding_h))

            height_area = int(result['roi_height']/2) if result['roi_height'] > 0 else 10 # Ensure height_area is positive
            landmarks = result['landmarks']
            result_p = result['plate_no']
            if result['object_no'] == 0 : # Single layer
                 result_p += " "+result['plate_color']
            else: # Double layer
                 result_p += " "+result['plate_color']+"双层"
            
            for k in range(4): # Landmarks
                cv2.circle(orgimg, (int(landmarks[k][0]), int(landmarks[k][1])), 5, clors[k], -1)
            
            text_y_pos = rect_area_copy[1] - height_area - 10 if rect_area_copy[1] - height_area - 10 > 0 else rect_area_copy[3]
            if "危险品" in result_p:
                orgimg = cv2ImgAddText(orgimg, result_p, rect_area_copy[0], rect_area_copy[3],(0,0,255),height_area) # Red for danger
            else:
                orgimg = cv2ImgAddText(orgimg, result_p, rect_area_copy[0],text_y_pos ,(0,255,0),height_area) # Green for normal
        else: # Car
            height_area = int((rect_area_copy[3]-rect_area_copy[1])/20) if (rect_area_copy[3]-rect_area_copy[1]) > 20 else 5 # Ensure positive, min height
            car_color_str = "车辆颜色:"+result['car_color']
            orgimg = cv2ImgAddText(orgimg, car_color_str, rect_area_copy[0], rect_area_copy[1],(0,255,0),height_area)

        cv2.rectangle(orgimg,(rect_area_copy[0],rect_area_copy[1]),(rect_area_copy[2],rect_area_copy[3]),object_color[result['object_no']],2)
    return orgimg

def format_result_text(result_dict_list):
    """Formats a list of detection results as a single text string."""
    if not result_dict_list:
        return "未检测到目标。"
    
    texts = []
    for result in result_dict_list:
        if result['object_no'] == 2:  # Car
            texts.append(f"车辆 - 颜色: {result['car_color']} (检测置信度: {result.get('score', 0):.2f}, 颜色置信度: {result.get('color_conf', 0):.2f})")
        else:  # License plate
            plate_type = "单层" if result['object_no'] == 0 else "双层"
            texts.append(f"车牌: {result['plate_no']} - 颜色: {result['plate_color']} - 类型: {plate_type} (检测置信度: {result.get('score', 0):.2f})")
    return "\n".join(texts)

def _process_image_data(img_bgr, img_size_val):
    """Core processing for a single BGR NumPy image. Returns (image_rgb, text_result, dict_list)."""
    if img_bgr is None:
        return None, "错误：无法读取图像数据", []

    if img_bgr.shape[-1] == 4: # BGRA to BGR
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    dict_list = detect_recognition_plate_core(detect_model, img_bgr, device, plate_rec_model, img_size_val, car_rec_model)
    # Make a copy of img_bgr for drawing to avoid modifying the original one passed if it's used elsewhere
    result_img_to_draw = img_bgr.copy()
    result_img_bgr = draw_result(result_img_to_draw, dict_list)
    result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
    result_text = format_result_text(dict_list)
    
    return result_img_rgb, result_text, dict_list

# --- Gradio UI Functions ---
def process_single_image_ui(image_path, img_size_val):
    """UI wrapper for single image processing."""
    if image_path is None:
        return None, "请先上传图片。"
    img_bgr = cv_imread(str(image_path)) # Ensure image_path is string
    processed_img, text_res, _ = _process_image_data(img_bgr, img_size_val) # Ignore dict_list for single image UI text output
    return processed_img, text_res

def process_video_ui(video_path, img_size_val, progress=gr.Progress()):
    """UI wrapper for video processing."""
    if video_path is None:
        return None, "请先上传视频。"

    video_path_str = str(video_path) 
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return None, f"错误：无法打开视频文件 {video_path_str}"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_video_path = temp_out_file.name
    temp_out_file.close() 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out_writer.isOpened():
        cap.release()
        if os.path.exists(output_video_path):
            os.unlink(output_video_path) 
        return None, "错误: 无法创建视频写入器. 请检查编解码器支持 (e.g., mp4v, XVID)."

    processed_frames_count = 0
    for frame_idx in progress.tqdm(range(total_frames), desc="处理视频帧"):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        processed_frame_rgb, _, _ = _process_image_data(frame_bgr, img_size_val) # We only need the image for video
        if processed_frame_rgb is None: # Handle case where _process_image_data might fail
            out_writer.write(frame_bgr) # Write original frame on error
            continue

        processed_frame_bgr_out = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
        
        if processed_frame_bgr_out.shape[0] != frame_height or processed_frame_bgr_out.shape[1] != frame_width:
            processed_frame_bgr_out = cv2.resize(processed_frame_bgr_out, (frame_width, frame_height))

        out_writer.write(processed_frame_bgr_out)
        processed_frames_count += 1

    cap.release()
    out_writer.release()

    if processed_frames_count == 0:
        if os.path.exists(output_video_path):
            os.unlink(output_video_path)
        return None, "视频处理失败，未成功处理任何帧。"

    return output_video_path, f"视频处理完成！已处理 {processed_frames_count} 帧。输出到: {output_video_path}"

def process_batch_images_ui(folder_path_str, img_size_val, progress=gr.Progress()):
    """UI wrapper for batch image processing."""
    if not folder_path_str or not os.path.isdir(folder_path_str):
        return None, "请输入有效的文件夹路径。"

    supported_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(folder_path_str) if os.path.splitext(f)[1].lower() in supported_exts]

    if not image_files:
        return None, "文件夹中未找到支持的图片文件。"

    processed_gallery = []
    status_messages = []
    
    for i, img_file in enumerate(progress.tqdm(image_files, desc="批量处理图片")):
        img_path = os.path.join(folder_path_str, img_file)
        try:
            img_bgr = cv_imread(img_path)
            if img_bgr is None:
                status_messages.append(f"{img_file}: ❌ 读取失败")
                continue
            
            processed_img_rgb, text_result, dict_list_for_image = _process_image_data(img_bgr, img_size_val)
            
            if processed_img_rgb is not None:
                caption_for_gallery = Path(img_file).name # Default caption
                
                if dict_list_for_image:
                    best_caption_found = False
                    for item_info in dict_list_for_image:
                        if 'plate_no' in item_info: # Prioritize plate
                            plate_no = item_info.get('plate_no', 'N/A')
                            plate_color = item_info.get('plate_color', 'N/A')
                            confidence = item_info.get('score', 0.0) # Detection confidence
                            caption_for_gallery = f"{plate_no}-{plate_color} (置信度: {confidence:.2f})"
                            best_caption_found = True
                            break # Found a plate, use its info for caption
                    
                    if not best_caption_found: # No plate, but other detections might exist
                        if text_result and text_result.strip() and text_result != "未检测到目标。":
                            caption_for_gallery = text_result.splitlines()[0] # Use first line of summary
                        else: # No specific detections, but processing happened
                             caption_for_gallery = f"{Path(img_file).name} (未识别特定目标)"
                else: # No detections in dict_list
                    caption_for_gallery = f"{Path(img_file).name} (未检测到目标)"

                processed_gallery.append((processed_img_rgb, caption_for_gallery))
                status_messages.append(f"{img_file}: ✅ 处理成功")
            else:
                status_messages.append(f"{img_file}: ❌ 处理失败 - {text_result}")
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {str(e)}")
            status_messages.append(f"{img_file}: ❌ 处理出错 - {str(e)}")
            continue
            
    final_status = "\n".join(status_messages)
    if not processed_gallery:
         return None, "⚠️ 未成功处理任何图片。\n" + final_status
    return processed_gallery, f"✅ 批量处理完成!\n{final_status}"


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate", text_size="md")) as demo:
    gr.Markdown("# 🚙 车辆与车牌识别系统 🚗")
    gr.Markdown("上传图片、视频或指定文件夹进行检测与识别。")

    shared_img_size_slider = gr.Slider(128, 1024, value=640, step=32, label="图像/帧处理尺寸 (较大尺寸更准但较慢)")

    with gr.Tabs():
        with gr.TabItem("🖼️ 单张图片分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="filepath", label="上传图片", sources=["upload", "clipboard"])
                    process_image_btn = gr.Button("🔍 处理图片", variant="primary")
                with gr.Column(scale=1):
                    output_image_single = gr.Image(label="检测结果", type="numpy", interactive=False)
                    output_text_single = gr.Textbox(label="识别结果", interactive=False, lines=5)
            
            process_image_btn.click(
                process_single_image_ui,
                inputs=[input_image, shared_img_size_slider],
                outputs=[output_image_single, output_text_single]
            )
            gr.Examples(
                examples=[
                    ["./imgs/example1.jpg", 640],
                    ["./imgs/test_3.jpeg", 640],
                    ["./imgs/tset_2.jpeg", 640],
                ],
                inputs=[input_image, shared_img_size_slider], 
                outputs=[output_image_single, output_text_single], 
                fn=process_single_image_ui, 
                cache_examples=os.getenv("GRADIO_CACHE_EXAMPLES", "False").lower() == "true"
            )

        with gr.TabItem("🎬 视频分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="上传视频", sources=["upload"]) # Removed format="mp4" to allow more input types
                    process_video_btn = gr.Button("▶️ 处理视频", variant="primary")
                with gr.Column(scale=1):
                    output_video_processed = gr.Video(label="处理后视频", interactive=False) # Removed format="mp4" here too for consistency if output changes
                    output_video_status = gr.Textbox(label="视频处理状态", interactive=False, lines=3)

            process_video_btn.click(
                process_video_ui,
                inputs=[input_video, shared_img_size_slider],
                outputs=[output_video_processed, output_video_status]
            )
           

        with gr.TabItem("🗂️ 批量图片处理"):
            with gr.Row():
                input_folder = gr.Textbox(label="输入图片文件夹路径 (例如: ./imgs)", placeholder="请填入包含图片的文件夹路径")
            process_batch_btn = gr.Button("⚙️ 开始批量处理", variant="primary")
            
            output_gallery_batch = gr.Gallery(label="批量处理结果", show_label=True, elem_id="gallery", columns=4, height="auto", object_fit="contain")
            output_batch_status = gr.Textbox(label="批量处理状态", interactive=False, lines=5)

            process_batch_btn.click(
                process_batch_images_ui,
                inputs=[input_folder, shared_img_size_slider],
                outputs=[output_gallery_batch, output_batch_status]
            )
            gr.Examples(
                examples=[
                    ["./imgs", 640] 
                ],
                inputs=[input_folder, shared_img_size_slider],
                outputs=[output_gallery_batch, output_batch_status],
                fn=process_batch_images_ui,
                cache_examples=os.getenv("GRADIO_CACHE_EXAMPLES", "False").lower() == "true"
            )

if __name__ == "__main__":
    # Create dummy files and folders for examples if they don't exist
    Path("./imgs").mkdir(parents=True, exist_ok=True)
    
    example_images_to_create = {
        "example1.jpg": "Example 1",
        "test_3.jpeg": "Test 3",
        "tset_2.jpeg": "Test 2", # Assuming typo in original, keeping it as is for consistency with example list
    }

    for fname, text in example_images_to_create.items():
        fpath = Path("./imgs") / fname
        if not fpath.exists():
            try:
                dummy_img = np.zeros((200,300,3), dtype=np.uint8)
                color = (0,255,0) if "CAR" in text else (255,255,255)
                cv2.putText(dummy_img, text, (30,100), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
                cv2.imwrite(str(fpath), dummy_img)
                print(f"Created dummy {fpath} for example.")
            except Exception as e:
                print(f"Could not create dummy image {fpath}: {e}")

    if not Path("./imgs/test_video.mp4").exists():
        try:
            fourcc_dummy = cv2.VideoWriter_fourcc(*'mp4v')
            dummy_video_path = "./imgs/test_video.mp4"
            dummy_video = cv2.VideoWriter(dummy_video_path, fourcc_dummy, 1, (100,100))
            if dummy_video.isOpened():
                for i in range(5): 
                    dummy_frame = np.zeros((100,100,3), dtype=np.uint8)
                    cv2.putText(dummy_frame, f"Test {i+1}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                    dummy_video.write(dummy_frame)
                dummy_video.release()
                print(f"Created dummy {dummy_video_path} for example.")
            else:
                print(f"Could not open VideoWriter for dummy video {dummy_video_path}.")
        except Exception as e:
            print(f"Could not create dummy video: {e}")


    parser = argparse.ArgumentParser(description='车辆与车牌识别系统')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=7865, help='服务器端口')
    parser.add_argument('--share', action='store_true', help='是否创建公共链接')
    parser.add_argument('--cache-examples', action='store_true', help='是否缓存Gradio示例（用于生产环境）')
    args = parser.parse_args()

    if args.cache_examples:
        os.environ["GRADIO_CACHE_EXAMPLES"] = "true"

    print("=" * 50)
    print("🚀 车辆、车牌与视频识别系统启动中...")
    print(f"📡 服务器地址: http://{args.host}:{args.port}")
    print(f"🔗 共享链接: {'启用' if args.share else '禁用'}")
    print(f"📦 示例缓存: {'启用' if os.getenv('GRADIO_CACHE_EXAMPLES') == 'true' else '禁用'}")
    print("=" * 50)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )