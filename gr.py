import gradio as gr
import time
import random
import os
import shutil
import re # 导入正则表达式模块
from PIL import Image, ImageDraw, ImageFont

# --- 配置 ---
APP_TITLE = "自动任务检测"
RESULTS_CATEGORIES = ["夜间", "强光", "雨天", "雾天", "超分增强", "去残影"] # 顺序很重要
IMAGE_DIR = "images" # 存放示例图片的文件夹
OUTPUT_BASE_DIR = "output_processed_images"

# NEW: Define the new example filenames for single task analysis
SINGLE_TASK_EXAMPLE_FILENAMES = [
    "img_1_123.jpg", "img_2_123.jpg", "img_3_124.jpg",
    "img_4_241.jpg", "img_5_13213.jpg", "img_6_23123.jpg"
]
EXAMPLE_IMAGES = [os.path.join(IMAGE_DIR, fname) for fname in SINGLE_TASK_EXAMPLE_FILENAMES]

# 示例图片与预设结果的映射 (单任务分析用)
# Ensure this map uses the updated EXAMPLE_IMAGES paths
EXAMPLE_TO_RESULT_MAP = {
    EXAMPLE_IMAGES[0]: RESULTS_CATEGORIES[0], # img_1_123.jpg -> 夜间
    EXAMPLE_IMAGES[1]: RESULTS_CATEGORIES[1], # img_2_123.jpg -> 强光
    EXAMPLE_IMAGES[2]: RESULTS_CATEGORIES[2], # img_3_124.jpg -> 雨天
    EXAMPLE_IMAGES[3]: RESULTS_CATEGORIES[3], # img_4_241.jpg -> 雾天
    EXAMPLE_IMAGES[4]: RESULTS_CATEGORIES[4], # img_5_13213.jpg -> 超分增强
    EXAMPLE_IMAGES[5]: RESULTS_CATEGORIES[5], # img_6_23123.jpg -> 去残影
}

# --- 辅助函数：创建文件夹和示例图片 ---
def setup_directories_and_images():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
    for category in RESULTS_CATEGORIES:
        category_path = os.path.join(OUTPUT_BASE_DIR, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    for i, img_path in enumerate(EXAMPLE_IMAGES):
        if not os.path.exists(img_path):
            try:
                img = Image.new('RGB', (200, 150), color = (random.randint(50,200), random.randint(50,200), random.randint(50,200)))
                d = ImageDraw.Draw(img)
                try: font = ImageFont.truetype("simhei.ttf", 20)
                except IOError: font = ImageFont.load_default()
                text = f"示例\n({os.path.basename(img_path)})"
                text_bbox = d.textbbox((0,0), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                x, y = (img.width - text_width) / 2, (img.height - text_height) / 2
                d.text((x, y), text, fill=(255,255,0), font=font)
                img.save(img_path)
                print(f"创建了示例图片: {img_path}")
            except Exception as e:
                print(f"创建示例图片 {img_path} 失败: {e}. 请手动放置图片。")

# --- 单任务分析处理函数 (MODIFIED) ---
def single_task_analysis(image_path_single):
    if image_path_single is None:
        return "请上传一张图片进行分析。"

    image_basename = os.path.basename(image_path_single)
    gr.Info(f"🚀 正在分析图像: '{image_basename}'. 请稍候...")

    sleep_duration = random.uniform(1, 3)
    time.sleep(sleep_duration)

    result_category = ""
    determination_method = "" # To explain how the category was chosen

    # 1. Check if the provided path matches one of our predefined example paths
    if image_path_single in EXAMPLE_TO_RESULT_MAP:
        result_category = EXAMPLE_TO_RESULT_MAP[image_path_single]
        determination_method = "示例预设"
    else:
        # 2. If not an exact example path, try to parse the filename
        # This handles user uploads with the naming convention img_X_...
        match = re.search(r"img[_\-]?(\d+)", image_basename, re.IGNORECASE)
        if match:
            try:
                index = int(match.group(1))
                if 1 <= index <= len(RESULTS_CATEGORIES):
                    result_category = RESULTS_CATEGORIES[index - 1]
                    determination_method = "文件名解析"
                else:
                    print(f"文件名解析: 任务索引 {index} 超出范围 for {image_basename}")
            except ValueError:
                print(f"文件名解析: 解析任务索引失败 for {image_basename}")

        # 3. Fallback to random assignment if no category determined yet
        if not result_category:
            result_category = random.choice(RESULTS_CATEGORIES)
            determination_method = "随机分配"

    base_message = f"图片 '{image_basename}' 分析完成 ({determination_method}):\n任务类型: {result_category}"

    saved_path_info = ""
    if result_category: # Ensure result_category is not empty
        target_folder_path = os.path.join(OUTPUT_BASE_DIR, result_category)
        os.makedirs(target_folder_path, exist_ok=True)
        destination_path = os.path.join(target_folder_path, image_basename)
        try:
            shutil.copy(image_path_single, destination_path)
            saved_path_info = f"\n图片已保存至: {os.path.relpath(destination_path)}"
        except Exception as e:
            saved_path_info = f"\n保存图片失败: {e}"
            print(f"Error copying file in single_task_analysis: {e}") # Added for debugging
            print(f"Source: {image_path_single}, Destination: {destination_path}")

    return f"{base_message}\n(处理耗时: {sleep_duration:.2f}秒){saved_path_info}"


# --- 批量任务分析处理函数 ---
def batch_task_analysis(uploaded_file_data_list):
    if not uploaded_file_data_list:
        return "请上传至少一张图片进行批量分析。"

    results_output_list = []
    total_sleep_duration = 0
    processed_count = 0
    saved_count = 0

    print(f"批量任务分析接收到 {len(uploaded_file_data_list)} 个文件对象。")

    for i, file_data in enumerate(uploaded_file_data_list):
        try:
            # Gradio file uploads are TempFile objects with a .name attribute (path)
            # For Gradio versions that might return strings directly:
            if isinstance(file_data, str) and os.path.exists(file_data):
                temp_file_path = file_data
                original_filename = os.path.basename(file_data)
            elif hasattr(file_data, 'name') and os.path.exists(file_data.name):
                temp_file_path = file_data.name # This is the path to the temporary file
                original_filename = os.path.basename(file_data.name) # Get original name if needed, but for saving, temp path basename is fine
                # If you need the *original* uploaded filename (not the temp one):
                # This part is tricky as Gradio's gr.File might not always provide it directly
                # in older versions. For newer versions, `file_data.orig_name` might exist.
                # Let's assume for now the basename of the temp file is what we want to save.
                # If the user uploaded 'my_img_1.jpg', file_data.name might be '/tmp/gradio/.../my_img_1.jpg'
                # So os.path.basename(file_data.name) would be 'my_img_1.jpg'
            else:
                results_output_list.append(f"图片 '{str(file_data)}': 处理失败（文件无法访问或类型未知）")
                continue
        except Exception as e:
            results_output_list.append(f"图片 '{str(file_data)}': 处理失败（路径解析异常: {e}）")
            continue

        result_category = None
        determination_method = "文件名解析"
        match = re.search(r"img[_\-]?(\d+)", original_filename, re.IGNORECASE)
        if match:
            try:
                index = int(match.group(1))
                if 1 <= index <= len(RESULTS_CATEGORIES):
                    result_category = RESULTS_CATEGORIES[index - 1]
                else:
                    print(f"批量: 任务索引 {index} 超出范围 for {original_filename}")
                    determination_method = "随机分配 (索引超范围)"
            except ValueError:
                print(f"批量: 解析任务索引失败 for {original_filename}")
                determination_method = "随机分配 (解析失败)"

        if not result_category:
            result_category = random.choice(RESULTS_CATEGORIES)
            if determination_method == "文件名解析": # It means regex didn't match at all
                 determination_method = "随机分配 (无匹配)"


        sleep_duration = random.uniform(0.5, 1.5)
        time.sleep(sleep_duration)
        total_sleep_duration += sleep_duration

        target_folder = os.path.join(OUTPUT_BASE_DIR, result_category)
        os.makedirs(target_folder, exist_ok=True)
        save_path = os.path.join(target_folder, original_filename) # Use original_filename for saving

        try:
            shutil.copy(temp_file_path, save_path)
            saved_path_info = f"(已保存至: {os.path.relpath(save_path)})"
            saved_count += 1
        except Exception as e:
            saved_path_info = f"(保存失败: {e})"
            print(f"批量保存失败 for {original_filename}: {e}")

        results_output_list.append(f"图片 '{original_filename}': {result_category} ({determination_method}) {saved_path_info}")
        processed_count += 1

    summary = f"批量分析完成 (共 {processed_count} 张，成功保存 {saved_count} 张, 总耗时: {total_sleep_duration:.2f}秒):"
    return summary + "\n" + "\n".join(results_output_list)


# --- Gradio 界面 ---
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(f"""
这是一个基于MobileV4的自动任务检测系统。
处理后的图片将根据任务类型保存到 `{OUTPUT_BASE_DIR}/[任务类型]/` 文件夹下。
**单任务分析现在也会尝试根据文件名（如 img_1_xxx.jpg）进行分类。**
""")

    with gr.TabItem("单任务分析"):
        gr.Markdown("### 上传单张图片进行分析")
        with gr.Row():
            with gr.Column(scale=1):
                input_image_single = gr.Image(type="filepath", label="上传图片或选择下方示例", height=300, sources=["upload", "clipboard"])
            with gr.Column(scale=1):
                output_text_single = gr.Textbox(label="分析结果", lines=7, interactive=False)
        analyze_button_single = gr.Button("开始分析", variant="primary")
        gr.Examples(
            examples=[[img_path] for img_path in EXAMPLE_IMAGES],
            inputs=[input_image_single],
            outputs=[output_text_single], # This output is just for display, won't pre-fill the box before fn call
            fn=single_task_analysis, # The function will be called when an example is clicked
            cache_examples=False, # Re-run function for examples
            label="点击示例图片尝试 (新文件名格式)"
        )
        analyze_button_single.click(
            fn=single_task_analysis,
            inputs=[input_image_single],
            outputs=[output_text_single]
        )

    with gr.TabItem("批量任务分析"):
        gr.Markdown("### 上传多张图片进行批量分析")
        input_files_batch = gr.File(
            label="上传图片（可多选，文件名推荐符合 imgN 或 img_N_ 格式以便自动分类）",
            file_count="multiple",
            file_types=["image"]
        )
        output_text_batch = gr.Textbox(label="分析结果", lines=15, interactive=False)
        analyze_button_batch = gr.Button("开始批量分析", variant="primary")

        analyze_button_batch.click(
            fn=batch_task_analysis,
            inputs=[input_files_batch],
            outputs=[output_text_batch]
        )

if __name__ == "__main__":
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow library not fully installed. Text on generated images might be basic.")
        ImageFont = None # type: ignore

    setup_directories_and_images()

    print("单任务分析示例图片路径:")
    for p in EXAMPLE_IMAGES: print(f" - {p} (映射到: {EXAMPLE_TO_RESULT_MAP.get(p, '未映射')})")

    print(f"\n处理后的图片将保存到: {os.path.abspath(OUTPUT_BASE_DIR)} 目录下的对应子文件夹中。")
    print("\n对于批量处理和单任务上传，任务类型将尝试从原始文件名（如 img1_xxx.jpg -> 夜间）推断。")

    demo.launch(server_port=7866)