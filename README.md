车牌与车辆识别系统

项目概述：
本项目是一个基于深度学习的车牌与车辆识别系统，能够实现以下功能：
1. 车牌检测与识别
2. 车牌颜色识别
3. 车辆颜色识别
4. 危险品车牌检测
5. 支持单层和双层车牌识别
6. 支持图片和视频输入

主要功能模块：
1. 车牌检测：使用YOLOv8模型进行车牌定位
2. 车牌识别：识别车牌号码和颜色
3. 车辆识别：识别车辆颜色
4. 危险品检测：识别危险品车牌
5. 透视变换：对倾斜车牌进行校正
6. 双层车牌处理：对双层车牌进行分割和合并

环境要求：
- Python 3.7+
- PyTorch 1.7+
- OpenCV 4.0+
- CUDA（可选，推荐使用GPU加速）

使用方法：
1. 安装依赖：
   pip install -r requirements.txt

2. 运行程序：
   可视化界面: python app.py
   其它:
   python Car_recognition.py --image_path 输入图片路径 --output 输出路径 或  python Car_recognition.py --video 输入视频路径

参数说明：
--detect_model：车牌检测模型路径（默认：weights/detect.pt）
--rec_model：车牌识别模型路径（默认：weights/plate_rec_color.pth）
--car_rec_model：车辆识别模型路径（默认：weights/car_rec_color.pth）
--image_path：输入图片路径或目录
--video：输入视频路径
--img_size：推理尺寸（默认：384）
--output：输出结果保存路径

性能指标：
- 车牌检测准确率：>95%
- 车牌识别准确率：>90%
- 车辆颜色识别准确率：>85%
- 推理速度：GPU环境下约30ms/帧

注意事项：
1. 确保输入图片或视频路径正确
2. 建议使用GPU加速以提高处理速度
3. 输出结果将保存在指定目录中

项目结构：
Car_recognition.py：主程序
weights/：模型权重文件
utils/：工具函数
plate_recognition/：车牌识别相关模块
car_recognition/：车辆识别相关模块
