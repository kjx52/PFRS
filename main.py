# 字体识别系统
# 2025/6/12
# 项目已发布于 https://github.com/kjx52/PFRS
# 项目循序 GNU_GPL_v3 许可

import tkinter as tk
from ui import OCRSystemUI
from character_recognition import CharacterRecognizer
from dataset_loader import DatasetLoader
from image_processing import ImageProcessor

def main():
    # 创建数据集加载器
    dataset_loader = DatasetLoader()
    
    # 创建字符识别器（使用SVM）
    char_recognizer = CharacterRecognizer(use_svm=True)

    # 检查是否需要训练SVM
    if not char_recognizer.load_svm_classifier():
        print("未找到预训练模型，新建...")
        try:
            # 加载数据集
            samples, labels = dataset_loader.load_chars74k()
            # 训练SVM
            char_recognizer.train_svm_classifier(samples, labels)
        except Exception as e:
            print("\033[31m训练SVM失败：", e, "\033[0m")
            return -1
        print("完成。")
    
    # 创建图像处理器
    image_processor = ImageProcessor()
    print("字符识别器已准备就绪。")

    # 创建UI
    root = tk.Tk()
    OCRSystemUI(root, image_processor, char_recognizer)
    print("启动UI。")

    root.mainloop()
    print("退出。")

if __name__ == "__main__":
    main()
