import cv2
import os
from image_processing import deskew_image, split_lines, split_chars
from character_recognition import CharacterRecognizer
from dataset_loader import DatasetLoader


def test_system():
    # 加载测试数据集
    dataset_loader = DatasetLoader()
    test_samples, test_labels = dataset_loader.load_icdar2013()  # 需要实现

    # 创建字符识别器
    char_recognizer = CharacterRecognizer(use_svm=True)
    char_recognizer.load_svm_classifier()

    # 评估指标
    total_chars = 0
    correct_chars = 0

    # 测试每个样本
    for image_path, ground_truth in test_samples:
        # 读取图像
        image = cv2.imread(image_path)

        # 倾斜校正
        deskewed, _ = deskew_image(image)

        # 行分割
        _, line_images, _ = split_lines(deskewed)

        # 字符分割
        _, all_char_images = split_chars(line_images)

        # 字符识别
        recognized_text = char_recognizer.recognize_chars(all_char_images)

        # 计算准确率
        for i, (pred_char, true_char) in enumerate(zip(recognized_text, ground_truth)):
            if i >= len(ground_truth):
                break
            total_chars += 1
            if pred_char == true_char:
                correct_chars += 1

        # 打印结果
        print(f"图像: {os.path.basename(image_path)}")
        print(f"识别结果: {recognized_text}")
        print(f"真实文本: {ground_truth}")
        print("-" * 50)

    # 计算总体准确率
    accuracy = correct_chars / total_chars * 100
    print(f"总体字符准确率: {accuracy:.2f}%")
    print(f"正确字符数: {correct_chars}/{total_chars}")


if __name__ == "__main__":
    test_system()