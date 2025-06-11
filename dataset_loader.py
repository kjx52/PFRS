import os
import cv2
import tarfile
from config import DATASET_DIR, CHARS74K_ZIP

class DatasetLoader:
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        os.makedirs(self.dataset_dir, exist_ok=True)

    def load_chars74k(self):  # sourcery skip: low-code-quality
        """加载Chars74K数据集"""
        #dataset_path = os.path.join(self.dataset_dir, 'Chars74K')
        dataset_path = self.dataset_dir
        if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
            self._unzip_chars74k()

        print("加载Chars74K数据集...")

        samples = []
        labels = []

        # 数字
        for i in range(10):
            num_dir = os.path.join(dataset_path, 'English', 'Fnt', f'Sample{str(i + 1).zfill(3)}')
            if not os.path.exists(num_dir):
                continue
            for img_file in os.listdir(num_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(num_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        samples.append(img)
                        labels.append(str(i))

        # 大写字母
        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            char_dir = os.path.join(dataset_path, 'English', 'Fnt', f'Sample{str(i + 11).zfill(3)}')
            if not os.path.exists(char_dir):
                continue
            for img_file in os.listdir(char_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(char_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        samples.append(img)
                        labels.append(char)
        
        # 小写字母
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
            char_dir = os.path.join(dataset_path, 'English', 'Fnt', f'Sample{str(i + 37).zfill(3)}')
            if not os.path.exists(char_dir):
                continue
            for img_file in os.listdir(char_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(char_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        samples.append(img)
                        labels.append(char)

        print("完成。")
        return samples, labels

    def _unzip_chars74k(self):
        """解压Chars74K数据集"""
        path = CHARS74K_ZIP

        # 解压文件
        print("正在解压Chars74K数据集...")
        with tarfile.open(path) as tar:
            tar.extractall(path=self.dataset_dir)
        print("完成")

    def load_icdar2013(self):
        """加载ICDAR 2013数据集用于测试"""
        # 实现类似的数据集加载逻辑
        pass