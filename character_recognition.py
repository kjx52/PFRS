from tqdm import tqdm
import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from config import MODEL_PATH, MODEL_DIR, IMAGE_EXTENSIONS, TEMP_DIR


class CharacterRecognizer:
    def __init__(self, template_dir=TEMP_DIR, use_svm=True):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.use_svm = use_svm
        if use_svm:
            self.svm_model = None
            self.scaler = None
            self.pca = None
        else:
            self.templates = self.load_templates(template_dir)

    def preprocess_char_img(self, char_img):
        """统一字符图片为白底黑字，居中填充到32x32"""
        # 灰度化
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY) if len(char_img.shape) == 3 else char_img
        # 自适应二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 反色，确保白底黑字
        if np.mean(binary) > 127:
            binary = 255 - binary
        # 找到字符外接矩形并裁剪
        coords = cv2.findNonZero(binary)
        if coords is None:
            # 全白或全黑，返回空白
            return np.ones((32, 32), dtype=np.uint8) * 255
        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            # 防止宽高为0
            return np.ones((32, 32), dtype=np.uint8) * 255
        char_crop = binary[y:y+h, x:x+w]
        # 居中填充到32x32
        size = 32
        result = np.ones((size, size), dtype=np.uint8) * 255
        scale = min((size-4)/w, (size-4)/h)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        char_resize = cv2.resize(char_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        x_offset = (size - nw) // 2
        y_offset = (size - nh) // 2
        result[y_offset:y_offset+nh, x_offset:x_offset+nw] = char_resize
        return result

    def load_templates(self, template_dir):
        templates = {}
        if not os.path.exists(template_dir):
            return templates
        for fname in os.listdir(template_dir):
            if fname.lower().endswith((IMAGE_EXTENSIONS)):
                label = os.path.splitext(fname)[0]
                img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[label] = img
        return templates

    def train_svm_classifier(self, samples, labels):
        """使用公开数据集训练SVM分类器"""
        print("开始训练SVM分类器...")

        # 提取HOG特征
        features = []
        for img in tqdm(samples):
            pre_img = self.preprocess_char_img(img)
            fd = hog(pre_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False)
            features.append(fd)

        x = np.array(features)
        y = np.array(labels)

        # 特征标准化
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)

        # PCA降维
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        x_pca = self.pca.fit_transform(x_scaled)

        # 训练SVM
        self.svm_model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
        self.svm_model.fit(x_pca, y)

        print("完成，已保存到", MODEL_PATH)

        # 保存模型
        joblib.dump({
            'model': self.svm_model,
            'scaler': self.scaler,
            'pca': self.pca
        }, MODEL_PATH)

    def load_svm_classifier(self):
        """加载预训练的SVM分类器"""
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            self.svm_model = data['model']
            self.scaler = data['scaler']
            self.pca = data['pca']
            print("SVM分类器加载成功。")
            return True
        return False

    def extract_hog_features(self, char_img):
        pre_img = self.preprocess_char_img(char_img)
        return hog(
            pre_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )

    def recognize_char_svm(self, char_img):
        """使用SVM识别字符"""
        if self.svm_model is None and not self.load_svm_classifier():
            return '?'  # 回退到模板匹配

        # 提取特征
        features = self.extract_hog_features(char_img)

        # 标准化
        features_scaled = self.scaler.transform([features])

        # PCA降维
        features_pca = self.pca.transform(features_scaled)

        # 预测
        prediction = self.svm_model.predict(features_pca)
        return prediction[0]

    def recognize_char(self, char_img):
        """识别单个字符"""
        return self.recognize_char_svm(char_img) if self.use_svm else '?'

    def recognize_chars(self, char_images):
        """识别字符列表"""
        results = []
        for img in char_images:
            try:
                ch = self.recognize_char(img)
            except Exception:
                ch = '?'
            results.append(ch)
        return ''.join(results)