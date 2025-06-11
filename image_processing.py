import cv2
import numpy as np

class ImageProcessor:
    def preprocess_image(self, image):
        """图像预处理：灰度化、二值化、去噪"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def deskew_image(self, image):
        """使用投影法进行倾斜校正"""
        processed = self.preprocess_image(image)
        best_angle = 0
        best_score = -1
        angles = np.arange(-15, 15, 0.5)
        for angle in angles:
            rotated = self.rotate_image(processed, angle)
            horizontal_proj = np.sum(rotated, axis=1)
            score = np.var(horizontal_proj)
            if score > best_score:
                best_score = score
                best_angle = angle
        deskewed = self.rotate_image(image, best_angle)
        return deskewed, best_angle

    def rotate_image(self, image, angle):
        """旋转图像"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def split_lines(self, image):
        """
        行分割 - 自适应扩展+连通域辅助+智能padding

        水平投影和垂直投影拥有类似的缺陷，它会忽略字符的一些部分，
        如“j”和“g”的下半部分（“尾巴”）。在行分割中可能导致多重分
        割，在字符分割中则表现为遗漏或分割粘连。

        也许应该多放宽一些界限，让padding也可以向上扩展。
        """
        processed = self.preprocess_image(image)
        horizontal_proj = np.sum(processed, axis=1)
        threshold = np.max(horizontal_proj) * 0.1
        line_ranges = []
        start = -1
        for i, val in enumerate(horizontal_proj):
            if val > threshold and start == -1:
                start = i
            elif val <= threshold and start != -1:
                end = i
                if end - start > 5:
                    line_ranges.append((start, end))
                start = -1
        if start != -1:
            line_ranges.append((start, len(horizontal_proj)-1))
        # 合并有“尾巴”的行
        merged_ranges = []
        i = 0
        while i < len(line_ranges):
            s, e = line_ranges[i]
            line_height = e - s
            # padding为行高的20%，最小2像素
            padding = max(2, int(line_height * 0.2))
            e_ext = min(processed.shape[0], e + padding)
            # 检查扩展区是否有黑色像素
            ext_region = processed[e:e_ext, :]
            has_tail = np.sum(ext_region > 0) > 0
            # 如果扩展区有内容且与下一行重叠，合并
            if has_tail and i + 1 < len(line_ranges):
                next_s, next_e = line_ranges[i+1]
                if e_ext > next_s:
                    merged_ranges.append((s, next_e))
                    i += 2
                    continue
            # 如果扩展区无内容，回滚到未扩展
            merged_ranges.append((s, e_ext if has_tail else e))
            i += 1

        line_image = image.copy()
        line_images = []
        for s, e in merged_ranges:
            cv2.rectangle(line_image, (0, s), (image.shape[1], e), (0, 255, 0), 2)
            line_img = image[s:e, :]
            line_images.append(line_img)
        return line_image, line_images, horizontal_proj

    def split_chars(self, line_images):    # sourcery skip: low-code-quality
        """
        字符分割 - 垂直投影+连通域+极小值法细分粘连字符

        当字符较粗或间距较小时，简单的垂直投影法会把多个字符分为
        一块，导致分割失败。
        """
        chars_image = None
        all_char_images = []
        if not line_images:
            return chars_image, all_char_images
        for i, line_img in enumerate(line_images):
            processed = self.preprocess_image(line_img)
            vertical_proj = np.sum(processed, axis=0)
            threshold = np.max(vertical_proj) * 0.1
            char_ranges = []
            start = -1
            for j, val in enumerate(vertical_proj):
                if val > threshold and start == -1:
                    start = j
                elif val <= threshold and start != -1:
                    end = j
                    if end - start > 3:
                        char_ranges.append((start, end))
                    start = -1
            if start != -1:
                char_ranges.append((start, len(vertical_proj)-1))
            optimized_ranges = self.optimize_char_split(processed, char_ranges)
            line_img_copy = line_img.copy()
            char_images = []
            for start, end in optimized_ranges:
                # 粘连字符细分
                w = end - start
                char_img = line_img[:, start:end]
                h = char_img.shape[0]
                # 如果宽度过大，尝试在该区域内寻找极小值分割
                if w > h * 1.2:
                    # 在该区域内做二次垂直投影
                    sub_processed = self.preprocess_image(char_img)
                    sub_proj = np.sum(sub_processed, axis=0)
                    min_indices = [
                        k
                        for k in range(1, len(sub_proj) - 1)
                        if sub_proj[k] < sub_proj[k - 1]
                        and sub_proj[k] < sub_proj[k + 1]
                        and sub_proj[k] < threshold
                    ]
                    # 根据极小值点分割
                    last = 0
                    for idx in min_indices + [w]:
                        if idx - last > 3:
                            sub_char = char_img[:, last:idx]
                            if sub_char.shape[1] > 3:
                                char_images.append(sub_char)
                        last = idx
                elif 3 < w < h * 3 and h > 5:
                    char_images.append(char_img)
                cv2.rectangle(line_img_copy, (start, 0), (end, line_img.shape[0]), (0, 0, 255), 1)
            all_char_images.extend(char_images)
            if i == 0:
                chars_image = line_img_copy
            else:
                chars_image = np.vstack((chars_image, line_img_copy))
        return chars_image, all_char_images

    def optimize_char_split(self, binary_line, char_ranges):
        """使用连通域分析优化字符分割"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_line, connectivity=8)
        bboxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w > 3 and h > 5:
                bboxes.append((x, x + w))
        if not bboxes:
            return char_ranges
        all_boundaries = sorted(char_ranges + bboxes, key=lambda k: k[0])
        optimized = []
        current_start, current_end = all_boundaries[0]
        for start, end in all_boundaries[1:]:
            if start <= current_end + 5:
                current_end = max(current_end, end)
            else:
                optimized.append((current_start, current_end))
                current_start, current_end = start, end
        optimized.append((current_start, current_end))
        return optimized