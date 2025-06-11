import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import cv2
from PIL import Image, ImageTk
from PIL.Image import Resampling
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family='Microsoft YaHei')

class OCRSystemUI:
    def __init__(self, root, image_processor, char_recognizer):
        self.root = root
        self.root.title("印刷字体识别系统")
        self.root.geometry("1200x800")

        self.image_processor = image_processor
        self.char_recognizer = char_recognizer

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.image_list_label = ttk.Label(self.control_frame, text="图像列表:")
        self.image_list_label.pack(anchor=tk.W, padx=5, pady=5)

        self.image_listbox = tk.Listbox(self.control_frame, width=40, height=15)
        self.image_listbox.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.btn_frame = ttk.Frame(self.control_frame)
        self.btn_frame.pack(pady=10)

        self.load_btn = ttk.Button(self.btn_frame, text="加载图片", command=self.load_images)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(self.btn_frame, text="处理图像", command=lambda: self.process_image())
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.configure(state='disabled')

        # 显示面板
        self.display_frame = ttk.LabelFrame(self.main_frame, text="图像处理结果", width=800)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_control = ttk.Notebook(self.display_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.original_tab, text="原始图像")
        self.original_label = tk.Label(self.original_tab, bg='black')
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.deskew_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.deskew_tab, text="倾斜校正")
        self.deskew_label = tk.Label(self.deskew_tab, bg='black')
        self.deskew_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lines_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.lines_tab, text="行分割")
        self.lines_canvas = None  # 初始化延迟

        self.chars_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.chars_tab, text="字符分割")
        self.chars_label = tk.Label(self.chars_tab, bg='black')
        self.chars_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.result_tab, text="识别结果")
        self.result_text = tk.Text(self.result_tab, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.image_paths = []
        self.current_image = None
        self.deskewed_image = None
        self.lines_image = None
        self.chars_image = None
        self.recognized_text = ""
        self.horizontal_proj = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_image(self, image, label_widget):
        if image is None:
            label_widget.configure(image='')  # 清空显示
            label_widget.image = None
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 修复：窗口未渲染时宽高为1，设置默认值
        max_width = label_widget.winfo_width()
        max_height = label_widget.winfo_height()
        if max_width < 50 or max_height < 50:
            max_width, max_height = 600, 400
        pil_image.thumbnail((max_width - 20, max_height - 20), Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(pil_image)
        label_widget.configure(image=tk_image)
        label_widget.image = tk_image

    def process_image(self):
        selection = self.image_listbox.curselection()
        if not selection:
            self.status_var.set("请先选择一张图像")
            return

        index = selection[0]
        path = self.image_paths[index]

        try:
            self._extracted_from_process_image_11(path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set("处理失败")
            messagebox.showerror("处理错误", str(e))

    # TODO Rename this here and in `process_image`
    def _extracted_from_process_image_11(self, path):
        self.current_image = cv2.imread(path)
        if self.current_image is None:
            raise ValueError("无法读取图像")

        # 显示原始图像
        self.show_image(self.current_image, self.original_label)

        # 清理上一次结果
        self.deskewed_image = None
        self.lines_image = None
        self.chars_image = None
        self.recognized_text = ""
        self.horizontal_proj = None
        self.result_text.delete(1.0, tk.END)
        self.show_image(None, self.deskew_label)
        self.show_image(None, self.chars_label)
        if hasattr(self, 'lines_fig'):
            self.ax_lines_image.clear()
            self.ax_projection.clear()
            self.lines_canvas.draw()

        self.status_var.set("处理中...")
        self.root.update_idletasks()

        self.deskewed_image, angle = self.image_processor.deskew_image(self.current_image)
        self.show_image(self.deskewed_image, self.deskew_label)

        self.lines_image, line_imgs, self.horizontal_proj = self.image_processor.split_lines(self.deskewed_image)
        self.show_lines_tab()

        # split_chars 应该传入单行图片列表
        self.chars_image, char_imgs = (
            self.image_processor.split_chars(line_imgs)
            if isinstance(line_imgs, list) and len(line_imgs) > 0
            else (None, [])
        )
        self.show_image(self.chars_image, self.chars_label)

        if char_imgs:
            self.recognized_text = self.char_recognizer.recognize_chars(char_imgs)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, str(self.recognized_text))  # 强制转为字符串
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "未检测到字符")

        self.status_var.set(f"处理完成，倾斜角度为 {angle:.2f}°")


    def show_lines_tab(self):
        # 修复：首次调用时初始化
        if not hasattr(self, 'lines_fig') or self.lines_fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
            fig.tight_layout(pad=3.0)
            self.lines_fig = fig
            self.ax_lines_image = ax1
            self.ax_projection = ax2
            self.lines_canvas = FigureCanvasTkAgg(fig, master=self.lines_tab)
            self.lines_canvas.draw()
            self.lines_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_lines_image.clear()
        self.ax_projection.clear()

        if self.lines_image is not None:
            img_rgb = cv2.cvtColor(self.lines_image, cv2.COLOR_BGR2RGB)
            self.ax_lines_image.imshow(img_rgb)
            self.ax_lines_image.axis('off')

        if self.horizontal_proj is not None:
            self.ax_projection.plot(self.horizontal_proj, color='blue')
            self.ax_projection.set_title('水平投影')
            self.ax_projection.set_xlabel('行索引')
            self.ax_projection.set_ylabel('投影值')
            self.ax_projection.grid(True)

        self.lines_canvas.draw()

    def on_image_select(self, event):
        if selection := self.image_listbox.curselection():
            self.process_btn.configure(state='normal')
            # 动态显示原始图像
            index = selection[0]
            path = self.image_paths[index]
            image = cv2.imread(path)
            if image is not None:
                self.current_image = image
                self.show_image(self.current_image, self.original_label)
        else:
            self.process_btn.configure(state='disabled')
            self.show_image(None, self.original_label)

    def load_images(self):
        # 打开文件对话框选择图片
        filetypes = [("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        paths = filedialog.askopenfilenames(title="选择图片", filetypes=filetypes)
        if not paths:
            return
        self.image_paths = list(paths)
        self.image_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.image_listbox.insert(tk.END, os.path.basename(path))
        self.process_btn.configure(state='disabled')

    def on_close(self):
        self.root.quit()
        self.root.destroy()

        sys.exit(0)
