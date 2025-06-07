import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QPushButton,
    QComboBox, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理工具")
        self.setGeometry(100, 100, 1200, 600)

        self.image = None
        self.processed_image = None

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QGridLayout(main_widget)

        self.inputImageLabel = QGraphicsView()
        self.outputImageLabel = QGraphicsView()
        self.inputHistogram = QGraphicsView()
        self.outputHistogram = QGraphicsView()

        layout.addWidget(QLabel("原始图像"), 0, 0)
        layout.addWidget(self.inputImageLabel, 1, 0)
        layout.addWidget(QLabel("原始直方图"), 2, 0)
        layout.addWidget(self.inputHistogram, 3, 0)

        layout.addWidget(QLabel("处理后图像"), 0, 1)
        layout.addWidget(self.outputImageLabel, 1, 1)
        layout.addWidget(QLabel("处理后直方图"), 2, 1)
        layout.addWidget(self.outputHistogram, 3, 1)

        control_layout = QVBoxLayout()

        self.buttons = {
            "加载图像": self.load_image,
            "保存图像": self.save_image,
            "灰度化": self.to_gray,
            "二值化": self.to_binary,
            "直方图均衡化": self.histogram_equalization,
            "图像锐化": self.sharpen_image,
            "添加噪声": self.add_noise,
            "人脸检测": self.face_detect,
            "腐蚀": self.erosion_operation,
            "膨胀": self.dilation_operation,
            "开操作": self.open_operation,
            "闭操作": self.close_operation,
        }

        for text, func in self.buttons.items():
            btn = QPushButton(text)
            btn.clicked.connect(func)
            control_layout.addWidget(btn)

        self.edgeComboBox = QComboBox()
        self.edgeComboBox.addItems(["Canny", "Sobel", "Prewitt", "Roberts"])
        self.edgeComboBox.currentIndexChanged.connect(self.edge_detection)
        control_layout.addWidget(QLabel("边缘检测方法"))
        control_layout.addWidget(self.edgeComboBox)

        self.segmentComboBox = QComboBox()
        self.segmentComboBox.addItems(["Otsu", "K-means", "GrabCut"])
        self.segmentComboBox.currentIndexChanged.connect(self.segment_image)
        control_layout.addWidget(QLabel("图像分割方法"))
        control_layout.addWidget(self.segmentComboBox)

        layout.addLayout(control_layout, 1, 2, 3, 1)

    def display_image(self, image, view):
        if image.ndim == 2:
            qformat = QImage.Format_Grayscale8
        elif image.shape[2] == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
        out_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        pixmap = QPixmap.fromImage(out_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        view.setScene(scene)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def display_histogram(self, image, view):
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        plt.figure(figsize=(4, 3))
        plt.plot(histogram, color='blue')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("hist_temp.png")
        plt.close()

        hist_img = cv2.imread("hist_temp.png")
        self.display_image(hist_img, view)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "Images (*.png *.jpg *.bmp *.tif *.jpeg *.tiff);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("图像解码失败")

                self.image = img
                self.processed_image = img.copy()
                self.display_image(self.image, self.inputImageLabel)
                self.display_histogram(self.image, self.inputHistogram)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败！\n原因：{str(e)}")
                self.image = None
                self.processed_image = None

    def save_image(self):
        if self.processed_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.jpg *.png)")
            if path:
                cv2.imwrite(path, self.processed_image)
                QMessageBox.information(self, "保存成功", "图像已保存！")

    def to_gray(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        self.processed_image = gray
        self.display_image(gray, self.outputImageLabel)
        self.display_histogram(gray, self.outputHistogram)

    def to_binary(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        self.processed_image = binary
        self.display_image(binary, self.outputImageLabel)
        self.display_histogram(binary, self.outputHistogram)

    def histogram_equalization(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        self.processed_image = eq
        self.display_image(eq, self.outputImageLabel)
        self.display_histogram(eq, self.outputHistogram)

    def sharpen_image(self):
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(self.image.copy(), -1, kernel)
        self.processed_image = sharpened
        self.display_image(sharpened, self.outputImageLabel)
        self.display_histogram(sharpened, self.outputHistogram)

    def add_noise(self):
        row, col, ch = self.image.shape
        gauss = np.random.normal(0, 25, (row, col, ch)).astype(np.uint8)
        noisy = cv2.add(self.image.copy(), gauss)
        self.processed_image = noisy
        self.display_image(noisy, self.outputImageLabel)
        self.display_histogram(noisy, self.outputHistogram)

    def face_detect(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        img = self.image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        self.processed_image = img
        self.display_image(img, self.outputImageLabel)
        self.display_histogram(img, self.outputHistogram)

    def erosion_operation(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.erode(self.image.copy(), kernel, iterations=1)
        self.processed_image = result
        self.display_image(result, self.outputImageLabel)
        self.display_histogram(result, self.outputHistogram)

    def dilation_operation(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(self.image.copy(), kernel, iterations=1)
        self.processed_image = result
        self.display_image(result, self.outputImageLabel)
        self.display_histogram(result, self.outputHistogram)

    def open_operation(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(self.image.copy(), cv2.MORPH_OPEN, kernel)
        self.processed_image = result
        self.display_image(result, self.outputImageLabel)
        self.display_histogram(result, self.outputHistogram)

    def close_operation(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(self.image.copy(), cv2.MORPH_CLOSE, kernel)
        self.processed_image = result
        self.display_image(result, self.outputImageLabel)
        self.display_histogram(result, self.outputHistogram)

    def edge_detection(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        method = self.edgeComboBox.currentIndex()

        if method == 0:
            edges = cv2.Canny(gray, 100, 200)
        elif method == 1:
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        elif method == 2:
            kernelx = np.array([[1, 0, -1]]*3)
            kernely = np.array([[1]*3, [0]*3, [-1]*3])
            edges = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)
        elif method == 3:
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            edges = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)

        self.processed_image = edges
        self.display_image(edges, self.outputImageLabel)
        self.display_histogram(edges, self.outputHistogram)

    def segment_image(self):
        method = self.segmentComboBox.currentIndex()
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

        if method == 0:
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 1:
            Z = self.image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            result = centers[labels.flatten()].reshape(self.image.shape)
        elif method == 2:
            mask = np.zeros(self.image.shape[:2], np.uint8)
            bgModel = np.zeros((1, 65), np.float64)
            fgModel = np.zeros((1, 65), np.float64)
            rect = (50, 50, self.image.shape[1]-100, self.image.shape[0]-100)
            cv2.grabCut(self.image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            result = self.image * mask2[:, :, np.newaxis]

        self.processed_image = result
        self.display_image(result, self.outputImageLabel)
        self.display_histogram(result, self.outputHistogram)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())
