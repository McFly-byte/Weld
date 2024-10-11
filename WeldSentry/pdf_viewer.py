# FILE: WeldSentry/pdf_viewer
# USER: mcfly
# IDE: PyCharm
# CREATE TIME: 2024/10/11 15:23
# DESCRIPTION: 技术支持查看pdf文件

from PyQt5.QtWidgets import  QWidget, QLabel, QSlider, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import fitz  # PyMuPDF


class PDFViewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("技术支持")
        self.initUI()

    def initUI(self):
        self.doc = fitz.open('doc/大型金属焊接接口缺陷检测系统_用户使用手册_V1.0.pdf')
        self.page_count = len(self.doc)
        self.current_page = 0

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.page_count - 1)
        self.slider.valueChanged.connect(self.displayPage)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

        self.displayPage(0)

    def displayPage(self, page_number):
        self.current_page = page_number
        page = self.doc[page_number]
        pix = page.get_pixmap()
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(img)
        self.label.setPixmap(qpixmap)
        self.adjustSize()

    def resizeEvent(self, event):
        super(PDFViewWindow, self).resizeEvent(event)
        self.displayPage(self.current_page)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = Window()
#     window.show()
#     sys.exit(app.exec_())
