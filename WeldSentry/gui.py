# FILE: WeldSentry/gui.py
# USER: mcfly
# IDE: PyCharm
# CREATE TIME: 2024/10/8 20:27
# DESCRIPTION:  在weldsentry_window.ui的基础上实现信号/槽

from PyQt5.QtWidgets import QMainWindow, QFileDialog

from weldsentry_window import Ui_MainWindow


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # 设置UI界面
        self.setupUi(self)

        # 1. 连接信号和槽
        # 1.1 按钮
        #done   btn_photo
        #todo   btn_confirm
        #done   btn_cancel
        #todo   btn_start
        self.btn_photo.clicked.connect(self.open_folder_dialog)
        self.btn_cancel.clicked.connect(self.clear_line_edits)

    # 实现槽函数
    def open_folder_dialog(self):
        # 打开文件夹对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:  # 如果用户选择了文件夹
            self.le_photo.setText(folder_path)  # 更新QLineEdit组件le_photo中的文本

    def clear_line_edits(self):
        self.le1.clear()
        self.le2.clear()
        self.le3.clear()
        self.le4.clear()
