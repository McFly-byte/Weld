# GUIdemo1.py
# Demo1 of GUI by PqYt5
# Copyright 2021 Youcans, XUPT
# Crated：2021-10-06

from PyQt5 import QtWidgets
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 初始化界面
    MainWindow = QtWidgets.QWidget()  # 生成一个主窗口
    MainWindow.show()  # 显示主窗口
    sys.exit(app.exec_())  # 在主线程中退出
