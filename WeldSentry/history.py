# FILE: WeldSentry/history
# USER: mcfly
# IDE: PyCharm
# CREATE TIME: 2024/10/11 14:44
# DESCRIPTION:

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTableView, QVBoxLayout, QWidget
from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel

class HistoryBatchWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('历史记录')
        self.setGeometry(100, 100, 600, 400)

        # 创建表格视图
        self.table_view = QTableView()

        # 连接到数据库并设置模型
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("sentry.db")
        if not self.db.open():
            print("无法打开数据库")
            return

        self.model = QSqlQueryModel()
        self.model.setQuery("SELECT * FROM tb_bc")
        self.table_view.setModel(self.model)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)

class HistoryDetailWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('不合格品信息')
        self.setGeometry(100, 100, 600, 400)

        # 创建表格视图
        self.table_view = QTableView()

        # 连接到数据库并设置模型
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("sentry.db")
        if not self.db.open():
            print("无法打开数据库")
            return

        self.model = QSqlQueryModel()
        self.model.setQuery("SELECT * FROM tb_detail")
        self.table_view.setModel(self.model)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)