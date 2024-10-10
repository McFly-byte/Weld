import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import sqlite3

DATABASE_FILENAME = 'sentry.db'

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initDB()

    def initUI(self):
        self.setWindowTitle('PyQt5 with SQLite')
        self.setGeometry(100, 100, 600, 400)

    def initDB(self):
        # 创建或连接到数据库
        self.conn = sqlite3.connect('example.db')
        self.cursor = self.conn.cursor()

        # 创建一个表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER
            )
        ''')
        self.conn.commit()

    def insertUser(self, name, age):
        # 插入数据
        self.cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)', (name, age))
        self.conn.commit()
        QMessageBox.information(self, 'Success', 'User added successfully.')

    def closeEvent(self, event):
        # 关闭数据库连接
        self