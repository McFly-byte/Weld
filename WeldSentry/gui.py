# FILE: WeldSentry/gui.py
# USER: mcfly
# IDE: PyCharm
# CREATE TIME: 2024/10/8 20:27
# DESCRIPTION:  在weldsentry_window.ui的基础上实现信号/槽
import os
import sqlite3
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem, QAbstractItemView, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import QtSql
from PyQt5.QtSql import QSqlQuery


import threading

from weldsentry_window import Ui_MainWindow
from detect import detect
from history import HistoryBatchWindow, HistoryDetailWindow
from pdf_viewer import PDFViewWindow


# 多线程解决耗时程序卡gui主线程
class DetectionThread(QThread):
    finished_signal = pyqtSignal()
    update_table_signal = pyqtSignal(object)  # 通知更新table
    # pause_signal = pyqtSignal()  # 信号用于通知线程暂停
    # resume_signal = pyqtSignal()  # 信号用于通知线程恢复
    def __init__(self, weight, source, project):
        super().__init__()
        self.weight = weight
        self.source = source
        self.project = project
        self.paused = False
        self.pause_event = threading.Event()

    def run(self):
        # 调用 detect 函数，并传递回调函数
        def callback(result):
            self.pause_event.wait()  # 如果线程被暂停，将在这里阻塞
            print( "callback")
            self.update_table_signal.emit(result)  # 发送信号到主界面
        self.pause_event.set()
        detect(self.weight, self.source, self.project, callback=callback)
        self.finished_signal.emit()  # 操作完成后发送信号

    def pause(self):
        self.paused = True
        self.pause_event.clear()  # 清除事件，使线程暂停

    def resume(self):
        self.paused = False
        self.pause_event.set()  # 设置事件，使线程恢复

# 名为sentry.db的sqlite3数据库
DATABASE_FILENAME = "sentry.db"


class Window(QMainWindow, Ui_MainWindow):
    weight = ""
    source = ""
    project = ""
    exp_path = "" # 每次运行后生成的exp文件夹路径（方便处理其下图片）

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # 设置UI界面
        self.setupUi(self)

        # 补充一些designer中做不到的设置
        self.tb_bc.setColumnWidth(0, 40)
        self.tb_detail.setColumnWidth(0, 40)
        self.tb_bc.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lb_show.setPixmap(QPixmap())  # 初始为空

        # detect线程
        self.detection_thread = DetectionThread(self.weight, self.source, self.project)

        # 数据库
        self.conn = sqlite3.connect(DATABASE_FILENAME)
        self.cursor = self.conn.cursor()
        self.create_tables_if_not_exist()

        # 连接信号和槽
        #done   btn_photo
        #done   btn_confirm
        #done   btn_cancel
        #done   btn_start
        #done   btn_pause
        self.btn_photo.clicked.connect(self.open_folder_dialog1)
        self.btn_photo2.clicked.connect(self.open_folder_dialog2)
        self.btn_confirm.clicked.connect(self.save_texts)
        self.btn_cancel.clicked.connect(self.clear_line_edits)
        self.btn_start.clicked.connect(self.run_model)
        self.btn_pause.clicked.connect(self.pause_or_stop)
        # self.stackedWidget_2.currentChanged.connect(self.run_detect)
        self.tb_detail.cellClicked.connect(self.on_cell_clicked)
        # self.btn_pause.clicked.connect(self.toggle_thread)
        self.btn_close.clicked.connect( self.on_btn_close_clicked)
        # 历史记录
        self.btn_his1.clicked.connect(self.open_his_batch_window)
        self.btn_his_defect.clicked.connect(self.open_his_detail_window)
        # 技术支持
        self.btn_ast1.clicked.connect(self.open_pdf_viewer_window)
        # 联系方式
        self.btn_ast2.clicked.connect(self.showContactInfo)

##### 1. 实现槽函数
    # 选择source打开路径
    def open_folder_dialog1(self):
        # 打开文件夹对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:  # 如果用户选择了文件夹
            self.le4.setText(folder_path)  # 更新QLineEdit组件le_photo中的文本

    # 选择project存储路径
    def open_folder_dialog2(self):
        # 打开文件夹对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:  # 如果用户选择了文件夹
            self.le5.setText(folder_path)  # 更新QLineEdit组件le_photo中的文本

    # 点击取消清空
    def clear_line_edits(self):
        self.le1.clear()
        self.le2.clear()
        self.le3.clear()
        self.le4.clear()
        self.le5.clear()
        self.le_para1.clear()
        self.le_para2.clear()
        self.textBrowser.clear()

    # 输入显示到左边
    def save_texts(self):
        #TODO 如果所有lineedit都没内容，点击确认时不更新数据，弹出提示框

        #todo 配置textbrowser展示哪些其他信息
        labels = [self.lb1, self.lb2, self.lb3,
                  self.lb_para1, self.lb_para2]
        line_edits = [self.le1, self.le2, self.le3,
                      self.le_para1, self.le_para2]

        # 清空textBrowser的内容
        self.textBrowser.clear()

        # 遍历当前页面的QLabel和QLineEdit对象
        for i in range(len(labels)):
            label_text = labels[i].text()
            line_edit_text = line_edits[i].text()

            # 将QLabel和QLineEdit的文本添加到textBrowser
            self.textBrowser.append(f"{label_text}: {line_edit_text}")

    # 运行
    def run_model(self):
        self.remove_rows_except_first(0)
        self.remove_rows_except_first(1)

        #TODO 如果没有输入则跑default数据，弹窗提示
        #TODO 加载动画
        #done stack切换
        self.stackedWidget.setCurrentIndex(1)
        #done 按钮切换
        self.btn_start.setEnabled(False)
        self.stackedWidget_2.setCurrentIndex(1)
        #TODO weight的设定
        self.weight = "YOLOv7/weight/best.pt"
        self.source = self.le4.text() if self.le4.text() else "YOLOv7/images"
        self.project = self.le5.text() if self.le5.text() else "runs/detect"
        #done 启动线程
        self.detection_thread = DetectionThread(self.weight, self.source, self.project)
        self.detection_thread.update_table_signal.connect(self.update_table_slot)
        self.detection_thread.finished_signal.connect(self.on_detection_finished)
        self.detection_thread.start()

    #  将单个结果添加到QTableWidget中，同时向tb_bc和tb_detail添加数据
    def update_table_slot(self, result):
        print( "update_table_slot")
        bc, dt = [], []
        nbc = self.tb_bc.rowCount()
        nr = self.tb_detail.rowCount()
        # 给bc添加
        if nbc > 0 :
            item = self.tb_bc.item( nbc-1, 0 )
            bc.append( int(item.text())+1 )
        else :
            bc.append(1)
        bc.append(self.le3.text())
        bc.append(self.le1.text())
        bc.append(self.le2.text())
        bc.append(result[0])
        bc.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        is_qualified = len(result) <= 2
        print( result[0], is_qualified )
        bc.append( "合格" if is_qualified else "不合格")
        self.addRow_bc(bc)

        # 不合格就给detail添加
        if not is_qualified :
            # 打印到主屏幕
            # self.terminal.append(result)
            if nr > 0 :
                item = self.tb_detail.item( nr-1, 0 )
                dt.append( int(item.text())+1 )
            else :
                dt.append(1)
            dt.append( result[0] )
            tmp = ""
            for i in range(1, len(result)-1 ):
                tmp += result[i]
            # print( tmp )
            # print( self.string_combination(tmp) )
            dt.append(self.string_combination(tmp))
            dt.append(tmp)
            self.addRow_detail(dt)


    # 当检测完成时，恢复状态，展示图片
    def on_detection_finished(self):
        # 恢复左侧界面
        self.btn_start.setEnabled(True)
        self.stackedWidget_2.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(0)
        #todo 下方textbrowser打印信息 需要合格信息
        self.terminal.clear()
        self.terminal.append(f"Done.")
        # 获取结果图片路径方便后续操作
        path = self.project
        subfolders = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        self.exp_path = max(subfolders, key=lambda x: os.path.getmtime(x))
        # 用不着了，可以边detect边添加到表格 # tb_bc提供所有图片 需要合格信息
        # self.addTable_bc()
        #figure 更新数据库中内容
        self.save_bc_to_db()
        self.save_detail_to_db()
        print( "on_detection_finished" )


    #TODO 暂停/停止
    def pause_or_stop(self):
        self.toggle_thread()

    #figure  点击表格（内容为图片名）显示图片
    def on_cell_clicked(self, row, column):
        # print( row, column )
        if self.stackedWidget.currentIndex() != 1:
            self.stackedWidget.setCurrentIndex(1)
        if column == 1:
            # 获取单元格内容（图片路径）
            item = self.tb_detail.item(row, column)
            # print( item.text() )
            image_path = os.path.join( self.exp_path, item.text() ) if item else ""
            # print( image_path )
            # 加载图片并显示
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.lb_show.setPixmap(pixmap.scaled(self.lb_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_btn_close_clicked(self):
        self.stackedWidget.setCurrentIndex(0)

    # 打开新窗口查看历史记录
    def open_his_batch_window(self):
        self.his_batch_window = HistoryBatchWindow()
        self.his_batch_window.show()
    def open_his_detail_window(self):
        self.his_detail_window = HistoryDetailWindow()
        self.his_detail_window.show()
    # 技术支持之pdf查看
    def open_pdf_viewer_window(self):
        self.pdf_viewer_window = PDFViewWindow()
        self.pdf_viewer_window.show()
    # 技术支持之弹窗联系方式
    def showContactInfo(self):
        # 弹出消息框
        QMessageBox.information(self, '联系方式', 'tel: 18168035472')

    ##### 2. 工具性函数
    # 向tb_bc中添加一行数据
    def addRow_bc(self, data):
        # 获取当前行数
        current_row_count = self.tb_bc.rowCount()
        # 在表格的末尾插入一行
        self.tb_bc.insertRow(current_row_count)
        # 遍历数据，并将数据设置到新行的单元格中
        for col_index, item in enumerate(data):
            # 创建TableWidgetItem对象，并设置数据
            table_item = QTableWidgetItem(str(item))
            # 将TableWidgetItem添加到QTableWidget中
            self.tb_bc.setItem(current_row_count, col_index, table_item)

    # 想tb_detail添加一行数据
    def addRow_detail(self, data):
        # 获取当前行数
        current_row_count = self.tb_detail.rowCount()
        # 在表格的末尾插入一行
        self.tb_detail.insertRow(current_row_count)
        # 遍历数据，并将数据设置到新行的单元格中
        for col_index, item in enumerate(data):
            # 创建TableWidgetItem对象，并设置数据
            table_item = QTableWidgetItem(str(item))
            # 将TableWidgetItem添加到QTableWidget中
            self.tb_detail.setItem(current_row_count, col_index, table_item)

    # 向tb_bc里填满数据
    #todo 是否合格
    def addTable_bc(self):
        filepath = self.exp_path
        type = self.le3.text()
        bacth_id = self.le1.text()
        man = self.le2.text()
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = []
        # print( data )
        # print( filepath)
        # 遍历文件夹中的文件
        for index, filename in enumerate(os.listdir(filepath)):
            # 构建完整的文件路径
            full_path = os.path.join(filepath, filename)
            # 确保是文件而不是子文件夹
            if os.path.isfile(full_path):
                data.append((index, type, bacth_id, man, current_time_str, filename))
        # print( data)
        for line in data:
            print( "line:{}".format(line))
            self.addRow_bc(line)

    # "3 patches, 1 crosion" -> "patches/crosion"
    def string_combination(self, s) :
        # 使用列表推导式移除字符串中的数字和空格
        cleaned_str = "".join([char for char in s if not char.isdigit() and char != ' ' and char != ','])
        # print( cleaned_str)
        # 将处理后的字符串按单词分割
        words = cleaned_str.split()
        # print( "words:{}".format(words) )
        # 使用反斜杠将单词连接起来，最后一个单词后不加反斜杠
        result = '\\'.join(words)
        return result

    # 暂停
    # todo 加入弹窗选择停止or暂停
    def toggle_thread(self):
        if self.detection_thread.paused:
            self.detection_thread.resume()
            self.btn_pause.setText('暂停/停止')
        else:
            self.detection_thread.pause()
            self.btn_pause.setText('继续')

    # 清空表格
    def remove_rows_except_first(self, flag):
        if flag == 0 :
            row_count = self.tb_bc.rowCount()
            # 从最后一行开始删除，直到第二行
            for row in range(row_count - 1, 0, -1):
                self.tb_bc.removeRow(row)
        else :
            row_count = self.tb_detail.rowCount()
            # 从最后一行开始删除，直到第二行
            for row in range(row_count - 1, 0, -1):
                self.tb_detail.removeRow(row)


##### 数据库操作
    # 如不存在，创建新表
    def create_tables_if_not_exist(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tb_bc (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                batch_id TEXT,
                chief TEXT,
                photo TEXT,
                detect_time TEXT,
                is_qualified TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tb_detail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo TEXT,
                defect TEXT,
                detail TEXT
            )
        ''')
        self.conn.commit()

    def save_bc_to_db(self):
        print("开始准备数据...")
        for row in range(self.tb_bc.rowCount()):
            type = self.tb_bc.item(row, 1).text() if self.tb_bc.item(row, 1) else ''
            batch_id = self.tb_bc.item(row, 2).text() if self.tb_bc.item(row, 2) else ''
            chief = self.tb_bc.item(row, 3).text() if self.tb_bc.item(row, 3) else ''
            photo = self.tb_bc.item(row, 4).text() if self.tb_bc.item(row, 4) else ''
            detect_time = self.tb_bc.item(row, 5).text() if self.tb_bc.item(row, 5) else ''
            is_qualified = self.tb_bc.item(row, 6).text() if self.tb_bc.item(row, 6) else ''
            print("添加{}数据到tb_bc...".format(photo))
            self.cursor.execute('''
                INSERT INTO tb_bc (type, batch_id, chief, photo, detect_time, is_qualified)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (type, batch_id, chief, photo, detect_time, is_qualified))
        print( "开始提交")
        self.conn.commit()
        print("tb_bc添加完成")

    def save_detail_to_db(self):
        print("开始准备数据...")
        for row in range(self.tb_detail.rowCount()):
            photo = self.tb_detail.item(row, 1).text() if self.tb_bc.item(row, 1) else ''
            defect = self.tb_detail.item(row, 2).text() if self.tb_bc.item(row, 2) else ''
            detail = self.tb_detail.item(row, 3).text() if self.tb_bc.item(row, 3) else ''
            print("添加{}数据到tb_bc...".format(photo))
            self.cursor.execute('''
                            INSERT INTO tb_detail (photo, defect, detail)
                            VALUES (?, ?, ?)
                        ''', (photo, defect, detail))
        self.conn.commit()
        print("tb_detail添加完成")

    def closeEvent(self, event):
        #done 没有数据库的时候第一次运行数据正常添加，几个batch都可以；但是一旦关闭界面再重开，conn.commit时就会卡退。怀疑是这里除了问题，因为前面createcreate_tables_if_not_exist注释掉时仍不行
        self.conn.close()
        super().closeEvent(event)
        print( "安全退出" )

    # # 异常退出
    # def on_about_to_quit(self):
    #     self.conn.close()
    #     print( "异常退出" )



