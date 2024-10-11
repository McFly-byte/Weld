import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from gui import Window

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    # app.aboutToQuit.connect(window.on_about_to_quit)
    # window.conn.close()
    sys.exit(app.exec_())
