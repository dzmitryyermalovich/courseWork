from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import sys

class Window(QMainWindow):
    def __init__(self):
        super(Window,self).__init__()

        self.setWindowTitle("S1mple program")
        self.setGeometry(300,250,450,300)

        self.btn=QtWidgets.QPushButton(self)
        self.btn.move(70,150)
        self.btn.setText("open")
        self.btn.clicked.connect(self.selectFile)

        self.btn1=QtWidgets.QPushButton(self)
        self.btn1.move(90,180)
        self.btn1.setText("print")
        self.btn1.clicked.connect(self.print)

    def selectFile(self):
        self.path = QFileDialog.getOpenFileName()[0]
    def print(self):
        print(self.path)
def application():
    app=QApplication(sys.argv)
    window=Window()

    window.show()
    sys.exit(app.exec_())

application()
