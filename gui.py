from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import LltePlotFunc

app = QApplication([])
window = QWidget()
layout = QHBoxLayout()
layout.addWidget(LltePlotFunc)
buttons = QVBoxLayout()
buttons.addWidget(QPushButton('Zero'))
buttons.addWidget(QPushButton('Calibrate'))
layout.addWidget(buttons)
window.setLayout(layout)
window.show()
app.exec_()
