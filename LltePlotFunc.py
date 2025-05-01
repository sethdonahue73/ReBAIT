import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

class MainWindow(QtWidgets.QMainWindow):

    class HighlightArea(pg.GraphicsWidget):
        def __init__(self, y1, y2, color):
            super().__init__()
            self.y1 = y1
            self.y2 = y2
            self.color = color

        def paint(self, p, *args):
            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(self.color))
            vr = self.viewRect()
            p.drawRect(QtCore.QRectF(vr.left(), self.y1, vr.width(), self.y2 - self.y1))

    def __init__(self, dc, *args, **kwargs):
        self.dc = dc
        self.app = QtWidgets.QApplication([])

        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.x = [0,100]
        self.y = [0 for _ in range(5)]  # 5 data points
        
        # highlight_1 = HighlightArea(-1.5, 0.1,  'w')
        # highlight_2 = HighlightArea(-4.0, -1.5,  'yellow')
        # highlight_3 = HighlightArea(-8.0,-4.0,  'gray')

        self.graphWidget.setBackground('w')
        # self.graphWidget.setYRange(-8,0.0, padding = 0.0)
        self.graphWidget.setXRange(0,100, padding = 0.0)
        # Add the highlight area
        # self.graphWidget.addItem(highlight_1, ignoreBounds = False)
        # highlight_1.setZValue(-3)
        # self.graphWidget.addItem(highlight_2, ignoreBounds = False)
        # highlight_2.setZValue(-2)
        # self.graphWidget.addItem(highlight_3, ignoreBounds = False)
        # highlight_3.setZValue(-1)

        pen1 = pg.mkPen(color=(0, 0, 255,  25), width=6)
        pen2 = pg.mkPen(color=(0, 0, 255,  50), width=6)
        pen3 = pg.mkPen(color=(0, 0, 255,  100), width=6)
        pen4 = pg.mkPen(color=(0, 0, 255, 150), width=6)
        pen5 = pg.mkPen(color=(0, 0, 255, 255), width=6)
        pen_tar = pg.mkPen(color=(0, 0, 0), width=32)
        
        self.data_line1 =  self.graphWidget.plot([self.y[0]], pen=pen1)
        self.data_line2 =  self.graphWidget.plot([self.y[1]], pen=pen2)
        self.data_line3 =  self.graphWidget.plot([self.y[2]], pen=pen3)
        self.data_line4 =  self.graphWidget.plot([self.y[3]], pen=pen4)
        self.data_line5 =  self.graphWidget.plot([self.y[4]], pen=pen5)
        self.data_line_tar = self.graphWidget.plot([0], pen=pen_tar)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        if len(self.dc.llte_val) >= 5:
            try: # sometimes a foot strike is detected before position integration calc is completed, so there is no value to access and an index exception is raised. Just skip this frame update and try again next frame
                self.y = self.dc.llte_val[-5:] # plots total distance walked
                # min_y = min(self.y)
                # if min_y < -4.0:
                #     self.graphWidget.setYRange(-8,0, padding = 0.0)
                # elif min_y < -1.5:
                #     self.graphWidget.setYRange(-4,0, padding = 0.0)
                # else: 
                #     self.graphWidget.setYRange(-1.5,0, padding = 0.0)
                
                self.data_line1.setData(self.x, [-self.y[0],-self.y[0]])  # Update the data.
                self.data_line2.setData(self.x, [-self.y[1],-self.y[1]])  # Update the data.
                self.data_line3.setData(self.x, [-self.y[2],-self.y[2]])  # Update the data.
                self.data_line4.setData(self.x, [-self.y[3],-self.y[3]])  # Update the data.
                self.data_line5.setData(self.x, [-self.y[4],-self.y[4]])  # Update the data.
                self.update()
            except:
                pass

    def reset_plot(self):
        self.y = [0 for _ in self.y]
        self.data_line1.setData(self.x, [self.y[0],self.y[0]])  # Reset the data.
        self.data_line2.setData(self.x, [self.y[1],self.y[1]])  # Reset the data.
        self.data_line3.setData(self.x, [self.y[2],self.y[2]])  # Reset the data.
        self.data_line4.setData(self.x, [self.y[3],self.y[3]])  # Reset the data.
        self.data_line5.setData(self.x, [self.y[4],self.y[4]])  # Reset the data.
        self.data_line_tar.setData(self.x, [0,0])

    def start_plot(self):
        self.show()
        self.app.exec_()
