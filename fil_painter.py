import sys, argparse, os, pathlib, dataset
from PyQt5 import QtCore, QtGui, QtWidgets
import mrcfile
import numpy as np
from base64 import b64decode, b64encode

def get_area(x,y):
    #helper function to return index location based on mouse click
    new_x = x//64
    new_y = y//64
    return new_x, new_y

def encode(arr):
    #encodes the matrix using base64
    return b64encode(np.packbits(arr.flatten()).tobytes())

def decode(s):
    #decodes the matrix using base64
    return np.unpackbits(np.frombuffer(b64decode(s), dtype='uint8'), count=196).astype(bool).reshape((-1, 14))

class Picture(QtWidgets.QLabel):
    #QWidget that will hold the current image
    def __init__(self, parent):
        super().__init__(parent=parent)

    def load_new_image(self, item1, item2):
        try:
            file_loc = self.parent().cur_directory.joinpath(item1.text())
            tmp = mrcfile.open(file_loc)
            data = tmp.data.copy()
            d_min = np.min(data)
            d_max = np.max(data)
            data_norm = (data-d_min)/(d_max-d_min)
            data_256 = (data_norm * 255).astype('uint8')
            #does a simple 0-1 normalization, not the same as the standardization done for training
            self.image = QtGui.QImage(data_256, 924, 924, QtGui.QImage.Format_Grayscale8)
        except Exception as e:
            print(e)
            sys.exit(1)
        self.cur_image = self.parent().db['jpgs'].find_one(file=item1.text()) #gets the database entry for this image
        if self.cur_image['data'] == "":
            self.arr = np.zeros((14,14), dtype=bool)
        else:
            self.arr = decode(self.cur_image['data'])
        painter = QtGui.QPainter(self.image)
        for i in range(1,15):
            painter.drawLine(0,i*64,924,i*64)
            painter.drawLine(i*64,0,i*64,924)
            #draws a fake grid on the image
        painter.end()
        self.prepare_image()
        self.setPixmap(QtGui.QPixmap(self.image))
        self.show()

    def prepare_image(self):
        #this function dims the selected areas of the image
        for x in range(14):
            for y in range(14):
                if self.arr[x,y]:
                    for i in range(x*64, (x+1)*64):
                        for j in range(y*64, (y+1)*64):
                            color = self.image.pixelColor(i,j)
                            color.setRed(int(color.red()*.5))
                            color.setGreen(int(color.green()*.5))
                            color.setBlue(int(color.blue()*.5))
                            self.image.setPixelColor(i,j,color)

    def mouseReleaseEvent(self, e):
        #mouse interaction function. Marks or unmarks the selected square
        x,y = get_area(e.x(), e.y())
        if (x>13) or (y>13):
            return
        if self.arr[x,y] == 0:
            self.arr[x,y] = 1
            for i in range(x*64, (x+1)*64):
                for j in range(y*64, (y+1)*64):
                    color = self.image.pixelColor(i,j)
                    color.setRed(int(color.red()*.5))
                    color.setGreen(int(color.green()*.5))
                    color.setBlue(int(color.blue()*.5))
                    self.image.setPixelColor(i,j,color)
        else:
            self.arr[x,y] = 0
            for i in range(x*64, (x+1)*64):
                for j in range(y*64, (y+1)*64):
                    color = self.image.pixelColor(i,j)
                    color.setRed(color.red()*2)
                    color.setGreen(color.green()*2)
                    color.setBlue(color.blue()*2)
                    self.image.setPixelColor(i,j,color)
        self.cur_image['data'] = encode(self.arr)
        self.parent().db['jpgs'].update(self.cur_image, ['file'])
        self.setPixmap(QtGui.QPixmap(self.image))

class SideBox(QtWidgets.QGroupBox):
    #Side widget that holds the names for all the files
    def __init__(self, parent):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        sideVBox = QtWidgets.QVBoxLayout()
        pic_list = [i['file'] for i in self.parent().db['jpgs']]

        self.pic_list = QtWidgets.QListWidget()
        self.pic_list.addItems(pic_list)
        sideVBox.addWidget(QtWidgets.QLabel("<b>Pictures:</b>"))
        sideVBox.addWidget(self.pic_list)

        self.setLayout(sideVBox)

        self.pic_list.currentItemChanged.connect(self.parent().picture.load_new_image)

class MainWidget(QtWidgets.QWidget):
    #Widget that holds the other widget and is used by them to communicate to each other
    def __init__(self, args, parent):
        super().__init__(parent)

        self.db = None
        self.cur_directory = pathlib.Path(args.directory).expanduser().absolute()
        db_path = self.cur_directory.joinpath(args.db)
        if db_path.is_file() == False:
            #if the database does not exist, create a new one with all of the current mrc files as entries
            self.db = dataset.connect("sqlite:///{}".format(db_path))
            jpg_list = [i for i in os.listdir(self.cur_directory) if i.endswith(".mrc")]
            jpg_list.sort()
            dics = [dict(file=i, data="") for i in jpg_list]
            self.db['jpgs'].insert_many(dics)
        else:
            self.db = dataset.connect("sqlite:///{}".format(db_path))

        self.initUI()

    def initUI(self):
        self.picture = Picture(self)

        mainHLayout = QtWidgets.QHBoxLayout()

        self.sidebox = SideBox(self)

        mainHLayout.addWidget(self.sidebox, 1)

        mainHLayout.addWidget(self.picture, 2)

        self.setLayout(mainHLayout)

class MainWindow(QtWidgets.QMainWindow):
    #initializes a few things but MainWidget does most of the logic
    def __init__(self, args):
        super().__init__()
        self.initUI(args)

    def initUI(self, args):
        exit_action = QtWidgets.QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QtWidgets.qApp.quit)

        self.statusBar()

        if args.directory is None:
            ans = QtWidgets.QFileDialog.getExistingDirectory()
            if ans == "":
                sys.exit()
            args.directory = ans

        if args.db is None:
            ans = QtWidgets.QFileDialog.getOpenFileName(directory= args.directory, filter="*.db", )
            if ans[0] == "":
                sys.exit()
            args.db = pathlib.Path(ans[0]).name
            

        main_widget = MainWidget(args, self)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(exit_action)

        self.setCentralWidget(main_widget)
        self.resize(1000,1000)
        self.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, default=None)
    parser.add_argument("--db", type=str, help='name of database file to open', default=None)
    args = parser.parse_args()

    mw = MainWindow(args)
    sys.exit(app.exec_())
