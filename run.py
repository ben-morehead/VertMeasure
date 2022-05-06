import argparse
import pandas as pd
import numpy as np
import time, os, sys
from PyQt5.QtWidgets import QApplication
from ui import Window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen_width, screen_height = app.desktop().screenGeometry().width(), app.desktop().screenGeometry().height()
    app.setStyleSheet('''
        QWidget{
            font-size:24px;
        }
    ''')

    window = Window(screen_width, screen_height)
    window.show()

    sys.exit(app.exec_())