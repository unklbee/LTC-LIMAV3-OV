from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication


def setup_menubar(main_window):
    """
    Pasang menu bar ke QMainWindow yang diberikan.
    main_window harus memiliki method:
      - save_preset(self)
      - load_preset(self)
      - reset(self)
      - _show_about(self)
      - _show_help(self)
    """
    menubar = main_window.menuBar()

    # --- File Menu ---
    file_menu = menubar.addMenu("&File")

    save_action = QAction("Save Preset…", main_window)
    save_action.setShortcut("Ctrl+S")
    save_action.triggered.connect(main_window.save_preset)
    file_menu.addAction(save_action)

    load_action = QAction("Load Preset…", main_window)
    load_action.setShortcut("Ctrl+O")
    load_action.triggered.connect(main_window.load_preset)
    file_menu.addAction(load_action)

    file_menu.addSeparator()

    reset_action = QAction("Reset", main_window)
    reset_action.setShortcut("Ctrl+R")
    reset_action.triggered.connect(main_window.reset)
    file_menu.addAction(reset_action)

    file_menu.addSeparator()

    exit_action = QAction("Exit", main_window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(QApplication.instance().quit)
    file_menu.addAction(exit_action)

    # --- Help Menu ---
    help_menu = menubar.addMenu("&Help")

    guide_action = QAction("Panduan Pengguna", main_window)
    guide_action.setShortcut("F1")
    guide_action.triggered.connect(main_window._show_help)
    help_menu.addAction(guide_action)

    help_menu.addSeparator()

    about_action = QAction("About", main_window)
    about_action.triggered.connect(main_window._show_about)
    help_menu.addAction(about_action)
