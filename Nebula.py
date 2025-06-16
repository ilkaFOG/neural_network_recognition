import sys
import os
import cv2
import requests
import datetime
import torch
import pygame
import warnings
import hid
import numpy as np
from torchvision import transforms, models
from PIL import Image
from torch import nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMenuBar, 
                             QMenu, QAction, QVBoxLayout, QWidget, QActionGroup,
                             QInputDialog, QLineEdit, QMessageBox, QProgressDialog)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QIcon, QFont, QKeyEvent
from PyQt5.QtCore import Qt, QTimer, QPointF, QEvent, QPoint, QCoreApplication
from PyQt5.QtWidgets import QToolButton, QHBoxLayout

# Конфигурация нейросети
CLASS_NAMES = ["Био-дефекты", "Норма", "Ржавчина", "Трещины"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "defect_detection_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DefectDetector:
    def __init__(self):
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Загрузка предварительно обученной модели с использованием нового API torchvision"""
        try:
            # Используем новый API torchvision с weights=None вместо pretrained=False
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model = model.to(device)
                model.eval()
                print("Модель успешно загружена")
                return model
            return None
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None
    
    def predict(self, image_path):
        """Предсказание класса дефекта для изображения"""
        if self.model is None:
            return "Модель не загружена", 0.0
        
        try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = self.transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                _, predicted = torch.max(outputs, 1)
                
            return CLASS_NAMES[predicted[0]], probs[predicted[0]].item()
        
        except Exception as e:
            print(f"Ошибка классификации: {e}")
            return f"Ошибка: {str(e)}", 0.0


def clear_console():
    """Очищает консоль в зависимости от ОС"""
    os.system('cls' if os.name == 'nt' else 'clear')

# Очищаем консоль при запуске
clear_console()

# Настройки окружения для Steam Deck
os.environ.update({
    "QT_QUICK_BACKEND": "software",
    "QT_QPA_PLATFORM": "xcb",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    "OPENCV_LOG_LEVEL": "ERROR"
})

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._initialize_pygame()
        self._setup_ui()
        self._initialize_components()
        
    def _initialize_pygame(self):
        """Инициализация Pygame для работы с джойстиком"""
        pygame.init()
        pygame.joystick.init()
        warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
        warnings.filterwarnings("ignore", module="pygame")
    
    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("Nebula")
        self._set_window_icon()
        self._apply_styles()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.background_label = QLabel(alignment=Qt.AlignCenter)
        self.crosshair_label = QLabel(self.background_label, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.background_label)
        
        self._setup_control_buttons()
        self._setup_joystick_display()
    
    def _initialize_components(self):
        """Инициализация компонентов приложения"""
        self.control_mode = "Ручной"
        self.camera = None
        self.current_source = None
        self.timer = QTimer()
        self.joystick = None
        self.joystick_name = "Не подключен"
        self.options_button_id = 9
        self.menu_shown = False
        self.current_menu_item = 0
        self.menu_items = []
        self.last_hat_pos = (0, 0)
        self.menu_navigation_enabled = False
        self.left_joystick_pos = QPoint(0, 0)
        self.right_joystick_pos = QPoint(0, 0)
        
        # Инициализация таймера для джойстика
        self.joystick_timer = QTimer()  # Исправлено: правильное имя атрибута
        self.joystick_timer.timeout.connect(self._update_joystick_status)
        
        self.detector = DefectDetector()
        self.timer.timeout.connect(self._update_frame)
        
        self.set_static_background()
        self.draw_crosshair()
        self.showFullScreen()
        self._init_joystick()
        self.joystick_timer.start(10)  # Запуск таймера джойстика
    
    def _setup_control_buttons(self):
        """Настройка кнопок управления"""
        # Кнопка Options
        options_button = QToolButton()
        options_button.setText("Options")
        options_button.setStyleSheet("""
            QToolButton {
                color: white;
                background-color: rgba(26, 26, 26, 150);
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: rgba(51, 51, 51, 150);
            }
        """)
        options_button.setPopupMode(QToolButton.InstantPopup)
        options_button.setMenu(self._create_options_menu())
        
        # Кнопка закрытия
        close_button = QToolButton()
        close_button.setIcon(QIcon.fromTheme("window-close"))
        close_button.setStyleSheet("""
            QToolButton {
                border: none;
                padding: 5px;
                icon-size: 20px;
            }
            QToolButton:hover {
                background-color: #333;
            }
        """)
        close_button.clicked.connect(self.close)
        close_button.setToolTip("Закрыть приложение")
        
        # Контейнер для кнопок
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 10, 0)
        buttons_layout.addWidget(options_button)
        buttons_layout.addWidget(close_button)
        
        self.menuBar().setCornerWidget(buttons_widget, Qt.TopRightCorner)
    
    def _create_options_menu(self):
        """Создание меню Options"""
        menu = QMenu(self)
        
        # Подменю "Фон"
        bg_menu = menu.addMenu('Фон')
        bg_menu.addAction(self._create_action('Статичный фон', self.set_static_background))
        bg_menu.addAction(self._create_action('Локальная камера', self.set_camera_background))
        bg_menu.addAction(self._create_action('BeagleBone Stream', self.set_beaglebone_stream))
        
        # Подменю "Управление"
        control_menu = menu.addMenu('Управление')
        self.control_group = QActionGroup(self)
        
        for text, mode in [('Автоматический', "Автоматический"), ('Ручной', "Ручной")]:
            action = self._create_action(text, lambda checked, m=mode: self.set_control_mode(m), True)
            self.control_group.addAction(action)
            control_menu.addAction(action)
            if mode == "Ручной":
                action.setChecked(True)
        
        # Другие пункты меню
        menu.addAction(self._create_action("Распознать дефект", self.set_neural_network))
        menu.addAction(self._create_action("Повторный поиск джойстика", self.redetect_joystick))
        
        return menu
    
    def _create_action(self, text, callback, checkable=False):
        """Создание действия меню"""
        action = QAction(text, self, checkable=checkable)
        action.triggered.connect(callback)
        return action
    
    def _setup_joystick_display(self):
        """Настройка отображения положения джойстиков"""
        # Overlay для джойстиков
        self.joystick_overlay = QLabel(self)
        self.joystick_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.joystick_overlay.resize(self.size())
        self.joystick_overlay.move(0, 0)
        self.joystick_overlay.show()
        
        # Статус джойстика
        self.joystick_status_label = QLabel(self)
        self.joystick_status_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 150);
                padding: 15px;
                border-radius: 5px;
            }
        """)
        self.joystick_status_label.setFont(QFont("Arial", 10))
        self.joystick_status_label.move(20, 60)
        self.joystick_status_label.resize(330, 120)
        self.joystick_status_label.show()
    
    def _set_window_icon(self):
        """Установка иконки приложения"""
        try:
            self.setWindowIcon(QIcon('Icon.ico'))
        except:
            print("Иконка не найдена, запуск без неё")
    
    def _apply_styles(self):
        """Применение стилей интерфейса"""
        self.setStyleSheet("""
            QMainWindow { background-color: #020061; }
            QMenuBar { background-color: #1a1a1a; color: white; }
            QMenuBar::item { background-color: transparent; }
            QMenuBar::item:selected { background-color: #333; }
            QMenu { 
                background-color: black; 
                color: white; 
                border: 1px solid #444;
            }
            QMenu::item:selected { 
                background-color: #333;
                color: #fff;
                font-weight: bold;
            }
            QAction { color: white; }
            QToolButton:hover { background-color: #333; }
        """)
    
    def _init_joystick(self):
        """Инициализация джойстика через Pygame"""
        try:
            joystick_count = pygame.joystick.get_count()
            if joystick_count > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_name = self.joystick.get_name()
                print(f"Подключен джойстик: {self.joystick_name}")
            else:
                self.joystick_name = "Не подключен"
                print("Джойстик не обнаружен")
        except Exception as e:
            print(f"Ошибка инициализации джойстика: {e}")
            self.joystick_name = "Ошибка подключения"
    
    def redetect_joystick(self):
        """Повторный поиск подключенных джойстиков"""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            
            joystick_count = pygame.joystick.get_count()
            if joystick_count > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_name = self.joystick.get_name()
                QMessageBox.information(self, "Успех", f"Джойстик подключен: {self.joystick_name}")
            else:
                self.joystick = None
                self.joystick_name = "Не подключен"
                QMessageBox.information(self, "Информация", "Джойстики не обнаружены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при поиске джойстиков: {str(e)}")
            self.joystick = None
            self.joystick_name = "Ошибка подключения"
    
    def draw_crosshair(self):
        """Отрисовка прицела"""
        size = min(self.width(), self.height()) // 10
        crosshair = QPixmap(size, size)
        crosshair.fill(Qt.transparent)
        
        center = size // 2
        third = size // 3
        
        with QPainter(crosshair) as painter:
            painter.setPen(QPen(Qt.white, 1))
            painter.drawLine(center, 0, center, third)
            painter.drawLine(0, center, third, center)
            painter.drawLine(2*third, center, size, center)
            painter.drawEllipse(QPointF(center, center), size//6, size//6)
        
        self.crosshair_label.setPixmap(crosshair)
        self.crosshair_label.resize(size, size)
        self.crosshair_label.move(self.width()//2 - size//2, self.height()//2 - size//2)
    
    def resizeEvent(self, event):
        """Обработка изменения размера окна"""
        super().resizeEvent(event)
        self.draw_crosshair()
        if hasattr(self, 'joystick_overlay'):
            self.joystick_overlay.resize(self.size())
    
    def set_control_mode(self, mode):
        """Установка режима управления"""
        self.control_mode = mode
        QMessageBox.information(
            self,
            "Режим управления",
            f"Режим изменён на: {mode}"
        )
    
    def set_static_background(self):
        """Установка статичного фона"""
        self._stop_camera()
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#020061"))
        self.background_label.setPixmap(pixmap)
        self.draw_crosshair()
    
    def _update_joystick_status(self):
        """Обновление статуса джойстиков"""
        try:
            # Чтение состояния джойстиков
            left_x, left_y = self._read_joystick_axis(0)
            right_x, right_y = self._read_joystick_axis(1)

            # Проверка нажатия кнопки 0 (A) - имитация Enter
            if self.joystick and self.joystick.get_button(0):
                enter_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Enter, Qt.NoModifier)
                QApplication.postEvent(self, enter_event)
            
            # Обновление позиций джойстиков
            self.left_joystick_pos = QPoint(int(left_x * 100), int(left_y * 100))
            self.right_joystick_pos = QPoint(int(right_x * 100), int(right_y * 100))
            
            # Обработка кнопки Options
            if self.joystick:
                options_pressed = self.joystick.get_button(self.options_button_id)
                
                if options_pressed:
                    if not self.menu_shown:
                        self.menu_shown = True
                        self.show_menu()
                elif self.menu_shown:
                    self._handle_menu_navigation()
            
            # Обновление текста статуса
            buttons_status = []
            if self.joystick:
                for i in range(min(4, self.joystick.get_numbuttons())):
                    buttons_status.append(f"Кн{i+1}: {'Вкл' if self.joystick.get_button(i) else 'Выкл'}")
            
            status_text = (f"Джойстик: {self.joystick_name}\n"
                        f"Левый: X={left_x:.2f}, Y={left_y:.2f}\n"
                        f"Правый: X={right_x:.2f}, Y={right_y:.2f}\n"
                        f"{' | '.join(buttons_status)}")
            
            self.joystick_status_label.setText(status_text)
            self.update()  # Обновляем отрисовку джойстиков
            
        except Exception as e:
            print(f"Ошибка обновления статуса джойстика: {e}")
            self.joystick_status_label.setText("Ошибка чтения джойстика")
    
    def _handle_menu_navigation(self):
        """Обработка навигации по меню с помощью джойстика"""
        if not self.menu_shown or not hasattr(self, 'options_menu'):
            return
        
        try:
            hat_pos = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)
            
            if hat_pos != self.last_hat_pos:
                if hat_pos[1] == 1:  # Вверх
                    self._move_menu_selection(-1)
                elif hat_pos[1] == -1:  # Вниз
                    self._move_menu_selection(1)
                elif hat_pos[0] == 1:  # Вправо (для подменю)
                    if self.options_menu.activeAction() and self.options_menu.activeAction().menu():
                        self.options_menu.activeAction().menu().popup(self.options_menu.pos())
                elif hat_pos[0] == -1:  # Влево (выход из подменю)
                    if self.options_menu.activeAction() and self.options_menu.activeAction().menu():
                        self.options_menu.setActiveAction(None)
                
                self.last_hat_pos = hat_pos
            
            if self.joystick.get_button(0):  # Кнопка A
                if self.options_menu.activeAction():
                    self.options_menu.activeAction().trigger()
        except Exception as e:
            print(f"Ошибка навигации по меню: {e}")
    
    def _move_menu_selection(self, direction):
        """Перемещение выбора по меню"""
        if not hasattr(self, 'options_menu'):
            return
        
        actions = self.options_menu.actions()
        if not actions:
            return
        
        current_index = -1
        if self.options_menu.activeAction():
            current_index = actions.index(self.options_menu.activeAction())
        
        new_index = max(0, min(current_index + direction, len(actions) - 1))
        self.options_menu.setActiveAction(actions[new_index])
    
    def show_menu(self):
        """Показать меню по кнопке Options"""
        if not hasattr(self, 'options_menu'):
            self.options_menu = self._create_options_menu()
        
        menu_pos = self.mapToGlobal(self.rect().center())
        menu_pos.setX(menu_pos.x() - self.options_menu.sizeHint().width() // 2)
        menu_pos.setY(menu_pos.y() - self.options_menu.sizeHint().height() // 2)
        
        if self.options_menu.actions():
            self.options_menu.setActiveAction(self.options_menu.actions()[0])
        
        self.options_menu.exec_(menu_pos)
        self.menu_shown = False
    
    def _read_joystick_axis(self, joystick_num):
        """Чтение данных с джойстика через Pygame"""
        try:
            if self.joystick is None:
                self._init_joystick()
                if self.joystick is None:
                    return (0, 0)
            
            pygame.event.pump()
            
            if joystick_num == 0:  # Левый джойстик
                axis_x = self.joystick.get_axis(0)
                axis_y = self.joystick.get_axis(1)
            else:  # Правый джойстик
                axis_x = self.joystick.get_axis(3)
                axis_y = self.joystick.get_axis(4)
            
            return (axis_x, axis_y)
        except Exception as e:
            print(f"Ошибка чтения джойстика: {e}")
            self.joystick = None
            return (0, 0)
    
    def paintEvent(self, event):
        """Отрисовка положения джойстиков на экране"""
        super().paintEvent(event)
        
        overlay_pixmap = QPixmap(self.size())
        overlay_pixmap.fill(Qt.transparent)
        
        painter = QPainter(overlay_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Отрисовка джойстиков
        self._draw_joystick(painter, self.left_joystick_pos, QPoint(100, self.height() - 100), "Левый")
        self._draw_joystick(painter, self.right_joystick_pos, QPoint(self.width() - 100, self.height() - 100), "Правый")
        
        painter.end()
        self.joystick_overlay.setPixmap(overlay_pixmap)
    
    def _draw_joystick(self, painter, position, center, label):
        """Отрисовка одного джойстика"""
        # Окружность основания
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, 30, 30)
        
        # Позиция джойстика
        dot_x = center.x() + position.x() * 30 / 100
        dot_y = center.y() + position.y() * 30 / 100
        dot_pos = QPoint(int(dot_x), int(dot_y))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 0, 0, 250))
        painter.drawEllipse(dot_pos, 10, 10)
        
        # Подпись
        painter.setPen(Qt.white)
        painter.drawText(center.x() - 30, center.y() + 50, label)
    
    def set_camera_background(self):
        """Запуск видеопотока с локальной камеры"""
        self._start_video_stream(0)
    
    def set_beaglebone_stream(self):
        """Подключение к BeagleBone видеопотоку"""
        ip, ok = QInputDialog.getText(
            self, 'BeagleBone Stream', 'Введите IP адрес BeagleBone:', 
            QLineEdit.Normal, '192.168.1.17'
        )
        
        if ok and ip:
            if not self._check_beaglebone_connection(ip):
                QMessageBox.warning(self, "Ошибка", "Не удалось подключиться к BeagleBone")
                return
                
            stream_url = f"http://{ip}:5000/video_feed"
            self._start_video_stream(stream_url)
    
    def _check_beaglebone_connection(self, ip):
        """Проверка доступности BeagleBone"""
        try:
            response = requests.get(f"http://{ip}:5000/status", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _start_video_stream(self, source):
        """Запуск видео потока с обработкой разных источников"""
        self._stop_camera()
        self.current_source = source
        
        try:
            if isinstance(source, str) and ("http://" in source or "rtsp://" in source):
                if "beaglebone" in source.lower() or "http://" in source:
                    self.camera = self._create_beaglebone_capture(source)
                else:
                    self.camera = cv2.VideoCapture(source)
                    self._configure_network_stream()
            else:
                self.camera = cv2.VideoCapture(source)
            
            if self.camera.isOpened():
                self._configure_camera()
                self.timer.start(30)
            else:
                raise RuntimeError("Не удалось открыть видео поток")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка запуска видео потока:\n{str(e)}")
            self._stop_camera()
    
    def _create_beaglebone_capture(self, url):
        """Создание захвата видео для BeagleBone"""
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            camera = cv2.VideoCapture(url, backend)
            if camera.isOpened():
                return camera
            if camera:
                camera.release()
        
        return cv2.VideoCapture(url)
    
    def _configure_network_stream(self):
        """Конфигурация сетевого потока"""
        if self.camera:
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    
    def _configure_camera(self):
        """Конфигурация параметров камеры"""
        if self.camera:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def _stop_camera(self):
        """Остановка и освобождение ресурсов камеры"""
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
    
    def _update_frame(self):
        """Обновление кадра с обработкой ошибок"""
        try:
            ret, frame = self.camera.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch*w, QImage.Format_RGB888)
                
                self.background_label.setPixmap(
                    QPixmap.fromImage(qt_image).scaled(
                        self.background_label.size(), 
                        Qt.KeepAspectRatioByExpanding
                    )
                )
                self.draw_crosshair()
            else:
                self._handle_frame_error()
        except Exception as e:
            print(f"Ошибка при обновлении кадра: {e}")
            self._handle_frame_error()
    
    def _handle_frame_error(self):
        """Обработка ошибок получения кадра"""
        print("Ошибка получения кадра, попытка переподключения...")
        self._stop_camera()
        QTimer.singleShot(2000, self._reconnect)
    
    def _reconnect(self):
        """Повторное подключение к источнику видео"""
        if self.current_source is not None:
            self._start_video_stream(self.current_source)
    
    def set_neural_network(self):
        """Сделать снимок и классифицировать с помощью нейросети"""
        if self.control_mode != "Ручной":
            QMessageBox.warning(
                self, 
                "Ошибка", 
                "Классификация доступна только в ручном режиме управления!"
            )
            return
            
        progress = QProgressDialog("Выполняется классификация...", "Отмена", 0, 0, self)
        progress.setWindowTitle("Распознавание дефектов")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QCoreApplication.processEvents()
        
        try:
            os.makedirs("Foto", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Foto/shot_{timestamp}.jpg"
            
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    cv2.imwrite(filename, frame)
                else:
                    raise RuntimeError("Не удалось получить кадр с камеры")
            else:
                pixmap = self.background_label.pixmap()
                if pixmap:
                    pixmap.save(filename)
                else:
                    raise RuntimeError("Нет доступного изображения")
            
            prediction, confidence = self.detector.predict(filename)
            progress.close()
            
            result_msg = (f"<b>Результат классификации:</b><br><br>"
                        f"<b>Изображение:</b> {os.path.basename(filename)}<br>"
                        f"<b>Тип дефекта:</b> {prediction}<br>"
                        f"<b>Уверенность:</b> {confidence:.2f}%")
            
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Результат распознавания")
            msg_box.setTextFormat(Qt.RichText)
            msg_box.setText(result_msg)
            
            pixmap = QPixmap(filename).scaled(400, 300, Qt.KeepAspectRatio)
            msg_box.setIconPixmap(pixmap)
            msg_box.exec_()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, 
                "Ошибка", 
                f"Не удалось выполнить классификацию:\n{str(e)}"
            )
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        """Закрытие приложения"""
        if hasattr(self, 'joystick') and self.joystick:
            self.joystick.quit()
        if hasattr(self, 'options_menu'):
            self.options_menu.deleteLater()
        pygame.quit()
        self._stop_camera()
        event.accept()


if __name__ == '__main__':
    clear_console()
    
    try:
        app = QApplication(sys.argv)
        window = GameWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")