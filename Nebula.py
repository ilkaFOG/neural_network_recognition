import sys
import os
import cv2
import requests
import datetime
import torch
import pygame
import warnings
import hid
from torchvision import transforms, models
from PIL import Image
from torch import nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMenuBar, 
                             QMenu, QAction, QVBoxLayout, QWidget, QActionGroup,
                             QInputDialog, QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QIcon, QFont, QKeyEvent
from PyQt5.QtCore import Qt, QTimer, QPointF, QEvent, QPoint
from PyQt5.QtWidgets import QToolButton, QHBoxLayout


# Функция для очистки консоли
def clear_console():
    """Очищает консоль в зависимости от ОС"""
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/MacOS
        os.system('clear')

# Очищаем консоль при запуске
clear_console()

# Настройки окружения для Steam Deck
os.environ.update({
    "QT_QUICK_BACKEND": "software",
    "QT_QPA_PLATFORM": "xcb"
})

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Инициализация Pygame для работы с джойстиком
        pygame.init()
        pygame.joystick.init()
        
        self.control_mode = "Ручной"
        self.camera = None
        self.current_source = None
        self.timer = QTimer()
        self.joystick = None
        self.joystick_name = "Не подключен"
        
        self._setup_ui()
        self._setup_connections()
        self._initialize_app()
        self._init_joystick()

        self.joystick_timer = QTimer()
        self.left_joystick_pos = QPoint(0, 0)
        self.right_joystick_pos = QPoint(0, 0)
        self._setup_joystick_display()

        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

        warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
        warnings.filterwarnings("ignore", module="pygame")

        self.options_button_id = 9  # Идентификатор кнопки Options на джойстике
        self.menu_shown = False     # Флаг для отслеживания состояния меню
        self.current_menu_item = 0  # Текущий выбранный пункт меню
        self.menu_items = []        # Список пунктов меню
        self.last_hat_pos = (0, 0)  # Последнее положение HAT (крестовины)
        self.menu_navigation_enabled = False  # Флаг для навигации по меню

    # Конфигурация нейросети
    NUM_CLASSES = 3  # Количество классов дефектов
    CLASS_NAMES = ["Норма", "Трещина", "Коррозия"]  # Пример классификации
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _handle_menu_navigation(self):
        """Обработка навигации по меню с помощью джойстика"""
        if not self.menu_shown or not hasattr(self, 'options_menu'):
            return
        
        try:
            # Получаем состояние HAT (крестовины)
            hat_pos = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)
            
            # Определяем направление движения
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
            
            # Проверка нажатия кнопки выбора (обычно кнопка A)
            if self.joystick.get_button(0):  # Кнопка A (обычно 0)
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
        
        new_index = current_index + direction
        
        # Проверка границ
        if new_index < 0:
            new_index = len(actions) - 1
        elif new_index >= len(actions):
            new_index = 0
        
        self.options_menu.setActiveAction(actions[new_index])

    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("Nebula")
        self._set_window_icon()
        self._apply_styles()
        
        # Создаем кнопку Options
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
        
        # Создаем меню для кнопки Options
        options_menu = QMenu(self)
        
        # Добавляем подменю "Фон"
        bg_menu = options_menu.addMenu('Фон')
        bg_menu.addAction(self._create_action('Статичный фон', self.set_static_background))
        bg_menu.addAction(self._create_action('Локальная камера', self.set_camera_background))
        bg_menu.addAction(self._create_action('BeagleBone Stream', self.set_beaglebone_stream))
        
        # Добавляем подменю "Управление"
        control_menu = options_menu.addMenu('Управление')
        self.control_group = QActionGroup(self)
        
        modes = [('Автоматический', "Автоматический"), ('Ручной', "Ручной")]
        for text, mode in modes:
            action = self._create_action(text, lambda checked, m=mode: self.set_control_mode(m), True)
            self.control_group.addAction(action)
            control_menu.addAction(action)
            if mode == "Ручной":
                action.setChecked(True)
        
        # Добавляем пункт "Нейросеть"
        options_menu.addAction(self._create_action("Распознать дефект", self.set_neural_network))
        
        # Связываем меню с кнопкой
        options_button.setMenu(options_menu)
        
        # Создаем кнопку закрытия
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
        
        # Создаем контейнер для кнопок
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 10, 0)
        buttons_layout.addWidget(options_button)
        buttons_layout.addWidget(close_button)
        
        # Добавляем кнопки в правый верхний угол
        self.menuBar().setCornerWidget(buttons_widget, Qt.TopRightCorner)
        
        # Остальная часть UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.background_label = QLabel(alignment=Qt.AlignCenter)
        self.crosshair_label = QLabel(self.background_label, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.background_label)
        
        # Удаляем старый метод создания меню, так как теперь используем кнопку Options
        # self._create_menu()

        # Создаем отдельный QLabel для джойстиков поверх всех элементов
        self.joystick_overlay = QLabel(self)
        self.joystick_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.joystick_overlay.resize(self.size())
        self.joystick_overlay.move(0, 0)
        self.joystick_overlay.show()

    def _setup_connections(self):
        """Настройка сигналов и слотов"""
        self.timer.timeout.connect(self._update_frame)

    def _initialize_app(self):
        """Инициализация приложения"""
        self.set_static_background()
        self.draw_crosshair()
        self.showFullScreen()

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

    def _create_action(self, text, callback, checkable=False):
        """Фабрика действий меню"""
        action = QAction(text, self, checkable=checkable)
        action.triggered.connect(callback)
        return action

    def set_control_mode(self, mode):
        """Установка режима управления"""
        self.control_mode = mode
        print(f"Режим управления изменён на: {mode}")
        QMessageBox.information(
            self,
            "Режим управления",
            f"Режим изменён на: {mode}"
        )

    def _init_hid_device(self):
        """Инициализация HID устройства"""
        try:
            # Открываем устройство по пути
            self.hid_device = hid.device()
            self.hid_device.open_path('/dev/hidraw4')
            self.hid_device.set_nonblocking(1)  # Неблокирующий режим
            print("Успешно подключено к HID устройству")
        except Exception as e:
            print(f"Ошибка подключения к HID устройству: {e}")
            self.hid_device = None

    def _init_joystick(self):
        """Инициализация джойстика через Pygame"""
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.joystick_name = self.joystick.get_name()
            print(f"Подключен джойстик: {self.joystick_name}")
        else:
            self.joystick_name = "Не подключен"
            print("Джойстик не обнаружен")        

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

    def set_static_background(self):
        """Установка статичного фона"""
        self._stop_camera()
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#020061"))
        self.background_label.setPixmap(pixmap)
        self.draw_crosshair()

    def _setup_joystick_display(self):
        """Настройка отображения положения джойстиков"""
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
        self.joystick_status_label.resize(330, 120)  # Увеличиваем высоту для дополнительной информации
        self.joystick_status_label.show()
        
        # Запускаем таймер для обновления положения джойстиков
        self.joystick_timer.timeout.connect(self._update_joystick_status)
        self.joystick_timer.start(10)  # Обновлять каждые 100 мс

    def _check_joystick_connection(self):
        """Проверка подключения джойстиков"""
        try:
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            
            if joystick_count > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self.joystick_name = self._joystick.get_name()
                self.joystick_connected = True
            else:
                self.joystick_name = "Не подключен"
                self.joystick_connected = False
                
        except Exception as e:
            print(f"Ошибка проверки джойстика: {e}")
            self.joystick_name = "Ошибка подключения"
            self.joystick_connected = False    

    def _update_joystick_status(self):
        """Обновление статуса джойстиков"""
        try:
            # Чтение состояния джойстиков
            left_x, left_y = self._read_joystick_axis(0)
            right_x, right_y = self._read_joystick_axis(1)

             # Проверка нажатия кнопки 0 (A) - имитация Enter
            if self.joystick and self.joystick.get_button(0):
                # Создаем событие нажатия Enter
                enter_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Enter, Qt.NoModifier)
                QApplication.postEvent(self, enter_event)
                
            # Преобразование координат
            self.left_joystick_pos = QPoint(int(left_x * 100), int(left_y * 100))
            self.right_joystick_pos = QPoint(int(right_x * 100), int(right_y * 100))
            
            # Проверка состояния кнопки Options
            if self.joystick:
                options_pressed = self.joystick.get_button(self.options_button_id)
                
                # Если кнопка Options нажата
                if options_pressed:
                    if not self.menu_shown:
                        self.menu_shown = True
                        self.show_menu()
                    else:
                        # Если меню уже показано - закрываем его
                        self.menu_shown = False
                        if hasattr(self, 'options_menu'):
                            self.options_menu.close()
                else:
                    # Обработка навигации по меню только если меню показано
                    if self.menu_shown:
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
            
        except Exception as e:
            print(f"Ошибка обновления статуса джойстика: {e}")
            self.joystick_status_label.setText("Ошибка чтения джойстика")

    def show_menu(self):
        """Показать меню по кнопке Options"""
        if not hasattr(self, 'options_menu'):
            # Создаем меню при первом вызове
            self.options_menu = QMenu(self)
            
            # Добавляем подменю "Фон"
            bg_menu = self.options_menu.addMenu('Фон')
            bg_menu.addAction(self._create_action('Статичный фон', self.set_static_background))
            bg_menu.addAction(self._create_action('Локальная камера', self.set_camera_background))
            bg_menu.addAction(self._create_action('BeagleBone Stream', self.set_beaglebone_stream))
            
            # Добавляем подменю "Управление"
            control_menu = self.options_menu.addMenu('Управление')
            self.control_group = QActionGroup(self)
            
            modes = [('Автоматический', "Автоматический"), ('Ручной', "Ручной")]
            for text, mode in modes:
                action = self._create_action(text, lambda checked, m=mode: self.set_control_mode(m), True)
                self.control_group.addAction(action)
                control_menu.addAction(action)
                if mode == "Ручной":
                    action.setChecked(True)
            
            # Добавляем пункт "Нейросеть"
            self.options_menu.addAction(self._create_action("Распознать дефект", self.set_neural_network))
        
        # Показываем меню в центре экрана
        menu_pos = self.mapToGlobal(self.rect().center())
        menu_pos.setX(menu_pos.x() - self.options_menu.sizeHint().width() // 2)
        menu_pos.setY(menu_pos.y() - self.options_menu.sizeHint().height() // 2)
        
        # Устанавливаем первый пункт меню активным
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
            
            # Обработка событий Pygame
            pygame.event.pump()
            
            # Чтение осей (зависит от конфигурации джойстика)
            if joystick_num == 0:  # Левый джойстик
                axis_x = self.joystick.get_axis(0)
                axis_y = self.joystick.get_axis(1)
            else:  # Правый джойстик
                axis_x = self.joystick.get_axis(3)  # Обычно ось 3 и 4 для правого джойстика
                axis_y = self.joystick.get_axis(4)
            
            return (axis_x, axis_y)
            
        except Exception as e:
            print(f"Ошибка чтения джойстика: {e}")
            self.joystick = None
            return (0, 0)
            
        except Exception as e:
            print(f"Ошибка чтения HID устройства: {e}")
            self.hid_device = None
            return (0, 0)

    def paintEvent(self, event):
        """Отрисовка положения джойстиков на экране"""
        super().paintEvent(event)
        
        # Создаем QPixmap для отрисовки джойстиков
        overlay_pixmap = QPixmap(self.size())
        overlay_pixmap.fill(Qt.transparent)
        
        painter = QPainter(overlay_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Отрисовка левого джойстика
        self._draw_joystick(painter, self.left_joystick_pos, QPoint(100, self.height() - 100), "Левый")
        
        # Отрисовка правого джойстика
        self._draw_joystick(painter, self.right_joystick_pos, QPoint(self.width() - 100, self.height() - 100), "Правый")
        
        painter.end()
        
        # Устанавливаем pixmap в overlay label
        self.joystick_overlay.setPixmap(overlay_pixmap)

    def _draw_joystick(self, painter, position, center, label):
        """Отрисовка одного джойстика"""
        # Окружность основания
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, 30, 30)
        
        # Преобразуем координаты из диапазона [-100,100] в [-30,30]
        dot_x = center.x() + position.x() * 30 / 100
        dot_y = center.y() + position.y() * 30 / 100
        dot_pos = QPoint(int(dot_x), int(dot_y))
        
        # Позиция джойстика
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
            # Для BeagleBone используем специальные параметры
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
        """Создание захвата видео для BeagleBone с разными вариантами"""
        # Попробуем несколько вариантов подключения
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            camera = cv2.VideoCapture(url, backend)
            if camera.isOpened():
                return camera
            if camera:
                camera.release()
        
        # Если ничего не сработало, пробуем стандартный метод
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
        """Сделать снимок, сохранить его и классифицировать с помощью нейросети"""
        if self.control_mode != "Ручной":
            QMessageBox.warning(
                self, 
                "Ошибка", 
                "Классификация доступна только в ручном режиме управления!"
            )
            return
            
        try:
            # Создаем папки если их нет
            os.makedirs("Foto", exist_ok=True)
            os.makedirs("data/train", exist_ok=True)
            
            # Делаем снимок
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
            
            print(f"Изображение сохранено: {filename}")
            
            # Загружаем модель
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.NUM_CLASSES)
            model.load_state_dict(torch.load("model_weights.pth"))
            model.to(self.device)
            model.eval()
            
            # Классифицируем изображение
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image = Image.open(filename)
            image = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = self.CLASS_NAMES[predicted[0]]
            
            QMessageBox.information(
                self, 
                "Результат классификации", 
                f"Изображение: {filename}\nДефект: {prediction}"
            )
            
        except Exception as e:
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
    try:
        # Очищаем консоль перед запуском приложения
        clear_console()
        
        app = QApplication(sys.argv)
        window = GameWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")