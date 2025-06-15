import sys
import os
import cv2
import requests
import datetime
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn  # Добавьте эту строку в секцию импортов
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMenuBar, 
                             QMenu, QAction, QVBoxLayout, QWidget, QActionGroup,
                             QInputDialog, QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QIcon
from PyQt5.QtCore import Qt, QTimer, QPointF

# Настройки окружения для Steam Deck
os.environ.update({
    "QT_QUICK_BACKEND": "software",
    "QT_QPA_PLATFORM": "xcb"
})

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.control_mode = "Ручной"  # Изменено на ручной режим по умолчанию
        self.camera = None
        self.current_source = None
        self.timer = QTimer()
        
        self._setup_ui()
        self._setup_connections()
        self._initialize_app()

    # Конфигурация нейросети
    NUM_CLASSES = 3  # Количество классов дефектов
    CLASS_NAMES = ["Норма", "Трещина", "Коррозия"]  # Пример классификации
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        self._create_menu()

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
            QMenu { background-color: black; color: white; border: 1px solid #444; }
            QMenu::item:selected { background-color: #333; }
            QAction { color: white; }
        """)

    def _create_menu(self):
        """Создание меню приложения"""
        menubar = self.menuBar()
        
        bg_menu = menubar.addMenu('Фон')
        bg_menu.addAction(self._create_action('Статичный фон', self.set_static_background))
        bg_menu.addAction(self._create_action('Локальная камера', self.set_camera_background))
        bg_menu.addAction(self._create_action('BeagleBone Stream', self.set_beaglebone_stream))
        
        control_menu = menubar.addMenu('Управление')
        self.control_group = QActionGroup(self)
        
        modes = [('Автоматический', "Автоматический"), ('Ручной', "Ручной")]
        for text, mode in modes:
            action = self._create_action(text, lambda checked, m=mode: self.set_control_mode(m), True)
            self.control_group.addAction(action)
            control_menu.addAction(action)
            if mode == "Ручной":  # Выбираем ручной режим по умолчанию
                action.setChecked(True)

        neural_network = menubar.addMenu('Нейросеть')
        neural_action = self._create_action("Распознать дефект", self.set_neural_network)
        neural_network.addAction(neural_action)
        
        menubar.addAction(self._create_action('Выход', self.close))

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
        super().resizeEvent(event)
        self.draw_crosshair()

    def set_static_background(self):
        """Установка статичного фона"""
        self._stop_camera()
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#020061"))
        self.background_label.setPixmap(pixmap)
        self.draw_crosshair()

    def set_camera_background(self):
        """Запуск видеопотока с локальной камеры"""
        self._start_video_stream(0)

    def set_beaglebone_stream(self):
        """Подключение к BeagleBone видеопотоку"""
        ip, ok = QInputDialog.getText(
            self, 'BeagleBone Stream', 'Введите IP адрес BeagleBone:', 
            QLineEdit.Normal, '192.168.1.100'
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
        self._stop_camera()
        event.accept()

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = GameWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")