import sys
import os
import random
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QMessageBox,
                             QHBoxLayout, QProgressBar, QTabWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image

class DefectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Detection App")
        self.setGeometry(100, 100, 900, 700)
        
        # Конфигурация
        self.DATA_DIR = "data"
        self.BATCH_SIZE = 32
        self.EPOCHS = 20
        self.LEARNING_RATE = 0.001
        self.CLASS_NAMES = ["Норма", "Ржавчина", "Трещины", "Био-дефекты"]
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализация UI
        self.init_ui()
        
        # Загрузка модели (если существует)
        self.model = None
        self.load_model()
    
    def init_ui(self):
        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.detection_tab = QWidget()
        self.training_tab = QWidget()
        
        self.init_detection_tab()
        self.init_training_tab()
        
        self.tabs.addTab(self.detection_tab, "Обнаружение дефектов")
        self.tabs.addTab(self.training_tab, "Обучение модели")
        
        main_layout.addWidget(self.tabs)
        central_widget.setLayout(main_layout)
    
    def init_detection_tab(self):
        layout = QVBoxLayout()
        
        # Элементы интерфейса для обнаружения
        self.title_label = QLabel("Обнаружение дефектов")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        
        self.result_label = QLabel("Результат: ")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px;")
        
        self.confidence_label = QLabel("Уверенность: ")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setStyleSheet("font-size: 14px;")
        
        # Добавление элементов в layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.select_button)
        
        self.detection_tab.setLayout(layout)
    
    def init_training_tab(self):
        layout = QVBoxLayout()
        
        # Элементы интерфейса для обучения
        self.train_title_label = QLabel("Обучение модели")
        self.train_title_label.setAlignment(Qt.AlignCenter)
        self.train_title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        self.dataset_status_label = QLabel("Статус датасета: Не проверен")
        self.model_status_label = QLabel("Статус модели: Не загружена")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.train_button = QPushButton("Обучить модель")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setStyleSheet("font-size: 14px;")
        
        self.check_dataset_button = QPushButton("Проверить датасет")
        self.check_dataset_button.clicked.connect(self.check_dataset)
        self.check_dataset_button.setStyleSheet("font-size: 14px;")
        
        self.split_data_button = QPushButton("Разделить данные")
        self.split_data_button.clicked.connect(self.split_data)
        self.split_data_button.setStyleSheet("font-size: 14px;")
        
        # Горизонтальный layout для кнопок
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.train_button)
        buttons_layout.addWidget(self.check_dataset_button)
        buttons_layout.addWidget(self.split_data_button)
        
        # Добавление элементов в layout
        layout.addWidget(self.train_title_label)
        layout.addWidget(self.dataset_status_label)
        layout.addWidget(self.model_status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(buttons_layout)
        
        self.training_tab.setLayout(layout)
    
    def load_model(self):
        """Загрузка предварительно обученной модели"""
        model_path = "defect_detection_model.pth"
        
        if os.path.exists(model_path):
            try:
                model = models.resnet18(pretrained=True)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.NUM_CLASSES)
                model = model.to(self.device)
                
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                self.model = model
                self.model_status_label.setText("Статус модели: Загружена")
                print("Модель успешно загружена")
            except Exception as e:
                self.model_status_label.setText("Статус модели: Ошибка загрузки")
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {str(e)}")
        else:
            self.model_status_label.setText("Статус модели: Не найдена")
    
    def select_image(self):
        """Выбор изображения через проводник"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, image_path):
        """Обработка и классификация изображения"""
        if self.model is None:
            QMessageBox.warning(self, "Предупреждение", "Модель не загружена. Пожалуйста, обучите модель сначала.")
            return
            
        try:
            # Отображение изображения
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(), 
                Qt.KeepAspectRatio))
            
            # Классификация
            prediction, confidence = self.classify_image(image_path)
            
            # Обновление результатов
            self.result_label.setText(f"Результат: {prediction}")
            self.confidence_label.setText(f"Уверенность: {confidence:.2f}%")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки изображения: {str(e)}")
    
    def classify_image(self, image_path):
        """Классификация изображения"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Открываем изображение и конвертируем RGBA в RGB
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        return self.CLASS_NAMES[predicted[0]], prob[predicted[0]].item()
    
    def create_dataset_structure(self):
        """Создает правильную структуру папок если её нет"""
        os.makedirs(os.path.join(self.DATA_DIR, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "test"), exist_ok=True)
        
        for split in ["train", "test"]:
            for class_name in self.CLASS_NAMES:
                os.makedirs(os.path.join(self.DATA_DIR, split, class_name), exist_ok=True)
    
    def split_data(self):
        """Разделение данных на train/test"""
        source_dir = QFileDialog.getExistingDirectory(self, "Выберите папку с исходными данными")
        if not source_dir:
            return
            
        try:
            self.create_dataset_structure()
            
            for class_name in self.CLASS_NAMES:
                class_path = os.path.join(source_dir, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                random.shuffle(files)
                split_idx = int(len(files) * 0.8)  # 80% train, 20% test
                
                # Копируем в train
                for f in files[:split_idx]:
                    src = os.path.join(class_path, f)
                    dst = os.path.join(self.DATA_DIR, "train", class_name, f)
                    shutil.copy2(src, dst)
                
                # Копируем в test
                for f in files[split_idx:]:
                    src = os.path.join(class_path, f)
                    dst = os.path.join(self.DATA_DIR, "test", class_name, f)
                    shutil.copy2(src, dst)
            
            QMessageBox.information(self, "Успех", "Данные успешно разделены на train и test наборы!")
            self.check_dataset()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при разделении данных: {str(e)}")
    
    def check_dataset(self):
        """Проверяет структуру датасета"""
        try:
            valid = True
            message = "Статус датасета:\n"
            
            for split in ["train", "test"]:
                split_path = os.path.join(self.DATA_DIR, split)
                if not os.path.exists(split_path):
                    message += f"\nПапка {split} не найдена!"
                    valid = False
                    continue
                
                existing_classes = os.listdir(split_path)
                missing_classes = set(self.CLASS_NAMES) - set(existing_classes)
                
                if missing_classes:
                    message += f"\nВ папке {split} отсутствуют классы: {', '.join(missing_classes)}"
                    valid = False
                
                message += f"\n\n{split}:"
                for class_name in self.CLASS_NAMES:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        message += f"\n  {class_name}: {len(images)} изображений"
                    else:
                        message += f"\n  {class_name}: папка не найдена"
            
            if valid:
                self.dataset_status_label.setText("Статус датасета: Корректный")
                QMessageBox.information(self, "Проверка датасета", message)
            else:
                self.dataset_status_label.setText("Статус датасета: Ошибка")
                QMessageBox.warning(self, "Проверка датасета", message)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при проверке датасета: {str(e)}")
    
    def start_training(self):
        """Запуск обучения в отдельном потоке"""
        if not hasattr(self, 'train_loader') or not hasattr(self, 'test_loader'):
            try:
                self.train_loader, self.test_loader, _ = self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {str(e)}")
                return
        
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = TrainingThread(self.model, self.train_loader, self.test_loader, 
                                   self.NUM_CLASSES, self.device, self.EPOCHS, self.LEARNING_RATE)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.training_finished.connect(self.training_finished)
        self.worker.start()
    
    def update_progress(self, epoch, total_epochs, loss, accuracy):
        """Обновление прогресс бара и информации"""
        progress = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress)
        self.model_status_label.setText(
            f"Статус модели: Обучение (Эпоха {epoch}/{total_epochs}), Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    def training_finished(self, model, test_accuracy):
        """Завершение обучения"""
        self.model = model
        self.progress_bar.setValue(100)
        self.train_button.setEnabled(True)
        self.model_status_label.setText(f"Статус модели: Обучена (Точность: {test_accuracy:.2f}%)")
        
        # Сохранение модели
        torch.save(model.state_dict(), "defect_detection_model.pth")
        QMessageBox.information(self, "Обучение завершено", 
                               f"Модель успешно обучена!\nТочность на тестовых данных: {test_accuracy:.2f}%")
    
    def load_data(self):
        """Загрузка и предобработка данных"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.DATA_DIR, "train"),
            transform=train_transform
        )
        
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.DATA_DIR, "test"),
            transform=test_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        return train_loader, test_loader, train_dataset.classes

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, int, float, float)  # epoch, total_epochs, loss, accuracy
    training_finished = pyqtSignal(object, float)  # model, test_accuracy
    
    def __init__(self, model, train_loader, test_loader, num_classes, device, epochs, learning_rate):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        if model is None:
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.model = self.model.to(device)
        else:
            self.model = model
    
    def run(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Обучение
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            
            self.progress_updated.emit(epoch + 1, self.epochs, epoch_loss, epoch_acc)
        
        # Тестирование
        test_accuracy = self.test_model()
        self.training_finished.emit(self.model, test_accuracy)
    
    def test_model(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DefectDetectionApp()
    window.show()
    sys.exit(app.exec_())