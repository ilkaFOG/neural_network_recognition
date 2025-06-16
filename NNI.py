import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os
import random
from PIL import Image
import shutil

# Конфигурация
DATA_DIR = "data"  # Основная папка с данными
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CLASS_NAMES = ["Ржавчина", "Трещины", "Био-дефекты"]  # Наши классы
NUM_CLASSES = len(CLASS_NAMES)

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

def create_dataset_structure():
    """Создает правильную структуру папок если её нет"""
    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "test"), exist_ok=True)
    
    for split in ["train", "test"]:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(DATA_DIR, split, class_name), exist_ok=True)

def manual_train_test_split(source_dir, test_ratio=0.2):
    """Ручное разделение данных на train/test без sklearn"""
    for class_name in CLASS_NAMES:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        split_idx = int(len(files) * (1 - test_ratio))
        
        # Копируем в train
        for f in files[:split_idx]:
            src = os.path.join(class_path, f)
            dst = os.path.join(DATA_DIR, "train", class_name, f)
            shutil.copy2(src, dst)
        
        # Копируем в test
        for f in files[split_idx:]:
            src = os.path.join(class_path, f)
            dst = os.path.join(DATA_DIR, "test", class_name, f)
            shutil.copy2(src, dst)

def check_dataset():
    """Проверяет структуру датасета"""
    print("\nПроверка структуры датасета:")
    for split in ["train", "test"]:
        split_path = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_path):
            print(f"Ошибка: Папка {split} не найдена!")
            return False
        
        existing_classes = os.listdir(split_path)
        missing_classes = set(CLASS_NAMES) - set(existing_classes)
        
        if missing_classes:
            print(f"Ошибка: В папке {split} отсутствуют классы: {missing_classes}")
            return False
            
        print(f"\n{split}:")
        for class_name in CLASS_NAMES:
            class_path = os.path.join(split_path, class_name)
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {class_name}: {len(images)} изображений")
            if len(images) == 0:
                print(f"Предупреждение: Нет изображений в {class_path}")
    
    return True

def load_data():
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
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

def create_model(load_if_exists=True):
    """Создание или загрузка модели"""
    model_path = "defect_detection_model.pth"
    
    # Инициализация модели
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)
    
    # Попытка загрузить сохраненную модель
    if load_if_exists and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"\nЗагружена предварительно обученная модель из {model_path}")
            return model
        except Exception as e:
            print(f"\nОшибка при загрузке модели: {e}. Будет создана новая модель.")
    
    return model

def train_model(model, train_loader, criterion, optimizer):
    """Обучение модели"""
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

def test_model(model, test_loader):
    """Тестирование модели"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nТочность на тестовых данных: {accuracy:.2f}%")

def classify_image(model, image_path):
    """Классификация одного изображения с обработкой RGBA"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Открываем изображение и конвертируем RGBA в RGB
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            # Создаем белый фон и накладываем изображение
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Используем альфа-канал как маску
            image = background
        
        image = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        return CLASS_NAMES[predicted[0]], prob[predicted[0]].item()
    
    except Exception as e:
        raise Exception(f"Ошибка обработки изображения: {str(e)}")

def main():
    # Создаем структуру папок если её нет
    create_dataset_structure()
    
    # Проверяем датасет
    if not check_dataset():
        print("\nОшибка: Датасет не соответствует требуемой структуре!")
        print(f"Требуемая структура: data/{{train,test}}/{{{'/'.join(CLASS_NAMES)}}}")
        print("Пожалуйста, организуйте данные правильно и попробуйте снова.")
        return

    # Загрузка данных
    print("\nЗагрузка данных...")
    train_loader, test_loader, loaded_classes = load_data()
    print(f"Загруженные классы: {loaded_classes}")

    # Проверка соответствия классов
    if set(loaded_classes) != set(CLASS_NAMES):
        print("Ошибка: Загруженные классы не соответствуют ожидаемым!")
        return

    # Создание/загрузка модели
    print("\nИнициализация модели...")
    model = create_model(load_if_exists=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Проверяем, нужно ли обучать модель
    model_path = "defect_detection_model.pth"
    if not os.path.exists(model_path):
        # Обучение
        print("\nНачало обучения...")
        train_model(model, train_loader, criterion, optimizer)

        # Тестирование
        print("\nТестирование модели...")
        test_model(model, test_loader)

        # Сохранение модели
        torch.save(model.state_dict(), model_path)
        print(f"\nМодель сохранена как '{model_path}'")
    else:
        # Только тестирование если модель была загружена
        print("\nТестирование загруженной модели...")
        test_model(model, test_loader)

    # Пример классификации
    while True:
        image_path = input("\nВведите путь к изображению для классификации (или 'exit' для выхода): ")
        if image_path.lower() == 'exit':
            break
            
        if not os.path.exists(image_path):
            print("Ошибка: Файл не найден!")
            continue
            
        try:
            prediction, confidence = classify_image(model, image_path)
            print(f"Результат: {prediction} (уверенность: {confidence:.2f}%)")
        except Exception as e:
            print(f"Ошибка при классификации: {str(e)}")

if __name__ == "__main__":
    main()