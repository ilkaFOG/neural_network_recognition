import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from PIL import Image

# Конфигурация
DATA_DIR = "data"  # Папка с данными (должна содержать подпапки train и test)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Укажите ваше количество классов
CLASS_NAMES = ["Класс 1", "Класс 2", "Класс 3"]  # Замените на свои названия

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Загрузка и предобработка данных
def load_data():
    # Трансформации для обучающих данных (аугментация)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Трансформации для тестовых данных (без аугментации)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Загрузка датасетов
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=test_transform
    )

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

# Модель нейронной сети (на основе ResNet18)
def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Заменяем последний слой
    model = model.to(device)
    return model

# Обучение модели
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Тестирование модели
def test_model(model, test_loader):
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
    print(f"Точность на тестовых данных: {accuracy:.2f}%")

# Классификация нового изображения
def classify_image(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted[0]]

# Основная функция
def main():
    # Загрузка данных
    train_loader, test_loader, class_names = load_data()
    print(f"Классы: {class_names}")

    # Создание модели
    model = create_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучение
    print("Начинаем обучение...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    # Тестирование
    print("\nТестирование...")
    test_model(model, test_loader)

    # Сохранение модели
    torch.save(model.state_dict(), "model_weights.pth")
    print("Модель сохранена как 'model_weights.pth'")

    # Пример классификации нового изображения
    while True:
        image_path = input("\nВведите путь к изображению (или 'exit' для выхода): ")
        if image_path.lower() == 'exit':
            break
        try:
            prediction = classify_image(model, image_path, class_names)
            print(f"Результат классификации: {prediction}")
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()