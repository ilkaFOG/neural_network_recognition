import cv2
import threading
from flask import Flask, Response
import time
import logging
from queue import Queue, Empty

app = Flask(__name__)

class VideoStreamer:
    def __init__(self):
        self.camera = None
        self.frame_queue = Queue(maxsize=2)  # Буфер на 2 кадра
        self.running = False
        self.thread = None
        self.last_frame_time = 0
        self.frame_interval = 0.1  # 10 FPS (0.1 сек между кадрами)
        
    def initialize_camera(self):
        """Инициализация камеры с улучшенными параметрами"""
        # Пробуем разные варианты открытия камеры
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            self.camera = cv2.VideoCapture(0, backend)
            if self.camera.isOpened():
                # Оптимальные настройки для BeagleBone
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 10)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                return True
            if self.camera:
                self.camera.release()
        return False
        
    def start(self):
        if not self.initialize_camera():
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()
        return True
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()
            
    def update_frame(self):
        """Поток для захвата кадров с обработкой ошибок"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.01)
                    continue
                    
                ret, frame = self.camera.read()
                if not ret:
                    logging.warning("Ошибка чтения кадра, попытка переподключения...")
                    self.camera.release()
                    time.sleep(1)
                    if not self.initialize_camera():
                        time.sleep(5)
                        continue
                    continue
                    
                # Сжимаем кадр для уменьшения размера
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                # Очищаем очередь, если она переполнена
                while self.frame_queue.qsize() > 1:
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break
                        
                self.frame_queue.put(jpeg.tobytes())
                self.last_frame_time = current_time
                
            except Exception as e:
                logging.error(f"Ошибка в потоке захвата: {str(e)}")
                time.sleep(1)
                
    def get_frame(self):
        """Получение последнего доступного кадра"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

video_streamer = VideoStreamer()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = video_streamer.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
                
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return Response("OK", status=200)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    if not video_streamer.start():
        logging.error("Не удалось инициализировать камеру!")
        exit(1)
        
    try:
        ip = open('/etc/hostname').read().strip()
        logging.info(f"Сервер запущен на http://{ip}:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        video_streamer.stop()