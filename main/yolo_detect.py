import datetime
import threading
import cv2
import os
import numpy as np
from multiprocessing import Process, JoinableQueue, current_process
import torch
from ultralytics import YOLO  # Import YOLOv8 từ thư viện ultralytics
import logging
import time  # Thêm thư viện time để đo thời gian

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn đến mô hình YOLOv8 đã được huấn luyện cho phát hiện khuôn mặt
model_name = "yolov8n-face.pt"  # Đảm bảo bạn đã tải mô hình này

video_folder_path = "./yolo-video"

lock = threading.Lock()  # Khóa để đồng bộ hóa ghi file

def write_image(task_id,video_file,current_second, frame,index_face,bbox):
    """
    Hàm để lưu frame vào thư mục /tasks/<task_id>/frames/<second>.png
    """
        # Lưu khuôn mặt vào /tasks/<task_id>/faces/<second>_<index>.png
    faces_dir = os.path.join('tasks', task_id, 'faces')
    folder_path_task = os.path.join('tasks', task_id)
    output_file = os.path.join(folder_path_task, 'output.txt')
    # os.makedirs(faces_dir, exist_ok=True)
    face_filename = os.path.join(faces_dir, f"{int(current_second)}_{index_face}.png")
    cv2.imwrite(face_filename, frame)
    with lock:  # Sử dụng khóa để đồng bộ hóa ghi file
        with open(output_file, 'a') as f:
            f.write(f'Video: {video_file}, Face: {face_filename}, Time: {current_second:.2f} s, BBox: {bbox}\n')

# Hàm để phát hiện khuôn mặt trong mỗi process
def detect_faces(detection_queue, model_name, gpu_id):
    """
    Hàm để xử lý các frame từ hàng đợi cho việc phát hiện khuôn mặt.
    Mỗi process sẽ lấy một frame từ hàng đợi, thực hiện phát hiện và lưu kết quả.
    """
    # Set the CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        if torch.cuda.is_available():
            logging.info(f"{current_process().name}: Sử dụng GPU để phát hiện khuôn mặt.")
        else:
            logging.info(f"{current_process().name}: Không tìm thấy GPU, sẽ sử dụng CPU.")
        # Tải mô hình YOLOv8 trong mỗi process để tránh xung đột
        model = YOLO(model_name).cuda()
        logging.info(f"{current_process().name}: Đã tải mô hình YOLOv8 thành công.")
    except Exception as e:
        logging.error(f"{current_process().name}: Lỗi khi tải mô hình YOLOv8: {e}")
        return

    while True:
        item = detection_queue.get()
        if item is None:
            # Tín hiệu để kết thúc xử lý cho process này
            logging.info(f"{current_process().name}: Nhận tín hiệu kết thúc.")
            detection_queue.task_done()
            break
        task_id,video_file, current_time_sec, frame = item
        time_start = datetime.datetime.now().timestamp()
        try:
            # Thay đổi kích thước frame về 640x640 như yêu cầu của YOLOv8
            resized_frame = cv2.resize(frame, (640, 640))

            # Chuyển đổi frame sang tensor và đưa lên GPU
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float().cuda() / 255.0

            # Thực hiện phát hiện
            detections = model(frame_tensor.unsqueeze(0))  # Thêm batch dimension

            face_detected = False

            for result in detections:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for i,detection in enumerate(result.boxes):
                        if detection.cls == 0:  # Giả định lớp 0 là khuôn mặt
                            x1, y1, x2, y2 = map(int, detection.xyxy[0])

                            # Điều chỉnh tọa độ về frame gốc
                            x1 = int(x1 * (frame.shape[1] / 640))
                            x2 = int(x2 * (frame.shape[1] / 640))
                            y1 = int(y1 * (frame.shape[0] / 640))
                            y2 = int(y2 * (frame.shape[0] / 640))

                            # Cắt khuôn mặt
                            face_image = frame[y1:y2, x1:x2]
                            if face_image.size == 0:
                                continue

                            # run in thread
                            file_write = threading.Thread(target=write_image, args=(task_id,video_file,current_time_sec, face_image,i,f"({x1},{y1}),({x2},{y2})"))
                            file_write.start()

        except Exception as e:
            logging.error(f"Lỗi khi xử lý frame tại giây {current_time_sec} cho task_id {task_id}: {e}")

        finally:
            detection_queue.task_done()
            time_end = datetime.datetime.now().timestamp()
            processing_time = time_end - time_start
            logging.info(f"{current_process().name}: Xử lý frame tại giây {current_time_sec} cho task_id {task_id} trong {processing_time:.2f} giây. queue size: {detection_queue.qsize()}")

# Hàm để tách frame từ video bằng OpenCV và đưa frame vào hàng đợi
def extract_frames(video_path, task_id, detection_queue):
    """
    Hàm để tách frame từ video bằng OpenCV và đặt frame vào hàng đợi để xử lý.
    Mỗi process sẽ xử lý một video riêng biệt.
    """
    try:
        start_time = datetime.datetime.now()
        faces_dir = os.path.join('tasks', task_id, 'faces')
        os.makedirs(faces_dir, exist_ok=True)
        # Mở video bằng OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Không thể mở video: {video_path}")
            return

        # Lấy FPS của video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Số frame cần bỏ qua để lấy 1 frame mỗi giây

        logging.info(f"FPS của video {video_path}: {fps:.2f}")
        frame_count = 0
        current_time_sec = 0
        time_start = datetime.datetime.now().timestamp()
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Video đã kết thúc

            if frame_count % frame_interval == 0:
                # Đưa frame vào hàng đợi để xử lý
                time_end = datetime.datetime.now().timestamp()
                logging.info(f"Đã đọc frame tại giây {current_time_sec} cho task_id {task_id}. Thời gian đọc: {time_end - time_start:.4f} giây.")
                detection_queue.put((task_id,video_path, current_time_sec, frame))
                current_time_sec += 1

            frame_count += 1

        cap.release()
        # Ghi lại thời gian xử lý cho video
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"Hoàn thành xử lý video: {video_path}. Thời gian xử lý: {processing_time:.2f} giây.")

    except Exception as e:
        logging.error(f"Lỗi khi xử lý video {video_path}: {e}")

# Hàm chính
def main():
    """
    Hàm chính để xử lý tất cả các video trong thư mục.
    """
    time_start = datetime.datetime.now().timestamp()

    # Kiểm tra nếu mô hình tồn tại
    if not os.path.exists(model_name):
        logging.error(f"Mô hình YOLOv8 không tồn tại tại: {model_name}")
        logging.info("Vui lòng tải mô hình YOLOv8-Face và đặt nó tại đường dẫn trên.")
        return

    # Lấy danh sách tất cả các video trong thư mục
    video_files = [f for f in os.listdir(video_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]

    if not video_files:
        logging.warning("Không tìm thấy video nào trong thư mục.")
        return

    num_detection_processes = 4  # Số lượng process cho phát hiện khuôn mặt
    detection_queue = JoinableQueue()

    # Khởi tạo các process phát hiện khuôn mặt
    detection_processes = []
    for i in range(num_detection_processes):
        p = Process(target=detect_faces, args=(detection_queue, model_name, i), name=f"Detection-Process-{i+1}")
        p.start()
        detection_processes.append(p)
        logging.info(f"Đã khởi động process phát hiện khuôn mặt {i+1}/{num_detection_processes}.")

    # Khởi tạo các process tách frame (mỗi process xử lý một video)
    extraction_processes = []
    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        task_id = os.path.splitext(video_file)[0]  # Sử dụng tên file không có phần mở rộng làm task_id

        # Bắt đầu process tách frame cho video này
        p = Process(target=extract_frames, args=(video_path, task_id, detection_queue), name=f"Extractor-{task_id}")
        p.start()
        extraction_processes.append(p)
        logging.info(f"Đã khởi động process tách frame cho video: {video_file}")

    # Chờ cho đến khi tất cả các process tách frame hoàn thành
    for p in extraction_processes:
        p.join()
        logging.info(f"{p.name} đã hoàn thành.")

    # Chờ cho đến khi tất cả các frame được xử lý
    detection_queue.join()

    # Sau khi tất cả các frame đã được xử lý, gửi tín hiệu kết thúc cho các process phát hiện
    for _ in detection_processes:
        detection_queue.put(None)

    # Chờ cho các process phát hiện hoàn thành
    for p in detection_processes:
        p.join()
        logging.info(f"{p.name} đã kết thúc.")

    time_end = datetime.datetime.now().timestamp()
    total_processing_time = time_end - time_start

    logging.info(f"Đã hoàn thành xử lý tất cả các video. Tổng thời gian: {total_processing_time:.2f} giây.")

if __name__ == "__main__":
    main()

