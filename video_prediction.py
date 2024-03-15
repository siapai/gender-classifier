import cv2
import youtube_dl
import insightface
from insightface.app import FaceAnalysis
from keras.models import load_model
import numpy as np
import onnxruntime
import time as t

app = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'],
                   allowed_modules=['detection'])  # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.7)

onnx_model_path = 'models/inception_v3_categorical_epoch_20.onnx'
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# URL YouTube
url = "https://www.youtube.com/watch?v=ob-qmfvnQVo&ab_channel=As%2FIs"
# url = "https://www.youtube.com/watch?v=cvb49-Csq1o&t=61s&ab_channel=Apple"
# url = "https://www.youtube.com/watch?v=4-7jSoINyq4&ab_channel=Apple"


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('files/video/model_1.mp4', fourcc, 24.0, (640, 360))


def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        # 'format': 'best',
        'nocheckcertificate': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
    return filename


video_filename = download_video(url)

cap = cv2.VideoCapture(video_filename)

scale_factor = 1.8
num_frames_to_process = 5
current_frame = 0
processing_frame = 0
latency = 0
fps = 0
pps = 0
confidence = 0

detection = None

prediction_results = []
start_time = t.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1

    frame_copy = frame.copy()
    print(frame_copy.shape)

    result = app.get(frame_copy)

    for id, res in enumerate(result[:1]):
        x1, y1, x2, y2 = res['bbox'].astype(int)

        is_present = any(id == obj["id"] for obj in prediction_results)

        if not is_present:
            prediction_results.append(
                {"id": id, "gender": None}
            )

        face_width = x2 - x1
        face_height = y2 - y1

        new_face_width = int(face_width * scale_factor)
        new_face_height = int(face_height * scale_factor * 0.8)

        new_x1 = max(0, x1 - int((new_face_width - face_width) // 2))
        new_y1 = max(0, y1 - int((new_face_height - face_height) // 2))

        new_x2 = min(frame.shape[1], new_x1 + new_face_width)
        new_y2 = min(frame.shape[0], new_y1 + new_face_height)

        face_image = frame[new_y1:new_y2, new_x1:new_x2]

        if current_frame % num_frames_to_process == 0:
            processing_frame += 1
            end_time = t.time()
            elapsed_time = end_time - start_time
            pps = processing_frame / elapsed_time
            fps = current_frame / elapsed_time

            target_size = (299, 299)
            resized_face = cv2.resize(face_image, target_size)
            img_array = resized_face.astype("float32")
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.  # N
            start_lat = t.time()
            predictions = session.run([output_name], {input_name: img_array})[0]
            end_lat = t.time()
            latency = end_lat - start_lat
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            class_labels = ['FEMALE', 'MALE']
            gender = class_labels[predicted_class]
            new_result = {"id": id, "gender": gender}

            for item in prediction_results:
                if item["id"] == id:
                    item.update(new_result)
                    break

        cv2.rectangle(frame_copy, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0))
        result = prediction_results[id]
        if result is not None:
            if result["gender"] is not None:
                color = (255, 0, 0)
                if result["gender"] == "FEMALE":
                    color = (0, 0, 255)
                cv2.putText(frame_copy, f'{result["gender"]} ', (new_x1 + 15, new_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            color, 2)
    cv2.putText(frame_copy, f"FPS: {fps:.2f} | PPS: {pps:.2f} | SKIP: {num_frames_to_process} | LAT: {latency:.4f}s | "
                            f"CONF: {confidence:.4f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # out.write(frame_copy)
    cv2.imshow('Video', frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
