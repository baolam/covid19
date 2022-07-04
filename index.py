import time, board, busio
import numpy as np
import adafruit_mlx90640
import imutils
import cv2
import socketio
import threading

from imutils.video import VideoStream
from scipy import ndimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

vs = VideoStream(src=0).start()
client = socketio.Client()
i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

mlx_interp_val = 15
mlx_shape = (24, 32)
mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

from detect_mask_video import detect_and_predict_mask

def get_temp(x, y, w, h, frame):
  temp = mlx.getFrame(frame)
  data_array = np.fliplr(
    np.reshape(temp, mlx_shape)
  )
  data_array = ndimage.zoom(data_array, mlx_interp_val)
  return round(np.max(data_array[y:y+h, x:x+w]), 2), data_array
  
def _run_cli():
  client.connect("")

threading.Thread(name="_run_cli", daemon=True, target=_run_cli) \
  .start()

while True:
  frame = vs.read()
  frame = imutils.resize(frame, width = mlx_interp_shape[0], height = mlx_interp_shape[1])
  (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
  
  for (box, pred) in zip(locs, preds):
    (startX, startY, endX, endY) = box
    (mask, withoutMask) = pred
    
    has_mask = mask > withoutMask
    temperature, xy = get_temp(x, y, w, h, frame)
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("FACE", frame)
    cv2.imshow("TEMP", xy)
    
    break
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break