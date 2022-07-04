import time, board, busio
import numpy as np
import adafruit_mlx90640
import imutils
import cv2
import socketio
import threading
import matplotlib as mpl

from imutils.video import VideoStream
from scipy import ndimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib import patches

vs = VideoStream(src=0).start()
client = socketio.Client()
i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ

mlx_interp_val = 10
mlx_shape = (24, 32)
mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)

# load our serialized face detector model from disk
mpl.rcParams["toolbar"] = 'None'
NAMESPACES = "/raspberrypi"
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

from detect_mask_video import detect_and_predict_mask

frame = np.zeros(mlx_shape[0] * mlx_shape[1])

# Cấu hình matplotlib
fig = plt.figure(figsize=(8, 6)) 
fig.canvas.set_window_title("Giám sát nhiệt độ")
fig.canvas.toolbar_visible = False

ax1 = fig.add_subplot(1, 2, 1)                  
ax2 = fig.add_subplot(1, 2, 2)

fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)

therm1 = ax1.imshow(
  np.zeros(mlx_interp_shape),
  interpolation = 'none',
  cmap = plt.cm.bwr,
  vmin = 25,
  vmax = 45
)

therm2 = ax2.imshow(
  np.zeros(mlx_interp_shape),
  interpolation = 'none',
  cmap = plt.cm.bwr,
  vmin = 25,
  vmax = 45
)

fig.canvas.draw()
ax_background = fig.canvas.copy_from_bbox(ax1.bbox)
fig.show()

def get_temp(x, y, w, h):
  mlx.getFrame(frame)
  data_array = np.fliplr(
    np.reshape(frame, mlx_shape)
  )
  
  vmin = round(np.min(data_array), 2)
  vmax = round(np.max(data_array), 2)
  
  data_array = ndimage.zoom(data_array, mlx_interp_val)
  return round(np.max(data_array[y:y+h, x:x+w]), 2), data_array, vmin, vmax

img = None
def run_mask():
  global img
  (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
  
  for (box, pred) in zip(locs, preds):
    (x, y, endX, endY) = box
    (mask, withoutMask) = pred
    
    w = endX - x
    h = endY - y
    
    has_mask = mask > withoutMask
    temperature, xy, vmin, vmax = get_temp(x, y, w, h)
    
    fig.canvas.restore_region(ax_background)
    
    therm2.set_array(xy)
    therm2.set_clim(vmin=vmin, vmax=vmax)
    ax2.draw_artist(therm2)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    msk = "Không"
    if has_mask:
      msk = "Có"
      
    msg = "Nhiệt độ = {}. {} đeo khẩu trang".format(temperature, msk)
    
    bbox_setting = {
      "facecolor" : "yellow",
      "alpha" : 1,
      "pad" : 20
    }
    
    if temperature > 38:
      bbox_setting["facecolor"] = "red"
        
    therm1.set_data(
     cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    ax1.text(250, -100, msg, bbox=bbox_setting)
    
    fig.canvas.blit(ax1.bbox)
    fig.canvas.flush_events()      
    
    break
  
def _run_cli():
  client.on("has_new_form", handler=run_mask, namespaces=NAMESPACES)
  client.connect("http://192.168.1.20:3000", namespaces=NAMESPACES)

threading.Thread(name="_run_cli", daemon=True, target=_run_cli) \
  .start()

plt.ion()

while True:
  img = vs.read()
  img = imutils.resize(img, width = mlx_interp_shape[1], height = mlx_interp_shape[0])
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
    plt.show()
    cv2.destroyAllWindows()
    break
