import cv2

# RTSP流地址
rtsp_url = 'rtsp://192.168.200.200/live'

# 创建VideoCapture对象
cap = cv2.VideoCapture(1)

# 检查是否成功连接到RTSP流
if not cap.isOpened():
    print("无法连接到RTSP流")

while True:
    # 从RTSP流中捕获帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("无法从流中获取帧（流可能已结束）")
        continue
    # 显示帧
    cv2.imshow('RTSP Video Stream', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
