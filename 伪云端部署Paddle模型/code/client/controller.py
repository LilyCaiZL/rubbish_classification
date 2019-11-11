from time import sleep
import RPi.GPIO as GPIO
import base64
import requests
import cv2
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

def setServoAngle(servo, angle):
    pwm = GPIO.PWM(servo, 50)
    pwm.start(0)
    dutyCycle = angle / 18 + 3
    pwm.ChangeDutyCycle(angle)
    sleep(0.3)
    pwm.stop()
while(1):
    ret,frame = cap.read()
    sleep(3)
    cv2.imwrite("01.jpg",frame)
    img = cv2.imread("01.jpg")
    cap.release()
    break
with open("01.jpg","rb") as f:
    img = base64.b64encode(f.read()).decode()
image = []
image.append(img)
res = {"image":image}
url = ""

result = requests.post(url,data=res)
label = str(result.content,"utf-8")

if label == '0':
    print("plastic")
    servo = int(27)
    GPIO.setup(servo, GPIO.OUT)
    setServoAngle(servo, int(8))
    sleep(2)
    setServoAngle(servo, int(6))

elif label == '1':
    print("metal")
    servo = int(22)
    GPIO.setup(servo, GPIO.OUT)
    setServoAngle(servo, int(8))
    sleep(2)
    setServoAngle(servo, int(6))

elif label == '2':
    print("glass")
    servo = int(17)
    GPIO.setup(servo, GPIO.OUT)
    setServoAngle(servo, int(8))
    sleep(2)
    setServoAngle(servo, int(6))

elif label == '3':
    print("paper")
    servo = int(4)
    GPIO.setup(servo, GPIO.OUT)
    setServoAngle(servo, int(8))
    sleep(2)
    setServoAngle(servo, int(6))
else:
    pass
