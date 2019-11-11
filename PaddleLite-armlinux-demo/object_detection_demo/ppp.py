import re
from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def setServoAngle(servo, angle):
	pwm = GPIO.PWM(servo, 50)
	pwm.start(0)
	dutyCycle = angle / 18 + 3
	pwm.ChangeDutyCycle(angle)
	sleep(0.3)
	pwm.stop()


f = open('myhistory','r')
k = f.read()
pat = re.compile(r'paper')
pat2 = re.compile(r'metal')
pat3 = re.compile(r'plastic')
pat4 = re.compile(r'glass')
patlist = []
def intolist(m):
    if m != []:
        patlist.append(m[0])
intolist(pat.findall(k))
intolist(pat2.findall(k))
intolist(pat3.findall(k))
intolist(pat4.findall(k))


print(patlist)
numlist={'plastic':int(27),'metal':int(22),'glass':int(17),'paper':int(4)}
for i in patlist:
	GPIO.setup(numlist[i], GPIO.OUT)
	setServoAngle(numlist[i], int(8))

sleep(2)
for i in patlist:
	GPIO.setup(numlist[i], GPIO.OUT)
	setServoAngle(numlist[i], int(6))
	

