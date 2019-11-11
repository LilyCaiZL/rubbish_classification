#import os
#main = '/home/pi/cmake-3.10.3/Paddle/PaddleLite-armlinux-demo/object_detection_demo/build/object_detection_demo'
#f = os.popen(main)
#data = f.readlines()
#f.close()
#print(data)
import subprocess
import os
#os.system("cd /home/pi/cmake-3.10.3/Paddle/PaddleLite-armlinux-demo/object_detection_demo/")
os.system('./run.sh armv7hf >> history')
print("start")
os.system
#s = subprocess.Popen('./run.sh armv7hf',stdin = subprocess.PIPE,stdout=subprocess.PIPE,shell=True)
#print(s.stdout.readline())





