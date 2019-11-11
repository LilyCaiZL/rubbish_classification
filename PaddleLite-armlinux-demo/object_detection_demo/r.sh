fswebcam -s -r 500*500 images/test1.jpg
echo ok
rm -rf myhistory
#raspistill -w 500 -h 500 -o images/test1.jpg
./run.sh armv7hf >> myhistory
python3 ppp.py
