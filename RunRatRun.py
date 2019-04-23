import RPi.GPIO as GPIO
import time
import datetime
import io

CH1 = 17
CH2 = 18
bounce_time = 300

counter1 = 0
counter2 = 0

f1=open('cage1.txt', 'a+')
f2=open('cage2.txt', 'a+')

def my_callback(channel):
    global counter1
    global counter2
    timestamp = datetime.datetime.now()
    if channel==CH1:
        counter1 += 1
        f1.write(str(timestamp)+'\n')
        f1.flush()
        print('[Cage1] ' +str(counter1)+ ' HIT at '+str(timestamp))
    elif channel==CH2:
        counter2 += 1
        f2.write(str(timestamp)+'\n')
        f2.flush()
        print('[Cage2] ' +str(counter2)+ ' HIT at '+str(timestamp))

GPIO.setmode(GPIO.BCM)
print("GPIO.setmode("+str(GPIO.BCM)+")");

GPIO.setup(CH1,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(CH2,GPIO.IN,pull_up_down=GPIO.PUD_UP)

GPIO.add_event_detect(CH1,GPIO.FALLING,callback=my_callback,bouncetime=bounce_time)
print("GPIO.add_event("+str(CH1)+","+str(GPIO.RISING)+")");

GPIO.add_event_detect(CH2,GPIO.FALLING,callback=my_callback,bouncetime=bounce_time)
print("GPIO.add_event("+str(CH2)+","+str(GPIO.RISING)+")");

try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    GPIO.cleanup()

f1.close()
f2.close()
print("Goodbye!")
