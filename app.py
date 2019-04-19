#! /usr/bin python3
# coding:utf-8
from box_interface import *
import RPi.GPIO as GPIO
from datetime import datetime, timedelta
from random import seed, choice

# define
DEBUG = True
reward = 100
today = datetime.today()
reset_time = datetime(today.year, today.month, today.day + 1, 10, 0, 0)
ex_limit = {True: [3, 3], False: [50, 100]}


# while pelet > 0 and not datetime.now().time().hour == 10:

def run(terminate="T0"):
    setup()
    ex_flow = ["T0",
               "T1", "T2"]
    if terminate in ex_flow[1:2]:
        i = ex_flow.index(terminate)
        del ex_flow[0:i - 1]
    # for term in ex_flow:
    #     eval("{}()".format(term))
    T0()

def T0():
    print("T0 start")
    global reward
    times = 0
    GPIO.output(dispenser_lamp, GPIO.LOW)
    hole_lamp_all("off")
    for times in range(0, ex_limit[DEBUG][0]):
        if reset_time <= datetime.now():
            dispense_all(reward)
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        dispense_pelet()
        while not GPIO.input(dispenser_sensor):
            sleep(0.1)
        print("poked")
        GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - times
    print("T0 end")

def T1():
    print("T1 start")
    global reward
    hole_lamp_all("off")
    times = 0
    for times in range(0, ex_limit[DEBUG][0]):
        if reset_time <= datetime.now():
            dispense_all(reward)
        hole_lamp_all("on")
        while not is_holes_poked():
            sleep(0.1)
        hole_lamp_all("off")
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        dispense_pelet()
        while not GPIO.input(dispenser_sensor):
            sleep(0.1)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - times
    print("T1 end")

def T2():
    print("T2 start")
    global reward
    hole_lamp_all("off")
    times = 0
    for times in range(0, ex_limit[DEBUG][1]):
        if reset_time <= datetime.now():
            dispense_all(reward)
        # start
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        while GPIO.input(dispenser_sensor) == GPIO.LOW:
            sleep(0.1)
        sleep(5)
        # random lamp on
        num = hole_lamp_rand()
        end_time = datetime.now() + timedelta(seconds=20)
        # nosepoke detect
        while not is_hole_poked(num):
            sleep(0.1)
        hole_lamp_all("off")
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        dispense_pelet()
        while not GPIO.input(dispenser_sensor):
            sleep(0.1)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        sleep(20)
    reward = reward - times
    print("T2 end")

def dispense_all(feed):
    for f in range(feed):
        dispense_pelet()
    global reward
    reward = 100
    # ここで実験ぶった切るならexit()


def ITI():
    secs = [4, 8, 16, 32]
    seed()
    sleep(choice(secs))


if __name__ == "__main__":
    try:
        run()
    finally:
        shutdown()