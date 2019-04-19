#! /usr/bin python3
import RPi.GPIO as GPIO
from time import sleep
from random import choice

# output
hole_lamp = {1: 22, 3: 18, 5: 23, 7: 24, 9: 25}
# for i in range(1, 9, 2):
#     hole_lamp[i] = 0

dispenser_magazine = 4
dispenser_lamp = 17
house_lamp = 27
# input
dispenser_sensor = 5
hole_sensor = {3: 6, 5: 13, 7: 19}


# for i in range(1, 9, 2):
#     hole_sensor[i] = 0


def setup():
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    GPIO.setmode(GPIO.BCM)
    # output
    for outputs in [hole_lamp, dispenser_magazine, dispenser_lamp, house_lamp]:
        if type(outputs) == type({}):
            for no in list(outputs.keys()):
                print("dict detected")
                GPIO.setup(outputs[no], GPIO.OUT, initial=GPIO.LOW)
            continue
        GPIO.setup(outputs, GPIO.OUT, initial=GPIO.LOW)
    # input
    for inputs in [dispenser_sensor, hole_sensor]:
        if type(inputs) == type({}):
            print("dict detected")
            for no in inputs.keys():
                GPIO.setup(inputs[no], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            continue
        GPIO.setup(inputs, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


def shutdown():
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    for outputs in [hole_lamp, dispenser_magazine, dispenser_lamp]:
        if type(outputs) == type(dict):
            for no in outputs.keys():
                GPIO.output(outputs[no], GPIO.LOW)
            continue
        GPIO.setup(outputs, GPIO.LOW)
    GPIO.cleanup()


def dispense_pelet():
    global hole_lamp,dispenser_lamp,house_lamp,dispenser_sensor,hole_sensor
    GPIO.output(dispenser_magazine, GPIO.HIGH)
    GPIO.output(dispenser_magazine, GPIO.LOW)
    sleep(3)


def hole_lamp_turn(no: int, switch: str):
    global hole_lamp,dispenser_lamp,house_lamp,dispenser_sensor,hole_sensor
    do = {"on": GPIO.HIGH, "off": GPIO.LOW}
    GPIO.output(hole_lamp[no], do[switch])


def hole_lamp_all(switch: str):
    global hole_lamp,dispenser_lamp,house_lamp,dispenser_sensor,hole_sensor
    for no in hole_lamp.keys():
        hole_lamp(no, switch)


def hole_lamp_rand():
    global hole_lamp,dispenser_lamp,house_lamp,dispenser_sensor,hole_sensor
    holes = hole_lamp.keys()
    num = choice(list(holes))
    hole_lamp_turn(num, "on")
    return num


def is_hole_poked(no: int):
    if GPIO.input(no):
        return True


def is_holes_poked():
    global hole_sensor
    for hole in hole_sensor.values():
        if is_hole_poked(hole):
            return True
    return False
