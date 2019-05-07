#! /usr/bin python3
from typing import Union
from file_io import *
import RPi.GPIO as GPIO
from time import sleep
from random import choice
from app import DEBUG

# output
# hole_lamp = {1: 22, 3: 18, 5: 23, 7: 24, 9: 25}
hole_lamp = {3: 18, 5: 23, 7: 24}

dispenser_magazine = 4
dispenser_lamp = 17
house_lamp = 27
# input
dispenser_sensor = 5
hole_sensor = {3: 6, 5: 26, 7: 19}


def setup():
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    GPIO.setmode(GPIO.BCM)
    # output
    for outputs in [hole_lamp, dispenser_magazine, dispenser_lamp, house_lamp]:
        if type(outputs) == type({}):
            for no in list(outputs.keys()):
                GPIO.setup(outputs[no], GPIO.OUT, initial=GPIO.LOW)
                print("hole lamp [{\033[32m" + str(no) + "\033[0m}] = GPIO output [" + str(outputs[no]) + "]")
            continue
        GPIO.setup(outputs, GPIO.OUT, initial=GPIO.LOW)
        print("output [" + str(outputs) + "] = GPIO output [" + str(outputs) + "]")
    # input
    for inputs in [dispenser_sensor, hole_sensor]:
        if isinstance(inputs, dict):
            for no in inputs.keys():
                GPIO.setup(inputs[no], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                print("hole sensor [{\033[32m" + str(no) + "\033[0m}] = GPIO input [" + str(inputs[no]) + "]")
                holes_event_setup(inputs[no])
            continue
        GPIO.setup(inputs, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        print("input [" + str(inputs) + "] = GPIO input [" + str(inputs) + "]")
    set_dir()

def shutdown():
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    for outputs in [hole_lamp, dispenser_magazine, dispenser_lamp]:
        if type(outputs) == type({}):
            for no in outputs.keys():
                GPIO.output(outputs[no], GPIO.LOW)
            continue
        GPIO.setup(outputs, GPIO.LOW)
    GPIO.cleanup()


def dispense_pelet(reason="reward"):
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    if not DEBUG:
        GPIO.output(dispenser_magazine, GPIO.HIGH)
        sleep(0.1)
        GPIO.output(dispenser_magazine, GPIO.LOW)
        magagine_log(reason)


def hole_lamp_turn(target: Union[int, str], switch: str):
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    do = {"on": GPIO.HIGH, "off": GPIO.LOW}
    if isinstance(target, int):
        GPIO.output(hole_lamp[target], do[switch])
    elif "lamp" in target:
        exec("GPIO.output({},do[switch])".format(target))


def hole_lamps_turn(switch: str, target=[]):
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    if len(target) == 0:
        for no in hole_lamp.keys():
            hole_lamp_turn(no, switch)
    else:
        for no in target:
            hole_lamp_turn(no, switch)


def hole_lamp_rand():
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    holes = hole_lamp.keys()
    num = choice(list(holes))
    hole_lamp_turn(num, "on")
    return num


def is_hole_poked(no: Union[int, str]):
    if isinstance(no, int):
        if GPIO.input(no) == GPIO.LOW:
            return True
    elif isinstance(no, str):
        if eval("GPIO.input({})".format(no)) == GPIO.LOW:
            return True
    return False


def is_holes_poked(holes: list):
    global hole_sensor
    #    print(str(hole_sensor))
    #    for hole in hole_sensor.values():
    holes = list(hole_sensor.keys()) if len(holes) == 0 else holes
    if None in holes:
        return False
    for hole in holes:
        #        print(str(hole))
        if is_hole_poked(hole_sensor[hole]):
            return hole
    return False


def holes_event_setup(gpio_no: int):
    GPIO.add_event_detect(gpio_no, GPIO.RISING, callback=callback_rising, bouncetime=50)
    # GPIO.add_event_detect(gpio_no, GPIO.FALLING, callback=callback_falling, bouncetime=50)
