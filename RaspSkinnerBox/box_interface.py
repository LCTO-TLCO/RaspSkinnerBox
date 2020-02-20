#! /usr/bin python3
from typing import Union
from file_io import *
from time import sleep
from random import choice
from app import DEBUG
import logging

if not DEBUG:
    import RPi.GPIO as GPIO
if DEBUG:
    import msvcrt

logger = logging.getLogger(__name__)
# output
hole_lamp = {1: 22, 3: 18, 5: 23, 7: 24, 9: 25}
# hole_lamp = {3: 18, 5: 23, 7: 24}

dispenser_magazine = 4
dispenser_lamp = 17
house_lamp = 27
# input
dispenser_sensor = 5
# hole_sensor = {3: 6, 5: 26, 7: 19}
hole_sensor = {1: 12, 3: 6, 5: 26, 7: 19, 9: 16}
sensor_pins = dict(zip(list(hole_sensor.values()) + [dispenser_sensor], list(hole_sensor.keys()) + ["dispenser"]))


def setup():
    if not DEBUG:
        global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
        GPIO.setmode(GPIO.BCM)
        # output
        for outputs in [hole_lamp, dispenser_magazine, dispenser_lamp, house_lamp]:
            if isinstance(outputs, dict):
                for no in list(outputs.keys()):
                    GPIO.setup(outputs[no], GPIO.OUT, initial=GPIO.LOW)
                    print("hole lamp [{\033[32m" + str(no) + "\033[0m}] = GPIO output [" + str(outputs[no]) + "]")
                continue
            GPIO.setup(outputs, GPIO.OUT, initial=GPIO.LOW)
            print("output [" + str(outputs) + "] = GPIO output [" + str(outputs) + "]")
        # input
        for pin, name in sensor_pins.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            holes_event_setup(pin)
            if isinstance(name, int):
                print("hole sensor [{\033[32m" + str(name) + "\033[0m}] = GPIO input [" + str(pin) + "]")
            elif isinstance(name, str):
                print("input [" + str(pin) + "] = GPIO input [" + str(pin) + "]")
        set_dir()


def shutdown():
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    if not DEBUG:
        for outputs in [hole_lamp, dispenser_magazine, dispenser_lamp]:
            if isinstance(outputs, dict):
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
    else:
        print("dispence for {}: {}".format(reason, 1))


def hole_lamp_turn(target: Union[int, str], switch: str):
    global hole_lamp, dispenser_lamp, house_lamp, dispenser_sensor, hole_sensor
    if not DEBUG:
        do = {"on": GPIO.HIGH, "off": GPIO.LOW}
        if isinstance(target, int):
            GPIO.output(hole_lamp[target], do[switch])
        elif "lamp" in target:
            exec("GPIO.output({},do[switch])".format(target))
    elif DEBUG:
        print("debug: {} hole turn {}".format(target, switch))


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
    if not DEBUG:
        if isinstance(no, int):
            if GPIO.input(no) == GPIO.LOW:
                return True
        elif isinstance(no, str):
            if eval("GPIO.input({})".format(no)) == GPIO.LOW:
                return True
        return False
    elif DEBUG:
        input("debug mode: type ENTER key")
        return True


def is_holes_poked(holes: list, dev_poke=True, dev_stop=False):
    if not DEBUG:
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
    elif DEBUG:
        import msvcrt
        return choice(holes) * msvcrt.kbhit() if not dev_stop else choice(holes) * bool(
            input("debug mode: type ENTER key"))
    return False


def holes_event_setup(gpio_no: int):
    if not DEBUG:
        GPIO.add_event_detect(gpio_no, GPIO.RISING, callback=callback_rising, bouncetime=50)
        # GPIO.add_event_detect(gpio_no, GPIO.FALLING, callback=callback_falling, bouncetime=50)
