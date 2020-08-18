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
lever_lamp = {1: 27, 2: 22}
# レバーひっこめる・出す
lever_out = {1: 18, 2: 23}

dispenser_magazine = 4
white_noise = 24
house_lamp = 17



# input
dispenser_sensor = 5
# hole_sensor = {3: 6, 5: 26, 7: 19}
# hole_sensor = {1: 12, 3: 6, 5: 26, 7: 19, 9: 16}
# レバー押す・押してない
lever_in = {1: 6, 2: 19}
sensor_pins = dict(zip(list(lever_in.values()) + [dispenser_sensor], list(lever_in.keys()) + ["dispenser"]))

# 実機デバッグ用, 入力無効
RASP_DEBUG = False


def setup():
    if not DEBUG:
        global lever_lamp, lever_in, white_noise, house_lamp, dispenser_sensor, lever_out
        GPIO.setmode(GPIO.BCM)
        # output
        for outputs in [lever_lamp, lever_out, dispenser_magazine, white_noise, house_lamp]:
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
    global lever_lamp, lever_in, white_noise, house_lamp, dispenser_sensor, lever_out
    if not DEBUG:
        for outputs in [lever_lamp, lever_out, dispenser_magazine, white_noise]:
            if isinstance(outputs, dict):
                for no in outputs.keys():
                    GPIO.output(outputs[no], GPIO.LOW)
                continue
            GPIO.setup(outputs, GPIO.LOW)
        GPIO.cleanup()


def dispense_pelet(reason="reward"):
    global lever_lamp, lever_in, white_noise, house_lamp, dispenser_sensor, lever_out
    if not DEBUG:
        GPIO.output(dispenser_magazine, GPIO.HIGH)
        sleep(0.1)
        GPIO.output(dispenser_magazine, GPIO.LOW)
        magagine_log(reason)
    else:
        print("dispence for {}: {}".format(reason, 1))


def hole_lamp_turn(target: Union[int, str], switch: str):
    global lever_lamp, lever_in, white_noise, house_lamp, dispenser_sensor, lever_out
    if not DEBUG:
        do = {"on": GPIO.HIGH, "off": GPIO.LOW}
        if isinstance(target, int):
            GPIO.output(lever_lamp[target], do[switch])
            GPIO.output(lever_out[target], not do[switch])
        elif "lamp" in target:
            exec("GPIO.output({},do[switch])".format(target))
    elif DEBUG:
        print("debug: {} hole turn {}".format(target, switch))


def hole_lamps_turn(switch: str, target=[]):
    global lever_lamp, lever_in, white_noise, house_lamp, dispenser_sensor, lever_out
    if len(target) == 0:
        for no in lever_lamp.keys():
            hole_lamp_turn(no, switch)
    else:
        for no in target:
            hole_lamp_turn(no, switch)


def hole_lamp_rand():
    global lever_lamp, lever_in, white_noise, house_lamp, dispenser_sensor, lever_out
    holes = lever_lamp.keys()
    num = choice(list(holes))
    hole_lamp_turn(num, "on")
    return num


def is_hole_poked(no: Union[int, str]):
    if DEBUG or RASP_DEBUG:
        input("debug mode: type ENTER key")
        return True
    elif not DEBUG:
        if isinstance(no, int):
            if GPIO.input(no) == GPIO.LOW:
                return True
        elif isinstance(no, str):
            if eval("GPIO.input({})".format(no)) == GPIO.LOW:
                return True
        return False


def is_holes_poked(holes: list, dev_poke=True, dev_stop=False):
    if not DEBUG and not RASP_DEBUG:
        global lever_in
        #    print(str(lever_out))
        #    for hole in lever_out.values():
        holes = list(lever_in.keys()) if len(holes) == 0 else holes
        if None in holes:
            return False
        for hole in holes:
            #        print(str(hole))
            if is_hole_poked(lever_in[hole]):
                return hole
    elif DEBUG:
        import msvcrt
        return choice(holes) * msvcrt.kbhit() if not dev_stop else choice(holes) * bool(
            input("debug mode: type ENTER key"))
    elif RASP_DEBUG:
        return choice(holes)
    return False


def holes_event_setup(gpio_no: int):
    if not DEBUG:
        GPIO.add_event_detect(gpio_no, GPIO.BOTH, callback=callback_both, bouncetime=50)
        # GPIO.add_event_detect(gpio_no, GPIO.RISING, callback=callback_rising, bouncetime=50)
        # GPIO.add_event_detect(gpio_no, GPIO.FALLING, callback=callback_falling, bouncetime=50)
