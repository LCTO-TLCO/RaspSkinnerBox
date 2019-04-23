#! /usr/bin python3
# coding:utf-8
from box_interface import *
import RPi.GPIO as GPIO
from datetime import datetime, timedelta
from random import seed, choice
import random
import time # added
import io # added

# define
DEBUG = False
reward = 70 # changed
today = datetime.today()
reset_time = datetime(today.year, today.month, today.day + 1, 10, 0, 0)
#ex_limit = {True: [1, 3], False: [50, 100]}
ex_limit = {True: [1, 3, 3, 3, 3, 3, 3], False: [50, 50, 50, 50, 100, 300, 300]} # updated

# while pelet > 0 and not datetime.now().time().hour == 10:

# TODO
# 1. 構造が同じタスクで、パラメータのみが違う場合のコピペをなくしたい(T1-Q4 -> same task, different param., task call=0/1, limited hold=unlimited/60s/10s/5s, target hole choice, ITI, upper limit, reward rates)
# 2. CSVログ出力と標準出力表示を一元管理 (標準出力は現在のCSVログ出力と同じ内容) (呼び出し元タスク名を自動埋め込み)
# 3. TTL入出力部のラッピング(残っている部分をなくす)
# 4. 再開するタスク名 and remained dutyをコマンドラインから引数で渡す
# 5. ITIの種類対応
# 6. 未払い給与支払いを再確認 (on/off control)
# 7. dispenseでログ出力
# 8. error, omission log
# 9. task setting log

# Task 
# T0. magazineからpelletを出す, ITI1, 50回
# T1. すべてのholeが点灯しつづけ, いずれかにnose pokeするとreward, ITI1, 50回 (Day1)
# T3. magazineでtaskをcallして、5th hole点灯, 5th hole nose pokeでreward, ITI3, limited hold 60s, 50回
# T4. limited hold 20s, ITI3, 50回(Day2)
# Q1. 5th hole点灯, 5th hole nose pokeで必ずreward, ITI3, limited hold 10s, 100回(Day3), ITI3
# Q2. 3,5,7th hole点灯, 3,5,7th hole nose pokeで{30%, 0%, 70%}の確率でreward, ITI3, limited hold 5s, 300回 (3日分)
# Q3. 3,5,7th hole点灯, 3,5,7th hole nose pokeで{30%, 0%, 0%}の確率でreward, ITI3, limited hold 5s, 300回 (3日分)
# Q4. 3,5,7th hole点灯, 3,5,7th hole nose pokeで{30%, 0%, 70%}の確率でreward, ITI3, limited hold 5s, 300回 (3日分)

# ITI1 4,8,16,32 s
# ITI3 1 s

seed(32)
logfile = open('epsilon-greedy.txt', 'a+')

def run(terminate="T3"):
    setup()
    ex_flow = ["T0", "T1", "T3", "T4", "Q1", "Q2"]
    if terminate in ex_flow:
        i = ex_flow.index(terminate)
        print("i="+str(i))
        del ex_flow[0:i]
    for term in ex_flow:
        eval("{}()".format(term))

def T0():
    print("T0 start")
    global reward
    times = 0
    GPIO.output(dispenser_lamp, GPIO.LOW)
    logfile.write(str(datetime.now())+',T1,'+str(times)+',start,0\n')
    logfile.flush()
    hole_lamp_all("off")
    for times in range(0, ex_limit[DEBUG][0]):
        print(times)
        if reset_time <= datetime.now():
            dispense_all(reward)
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        dispense_pelet()
        while GPIO.input(dispenser_sensor) == GPIO.HIGH:
            sleep(0.01)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - times
    print("T0 end")

def T1():
    print("T1 start")
    global reward
    hole_lamp_all("off")
    times = 0
    for times in range(0, ex_limit[DEBUG][1]):
        print(times)
        logfile.write(str(datetime.now())+',T1,'+str(times)+',start,0\n')
        logfile.flush()

        if reset_time <= datetime.now():
            dispense_all(reward)
        hole_lamp_all("on")
        while not is_holes_poked():
            sleep(0.01)

        logfile.write(str(datetime.now())+',T1,'+str(times)+',nose poked,0\n')
        logfile.flush()

        hole_lamp_all("off")
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        dispense_pelet()
        while GPIO.input(dispenser_sensor) == GPIO.HIGH:
            sleep(0.01)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        logfile.write(str(datetime.now())+',T1,'+str(times)+',magazine nose poked,0\n')
        logfile.flush()
        ITI()
    reward = reward - times
    print("T1 end")

def T2():
    print("T2 start")
    global reward
    hole_lamp_all("off")
    correct_times = 0
    while correct_times <= ex_limit[DEBUG][2]:
        print(correct_times)
        logfile.write(str(datetime.now())+',T2,'+str(correct_times)+',start,0\n')
        logfile.flush()
#        if reset_time <= datetime.now():
#            dispense_all(reward)

        # hole
        default_hole = 5
        hole_lamp_turn(default_hole,"on")
        end_time = datetime.now() + timedelta(seconds=60)
        hole_poked = False
        time_over = False
        while not(hole_poked or time_over):
            if is_hole_poked(default_hole):
                hole_poked = True
                correct_times += 1
                logfile.write(str(datetime.now())+',T2,'+str(correct_times)+',nose poked,'+str(default_hole)+'\n')
                logfile.flush()
            if end_time < datetime.now():
                time_over = True
                logfile.write(str(datetime.now())+',T2,'+str(correct_times)+',time over,0\n')
                logfile.flush()
            sleep(0.01)

        hole_lamp_all("off")
        if hole_poked:
            GPIO.output(dispenser_lamp, GPIO.HIGH)
            dispense_pelet()
            while GPIO.input(dispenser_sensor) == GPIO.HIGH:
                sleep(0.01)
            logfile.write(str(datetime.now())+',T2,'+str(correct_times)+',magazine nose poked,0\n')
            logfile.flush()
            sleep(20)
            GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - correct_times
    print("T2 end")

def T3():
    print("T3 start")
    global reward
    hole_lamp_all("off")
    correct_times = 0
    while correct_times <= ex_limit[DEBUG][3]:       
        print(correct_times)
        logfile.write(str(datetime.now())+',T3,'+str(correct_times)+',start,0\n')
        logfile.flush()
#        if reset_time <= datetime.now():
#            dispense_all(reward)

        # calling task
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        while GPIO.input(dispenser_sensor) == GPIO.HIGH:
            sleep(0.01)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        sleep(1)

        # hole
        default_hole = 5
        hole_lamp_turn(default_hole,"on")
        end_time = datetime.now() + timedelta(seconds=60)
        hole_poked = False
        time_over = False
        while not(hole_poked or time_over):
            if is_hole_poked(default_hole):
                hole_poked = True
                correct_times += 1
                logfile.write(str(datetime.now())+',T3,'+str(correct_times)+',nose poked,'+str(default_hole)+'\n')
                logfile.flush()
            if end_time < datetime.now():
                time_over = True
                logfile.write(str(datetime.now())+',T3,'+str(correct_times)+',time over,0\n')
                logfile.flush()
            sleep(0.01)

        hole_lamp_all("off")
        if hole_poked:
            GPIO.output(dispenser_lamp, GPIO.HIGH)
            dispense_pelet()
            while GPIO.input(dispenser_sensor) == GPIO.HIGH:
                sleep(0.01)
            logfile.write(str(datetime.now())+',T3,'+str(correct_times)+',magazine nose poked,0\n')
            logfile.flush()
            sleep(20)            
            GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - correct_times
    print("T3 end")

def T4():
    print("T4 start")
    global reward
    hole_lamp_all("off")
    correct_times = 0
    while correct_times <= ex_limit[DEBUG][3]:       
        print(correct_times)
        logfile.write(str(datetime.now())+',T4,'+str(correct_times)+',start,0\n')
#        if reset_time <= datetime.now():
#            dispense_all(reward)

        # calling task
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        while GPIO.input(dispenser_sensor) == GPIO.HIGH:
            sleep(0.01)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        sleep(1)

        # hole
        default_hole = 5
        hole_lamp_turn(default_hole,"on")
        end_time = datetime.now() + timedelta(seconds=20)
        hole_poked = False
        time_over = False
        while not(hole_poked or time_over):
            if is_hole_poked(default_hole):
                hole_poked = True
                correct_times += 1
                logfile.write(str(datetime.now())+',T4,'+str(correct_times)+',nose poked,'+str(default_hole)+'\n')
                logfile.flush()
            if end_time < datetime.now():
                time_over = True
                logfile.write(str(datetime.now())+',T4,'+str(correct_times)+',time over,0\n')
                logfile.flush()
            sleep(0.01)

        hole_lamp_all("off")
        if hole_poked:
            GPIO.output(dispenser_lamp, GPIO.HIGH)
            dispense_pelet()
            while GPIO.input(dispenser_sensor) == GPIO.HIGH:
                sleep(0.01)
            logfile.write(str(datetime.now())+',T4,'+str(correct_times)+',magazine nose poked,0\n')
            logfile.flush()
            sleep(20)
            GPIO.output(dispenser_lamp, GPIO.LOW)            
        ITI()
    reward = reward - correct_times
    print("T4 end")

def Q1():
    print("Q1 start")
    global reward
    hole_lamp_all("off")
    correct_times = 0
    while correct_times <= ex_limit[DEBUG][4]:       
        print(str(correct_times))
        logfile.write(str(datetime.now())+',Q1,'+str(correct_times)+',start,0\n')
#        if reset_time <= datetime.now():
#            dispense_all(reward)

        # calling task
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        while GPIO.input(dispenser_sensor) == GPIO.HIGH:
            sleep(0.01)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        logfile.write(str(datetime.now())+',Q1,'+str(correct_times)+',task called,0\n')
        logfile.flush()
        sleep(1)

        # hole
        target_holes = {5:1.0}
        q_holes = []
        for h in target_holes.keys():
            hole_lamp_turn(h,"on")
            if random.random() < target_holes[h]:
                q_holes.append(h)
        end_time = datetime.now() + timedelta(seconds=10)
        hole_poked = False
        time_over = False
        while not (hole_poked or time_over):
            for h in q_holes:
                if is_hole_poked(h):
                    hole_poked = True
                    correct_times += 1
                    logfile.write(str(datetime.now())+',Q1,'+str(correct_times)+',nose poked,'+str(h)+'\n')
                    logfile.flush()
                    break
            if end_time < datetime.now():
                time_over = True
                logfile.write(str(datetime.now())+',Q1,'+str(correct_times)+',time over,0\n')
                logfile.flush()
            sleep(0.01)

        hole_lamp_all("off")
        if hole_poked:
            GPIO.output(dispenser_lamp, GPIO.HIGH)
            dispense_pelet()
            while GPIO.input(dispenser_sensor) == GPIO.HIGH:
                sleep(0.01)
            logfile.write(str(datetime.now())+',Q1,'+str(correct_times)+',magazine nose poked,0\n')
            logfile.flush()
            sleep(20)
            GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - correct_times
    print("Q1 end")


def Q2():
    print("Q2 start")
    global reward
    hole_lamp_all("off")
    correct_times = 0
    while correct_times <= ex_limit[DEBUG][5]:
        print(str(correct_times))
        logfile.write(str(datetime.now())+',Q2,'+str(correct_times)+',start,0\n')
#        if reset_time <= datetime.now():
#            dispense_all(reward)

        # calling task
        GPIO.output(dispenser_lamp, GPIO.HIGH)
        while GPIO.input(dispenser_sensor) == GPIO.HIGH:
            sleep(0.01)
        GPIO.output(dispenser_lamp, GPIO.LOW)
        logfile.write(str(datetime.now())+',Q2,'+str(correct_times)+',task called,0\n')
        logfile.flush()
        sleep(1)

        # hole
        target_holes = {3:0.3, 5:0.0, 7:0.7}
        q_holes = []
        for h in target_holes.keys():
            hole_lamp_turn(h,"on")
            if random.random() < target_holes[h]:
                q_holes.append(h)
        end_time = datetime.now() + timedelta(seconds=10)
        hole_poked = False
        time_over = False
        while not (hole_poked or time_over):
            for h in q_holes:
                if is_hole_poked(h):
                    hole_poked = True
                    correct_times += 1
                    logfile.write(str(datetime.now())+',Q2,'+str(correct_times)+',nose poked,'+str(h)+'\n')
                    logfile.flush()
                    break
            if end_time < datetime.now():
                time_over = True
                logfile.write(str(datetime.now())+',Q2,'+str(correct_times)+',time over,0\n')
                logfile.flush()
            sleep(0.01)

        hole_lamp_all("off")
        if hole_poked:
            GPIO.output(dispenser_lamp, GPIO.HIGH)
            dispense_pelet()
            while GPIO.input(dispenser_sensor) == GPIO.HIGH:
                sleep(0.01)
            logfile.write(str(datetime.now())+',Q2,'+str(correct_times)+',magazine nose poked,0\n')
            logfile.flush()
            sleep(20)        
            GPIO.output(dispenser_lamp, GPIO.LOW)
        ITI()
    reward = reward - correct_times
    print("Q2 end")

def dispense_all(feed):
    for f in range(feed):
        dispense_pelet()
    global reward
    reward = 100
    # ここで実験ぶった切るならexit()


def ITI():
    secs = [4, 8, 16, 32]
    if DEBUG:
        secs = [2] # changed
    sleep(choice(secs))


if __name__ == "__main__":
    try:
        run()
    finally:
        shutdown()
        logfile.close()
