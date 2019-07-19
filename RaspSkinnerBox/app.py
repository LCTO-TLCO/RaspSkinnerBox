#! /usr/bin python3
# coding:utf-8
import sys
from box_interface import *
from datetime import *
from random import seed, choice
import random
from file_io import *

# define
DEBUG = False
reward = 70  # changed
today = datetime.today()
reset_time = datetime(today.year, today.month, today.day, 6, 0, 0) + timedelta(days=1)
# ex_limit = {True: [1, 3], False: [50, 100]}
ex_limit = {True: [1, 3, 3, 3, 3, 3, 3], False: [50, 50, 50, 50, 100, 300, 300]}  # updated
mouse_no = "10"
limit = {True: 25, False: 1}
# while pelet > 0 and not datetime.now().time().hour == 10:

seed(32)


def run(terminate="", remained=-1):
    global mouse_no
    setup()
    file_setup(mouse_no)
    if terminate in list(ex_flow.keys()):
        i = list(ex_flow.keys()).index(terminate)
        print("i=" + str(i))
        for delete_task in list(ex_flow.keys())[0:i]:
            del ex_flow[delete_task]
    for term in ex_flow:
        if term == "T0":
            T0()
        else:
            task(term, remained)
            remained = -1


def task(task_no: str, remained: int):
    global reward
    current_task = ex_flow[task_no]
    print("{} start".format(task_no))
    hole_lamps_turn("off")
    session_no = last_session_id()
    begin = 0
    if remained == -1:
        begin = 0
    else:
        begin = int(current_task["upper_limit"] / limit[DEBUG]) - remained
    if begin < 0:
        begin = 0
    correct_times = begin
    while correct_times <= int(current_task["upper_limit"] / limit[DEBUG]):

        export(task_no, session_no, correct_times, "start")
        # if reset_time <= datetime.now():
        #     dispense_all(reward)

        # task call
        if current_task["task_call"]:
            hole_lamp_turn("dispenser_lamp", "on")
            while not is_hole_poked("dispenser_sensor"):
                sleep(0.01)
            hole_lamp_turn("dispenser_lamp", "off")
            export(task_no, session_no, correct_times, "task called")
            hole_lamp_turn("house_lamp", "on")
            sleep(1)

        # hole setup
        target_holes = current_task["target_hole"]
        q_holes = []
        hole_lamps_turn("on", target_holes)

        # reward holes setup
        if not len(current_task["reward_late"]) == 0:
            [q_holes.append(h) for h in target_holes if
             random.random() * 100 <= current_task["reward_late"][target_holes.index(h)]]
            q_holes.append(None) if len(q_holes) == 0 else None
        else:
            q_holes = target_holes
        export(task_no, session_no, correct_times, "correct holes", '/'.join([str(s) for s in q_holes]))

        # time
        end_time = False
        if current_task["limited_hold"] >= 0:
            end_time = datetime.now() + timedelta(seconds=current_task["limited_hold"])
        hole_poked = False
        is_correct = False
        time_over = False
        while not (hole_poked or time_over):
            h = is_holes_poked(target_holes)
            if not h == False:
                hole_poked = True
                if h in q_holes:
                    is_correct = True
                    correct_times += 1
                    export(task_no, session_no, correct_times, "nose poke", h)
                    export(task_no, session_no, correct_times, "reward", h)
                else:
                    export(task_no, session_no, correct_times, "nose poke", h)
                    export(task_no, session_no, correct_times, "failure", h)
            # time over
            if not end_time == False:
                if end_time < datetime.now():
                    time_over = True
                    export(task_no, session_no, correct_times, "time over")
            sleep(0.01)
        # end
        hole_lamps_turn("off")
        if is_correct:
            hole_lamp_turn("dispenser_lamp", "on")
            dispense_pelet()

            # perseverative response measurement after reward & magazine nose poke detection
            while not is_hole_poked("dispenser_sensor"):
                h = is_holes_poked(target_holes)
                if not h == False:
                    export(task_no, session_no, correct_times, "nose poke after rew", h)
                    sleep(0.5)
                sleep(0.01)
            export(task_no, session_no, correct_times, "magazine nose poked")
            hole_lamp_turn("dispenser_lamp", "off")
            actualITI = ITI(current_task["ITI_correct"])
            export(task_no, session_no, correct_times, "ITI", actualITI)
            hole_lamp_turn("house_lamp", "off")
        else:
            #            sleep(int(20/limit[DEBUG]))
            hole_lamp_turn("house_lamp", "off")
            actualITI = ITI(current_task["ITI_failure"])
            export(task_no, session_no, correct_times, "ITI", actualITI)
        session_no += 1
    reward = reward - correct_times
    print("{} end".format(task_no))


def T0():
    print("T0 start")
    global reward
    times = 0
    session_no = 0
    task_no = "T0"
    hole_lamp_turn("dispenser_lamp", "on")
    export(task_no, session_no, times, "start")
    hole_lamps_turn("off")
    for times in range(0, ex_limit[DEBUG][0]):
        #        if reset_time <= datetime.now():
        #            dispense_all(reward)
        hole_lamp_turn("dispenser_lamp", "on")
        while not is_hole_poked("dispenser_sensor"):
            sleep(0.01)
        dispense_pelet()
        export(task_no, session_no, times, "reward")
        hole_lamp_turn("dispenser_lamp", "off")
        ITI([4, 8, 16, 32])
        session_no += 1
    reward = reward - times
    print("T0 end")


def is_execution_time(start_end: list):
    """ 実行時刻の開始終了リストを引数にして今実行時間かどうかを判定する """
    start, end = [datetime.combine(datetime.today(), datetime.strptime(time, "%H:%M").time()) for time in start_end]
    # 日付繰り上がりの処理
    end = end + timedelta(days=int(start > end))
    return start <= datetime.now() and end >= datetime.now()


def dispense_all(feed):
    global reset_time
    for f in range(feed):
        dispense_pelet("payoff")
    global reward
    reward = 100
    reset_time = datetime(today.year, today.month, today.day + 1, 10, 0, 0)
    # ここで実験ぶった切るならexit()


def ITI(secs: list):
    if DEBUG:
        secs = [2]  # changed
    selected = choice(secs)
    sleep(selected)
    return selected


def unpayed_feeds_calculate():
    """ 直前の精算時間までに吐き出した餌の数を計上し足りなければdispense_all """
    """ 未デバッグ """

    # calc len
    # dispense
    None


if __name__ == "__main__":
    try:
        terminate_task = ""
        remained = -1
        # if len(sys.argv) == 1:
        #     print("usage: python app.py mouse_No terminate_task_No remained_number_of_tasks")
        #     sys.exit()
        # mouse_no = sys.argv[1]
        if len(sys.argv) >= 2:
            terminate_task = sys.argv[1]
        if len(sys.argv) == 3:
            remained = int(sys.argv[2])
            print("remained{}".format(remained))
        run(terminate_task, remained)
    except Exception as e:
        # error log
        error_log(e)
    finally:
        shutdown()
