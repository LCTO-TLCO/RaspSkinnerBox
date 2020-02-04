#! /usr/bin python3
# coding:utf-8
import sys
import schedule
import pandas as pd
from datetime import *
from random import seed, choice
import random
from defines import mouse_no

DEBUG = False

from box_interface import *
from file_io import *

# define

# reset_time = datetime(today.year, today.month, today.day, 6, 0, 0) + timedelta(days=1)
# ex_limit = {True: [1, 3], False: [50, 100]}
ex_limit = {True: [1, 3, 3, 3, 3, 3, 3], False: [50, 50, 50, 50, 100, 300, 300]}  # updated
limit = {True: 25, False: 1}
is_time_limit_task = False
current_task_name = ""
exist_reserved_payoff = False
feeds_today = 0
current_reset_time = None
reward = 70
# while pelet > 0 and not datetime.now().time().hour == 10:

seed(32)


def run(terminate="", remained=-1):
    global mouse_no
    setup()
    file_setup(mouse_no)
    # unpayed_feeds_calculate()
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
    # initialize
    global reward, is_time_limit_task, current_task_name, feeds_today, current_reset_time
    current_task = ex_flow[task_no]
    print("{} start".format(task_no))
    hole_lamps_turn("off")
    session_no = last_session_id()
    current_task_name = task_no
    current_reset_time = current_task.get("reset_time", "07:00")
    schedule.every().day.at("{}".format(current_reset_time)).do(unpayed_feeds_calculate)
    feeds_today = int(calc_todays_feed(select_basetime(current_reset_time)))
    begin = 0
    is_time_limit_task = "time" in current_task
    if remained == -1:
        begin = 0
    else:
        begin = int(current_task["upper_limit"] / limit[DEBUG]) - remained
    if begin < 0:
        begin = 0
    correct_times = begin
    reward = ex_flow[current_task_name].get("feed_upper", 70)

    # main
    while correct_times <= int(current_task["upper_limit"] / limit[DEBUG]):
        # task start
        schedule.run_pending()
        if overpayed_feeds_calculate():
            sleep(5)
            continue
        if is_time_limit_task:
            if (not any(list(map(is_execution_time, current_task["time"])))) and exist_reserved_payoff:
                unpayed_feeds_calculate()
                continue
            elif not any(list(map(is_execution_time, current_task["time"]))):
                if all([datetime.now().minute % 5, datetime.now().second == 0]):
                    print("pending ... not in {}".format(current_task["time"]))
                sleep(5)
                continue

        export(task_no, session_no, correct_times, "start")

        # task call
        if current_task["task_call"]:
            hole_lamp_turn("dispenser_lamp", "on")
            if not DEBUG:
                while not is_hole_poked("dispenser_sensor"):
                    sleep(0.01)
            elif DEBUG:
                print("debug mode: type any key")
                input()
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
            feeds_today += 1
            # perseverative response measurement after reward & magazine nose poke detection
            while not is_hole_poked("dispenser_sensor"):
                h = is_holes_poked(target_holes)
                if h:
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

    # task end
    # reward = reward - correct_times
    schedule.clear()
    print("{} end".format(task_no))


def T0():
    print("T0 start")
    global reward, current_task_name
    current_task_name = "T0"
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


# time
def is_execution_time(start_end: list):
    """ 実行時刻の開始終了リストを引数にして今実行時間かどうかを判定する """
    start, end = [datetime.combine(datetime.today(), datetime.strptime(time, "%H:%M").time()) for time in start_end]
    # 日付繰り上がりの処理
    start -= timedelta(days=int(start.hour > end.hour)) if datetime.now().hour < end.hour else timedelta(days=0)
    end += timedelta(days=int(start.hour > end.hour)) if datetime.now().hour > end.hour else timedelta(days=0)
    return start <= datetime.now() <= end


def select_basetime(times="07:00"):
    hours = int(times.split(":")[0])
    minutes = int(times.split(":")[1])

    today = datetime.today() if datetime.now().time() >= time(hours,
                                                              minutes) else datetime.today() - timedelta(
        days=1)
    return datetime.combine(today, time(hours, minutes))


def ITI(secs: list):
    if DEBUG:
        secs = [2]  # changed
    selected = choice(secs)
    sleep(selected)
    return selected


def dispense_all(feed):
    for f in range(feed):
        dispense_pelet("payoff")
        sleep(5)


def unpayed_feeds_calculate():
    """ 直前の精算時間までに吐き出した餌の数を計上し足りなければdispense_all """
    global current_task_name, reward, exist_reserved_payoff, feeds_today, current_reset_time
    if ex_flow[current_task_name].get("payoff", False):
        # リスケ
        if any(list(map(is_execution_time, ex_flow[current_task_name].get("time", [["00:00", "00:00"]])))):
            exist_reserved_payoff = True
            return
        exist_reserved_payoff = False
        # calc remain
        reward = reward - calc_todays_feed(select_basetime(current_reset_time))
        # dispense
        #    sleep(5 * 60)
        if DEBUG:
            reward = 3
        while reward > 0:
            # if any(list(map(is_execution_time, ex_flow[current_task_name]["time"]))):
            #     daily_log(select_basetime(current_reset_time))
            #     exist_reserved_payoff = True
            #     return
            print("reward = {}".format(reward)) if DEBUG else None
            dispense_all(min(1, reward))
            reward -= 1

            sleep(1 * 60)
        reward = ex_flow[current_task_name].get("feed_upper", 70)
        feeds_today = 0
    daily_log(select_basetime(current_reset_time))


def overpayed_feeds_calculate():
    global feeds_today, current_task_name
    return feeds_today >= ex_flow[current_task_name].get("feed_upper", 100)


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
