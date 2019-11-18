#! /usr/bin python3
# coding:utf-8
import sys
import schedule
from datetime import *
from random import seed, choice
import random
from defines import mouse_no
import logging

# logger setting
logging.basicConfig(
    level=logging.DEBUG,
    filename='infos.txt',
    filemode='a',
    format='%(asctime)s %(levelname)s %(module)s %(funcName)s line%(lineno)d %(message)s')

console = logging.StreamHandler()
console.setLevel(getattr(logging, 'INFO'))
console_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(module)s %(funcName)s line:%(lineno)d %(message)s')
console.setFormatter(console_formatter)
logger = logging.getLogger(__name__)
logger.addHandler(console)
logging.getLogger("box_interface").addHandler(console)

# debug mode
DEBUG = False

from box_interface import *
from file_io import *

# define

# reset_time = datetime(today.year, today.month, today.day, 6, 0, 0) + timedelta(days=1)
# ex_limit = {True: [1, 3], False: [50, 100]}
limit = {True: 1, False: 1}
is_time_limit_task = False
current_task_name = ""
exist_reserved_payoff = False
feeds_today = 0
current_reset_time = None
reward = 70
payoff_flag = False

seed(32)


def run(terminate="", remained=-1):
    global mouse_no
    setup()
    file_setup(mouse_no)
    # unpayed_feeds_calculate()
    if terminate in list(ex_flow.keys()):
        i = list(ex_flow.keys()).index(terminate)
        logger.info("i=" + str(i))
        for delete_task in list(ex_flow.keys())[0:i]:
            del ex_flow[delete_task]
    for term in ex_flow:
        if term == "T0":
            T0()
            continue
        task(term, remained)
        remained = -1


def task(task_no: str, remained: int):
    # initialize
    global reward, is_time_limit_task, current_task_name, feeds_today, current_reset_time, payoff_flag
    current_task = ex_flow[task_no]
    logger.info("task {} start".format(task_no))
    hole_lamps_turn("off")
    session_no = last_session_id()
    current_task_name = task_no
    current_reset_time = current_task.get("reset_time", "07:00")
    schedule.every().day.at(current_reset_time).do(unpayed_feeds_calculate)
    feeds_today = int(calc_todays_feed(select_basetime(current_reset_time) + timedelta(days=1)))
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
    # start time
    start_time = ex_flow[task_no].get("start_time", False)
    # タスク指定開始時のみbasetimeを過去にする
    if start_time:
        start_time = (select_basetime(start_time) if list(ex_flow.keys())[0] == current_task_name else select_basetime(
            start_time) + timedelta(days=1))
    payoff_flag = False

    # main
    while correct_times < int(current_task["upper_limit"] / limit[DEBUG]):
        # task start
        schedule.run_pending()
        if current_task.get("terminate_when_payoff", False) and payoff_flag:
            logger.info("terminate with payoff!".format())
            break
        payoff_flag = False
        if overpayed_feeds_calculate():
            if all([datetime.now().minute % 5, datetime.now().second == 0]):
                logger.info("over payed ...")
            sleep(5)
            continue
        if is_time_limit_task:
            if (not any(list(map(is_execution_time, current_task["time"])))) and exist_reserved_payoff:
                unpayed_feeds_calculate()
                logger.info("terminate payoff complete!")
                continue
            elif not any(list(map(is_execution_time, current_task["time"]))):
                if all([datetime.now().minute % 5, datetime.now().second == 0]):
                    logger.info("pending ... not in {}".format(current_task["time"]))
                sleep(5)
                continue
        if start_time:
            if start_time > datetime.now():
                sleep(5)
                if all([datetime.now().minute % 5, datetime.now().second == 0]):
                    logger.info("task stopping ... not after {}".format(start_time))
                continue
        export(task_no, session_no, correct_times, "start")
        hole_lamp_turn("house_lamp", "off")
        hole_lamp_turn("dispenser_lamp", "on")

        # task call
        if current_task.get("task_call", False):
            if not DEBUG:
                while not is_hole_poked("dispenser_sensor"):
                    if not any(list(map(is_execution_time, current_task.get("time", [["00:00", "23:59"]])))):
                        break
                    sleep(0.01)
            elif DEBUG:
                input("type ENTER key to call task")
            if current_task.get("time", False) and not any(
                    list(map(is_execution_time, current_task.get("time", [["00:00", "23:59"]])))):
                continue
            export(task_no, session_no, correct_times, "task called")
            hole_lamp_turn("house_lamp", "off")
            hole_lamp_turn("dispenser_lamp", "off")

            # cue delay
            premature = False
            timelimit = False
            base_time = datetime.now()
            cue_delay = current_task.get("cue_delay", 5) if isinstance(current_task.get("cue_delay", 5),
                                                                       int) else choice(current_task["cue_delay"])
            while not (premature or timelimit):
                if (datetime.now() - base_time).seconds >= cue_delay:
                    timelimit = True
                if is_holes_poked(current_task["target_hole"], False):
                    premature = True
                    export(task_no, session_no, correct_times, "premature")
                sleep(0.05)
            if premature:
                continue

        hole_lamp_turn("house_lamp", "off")
        hole_lamp_turn("dispenser_lamp", "off")

        # hole setup
        target_holes = current_task["target_hole"]
        hole_lamps_turn("on", target_holes)
        export(task_no, session_no, correct_times, "pokelight on")

        # time
        end_time = False
        houselamp_end_time = False
        if current_task["limited_hold"] >= 0:
            end_time = datetime.now() + timedelta(seconds=max([current_task["limited_hold"], 5]))
            houselamp_end_time = datetime.now() + timedelta(seconds=current_task["limited_hold"])
        hole_poked = False
        is_correct = False
        time_over = False
        while not (hole_poked or time_over):
            h = is_holes_poked(target_holes)
            if h:
                hole_poked = True
                is_correct = True
                export(task_no, session_no, correct_times, "nose poke", h)
                export(task_no, session_no, correct_times, "reward", h)
                correct_times += 1
            # time over
            elif end_time:
                if end_time < datetime.now():
                    time_over = True
                    export(task_no, session_no, correct_times, "time over")
            elif houselamp_end_time:
                if houselamp_end_time < datetime.now():
                    hole_lamps_turn("off", target_holes)
                    houselamp_end_time = False
            sleep(0.01)
        # end
        hole_lamps_turn("off", target_holes)
        hole_lamp_turn("house_lamp", "off")
        hole_lamp_turn("dispenser_lamp", "on")
        if is_correct:
            dispense_pelet()
            feeds_today += 1
            sleep(1)
            # perseverative response measurement after reward & magazine nose poke detection
            while not is_hole_poked("dispenser_sensor"):
                if is_holes_poked(target_holes):
                    export(task_no, session_no, correct_times, "nose poke after rew", h)
                    while is_holes_poked(target_holes):
                        sleep(0.01)
                sleep(0.01)
            export(task_no, session_no, correct_times, "magazine nose poked")
            actualITI = ITI(current_task["ITI_correct"])
            export(task_no, session_no, correct_times, "ITI", actualITI)
        else:
            #            sleep(int(20/limit[DEBUG]))
            actualITI = ITI(current_task["ITI_failure"])
            export(task_no, session_no, correct_times, "ITI", actualITI)
        session_no += 1

    # task end
    # reward = reward - correct_times
    schedule.clear()
    logger.info("{} end".format(task_no))


def T0():
    logger.info("T0 start")
    global reward, current_task_name, feeds_today
    current_task_name = "T0"
    times = 0
    session_no = 0
    task_no = "T0"
    current_task = ex_flow[task_no]
    hole_lamp_turn("dispenser_lamp", "on")
    export(task_no, session_no, times, "start")
    hole_lamps_turn("off")
    for times in range(0, int(current_task["upper_limit"] / limit[DEBUG])):
        #        if reset_time <= datetime.now():
        #            dispense_all(reward)
        hole_lamp_turn("dispenser_lamp", "on")
        while not is_hole_poked("dispenser_sensor"):
            sleep(0.01)
        dispense_pelet()
        feeds_today += 1
        export(task_no, session_no, times, "reward")
        hole_lamp_turn("dispenser_lamp", "off")
        ITI([4, 8, 16, 32])
        session_no += 1
    reward = reward - times
    logger.info("T0 end")


# time
def is_execution_time(start_end: list):
    """ 実行時刻の開始終了リストを引数にして今実行時間かどうかを判定する """
    start, end = [datetime.combine(datetime.today(), datetime.strptime(time, "%H:%M").time()) for time in start_end]
    # 日付繰り上がりの処理
    start -= timedelta(days=int(start.hour > end.hour)) if datetime.now().hour < end.hour else timedelta(days=0)
    end += timedelta(days=int(start.hour > end.hour)) if datetime.now().hour >= end.hour else timedelta(days=0)
    return start <= datetime.now() <= end


def select_basetime(times="07:00"):
    # (past) last datetime
    hours = int(times.split(":")[0])
    minutes = int(times.split(":")[1])

    today = datetime.today() if datetime.now().time() >= time(hours,
                                                              minutes) else datetime.today() - timedelta(days=1)
    return datetime.combine(today, time(hours, minutes))


def ITI(secs: list):
    if DEBUG:
        secs = [2]  # changed
    selected = choice(secs) if isinstance(secs, list) else secs
    sleep(selected)
    return selected


def dispense_all(feed):
    for f in range(feed):
        dispense_pelet("payoff")
        sleep(5)


def unpayed_feeds_calculate():
    """ 直前の精算時間までに吐き出した餌の数を計上し足りなければdispense_all """
    global current_task_name, reward, exist_reserved_payoff, feeds_today, current_reset_time, payoff_flag
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
            logger.info("reward = {}".format(reward)) if DEBUG else None
            dispense_all(min(1, reward))
            reward -= 1
            payoff_flag = True
            sleep(1 * 60)
        reward = ex_flow[current_task_name].get("feed_upper", 70)
        feeds_today = 0
    daily_log(select_basetime(current_reset_time))


def overpayed_feeds_calculate():
    global feeds_today, current_task_name
    return feeds_today > ex_flow[current_task_name].get("feed_upper", 70)


if __name__ == "__main__":
    try:
        terminate_task = ""
        remained = -1
        # if len(sys.argv) == 1:
        #     logger.info("usage: python app.py mouse_No terminate_task_No remained_number_of_tasks")
        #     sys.exit()
        # mouse_no = sys.argv[1]
        if len(sys.argv) >= 2:
            terminate_task = sys.argv[1]
        if len(sys.argv) == 3:
            remained = int(sys.argv[2])
            logger.info("remained{}".format(remained))
        run(terminate_task, remained)
        print("hello")
    except Exception as e:
        # error log
        print("error occured")
        error_log(e)
        raise
    finally:
        shutdown()
