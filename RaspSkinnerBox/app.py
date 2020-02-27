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
    session_no = last_session_id(current_task_name)
    current_task_name = task_no
    current_reset_time = current_task.get("reset_time", "07:00") if not DEBUG else "06:00"
    schedule.every().day.at(current_reset_time).do(unpayed_feeds_calculate)
    feeds_today = int(calc_todays_feed(select_basetime(current_reset_time) + timedelta(days=1)))
    correct_times = 0
    is_time_limit_task = "time" in current_task
    if remained == -1:
        correct_times = 0
    else:
        correct_times = int(current_task.get("upper_limit", 0) / limit[DEBUG]) - remained
    if "criterion" in current_task:
        correct_times = remained
    if correct_times < 0:
        correct_times = 0

    reward = ex_flow[current_task_name].get("feed_upper", 70)
    # start time
    start_time = ex_flow[task_no].get("start_time", False)
    # タスク指定開始時のみbasetimeを過去にする
    if start_time:
        start_time = (select_basetime(start_time) if list(ex_flow.keys())[0] == current_task_name else select_basetime(
            start_time) + timedelta(days=1))
    payoff_flag = False
    feeds_today = 0 if DEBUG else feeds_today
    rising = False

    def is_continue():
        if current_task.get("criterion", False):
            session_data = select_last_session_log(20, current_task_name)
            crit_and = []
            crit_or = []
            crit_and.append(current_task.get("trials", True) < session_no)
            crit_and.append(current_task.get("accuracy", True) / 100 < session_data["accuracy"])
            crit_or.append(
                current_task.get("or", {"omission": False}).get("omission", False) / 100 > session_data["omission"])
            crit_or.append(current_task.get("or", {"correct": False}).get("correct", False) <= correct_times)
            japanese_dict = {True: "達成中", False: "未達成"}
            if not (overpayed_feeds_calculate() or start_time or not any(
                    list(map(is_execution_time, current_task.get("time", [["00:00", "23:59"]]))))):
                export_crit(task_no, session_no, session_data["accuracy"] * 100, session_data["omission"] * 100,
                            correct_times)
                print("trials:{0} & accuracy:{1} & (omission:{2} or correct num:{3})".format(
                    japanese_dict[crit_and[0]], japanese_dict[crit_and[1]], japanese_dict[crit_or[0]],
                    japanese_dict[crit_or[1]]))
            return not all([all(crit_and), any([any(crit_or), not bool(current_task.get("or", False))])])
        return correct_times < int(current_task["upper_limit"] / limit[DEBUG])

    # main
    while is_continue():
        # task start
        if start_time:
            if start_time > datetime.now():
                if is_holes_poked(current_task["target_hole"]):
                    if not rising:
                        export(task_no, session_no, correct_times, "nosepoke before task start time",
                               is_holes_poked(current_task["target_hole"]))
                        rising = True
                else:
                    rising = False
                if all([datetime.now().minute % 5, datetime.now().second == 0]):
                    logger.info("task stopping ... not after {}".format(start_time))
                sleep(5)
                continue
        schedule.run_pending()
        if current_task.get("terminate_when_payoff", False) and payoff_flag:
            logger.info("terminate with payoff!")
            break
        payoff_flag = False
        if is_time_limit_task:
            if (not any(list(map(is_execution_time, current_task["time"])))) and exist_reserved_payoff:
                unpayed_feeds_calculate()
                logger.info("terminate payoff complete!")
                continue
            elif not any(list(map(is_execution_time, current_task["time"]))):
                if is_holes_poked(current_task["target_hole"]):
                    if not rising:
                        export(task_no, session_no, correct_times, "nosepoke not between execution time",
                               is_holes_poked(current_task["target_hole"]))
                        rising = True
                else:
                    rising = False
                if all([datetime.now().minute % 5, datetime.now().second == 0]):
                    logger.info("pending ... not in {}".format(current_task["time"]))
                sleep(1)
                continue
        if overpayed_feeds_calculate():
            if is_holes_poked(current_task["target_hole"]):
                if not rising:
                    export(task_no, session_no, correct_times, "nosepoke when overpayed",
                           is_holes_poked(current_task["target_hole"]))
                    rising = True
            else:
                rising = False
            if all([datetime.now().minute % 5, datetime.now().second == 0]):
                logger.info("over payed ...")
            sleep(1)
            continue
        export(task_no, session_no, correct_times, "start")
        hole_lamp_turn("dispenser_lamp", "on")

        # task call
        rising = False
        if current_task.get("task_call", False):
            if not DEBUG:
                while not is_hole_poked("dispenser_sensor"):
                    if is_holes_poked(current_task["target_hole"]):
                        if not rising:
                            export(task_no, session_no, correct_times, "nosepoke before task call",
                                   is_holes_poked(current_task["target_hole"]))
                            rising = True
                    else:
                        rising = False
                    if not any(list(map(is_execution_time, current_task.get("time", [["00:00", "23:59"]])))):
                        break
                    sleep(0.01)
            elif DEBUG:
                input("type ENTER key to call task")
            if current_task.get("time", False) and not any(
                    list(map(is_execution_time, current_task.get("time", [["00:00", "23:59"]])))):
                hole_lamp_turn("dispenser_lamp", "off")
                continue
            export(task_no, session_no, correct_times, "task called")
            hole_lamp_turn("dispenser_lamp", "off")

            # cue delay
            premature = False
            timelimit = False
            base_time = datetime.now()
            cue_delay = current_task.get("cue_delay", 5) if isinstance(current_task.get("cue_delay", 5),
                                                                       int) else choice(current_task["cue_delay"])
            export(task_no, session_no, correct_times, "cue delay", cue_delay)
            while not (premature or timelimit):
                if (datetime.now() - base_time).seconds >= cue_delay:
                    timelimit = True
                if is_holes_poked(current_task["target_hole"], False) and current_task.get("premature", False):
                    premature = True
                    export(task_no, session_no, correct_times, "premature",
                           is_holes_poked(current_task["target_hole"]))
                sleep(0.05)
            if premature:
                continue

        # hole setup
        q_holes = [choice(current_task["target_hole"])]
        target_holes = current_task["target_hole"]
        stimule_holes = target_holes if current_task.get("stimulate_all", False) else q_holes
        check_holes = target_holes if current_task.get("check_all", False) else q_holes

        hole_lamps_turn("on", stimule_holes)
        export(task_no, session_no, correct_times, "pokelight on", "/".join([str(x) for x in stimule_holes]))

        # time
        end_time = False
        holelamp_end_time = False
        if current_task["limited_hold"] >= 0:
            # バグを吐くようにする
            end_time = datetime.now() + timedelta(seconds=current_task["limited_hold2"])
            holelamp_end_time = datetime.now() + timedelta(seconds=current_task["limited_hold"])
        hole_poked = False
        is_correct = False
        time_over = False
        while not (hole_poked or time_over):
            h = is_holes_poked(target_holes)
            # if h in q_holes:
            if h:
                export(task_no, session_no, correct_times, "nose poke", h)
                hole_poked = True
                is_correct = h in check_holes
                export(task_no, session_no, correct_times, "reward" if is_correct else "failure", h)
                correct_times += is_correct
            # time over
            if end_time:
                if end_time < datetime.now():
                    time_over = True
                    export(task_no, session_no, correct_times, "time over")
            if holelamp_end_time and not time_over:
                if holelamp_end_time < datetime.now():
                    hole_lamps_turn("off", stimule_holes)
                    export(task_no, session_no, correct_times, "pokelight off",
                           "/".join([str(x) for x in stimule_holes]))
                    holelamp_end_time = False
            sleep(0.01)
        # end
        hole_lamps_turn("off", stimule_holes)
        # hole_lamp_turn("house_lamp", "off")
        if is_correct:
            hole_lamp_turn("dispenser_lamp", "on")
            dispense_pelet()
            feeds_today += 1
            sleep(1)
            # perseverative response measurement after reward & magazine nose poke detection
            while not is_hole_poked("dispenser_sensor"):
                if is_holes_poked(target_holes):
                    export(task_no, session_no, correct_times, "nose poke after rew", is_holes_poked(target_holes))
                    while is_holes_poked(target_holes):
                        sleep(0.01)
                sleep(0.01)
            export(task_no, session_no, correct_times, "magazine nose poked")
            hole_lamp_turn("dispenser_lamp", "off")
            actualITI = ITI(current_task["ITI_correct"], correct_times=correct_times)
            export(task_no, session_no, correct_times, "ITI", actualITI)
        else:
            #            sleep(int(20/limit[DEBUG]))
            actualITI = ITI(current_task["ITI_failure"], correct_times=correct_times)
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
    for times in range(0, int(current_task.get("upper_limit", 50) / limit[DEBUG])):
        #        if reset_time <= datetime.now():
        #            dispense_all(reward)
        hole_lamp_turn("dispenser_lamp", "on")
        while not is_hole_poked("dispenser_sensor"):
            sleep(0.01)
        dispense_pelet()
        feeds_today += 1
        export(task_no, session_no, times, "reward")
        hole_lamp_turn("dispenser_lamp", "off")
        ITI(current_task.get("ITI", [4, 8, 16, 32]), correct_times=times)
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


def ITI(secs: list, correct_times=-1):
    global current_task_name, ex_flow
    if DEBUG:
        secs = [2]  # changed
    selected = choice(secs) if isinstance(secs, list) else secs
    current_task = ex_flow[current_task_name]
    end_time = datetime.now() + timedelta(seconds=selected)
    rising = False
    while datetime.now() <= end_time:
        if is_holes_poked(current_task["target_hole"]):
            if not rising:
                export(current_task_name, last_session_id(current_task_name) - 1, correct_times,
                       "nosepoke when overpayed",
                       is_holes_poked(current_task["target_hole"]))
                rising = True
        else:
            rising = False
        sleep(0.1)
    return selected


def dispense_all(feed):
    global current_task_name
    for f in range(feed):
        dispense_pelet("payoff")
        export(current_task_name, -1, -1, "payoff")
        sleep(60)


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
        remain_reward = reward - feeds_today
        # dispense
        if DEBUG:
            reward = 3
        logger.info("payoff_num:{}".format(remain_reward))
        while remain_reward > 0:
            print("payoff: reward = {}".format(remain_reward)) if DEBUG else None
            export(current_task_name, -1, -1, "payoff")
            dispense_all(1)
            remain_reward -= 1
            payoff_flag = True
        reward = ex_flow[current_task_name].get("feed_upper", 70)
    feeds_today = 0
    daily_log(select_basetime(current_reset_time))
    logger.info("unpayed_feeds_calculate complete")


def overpayed_feeds_calculate():
    global feeds_today, current_task_name
    if ex_flow[current_task_name].get("overpay", True):
        return feeds_today > ex_flow[current_task_name].get("feed_upper", 70)
    return False


if __name__ == "__main__":
    try:
        terminate_task = ""
        remained = -1
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
