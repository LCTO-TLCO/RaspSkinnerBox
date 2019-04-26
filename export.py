#! /usr/bin python3
# coding:utf-8
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename='error.txt',
    filemode='a')

# define
bounce_time = 300
logfile = open('epsilon-greedy.txt', 'a+')
dispence_log_file = open('dispence_feed.csv', 'a+')

class pycolor:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'    

def export(task_no: str, session_no:int, times: int, event_type: str, hole_no=0):
    logstring = ','.join([str(datetime.now()), task_no, str(session_no), str(times), event_type, str(hole_no)])
    logfile.write(logstring+"\n")
    logfile.flush()
    print(logstring.replace('time over',pycolor.YELLOW+'time over'+pycolor.END).replace('reward',pycolor.GREEN+'reward'+pycolor.END).replace('failure',pycolor.YELLOW+'failure'+pycolor.END).replace('task called',pycolor.WHITE+'task called'+pycolor.END))

def magagine_log(reason, amount=1):
    """ 餌やりのログ 記入事項：「日付, 量(粒),報酬・精算」 """
    string = ','.join([str(datetime.now()), str(amount), reason])
    dispence_log_file.write(string)
    dispence_log_file.flush()


def error_log(error):
    logging.exception(error)
    raise


def files_close():
    logfile.close()
    dispence_log_file.close()
