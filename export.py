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


def export(task_no: str, times: int, event_type: str, hole_no=0):
    string = ','.join([str(datetime.now()), task_no, str(times), event_type, str(hole_no)])
    logfile.write(string)
    logfile.flush()
    print(string)


def magagine_log(reason, amount=1):
    """ 餌やりのログ 記入事項：「日付, 量(粒),報酬・精算」 """
    string = ','.join([str(datetime.now()), amount, reason])
    dispence_log_file.write(string)
    dispence_log_file.flush()


def error_log(error):
    logging.exception(error)
    raise


def files_close():
    logfile.close()
    dispence_log_file.close()
