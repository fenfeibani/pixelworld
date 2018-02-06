"""Simple logging code"""

from __future__ import print_function
from __future__ import absolute_import

import atexit
import datetime
import dateutil

import os

LOG_FILE = None
LOG_PREFIX = ""

def close_log():
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()
        LOG_FILE = None

def open_log(path, filename, prefix=""):
    global LOG_FILE
    global LOG_PREFIX
    if LOG_FILE is None:
        if not os.path.exists(path):
            os.makedirs(path)
        LOG_FILE = open(os.path.join(path, filename), 'a')
        LOG_PREFIX = prefix
        atexit.register(close_log)
    else:
        raise Exception("Log already opened")

def log(text):
    global LOG_FILE
    global LOG_PREFIX
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    log_line = '%s[%s] %s\n' % (LOG_PREFIX, timestamp, text)
    
    if LOG_FILE is None:
        print("!!!! Logging without log file set !!!!")
        print("!!!", log_line)
    else:
        LOG_FILE.write(log_line)
        print("!!!", log_line)
