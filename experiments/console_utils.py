"""Salida UTF-8 consistente en Windows, terminales y registros de procesos."""

import os
import sys


def utf8_console():
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONUTF8']='1'
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
