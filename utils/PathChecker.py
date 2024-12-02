#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from logging import getLogger, StreamHandler

# Logger
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../python')
    ))


def confirm_directory_path(directory_path, message):
    while not directory_path.is_dir():
        logger.error(
            '{0} directory {1} is not found!!'
            .format(message, directory_path)
            )
        directory_path_str = input(
            '''Please enter the path to the {0} directory.
If you want to exit, type "EXIT". >>> '''.format(message)
            )
        if directory_path_str == 'EXIT':
            sys.exit()
        directory_path = Path(directory_path_str)
    return directory_path


def confirm_file_path(file_path, message):
    while not file_path.is_file():
        logger.error(
            '{0} file {1} is not found!!'
            .format(message, file_path)
            )
        file_path_str = input(
            '''Please enter the path to the {0} file.
If you want to exit, type "EXIT". >>> '''.format(message)
            )
        if file_path_str == 'EXIT':
            sys.exit()
        file_path = Path(file_path_str)
    return file_path
