'''
Description: functions for train
Author: GuoYi
Date: 2021-06-15 12:07:08
LastEditTime: 2021-06-15 12:07:52
LastEditors: GuoYi
'''

import os 

## ----------------------------------------
def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)


