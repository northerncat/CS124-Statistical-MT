# LanguageModel.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This scripts builds the language models for English and Foreign languages.

import sys, math, json, re
from collections import defaultdict

def outputObject(outfile, dict_list): # causing all strings to become unicode
	file = open(outfile, 'w+')
	json.dump(dict_list, file, ensure_ascii=True, encoding="utf-8")
	file.close()

def inputObject(infile): # causing all strings to become unicode
	file = open(infile)
	dict_list = json.load(file, encoding="utf-8")
	file.close()
	return dict_list

def findSublist(sub, full):
	if len(sub) > len(full):
		return -1
	for i in xrange(len(full)-len(sub)+1):
		if sub[0] == full[i]:
			match = True
			for j in xrange(len(sub)):
				if sub[j] != full[i+j]:
					match = False
					break;
			if match:
				return i
	return -1

def isRangeOverlapped(x1, y1, x2, y2):
	return (x1 <= x2 and x2 <= y1) or (x2 <= x1 and x1 <= y2)