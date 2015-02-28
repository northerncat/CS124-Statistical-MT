# Utility.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This scripts builds the language models for English and Foreign languages.

import sys, math, json, re, string
from collections import defaultdict
import IBMModel1

def writeDictionary(dictionary, outfile):
	output = open(outfile, 'w+')
	for f in dictionary:
		e = dictionary[f]
		if isinstance(e, list):
			e = '\t'.join(e)
		print >>output, f + '\t' + e
	output.close()

def readDictionary(infile):
	dictionary = {}
	input = open(infile, 'r')
	for line in input:
		tokens = line.rstrip('\r\n').split('\t')
		f = tokens[0]
		e = tokens[1:] if len(tokens) > 2 else tokens[1]
		dictionary[f] = e
	input.close()
	return dictionary

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

def removeEndingNewline(tokens):
	last_token = tokens[-1]
	last_char = last_token[-1]
	if last_char == '\n' or last_char == '\r':
		if len(last_token) == 1:
			tokens = tokens[:-1]
		else:
			tokens[-1] = last_token[:-1]
	return tokens

def isFirstLetterCapital(token):
	return token[0].isupper()

def isAscii(token):
	for ch in token:
		if ch not in string.ascii_letters:
			return False
	return True

# remove consecutive duplicate
def removeConsecutiveDuplicate(tokens):
	lst = []
	for word in tokens:
		if not lst or word != lst[-1]:
			lst.append(word)
	return lst

def removeNULLfromTokens(etokens):
	return [e for e in etokens if e != 'NULL']

def removeNULLfromCorpus(corpus):
	return [(removeNULLfromTokens(eLine), fLine) for (eLine, fLine) in corpus]


bundle2 = [('a', 'the'), ('an', 'the'), ('a', 'an'), ('a', 'a'), ('an', 'an'), ('the', 'the'),
		   ('this', 'this'), ('that', 'that'), ('the', 'this'), ('the', 'that'), ('this', 'that'),
		   ('it', 'we'), ('a', 'their'), ('a', 'his'), ('a', 'her'), ('a', 'one'), ('an', 'one')
		]

AllCapitals = ['NATO']

Nations = [ 'America', 'American', 'Austria', 'Austrian', 'Belgium', 'Europe', 'Asia', 'Africa', 'India',
			'Britain', 'British', 'China', 'Chinese', 'Denmark', 'Danish',
			'Finland', 'Finnish', 'Greece', 'Greek', 'Germany', 'German', 'France', 'French',
			'Italy', 'Italian', 'Iraland', 'Irish', 'Japan', 'Japanese', 'Holland',
			'Norway', 'Netherland', 'Poland', 'Polish', 'Portugal', 'Portuguese', 'Russia', 'Russian',
			'Spain', 'Spanish', 'Switzerland', 'Swiss', 'Sweden', 'Swedish', 'Turk', 'Turkish', 'Ukraine', 'Ukranian',
			'Libya'
			]
Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

Capitalized = Nations + Months

def finalTranslationTokensFixup(etokens):
	"""Remove words that should not be together."""
	if not etokens:
		return etokens

	# check for all captials (this does not improve BLEU score)
	for i in xrange(len(etokens)):
		for x in AllCapitals:
			if etokens[i] == x.lower():
				etokens[i] = x
				break;
	
	# check for capitalized (this does not improve BLEU score)
	for i in xrange(len(etokens)):
		token = etokens[i]
		token = token.capitalize()
		for x in Capitalized:
			if token == x:
				etokens[i] = x
				break;

	# check for nonsense
	newtokens = etokens[:1] # only first element
	for i in xrange(1, len(etokens)):
		e1, e2 = etokens[i].lower(), etokens[i-1].lower()
		bad = False
		for (w1, w2) in bundle2:
			if (e1 == w1 and e2 == w2) or (e1 == w2 and e2 == w1):
				bad = True
				break
		if not bad:
			newtokens.append(etokens[i])
	etokens = newtokens

	# fix article from 'a' -> 'an' or 'an' -> a
	vowels = ['a', 'e', 'i', 'o']
	preps = ['as', 'at', 'in', 'it', 'of', 'on', 'only']
	for i in xrange(0, len(etokens)-1):
		curr, next = etokens[i], etokens[i+1]
		next0 = next[0]
		if curr == 'a':
			if next0 in vowels and next not in preps:
				etokens[i] = 'an'
		elif curr == 'an':
			if next0 not in vowels:
				etokens[i] = 'a'

	# first letter must be captical
	newtokens[0] = newtokens[0].capitalize()
	return newtokens


fixup_patterns = [
	(r'\, (a|the|an|this|that) \,', ','),
	(r' (a|the|an|this|that)( ,)? a ',    ' a '),
	(r' (a|the|an|this|that)( ,)? an ',   ' an '),
	(r' (a|the|an|this|that)( ,)? the ',  ' the '),
	(r' (a|the|an|this|that)( ,)? this ', ' this '),
	(r' not we .', ' we not .'),
	(r' not is ', ' is not '),
	(r' not not ', ' not '),
	(r' we we ', ' we '),
	(r', was to ', ', it was to '),
	]

def finalTranslationStringFixup(esentence):
	if not esentence:
		return esentence
	for (pat, repl) in fixup_patterns:
		esentence = re.sub(pat, repl, esentence)
	return esentence


