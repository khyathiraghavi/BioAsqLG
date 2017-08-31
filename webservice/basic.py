from flask import Flask, request, render_template
from flask import jsonify, render_template
import sys
import json
import csv
import sys
import os
import itertools
import numpy as np
import operator
import types
import cPickle as pickle
import pickle
from nltk.tokenize import sent_tokenize
import re
import lucene
from sklearn.cluster import AgglomerativeClustering
from java.nio.file import Paths
#from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher, LegacyNumericRangeQuery
from org.apache.lucene.index import MultiReader
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

from pymedtermino import *
from pymedtermino.snomedct import *
from pymedtermino.umls import *
from pymetamap import MetaMap

from Authentication import *
import requests
import json
from random import randint

#############################################

global gen_dir, cache, cuilist, username, password, version, AuthClient, tgt, uri
gen_dir = "generatedSummaries"

cache = {} #{cui:jsonData}
cuilist=[]

#umls authentication
username = "khyathi"
password = "Oaqa12#$"
#apikey = args.apikey
version = "2016AB"
#identifier = args.identifier
#source = args.source
AuthClient = Authentication(username,password)

#get TGT for our session
tgt = AuthClient.gettgt()
uri = "https://uts-ws.nlm.nih.gov"

"""
#initializing lucene parameters
lucene.initVM()
analyzer = StandardAnalyzer()
reader_first = DirectoryReader.open(SimpleFSDirectory(Paths.get("/home/khyathi/Projects/bioasq/medline17n-lucene1/")))
searcher = IndexSearcher(reader_first)
"""

#############################################

app = Flask(__name__)

@app.route('/initializeGlobalVariables')
def initializeGlobalVariables():
	gen_dir = "generatedSummaries"

def similarity(question, sentence):
	originalQuestion = question[0]
	expandedQuestion = question[1]
	originalSentence = sentence[0]
	expandedSentence = sentence[1]
	originalSimilarity = len(list(set(originalQuestion).intersection(originalSentence)))
	expandedSimilarity = len(list(set(expandedQuestion).intersection(expandedSentence)))
	similarityScore = originalSimilarity + (0.5*expandedSimilarity)
	#return similarityScore
	question1 = question[0] + question[1]
	sentence1 =  sentence[0] + sentence[1]
	#similarity with word2vec tools
	return len(list(set(question1).intersection(sentence1)))

def getUMLSSynonyms(cui):
	synonyms = []
	'''
	if umls_relations_dict.has_key(cui):
		print "found cui"
		jsonData = umls_relations_dict[str(cui)]
		for el in jsonData:
			#print el
			if el['relationLabel'] == 'RL' or el['relationLabel'] == 'RQ':
				# print el['relatedIdName'] , el['relationLabel']
				synonyms.append(el['relatedIdName'])
		return synonyms
	synonyms=[]
	'''
	content_endpoint = "/rest/content/"+str(version)+"/CUI/"+str(cui)
	##ticket is the only parameter needed for this call - paging does not come into play because we're only asking for one Json object
	query = {'ticket':AuthClient.getst(tgt)}
	r = requests.get(uri+content_endpoint,params=query)
	r.encoding = 'utf-8'
	items	= json.loads(r.text)
	jsonData = items["result"]	##ticket is the only parameter needed for this call - paging does not come into play because we're only asking for one Json object
	try:
		query = {'ticket':AuthClient.getst(tgt)}
		r = requests.get(jsonData["relations"],params=query)
		r.encoding = 'utf-8'
		items	= json.loads(r.text)
		jsonData = items["result"]
		#global cache
		cache[cui] = jsonData
		for el in jsonData:
			#if el['relationLabel'] == 'RL' or el['relationLabel'] == 'RQ':
			#if el['relationLabel'] == 'RB':
			#print el['relatedIdName']
			#print "just before raw input inside the if for synonyms"
			#raw_input()
			synonyms.append(el['relatedIdName'])
	except:
		pass
	return synonyms

#EXPANSION FUNCTIONS
def expandConceptsUMLS(sentence):
	try:
		sents = [sentence]
		expandedSent = []
		#sents = ['John had a leukemia']# and heart attack']
		metaConcepts,error = mm.extract_concepts(sents,[1,2])
		for mconcept in metaConcepts:
			cui = mconcept.cui
			#print cui
			#global cuilist
			cuilist.append(cui)
			synonyms = getUMLSSynonyms(cui)
			expandedSent += synonyms
		return list(set(expandedSent))
	except:
		return []


def expandConceptsSNOMEDCT(sentence):
  try:
    sents = [sentence]
    expandedSent = []
    metaConcepts,error = mm.extract_concepts(sents,[1,2])
    for mconcept in metaConcepts:
      cui = mconcept.cui
      terms = cui.terms
      prefName = str(mconcept).split("preferred_name='")[1].split("', cui=")[0] #preferred name
      conceptSyns = SNOMEDCT.search(prefName)  
      for uconcept in conceptSyns:
          for el in uconcept.terms: #.terms is also possible for all synonyms
            expandedSent.append( el )
    return list(set(expandedSent))
  except:
    return []


#CLUSTERING FUNCTION
def cluster(sentenceScoreDict,csumm):
	allSents = []
	totalScore = 0
	for key,value in sentenceScoreDict.iteritems():
		totalScore += value
	if len(sentenceScoreDict)!=0:
		avgScore = float(totalScore)/float(len(sentenceScoreDict))
	else:
		avgScore = float(totalScore)
	for key,value in sentenceScoreDict.iteritems():
		#if value > avgScore:
		allSents.append(key)
	#allSents = ["ac is my", "this is apple", "apple and bottle", "and"]
	sentencePairs = list(itertools.combinations(allSents,2))
	num_clusters=5
	summaryFinal = ""
	if len(allSents) < 10:
		#f = open(gen_dir+"bioasq."+str(csumm)+".txt",'w')
		for sent in allSents:
			#f.write(sent+"\n")
			summaryFinal += sent
		#f.close()
		return summaryFinal
	model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', connectivity=None, linkage='average', compute_full_tree='auto')
	X = np.zeros((len(allSents), len(allSents)))
	for i in range( len(sentencePairs) ):
		index1 = allSents.index(sentencePairs[i][0])
		index2 = allSents.index(sentencePairs[i][1])
		distance =	1- similarity( sentencePairs[i][0].split(), sentencePairs[i][1].split() )
		if index1 <= index2:
			X[index1, index2] = distance
		else:
			X[index2, index1] = distance
	class_labels = model.fit_predict(X)
	selectedSents = {}
	for i in range(len(allSents)):
		curClass = class_labels[i]
		curSent = allSents[i]
		if curClass in selectedSents:
			curScore = sentenceScoreDict[curSent]
			if curScore > selectedSents[curClass][1]:
				selectedSents[curClass] = ( curSent , sentenceScoreDict[curSent] )
		else:
			selectedSents[curClass] =	( curSent , sentenceScoreDict[curSent] )		 
		score = sentenceScoreDict[curSent]
	#print "OUR SUMMMAAARRRRRRRRRRRRRRRRYYYYYYYYYYYYYYYYYYYYYYYY"
	newselectedSents = {}
	for key,value in selectedSents.iteritems():
		newselectedSents[value[0]] = value[1]
	#print newselectedSents
	sorted_ss = sorted(newselectedSents.items(), key=operator.itemgetter(1), reverse=True)
	sumLen=0
	#f = open(gen_dir+"bioasq."+str(csumm)+".txt",'w')
	summaryFinal = ""
	for pair in sorted_ss:
		summarySentence = pair[0]
		sumLen += len(summarySentence.split())
		if sumLen<=200:
			#f.write(summarySentence+"\n")
			summaryFinal += summarySentence
	#f.close()
	return summaryFinal
	#print selectedSents
	#exit(1)



#@app.route('/readData', methods=['GET', 'POST'])
def readData(NoExpansionFlag, UmlsExpansionFlag, SnomedctExpansionFlag):
	if NoExpansionFlag == False and UmlsExpansionFlag ==  False and SnomedctExpansionFlag == False:
		return ("","","")
	#instantiating metamap
	mm = MetaMap.get_instance('/home/khyathi/installations/public_mm/bin/metamap')
	start_command = "/home/khyathi/installations/public_mm/bin/skrmedpostctl start"
	os.system(start_command)
	randomNumber = randint(0, 100)
	csumm=0
	infile = open(sys.argv[1],'r')
	data = json.load(infile)
	#c=1
	for (i, question) in enumerate(data['questions']):
		if question['type'] == 'summary':
			csumm +=1
		if csumm != randomNumber:
			continue
		#if csumm >=3:
		#	break
		quest = unicode(question['body']).encode("ascii","ignore")
		questionBow = quest.split()
		expandedQuestion = [questionBow] + [[]]
		if NoExpansionFlag == True:
			expandedQuestion = [questionBow] + [[]]
		elif UmlsExpansionFlag == True:
			expandedQuestion = [questionBow] + [expandConceptsUMLS(quest)]
		elif SnomedctExpansionFlag == True:
			expandedQuestion = [questionBow] + [expandConceptsSNOMEDCT(quest)]
		#print expandedQuestion
		#raw_input()
		ideal_summaries = question["ideal_answer"]
		ideal_answer_sents = []
		if isinstance(ideal_summaries, types.StringTypes):
			ideal_answer_sents = sent_tokenize(ideal_summaries)
		else:
			ideal_answer_sents = sent_tokenize(ideal_summaries[0])
		"""
		out = open("./ideal_summaries1/bioasq."+str(csumm)+".txt", "w")
		for sentence in ideal_answer_sents:
			out.write(unicode(sentence).encode("ascii","ignore")+"\n")
		out.close()
		"""
		snippets = question['snippets']
		#documents = question['documents']
		sentences = []
		sentenceScoreDict = {}
		snippetsText = []
		for snippet in question['snippets']:
			text = unicode(snippet["text"]).encode("ascii", "ignore")
			snippetsText.append(text)
			if text == "":
				continue
			try:
				sentences += sent_tokenize(text)
			except:
				sentences += text.split(". ")
			#print sentences
			#exit(1)
			#for document in question['documents']:
			#print document
			#abstractText = unicode( retrieve(document) ).encode("ascii","ignore")
			#if abstractText == "":
			#  continue
			#try:
			#  sentences += sent_tokenize(abstractText)
			#except:
			#  sentences += abstractText.split(". ")
		for sentence in sentences:
			sentenceBow = sentence.split()
			expandedSentence = [sentenceBow] + [[]]
			if NoExpansionFlag == True:
				expandedSentence = [sentenceBow] + [[]]
			elif UmlsExpansionFlag == True:
				expandedSentence = [sentenceBow] + [expandConceptsUMLS(sentence)]
			elif SnomedctExpansionFlag == True:
				expandedSentence = [sentenceBow] + [expandConceptsSNOMEDCT(sentence)]
			similarityScore = similarity(expandedQuestion, expandedSentence)
			sentenceScoreDict[sentence] = similarityScore  
		summaryFinal = cluster(sentenceScoreDict,csumm)
		#print "generated summary " + str(csumm)
		#question = "When does the antipeptic action of bisabolol occur with a pH-value?"
		pickle.dump(cache, open("cached_umls_json.pkl","wb"))
		pickle.dump(cuilist, open("cui.pkl","wb"))
		stop_command = "/home/khyathi/installations/public_mm/bin/skrmedpostctl stop"
		#stop_command = "~/public_mm/bin/skrmedpostctl stop"
		os.system(stop_command)
		#exit(1)
		return (quest, snippetsText, summaryFinal)

#helper function
def request_wants_json():
	best = request.accept_mimetypes.best_match(['application/json', 'text/html'])
	return best == 'application/json' and request.accept_mimetypes[best] > request.accept_mimetypes['text/html']

#http://127.0.0.1:5000/?expansion=UMLS
@app.route('/', methods=['GET', 'POST'])
def index():
	initializeGlobalVariables()
	NoExpansionFlag = False
	UmlsExpansionFlag = False
	SnomedctExpansionFlag = False
	#if UmlsExpansionFlag == False and NoExpansionFlag==False:
	#	return ("","","")
	"""
	if request.method == 'POST':
		if request.form['submit'] == 'UMLS Expansion':
			UmlsExpansionFlag = True
		elif request.form['submit'] == 'No Expansion':
			NoExpansionFlag = True
		elif request.form['submit'] == 'SNOMEDCT Expansion':
			SnomedctExpansionFlag = True
	"""
	expansionType = request.args.get('expansion')
	if expansionType == "UMLS":
		UmlsExpansionFlag = True
	elif expansionType == "SNOMEDCT":
		SnomedctExpansionFlag = True
	else:
		NoExpansionFlag = True
	question, snippets, summaryFinal = readData(NoExpansionFlag, UmlsExpansionFlag, SnomedctExpansionFlag)
	if request_wants_json():
		#summaryFinal = ["1","2"]
		#return jsonify(summaryFinal=[x.to_json() for x in summaryFinal])
		return jsonify(question = question, snippets = snippets, summaryFinal = summaryFinal)
	return render_template("bioasq.html", question = question, snippets = snippets, summaryFinal = summaryFinal)
	#return 'Welcome to the homepage of bioasq'


@app.route('/load', methods=['GET', 'POST'])
def load():
	return 'This is the image'

if __name__ == "__main__":
	app.run(debug = True)
