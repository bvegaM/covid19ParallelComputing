
### Library for read files
import fnmatch
import os.path
import json
import requests
from urllib.request import urlopen, Request
import urllib.request
from bs4 import BeautifulSoup
from os import remove

### Library for preprocessing
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

### Library for data manipulate
import pandas as pd
import math
import numpy as np
import time
import sys
import gc

### Library for paralell processing
import multiprocessing

import cudf
import cupy as cp
from cuml import TruncatedSVD
from cuml.decomposition import TruncatedSVD

class Preprocessing:

    def __init__(self,textValue):
        self.textValue=textValue

    def preprocessingFile(self,text):
        listTextProcessing = ()
        textProcessing     = self.wordTokenizer(self.textValue[text])
        textProcessing     = self.removeTrashWords(textProcessing)
        textProcessing     = self.removeTrashLetters(textProcessing)
        textProcessing     = self.wordLemantizer(textProcessing)
        textProcessing     = self.filterWordsShort(textProcessing)
        listTextProcessing=(text,textProcessing)
        return listTextProcessing

    def wordTokenizer(self,text):
        return text.split()
    
    def removeTrashWords(self,text):
        return [i for i in text if i not in stopwords.words('english')]
    
    def removeTrashLetters(self,text):
        return [i for i in text if i not in list(string.ascii_letters)]

    def wordLemantizer(self,text):
        return [WordNetLemmatizer().lemmatize(i, pos='v') for i in text]
    
    def filterWordsShort(self,text):
        return [i for i in text if len(i) > 2]
    
class ReadFile:

    def obtainPathFiles(self,path):
        filesPath=[]
        for roots,dirs,files in os.walk(path):
            for file in files:
                if fnmatch.fnmatch(file,'*.json'):
                    direction = roots+"/"+file
                    filesPath.append(direction)
        return filesPath
    
    def readFiles(self,filePath):
        listFiles     = ()
        read          = json.loads(open(filePath).read())
        finalText     = ""
        if('abstract' not in read.keys() or len(read['abstract'])==0):
            finalText = ""
            remove(filePath)
        else:
            text      = read['abstract']    
            for j in text:
                finalText = finalText+j['text']+" "
            finalText = finalText.lower()
            finalText = finalText.translate({ord(i): None for i in '!@#$%^&*()-_=+[{]}\|;:<,>.?/1234567890'})
        listFiles = (read['paper_id'],finalText)
        return listFiles

    def readWebPage(self,url):
        result  = ()
        hdr     = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
        req      = Request(url[0], headers=hdr)
        data     = urllib.request.urlopen(req).read()
        soup     =  BeautifulSoup(data,'html.parser')
        tags     = soup('p')
        finalText=""
        for tag in tags:
            finalText = finalText+tag.get_text()+" "
        finalText = finalText.lower()
        finalText = finalText.replace("’","")
        finalText = finalText.replace("”","")
        finalText = finalText.replace("“","")
        finalText = finalText.translate({ord(i): None for i in '!@#$%^&*()-_=+[{]}\|;:<,>.?/1234567890'})
        title=url[1]+" new"
        result=(title,finalText)
        return result

class TermFrequency:

    def __init__(self,textDictionary):
        self.textDictionary=textDictionary
        self.countDictionary={}

    def joinDictionary(self,text):
        joinDictionary={}
        for i in text:
            for j in i:
                joinDictionary[j]=i[j]
        return joinDictionary

    def wordsFrequency(self,textDictionary):
        wordDocument = {}
        for i in textDictionary.values():
            for j in i:
                if j not in wordDocument.keys():
                    wordDocument[j] = 1
                else:
                    wordDocument[j] = wordDocument[j] + 1
        return wordDocument

    def filterVocabulary(self,wordFrequency):
        return [(i,wordFrequency[i]) for i in wordFrequency if wordFrequency[i]>=5]

    def obtainWordsDocument(self):
        for i in self.textDictionary:
            aux={}
            aux[i]=self.textDictionary[i]
            self.countDictionary[i]=self.wordsFrequency(aux)
    
    def obtainTermDocument(self,document):
        aux={}
        termDocumentDictionary={}
        for i in self.countDictionary:
            if document[0] not in self.countDictionary[i].keys(): 
                aux[i]=0
            else:
                aux[i]=self.countDictionary[i][document[0]]
        termDocumentDictionary[document[0]]=aux
        return termDocumentDictionary

    def buildTermDocumentMatrix(self,termDocument):
        documents=termDocument[list(termDocument.keys())[0]].keys()
        dframe = pd.DataFrame([key for key in documents], columns=['words'])
        for i in termDocument:
            dframe[i]=[termDocument[i][j] for j in termDocument[i]]
        dframe=dframe.transpose()
        index=list(dframe.loc['words'])
        dframe.columns=index
        dframe=dframe.drop(['words'],axis=0)
        return dframe
       
class SvdProcess():

    def obtainSvdMatrix(self,tfidfMatrix,componentsN):
        tfidfMatrixTranspose = tfidfMatrix.transpose()
        values               = tfidfMatrixTranspose.values
        dataFrameCudf        = cudf.DataFrame([values[i] for i in range(len(values))],dtype=cp.float32)
        tsvdT_float          = TruncatedSVD(n_components = componentsN, algorithm = "jacobi", n_iter = 20, tol = 1e-9)
        tsvdT_float.fit(dataFrameCudf)        
        return tsvdT_float.transform(dataFrameCudf)

class TfidfMatrix:

    def __init__(self,len,term):
        self.lenFiles=len
        self.termDocument=term
    
    def obtainIdf(self,word):
        count=0
        idfValue=0
        for j in self.termDocument[word]:
            if(self.termDocument[word][j]!=0):
                count+=1
        if (count!=0):
            idfValue=math.log10(self.lenFiles/count)
        else:
            idfValue=0
        return idfValue
    
    def obtainTfidf(self,termDocumentMatrix,idf):
        for i in termDocumentMatrix:
            termDocumentMatrix[i]=termDocumentMatrix[i]*idf
        return termDocumentMatrix

class CosineSimilarity:
    
    def __init__(self,tfidfMatrix,tfIdfMatrixWeb):
        self.web = tfIdfMatrixWeb
        self.documents = tfidfMatrix
    
    def obtainCosineSimilarity(self,value):
        dataFrame=pd.DataFrame(self.web.values,columns=self.web.columns)
        result=[]
        for i in dataFrame:
            dot_product = np.dot(np.array(dataFrame[i]),np.array(self.documents[value]))
            norm_a = np.linalg.norm(np.array(dataFrame[i]))
            norm_b = np.linalg.norm(np.array(self.documents[value])) 
            cosine=dot_product/(norm_a*norm_b)
            result.append(cosine)
        return result
    
    def buildCosineMatrix(self,listCosine):
        dataFrame=pd.DataFrame(self.web.values,columns=self.web.columns)
        cosineDataFrame=pd.DataFrame(columns=dataFrame.columns)
        for i in listCosine:
            cosineDataFrame.loc[len(cosineDataFrame)]=i
        cosineDataFrame=cosineDataFrame.set_index(self.documents.columns)
        return cosineDataFrame
    

if __name__=="__main__":
    
    ## lectura de archivos JSON
    readFile       = ReadFile()
    direction      = "/home/usuario/Descargas/covid/documentos/pdf_json"
    start_time     = time.time()
    filesPath      = readFile.obtainPathFiles(direction)
    filesPath      = filesPath[0:7000]
    pool           = multiprocessing.Pool(processes=32)
    textList       = pool.map(readFile.readFiles,filesPath)
    pool.close() 
    pool.join()
    textList       = dict(textList)
    print('*'*10,"time - Read Files process = ",(time.time()-start_time),' seconds ','| final documents read: ',len(textList),' | memory used: ',sys.getsizeof(textList),'bytes ','*'*10)

    ## preprocesamiento de archivos JSON
    preprocessing      = Preprocessing(textList)
    start_time         = time.time()
    pool               = multiprocessing.Pool(processes=32)
    textProcessing     = pool.map(preprocessing.preprocessingFile,textList)
    pool.close() 
    pool.join()
    textProcessing     = dict(textProcessing)
    print('*'*10,"time - Preprocessing process = ",(time.time()-start_time),' seconds ','| final documents processed: ',len(textProcessing),' | memory used: ',sys.getsizeof(textProcessing),'bytes ','*'*10)
    

    ## frecuenciaTermino
    start_time         = time.time()
    termFrequency      = TermFrequency(textProcessing)
    frequency          = termFrequency.wordsFrequency(textProcessing)
    vocabulary         = termFrequency.filterVocabulary(frequency)
    print('*'*10,"time - vocabulary process = ",(time.time()-start_time),' seconds ','| terms in vocabulary: ',len(vocabulary),' | memory used: ',sys.getsizeof(vocabulary),'bytes ','*'*10)
    
    ## borrando variables innecesarias
    del textList
    del textProcessing
    del frequency
    gc.collect()

    ## Termino Documento
    start_time         = time.time()    
    termFrequency.obtainWordsDocument()
    pool               = multiprocessing.Pool(processes=32)
    termDocument       = pool.map(termFrequency.obtainTermDocument ,vocabulary)
    pool.close() 
    pool.join()
    termDocument       = termFrequency.joinDictionary(termDocument)
    termDocumentMatrix = termFrequency.buildTermDocumentMatrix(termDocument)
    print('*'*10,"time - matrix term Document process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(termDocumentMatrix),' ','*'*10)

    ## TF-IDF
    tfidf              = TfidfMatrix(len(filesPath),termDocument)
    start_time         = time.time()
    pool               = multiprocessing.Pool(processes=32)
    idf                = pool.map(tfidf.obtainIdf,termDocument)
    pool.close() 
    pool.join()
    print('*'*10,"time - matrix tf-idf process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(tfidfMatrix),' ','*'*10)
    tfidfMatrix        = tfidf.obtainTfidf(termDocumentMatrix,idf)

    ### SVD
    svdProcess         = SvdProcess()
    start_time         = time.time()
    svdJson            = svdProcess.obtainSvdMatrix(tfidfMatrix,300)
    svdJson.to_csv('tfJson.csv')
    print('*'*10,"time - matrix svd process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(svdJson),' ','*'*10)
    
    del termDocument
    del termDocumentMatrix
    del tfidfMatrix
    del tfidf
    del idf
    gc.collect()

    #########################################################################################################################################
    ##################################################### PROCESO PARA NOTICIAS #############################################################
    #########################################################################################################################################

    ## Creacion de una lista con url de páginas web
    urlList = []
    urlList.append(("https://www.biorxiv.org/content/10.1101/2020.04.03.022723v1.full","article 1"))
    urlList.append(("https://www.biorxiv.org/content/10.1101/2020.02.05.936013v1.full","article 2"))
    urlList.append(("https://www.biorxiv.org/content/10.1101/2020.04.10.036418v1.full","article 3"))
    urlList.append(("https://www.biorxiv.org/content/10.1101/2020.04.12.025577v1.full","article 4"))
    urlList.append(("https://www.biorxiv.org/content/10.1101/2020.03.03.975524v1.full","article 5"))
    urlList.append(("https://www.nytimes.com/2020/05/08/health/fda-coronavirus-spit-test.html?action=click&module=Top%20Stories&pgtype=Homepage","real 1"))
    urlList.append(("https://www.theguardian.com/world/2020/may/08/revealed-uk-scientists-fury-over-attempt-to-censor-covid-19-advice","real 2"))
    urlList.append(("https://www.nature.com/articles/d41586-020-01284-x","real 3"))
    urlList.append(("https://www.nytimes.com/2020/05/01/health/coronavirus-remdesivir.html?searchResultPosition=10","real 4"))
    urlList.append(("https://theconversation.com/coronavirus-treatments-what-drugs-might-work-against-covid-19-135352","real 5"))
    urlList.append(("https://futurism.com/pandemic-coronavirus-zombies","fake 1"))
    urlList.append(("https://www.dimsumdaily.hk/covid-19-apocalypse-coronavirus-pandemic-to-stay-for-years-even-with-vaccine-highest-single-day-1281-cases-in-venezuela-record-103-new-cases-in-south-korea/","fake 2"))
    urlList.append(("https://balkaninsight.com/2020/07/07/conspiratorial-corona-hoaxes-and-conspiracy-theories-in-the-balkans/","fake 3"))
    urlList.append(("https://www.euronews.com/2020/05/15/what-is-the-truth-behind-the-5g-coronavirus-conspiracy-theory-culture-clash","fake 4"))
    urlList.append(("https://www.forbes.com/sites/kenrapoza/2020/04/03/the-post-coronavirus-world-may-be-the-end-of-globalization/#329758d37e66","fake 5"))
    
    
    

    ## lectura de texto de paginas web
    start_time=time.time()
    pool                  = multiprocessing.Pool(processes=32)
    textWebList           = pool.map(readFile.readWebPage,urlList)
    pool.close() 
    pool.join()
    textWebList           = dict(textWebList)
    print('*'*10,"time - Read Web Page process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(textWebList),' ','*'*10)

    ## Preprocesamiento de textos de pagina web
    preprocessing      = Preprocessing(textWebList)
    start_time            = time.time()
    pool                  = multiprocessing.Pool(processes=32)
    textWebProcessing     = pool.map(preprocessing.preprocessingFile,textWebList)
    pool.close() 
    pool.join()
    textWebProcessing     = dict(textWebProcessing)
    print('*'*10,"time - Preprocessing process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(textWebProcessing),' ','*'*10)

    ## Termino Documento web
    start_time            = time.time()
    termFrequency         = TermFrequency(textWebProcessing)
    pool                  = multiprocessing.Pool(processes=32)
    termFrequency.obtainWordsDocument()
    termDocumentWeb       = pool.map(termFrequency.obtainTermDocument,vocabulary)
    pool.close() 
    pool.join()
    termDocumentWeb       = termFrequency.joinDictionary(termDocumentWeb)
    termDocumentMatrixWeb = termFrequency.buildTermDocumentMatrix(termDocumentWeb)
    print('*'*10,"time - matrix term Document process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(termDocumentMatrixWeb),' ','*'*10)

    ## TF-IDF Documentos Web
    start_time         = time.time()
    tfidf              = TfidfMatrix(len(urlList),termDocumentWeb)
    pool               = multiprocessing.Pool(processes=32)
    idfWeb             = pool.map(tfidf.obtainIdf,termDocumentWeb)
    pool.close() 
    pool.join()
    tfidfMatrixWeb     = tfidf.obtainTfidf(termDocumentMatrixWeb,idfWeb)
    print('*'*10,"time - matrix tf-idf process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(tfidfMatrixWeb),' ','*'*10)

    ### SVD - WEB
    start_time         = time.time()
    svdWeb             = svdProcess.obtainSvdMatrix(tfidfMatrixWeb,300)
    svdWeb.to_csv('tfWeb.csv')
    print('*'*10,"time - matrix svd process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(svdWeb),' ','*'*10)

    ## Borrar variables que consumen memoria
    del termDocumentMatrixWeb
    del urlList
    del filesPath
    del textWebList
    del textWebProcessing
    del vocabulary
    del idfWeb
    del tfidfMatrixWeb
    del svdWeb
    del svdJson
    gc.collect()

    ## Similitud del coseno
    start_time         = time.time()
    svdJson            = pd.read_csv('tfJson.csv')
    svdWeb             = pd.read_csv('tfWeb.csv')
    svdJson            = svdJson.transpose()
    svdWeb             = svdWeb.transpose()
    cosineSimilarity   = CosineSimilarity(svdJson,svdWeb)
    pool               = multiprocessing.Pool(processes=32)
    cosineSimMatrix    = pool.map(cosineSimilarity.obtainCosineSimilarity,svdJson.to_dict())
    pool.close() 
    pool.join()
    cosineSimMatrix    = cosineSimilarity.buildCosineMatrix(cosineSimMatrix)
    cosineSimMatrix    = cosineSimMatrix.describe()
    cosineSimMatrix.to_csv('CosineSimilarity.csv')
    print('*'*10,"time - matrix cosine similarity process = ",(time.time()-start_time),' seconds ',' | memory used: ',sys.getsizeof(cosineSimMatrix),' ','*'*10)
