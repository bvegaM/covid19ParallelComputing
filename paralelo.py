 
import fnmatch
import os.path
import json
import requests
from urllib.request import urlopen, Request
import urllib.request
from bs4 import BeautifulSoup

import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd
import math
import numpy as np
import time

import multiprocessing
#import cudf
#from cuml import TruncatedSVD
#from cuml.decomposition import TruncatedSVD

class Preprocessing:
    
    def tokenizarPalabras(self,texto):
        return texto.split()
    
    def removerPalabrasBasura(self,texto):
        return [i for i in texto if i not in stopwords.words('english')]
    
    def removerLetrasBasura(self,texto):
        return [i for i in texto if i not in list(string.ascii_letters)]
    
    def lemantizarPalabras(self,texto):
        return [WordNetLemmatizer().lemmatize(i, pos='v') for i in texto]
    

class JsonClass:
    
    def obtenerRutaArchivos(self,ruta):
        archivo=[]
        for roots,dirs,files in os.walk(ruta):
            for file in files:
                if fnmatch.fnmatch(file,'*.json'):
                    path = roots+"/"+file
                    archivo.append(path)
        return archivo  
    
    def leerArchivos(self,archivo):
        preprocesamiento  = Preprocessing()
        lista=()
        leer =  json.loads(open(archivo).read())  
        text=leer['body_text']
        texto=""
        for j in text:
            texto=texto+j['text']+" "
        texto  = texto.lower()
        texto  = texto.translate({ord(i): None for i in '!@#$%^&*()-_=+[{]}\|;:<,>.?/1234567890'})
        textoP = preprocesamiento.tokenizarPalabras(texto)
        textoP = preprocesamiento.removerLetrasBasura(textoP)
        textoP = preprocesamiento.removerPalabrasBasura(textoP)
        textoP = preprocesamiento.lemantizarPalabras(textoP)
        textoP = [w for w in textoP if len(w) > 2] 
        lista=(leer['paper_id'],textoP)
        texto=""
        textoP=[]
        return lista

class HtmlClass:
    
    def leerArchivosHtml(self,url):
        preprocesamiento  = Preprocessing()
        lista=()
        hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
        req = Request(url[0], headers=hdr)
        datos = urllib.request.urlopen(req).read()
        soup =  BeautifulSoup(datos,'html.parser')
        tags = soup('p')
        texto=""
        for tag in tags:
            texto=texto+tag.get_text()+" "
        texto=texto.lower()
        texto = texto.replace("’","")
        texto = texto.replace("”","")
        texto = texto.replace("“","")
        texto  = texto.translate({ord(i): None for i in '!@#$%^&*()-_=+[{]}\|;:<,>.?/1234567890'})
        textoP = preprocesamiento.tokenizarPalabras(texto)
        textoP = preprocesamiento.removerLetrasBasura(textoP)
        textoP = preprocesamiento.removerPalabrasBasura(textoP)
        textoP = preprocesamiento.lemantizarPalabras(textoP)
        textoP = [w for w in textoP if len(w) > 2]
        title=url[1]+" new"
        lista=(title,textoP)
        print("noticia procesada")
        return lista

class TerminoDocumento:
    
    def __init__(self,listaT):
        self.diccionarioTexto=listaT
        self.diccionarioContador={}
        
    def unirDiccionario(self,texto):
        union={}
        for i in texto:
            for j in i:
                union[j]=i[j]
        return union
    
    def obtenerPalabras(self,listaTexto):
            lista=[]
            for i in listaTexto:
                lista=lista+i
            return lista
    
    def frecuenciaPalabras(self,diccionarioTexto):
        palabrasXdocumento = {}
        for x in diccionarioTexto.values():
            for z in x:
                if z not in palabrasXdocumento.keys():
                    palabrasXdocumento[z] = 1
                else:
                    palabrasXdocumento[z] = palabrasXdocumento[z] + 1
        return palabrasXdocumento
    
    def obtenerVocabulario(self,frecuenciaTermino):
        return [(i,frecuenciaTermino[i]) for i in frecuenciaTermino if frecuenciaTermino[i]>10]
    
    def obtenerpalabrasDocumento(self):
        diccionarioContador={}
        for i in self.diccionarioTexto:
            aux={}
            aux[i]=self.diccionarioTexto[i]
            self.diccionarioContador[i]=self.frecuenciaPalabras(aux)
        
    def obtenerTerminoDocumento(self,frecuenciaTermino):
        aux={}
        diccionarioTermDoc={}
        for j in self.diccionarioContador:
            if frecuenciaTermino[0] not in self.diccionarioContador[j].keys(): 
                aux[j]=0
            else:
                aux[j]=self.diccionarioContador[j][frecuenciaTermino[0]]
        diccionarioTermDoc[frecuenciaTermino[0]]=aux
        return diccionarioTermDoc
    
    def matrizTerminoDocumento(seflf,terminoDocumento):
        documentos=terminoDocumento[list(terminoDocumento.keys())[0]].keys()
        dframe = pd.DataFrame([key for key in documentos], columns=['Palabras'])
        for i in terminoDocumento:
            dframe[i]=[terminoDocumento[i][j] for j in terminoDocumento[i]]
        
        dframe=dframe.transpose()
        indices=list(dframe.loc['Palabras'])
        dframe.columns=indices
        dframe=dframe.drop(['Palabras'],axis=0)
        return dframe
        
class TfIdf:
    
    def __init__(self,len,term):
        self.lenArchivos=len
        self.terminoDocumento=term
        
    def obtenerIdf(self,palabra):
        count=0
        valorIdf=0
        for j in self.terminoDocumento[palabra]:
            if(self.terminoDocumento[palabra][j]!=0):
                count+=1
        if (count!=0):
            valorIdf=math.log10(self.lenArchivos/count)
        else:
            valorIdf=0
        return valorIdf
    
    def obtenerTfIdf(self,terminoDocumentoM,idf):
        for i in terminoDocumentoM:
            terminoDocumentoM[i]=terminoDocumentoM[i]*idf
        return terminoDocumentoM
    
class SimilaridadCoseno:
    
    def __init__(self,tfIdfMatriz,tfIdfHtmlMatriz):
        self.html = tfIdfHtmlMatriz
        self.json = tfIdfMatriz
    
    def obtenerSimilaridadCoseno(self,valor):
        respaldoDos=pd.DataFrame(self.html.values,columns=self.html.columns)
        lista=[]
        for i in respaldoDos:
            dot_product = np.dot(np.array(respaldoDos[i]),np.array(self.json[valor]))
            norm_a = np.linalg.norm(np.array(respaldoDos[i]))
            norm_b = np.linalg.norm(np.array(self.json[valor])) 
            coseno=dot_product/(norm_a*norm_b)
            lista.append(coseno)
        return lista
    
    def armarMatrizCoseno(self,lista):
        respaldo=pd.DataFrame(self.html.values,columns=self.html.columns)
        cosenoFrame=pd.DataFrame(columns=respaldo.columns)
        for i in lista:
            cosenoFrame.loc[len(cosenoFrame)]=i
        cosenoFrame=cosenoFrame.set_index(self.json.columns)
        return cosenoFrame

if __name__=="__main__":
    listaT           = {}
    
    jsonClass         = JsonClass()
    termDoc           = TerminoDocumento(listaT)
    
    
    ruta="/Users/bvegam/Documents/proyecto/2000/"
    start_time=time.time()
    rutaArchivos            = jsonClass.obtenerRutaArchivos(ruta)
    pool                    = multiprocessing.Pool(processes=8)
    listaTextos             = pool.map(jsonClass.leerArchivos,rutaArchivos)
    pool.close() 
    pool.join()
    listaTextos             = dict(listaTextos)
    '''
    frecuenciaTermino       = termDoc.frecuenciaPalabras(listaTextos)
    vocabulario             = termDoc.obtenerVocabulario(frecuenciaTermino)
    termDoc                 = TerminoDocumento(listaTextos)
    pool                    = multiprocessing.Pool(processes=8)
    termDoc.obtenerpalabrasDocumento()
    terminoDocumento        = pool.map(termDoc.obtenerTerminoDocumento,vocabulario)
    pool.close() 
    pool.join()
    terminoDocumento        = termDoc.unirDiccionario(terminoDocumento) 
    matTerminoDocumento     = termDoc.matrizTerminoDocumento(terminoDocumento)

    tfidf                   = TfIdf(len(rutaArchivos),terminoDocumento)
    pool                    = multiprocessing.Pool(processes=8)
    idf                     = pool.map(tfidf.obtenerIdf,terminoDocumento)
    pool.close() 
    pool.join()
    tfIdfMatriz             = tfidf.obtenerTfIdf(matTerminoDocumento,idf)
    ss= tfIdfMatriz.transpose()
    a =ss.values
    cudtfUno                = cudf.DataFrame([a[i] for i in range(len(a))],dtype=float)
    del frecuenciaTermino
    del a
    del ss
    del tfIdfMatriz
    
    tsvd_float = TruncatedSVD(n_components = 8, algorithm = "jacobi", n_iter = 20, tol = 1e-9)
    tsvd_float.fit(cudtfUno)
    
    trans_gdf_float = tsvd_float.transform(cudtfUno)
    #print(f'Transformed matrix: {trans_gdf_float}')
    print("El proceso demoro = ",(time.time()-start_time))
    
    

    html              = HtmlClass()
    termDoc           = TerminoDocumento(listaT)
    
    lista = []
    lista.append(("https://www.biorxiv.org/content/10.1101/2020.04.03.022723v1.full","article 1"))
    lista.append(("https://www.biorxiv.org/content/10.1101/2020.02.05.936013v1.full","article 2"))
    lista.append(("https://www.biorxiv.org/content/10.1101/2020.04.10.036418v1.full","article 3"))
    lista.append(("https://www.biorxiv.org/content/10.1101/2020.04.12.025577v1.full","article 4"))
    lista.append(("https://www.biorxiv.org/content/10.1101/2020.03.03.975524v1.full","article 5"))
    lista.append(("https://www.nytimes.com/2020/05/08/health/fda-coronavirus-spit-test.html?action=click&module=Top%20Stories&pgtype=Homepage","real 1"))
    lista.append(("https://www.theguardian.com/world/2020/may/08/revealed-uk-scientists-fury-over-attempt-to-censor-covid-19-advice","real 2"))
    lista.append(("https://www.nature.com/articles/d41586-020-01284-x","real 3"))
    lista.append(("https://www.nytimes.com/2020/05/01/health/coronavirus-remdesivir.html?searchResultPosition=10","real 4"))
    lista.append(("https://theconversation.com/coronavirus-treatments-what-drugs-might-work-against-covid-19-135352","real 5"))
    
    start_time=time.time()
    pool                          = multiprocessing.Pool(processes=8)
    listaHtml                     = pool.map(html.leerArchivosHtml,lista)
    listaHtml                     = dict(listaHtml)
    termDoc                       = TerminoDocumento(listaHtml)
    pool                          = multiprocessing.Pool(processes=8)
    termDoc.obtenerpalabrasDocumento()
    terminoDocumentoHtml          = pool.map(termDoc.obtenerTerminoDocumento,vocabulario)
    pool.close() 
    pool.join()
    terminoDocumentoHtml          = termDoc.unirDiccionario(terminoDocumentoHtml) 
    matTerminoDocumentoHtml       = termDoc.matrizTerminoDocumento(terminoDocumentoHtml)
    
    matTerminoDocumentoHtml.to_csv(r'TermDocumentMatrix.csv')
    
    del vocabulario
    del rutaArchivos
    
    tfidf                         = TfIdf(len(lista),terminoDocumentoHtml)
    pool                          = multiprocessing.Pool(processes=4)
    idfHtml                       = pool.map(tfidf.obtenerIdf,terminoDocumentoHtml)
    pool.close() 
    pool.join()
    tfIdfMatrizHtml               = tfidf.obtenerTfIdf(matTerminoDocumentoHtml,idfHtml)
    ss = tfIdfMatrizHtml.tranpose()
    a                             =ss.values
    cudtfDos                      = cudf.DataFrame([a[i] for i in range(len(a))],dtype=float)
    
    tsvdT_float = TruncatedSVD(n_components = 3, algorithm = "jacobi", n_iter = 20, tol = 1e-9)
    tsvdT_float.fit(cudtfDos)
    
    trans_gdfD_float = tsvdT_float.transform(cudtfDos)
    del tfIdfMatrizHtml
    del a
    
    trans_gdf_float.to_csv('tfUno.csv')
    trans_gdfD_float.to_csv('tfDos.csv')
   
    svdUno=pd.read_csv('tfUno.csv')
    svdDos=pd.read_csv('tfDos.csv')

    cosenoS                       = SimilaridadCoseno(svdUno,svdDos)
    poolU                         = multiprocessing.Pool(processes=8)
    cosenoSimilaridadMatriz       = poolU.map(cosenoS.obtenerSimilaridadCoseno,svdUno.to_dict())
    poolU.close() 
    poolU.join()
    cosenoSimilaridadMatriz       = cosenoS.armarMatrizCoseno(cosenoSimilaridadMatriz)
    print("El proceso demoro = ",(time.time()-start_time))
    descripcionMatrizCoseno       = cosenoSimilaridadMatriz.transpose()
    descripcionMatrizCoseno.to_csv(r'cosenoSimilaridadMatriz.csv')
    '''
    
    
