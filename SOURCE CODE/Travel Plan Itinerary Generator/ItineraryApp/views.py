from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pymysql
from django.core.files.storage import FileSystemStorage
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
import os
from string import punctuation

from nltk.corpus import stopwords
import re
from image import search_images, download_images
from collections.abc import Sequence
from collections import defaultdict

import google.generativeai as genai
import PIL.Image

global uname, gemini_model, tokenizer, retriever, model
X = []
Y = []

#gemeini model to generate itinerary
genai.configure(api_key="AIzaSyCSrdHqh7wkS7TiOhA8NebxDFiQO41gCJ4")
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
print("Gemini Model Loaded")
stop_words = set(stopwords.words('english'))
tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
print("done")
#rag model for itinerary search
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
print("done")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
print("done")

#define function to clean text by removing stop words and other special symbols
def cleanText(data):
    data = data.split()
    data = [w for w in data if not w in stop_words]
    data = [word for word in data if len(word) > 3]
    data = ' '.join(data)
    return data
#function to scrape images from net for selected itinerary
def scrapeImages(location):
    arr = location.split("_")
    name = arr[1]
    if os.path.exists('ItineraryApp/static/location_images/'+name) == False:
        os.makedirs('ItineraryApp/static/location_images/'+name)
        image_urls = search_images(name, num_images=5)
        download_images(image_urls, save_dir='ItineraryApp/static/location_images/'+name)    


def loadData():
    global X, Y, gemini_model, tokenizer, retriever, model
    if os.path.exists('ItineraryApp/static/features/X.npy'):
        X = np.load('ItineraryApp/static/features/X.npy')
        Y = np.load('ItineraryApp/static/features/Y.npy')
    flag = False
    for root, dirs, directory in os.walk('ItineraryApp/static/model'):
        for j in range(len(directory)):
            if directory[j].lower() not in Y:
                with open(root+"/"+directory[j], "rb") as file:
                    data = file.read()
                file.close()
                data = data.decode()
                data = data.strip('\n').strip().lower() 
                data = re.sub('[^a-z]+', ' ', data)
                data = cleanText(data)
                if len(data) > 2500:
                    data = data[0:2500]
                inputs = tokenizer(data, return_tensors="pt")
                input_ids = inputs["input_ids"]
                question_hidden_states = model.question_encoder(input_ids)[0]
                question_hidden_states = question_hidden_states.detach().numpy().ravel()
                if len(X) > 0:
                    X = X.tolist()
                    Y = Y.tolist()
                Y.append(directory[j].lower())
                X.append(question_hidden_states)
                X = np.asarray(X)
                Y = np.asarray(Y)
                scrapeImages(directory[j])
                flag = True
                print(directory[j])
    if flag == True:
        np.save("ItineraryApp/static/features/X.npy", X)
        np.save("ItineraryApp/static/features/Y.npy", Y)
    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
loadData()

def TravelPlanAction(request):
    if request.method == 'POST':
        global uname, X, Y, gemini_model, tokenizer, retriever, model
        source = request.POST.get('t1', False)
        destination = request.POST.get('t2', False)
        budget = request.POST.get('t3', False)
        desc = request.POST.get('t4', False)
        name = source+"_"+destination+"_"+budget+".txt"
        print(Y)
        print(name)
        tt = name.lower() not in Y
        print(tt)
        if name.lower() not in Y:
            prompt = f"""Generate a trip plan for from {source} to {destination} with a budget of ${budget}.
             We are interested in a mix of historical sightseeing, cultural experiences, travel websites, hotel booking platforms, tourist guides,
             transportation schedules, weather forecasts and delicious food.
             Provide a detailed itinerary for hotels and flights
            """
            plan = ""
            response = gemini_model.generate_content(prompt)
            for chunk in response:
                plan += chunk.text
            with open("'ItineraryApp/static/model/"+source+"_"+destination+"_"+budget+".txt", "wb") as file:
                file.write(plan.encode())
            file.close()
            loadData()
        data = source+" "+destination+" "+desc
        data = data.strip('\n').strip().lower()
        data = re.sub('[^a-z]+', ' ', data)
        data = cleanText(data)
        if len(data) > 2500:
            data = data[0:2500]
        inputs = tokenizer(data, return_tensors="pt")
        input_ids = inputs["input_ids"]
        query = model.question_encoder(input_ids)[0]
        query = query.detach().numpy().ravel()
        plan_name = ""
        max_score = 0
        for i in range(len(X)):
            predict_score = dot(X[i], query)/(norm(X[i])*norm(query))
            if predict_score > max_score and destination.lower() in Y[i]:
                max_score = predict_score
                plan_name = Y[i]
        data = ""        
        with open('ItineraryApp/static/model/'+plan_name, "r") as file:
            for line in file:
                values = line.strip()
                if len(values) == 0:
                    data += "<br/>"
                else:
                    data += values+"<br/>"
        file.close()
        output = '<br/><table border=0 align=center><tr>'
        for root, dirs, directory in os.walk('ItineraryApp/static/location_images/'+destination.lower()):
            for j in range(len(directory)):
                output += '<td><img src="static/location_images/'+destination.lower()+'/'+directory[j]+'" width="200" height="200"/></td>'
        output += "</tr></table><br/><br/><br/><br/>"      
        data += output
        context= {'data': data}
        return render(request, 'UserScreen.html', context)
        
def TravelPlan(request):
    if request.method == 'GET':
       return render(request, 'TravelPlan.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'itinerary',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM register where username='"+username+"' and password='"+password+"'")
            rows = cur.fetchall()
            for row in rows:
                uname = username
                index = 1
                break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        
    
def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)        
        
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'itinerary',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'itinerary',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup task completed"
        context= {'data': status}
        return render(request, 'Register.html', context)

