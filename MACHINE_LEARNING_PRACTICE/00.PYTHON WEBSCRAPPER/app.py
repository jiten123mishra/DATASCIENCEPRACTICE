from flask import Flask,render_template,jsonify,request
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re 
app = Flask(__name__)
def web_scrapping(key):
    print("webscrapping")
    query=key+"customer review amazon india search"
    url=""
    for ele in search(query, stop=1):
         url=ele
    
    headers = requests.utils.default_headers()
    headers.update(
        {
            'User-Agent': 'My User Agent 1.0',
        }
    )
    print("Accessing URL=",url)
    page = requests.get(url,headers=headers)
    status=page.status_code
    if status==200:
        print("connection successful")
    elif status==503:
        print("website blocking you from scrapping")
    elif status==404:
        print("page not found")
        
    soup = BeautifulSoup(page.content, 'html.parser')
    soup1=soup.findAll('div',class_='a-section a-spacing-none review-views celwidget')
    #print(soup)
    ratings=[]
    for ele in soup1:
        date=ele.findAll('span', class_='a-size-base a-color-secondary review-date')
        dates=[x.get_text() for x in date]
        title=ele.findAll('a',class_='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold')
        titles=[x.find('span',class_='').get_text() for x in title]
        name=ele.findAll('span',class_='a-profile-name')
        names=[x.get_text() for x in name]
        rating=ele.findAll('span',class_='a-icon-alt')
        ratings=[x.get_text() for x in rating]
    
    
    print(len(dates))
    print(len(titles))
    print(len(names))
    print(len(ratings))
    out=[]
    for i in range(0,len(titles)):
        temp={}  
        temp["name"]=names[i]
        temp["date"]=dates[i]
        temp["title"]=titles[i]
        temp["rating"]=ratings[i]
        out.append(temp)
    return out
def update_mongodb(data,tablename):
    print("updating mongodb")
    from pymongo import MongoClient  
    try: 
        myclient = MongoClient() 
        print("Connected successfully!!!") 
    except:   
        print("Could not connect to MongoDB") 
        
    mydb = myclient.AMAZONDB
    mycoll = mydb[tablename]
    colllist=mydb.list_collection_names()
    if mycoll.name in colllist:
        print("record already exists, showing results ")
#        cursor = mycoll.find({}, {'_id': False})
#        for record in cursor: 
#            print(record) 
    else:
        print("record doesnot exists, updating results")
        mycoll.insert_many(data)
#        print("after inserting result is ")
#        cursor = mycoll.find({}, {'_id': False})
#        for record in cursor: 
#            print(record) 


@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form.to_dict()
        #print("result=",result)
        key=result['Name']
        tablename= re.sub("\s","_",key).lower()
        data=web_scrapping(key)        
        update_mongodb(data,tablename)
        myclient = MongoClient('localhost', 27017)
        mydb = myclient['AMAZONDB']
        reviewList = []
        reviewColl = mydb[tablename]
        for x in reviewColl.find():
            record = {"name": x["name"], "title": x["title"], "rating": x["rating"], "date": x["date"]}
            #print(record)
            reviewList.append(record) 
            #print(reviewList)
         
    return jsonify({'reviewList': reviewList}) 

if __name__ == '__main__':
   app.run(debug = True)

