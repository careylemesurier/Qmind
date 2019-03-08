from bs4 import BeautifulSoup as bs
import requests
import time
import csv
import pandas as pd
import numpy as np
import datetime

s=input("What ticker? ")
website='https://finance.yahoo.com/quote/{}/'.format(s)
'''page=requests.get(website)
soup = bs(page.content, 'html.parser')
class1='W(100%)'
class2='Ta(end) Fw(b) Lh(14px)'
col1 = soup.find_all('span', {'class': 'Trsdu(0.3s)'})'''
#while market is closed
cleanData=[]
#cleanData=np.asarray(cleanData)
a=0
while a < 5:
    page = requests.get(website)
    soup = bs(page.content, 'html.parser')
    class1 = 'W(100%)'
    class2 = 'Ta(end) Fw(b) Lh(14px)'
    col1 = soup.find_all('span', {'class': 'Trsdu(0.3s)'})
    row=[]
    currentDT=datetime.datetime.now()
    print(currentDT)
    row.append(currentDT.strftime("%m/%d/%Y"))
    row.append(currentDT.strftime("%H:%M:%S"))
    for x in range(len(col1)):
        #if (x!=2):
            row.append(col1[x].decode_contents(formatter="html"))
    a+=1
    cleanData.append(row)
    #row = np.asarray(row)
    #np.stack((cleanData,row))
    time.sleep(30)
    print(cleanData)

numpyCD=np.array(cleanData)
print(numpyCD.shape)
print(numpyCD)
df=pd.DataFrame(numpyCD,columns=['Date', 'Time', 'Trading Price', 'Daily Change', 'Previous Close', 'Open', 'Bid', 'Ask', 'Volume', 'Avg. Volume', 'Market Cap', 'Beta (3Y Monthly)', 'PE Ratio (TTM)', 'EPS (TTM)', 'Ex-Dividend Date', '1y Target Est'])
df.to_csv('data.csv')
print (df)

#Forward Dividend & Yield
#Write a program to pull data from Apple's historical data
#Write a program to continuously run the program every minute, save data