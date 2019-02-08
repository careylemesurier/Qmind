from bs4 import BeautifulSoup as bs
import requests
import time
s=input("What ticker? ")
website='https://finance.yahoo.com/quote/{}/'.format(s)
page=requests.get(website)
soup = bs(page.content, 'html.parser')
class1='W(100%)'
class2='Ta(end) Fw(b) Lh(14px)'
col1 = soup.find_all('span', {'class': 'Trsdu(0.3s)'})
cleanData=[]
#cleanData=np.asarray(cleanData)
a=0
while a < 10:
    row=[]
    for x in range(len(col1)):
        row.append(col1[x].decode_contents(formatter="html"))
    a+=1
    cleanData.append(row)
    #row = np.asarray(row)
    #np.stack((cleanData,row))
    time.sleep(30)
    print(cleanData)
#price = soup.find('span', data-reactid_='35')
#col2=soup.find_all('td', class_=class2)
#print(col1.decode_contents(formatter="html"))

#Forward Dividend & Yield
#Write a program to pull data from Apple's historical data
#Write a program to continuously run the program every minute, save data