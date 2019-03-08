from bs4 import BeautifulSoup as bs
import pandas as pd
import requests

s=input("What ticker? ")
website='http://download.macrotrends.net/assets/php/stock_data_export.php?t={}'.format(s)
print(website)
biib_url = website
def download_stock_data(csv_url):
    #~ response = request.urlopen(csv_url)
    #~ csv = response.read()
    #~ csv_str = str(csv)
    csv_str = requests.get(csv_url).text
    lines = csv_str.split("\\n")
    dest_url = r'biib.csv'
    fx = open(dest_url, "w")
    for line in lines:
        fx.write(line + "\n")
    fx.close()

download_stock_data(biib_url)
#data = pd.read_csv(website)
'''
url = website
username = 'user'
password = 'pass'
p = urlopen.HTTPPasswordMgrWithDefaultRealm()

p.add_password(None, url, username, password)

handler = urlopen.HTTPBasicAuthHandler(p)
opener = urlopen.build_opener(handler)
urlopen.install_opener(opener)

page = urlopen.urlopen(url).read()'''





'''print(website)
data = pd.read_csv(website)


CSV_URL = website'''

'''
with requests.Session() as s:
    download = s.get(CSV_URL)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)
'''