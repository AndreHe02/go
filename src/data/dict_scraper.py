import urllib.request
from bs4 import BeautifulSoup
import re
from os import listdir
from os.path import isfile, join

dict_file =  "../../data/dict_file.txt"

url = "https://senseis.xmp.net/?GoTerms"

fp = urllib.request.urlopen(url)
mybytes = fp.read()
mystr = mybytes.decode("utf8")
fp.close()

soup = BeautifulSoup(mystr, features="html.parser")

li = soup.findAll('div', {'class': 'contentsize'})
children = li[1].findChildren("ul", recursive=True)
terms = []
for child in children[0]:
  links = child.findAll('a')
  for link in links:
    terms.append(link.string)

f = open(dict_file, "w+")
for term in terms:
  f.write(term)
  f.write("\n")
