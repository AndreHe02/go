import urllib.request
from bs4 import BeautifulSoup
import re
from os import listdir
from os.path import isfile, join

annotation_path = "../../data/annotations/"

scraped_files = [int(f.split(".")[0]) for f in listdir(annotation_path) if isfile(join(annotation_path, f))]
print("Using {} previously scraped files".format(len(scraped_files)))

counter = 0



for i in range(100):
  url = "https://gtl.xmp.net/reviews/by_index?f=" # 1&l=100"
  start = i*100 + 1
  end = (i+1) * 100
  url = url + str(start) + "&l=" + str(end)


  fp = urllib.request.urlopen(url)
  mybytes = fp.read()

  mystr = mybytes.decode("utf8")
  fp.close()

  soup = BeautifulSoup(mystr, features="html.parser")
  for link in soup.findAll('a'):
    if link.get("href").split(".")[-1] == "sgf" and link.get("href").split("/")[1] == "sgf":
      if not int(link.string) in scraped_files:
        url = "https://gtl.xmp.net" + link.get("href")
        path = annotation_path + link.string + ".sgf"
        try:
          urllib.request.urlretrieve(url, path)
          counter += 1
        except:
          print("Error retrieving {}".format(link.string))
   
  print("Num files scraped: {}".format(counter))
