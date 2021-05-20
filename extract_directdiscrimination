import pandas as pd
import re 
import csv

data_sentences = pd.read_pickle("data_no_duplicate.pkl")
direct_dict = dict()
index = 1

#List of regex based to search for direct discrimination in text
direct_regex = ["(?P<relstr>integer)",
              "(?P<relstr>betrouwbaar)",
              "(?P<relstr>accentloos\s.{0,30}snederlands)",
              "(?P<relstr>foutloos\s.{0,30}snederlands)",
              "(?P<relstr>vlekkeloze\s.{0,30}sbeheersing)",
              "(?P<relstr>native\sspeaker)",
              "(?P<relstr>moedertaal)",
              "(?P<relstr>representatief\s.{0,30}uiterlijk)",
              "(?P<relstr>neutrale uitstraling)",
              "(?P<relstr>verzorgd uiterlijk)",
              "(?P<relstr>je\sziet\ser\sverzorgd\suit)",
              "(?P<relstr>nederlandse\sachtergrond)",
              "(?P<relstr>turkse\sachtergrond)",
              "(?P<relstr>marokkaanse\sachtergrond)",
              "(?P<relstr>nederlandse\scultuur)",
              "(?P<relstr>westerse\scultuur)",
              "(?P<relstr>nederlandse\scultuur)",
              "(?P<relstr>\sneger\s)",
              "(?P<relstr>vrijdagmiddagborrel)",
              "(?P<relstr>de\skroeg\sin)"]

#Looping over the sentences to find these regexes
for reg in direct_regex:
    direct_disc = data_sentences[data_sentences["text"].str.contains(reg, regex = True)]
    text = direct_disc["text"].values
    direct_dict.update({(str(index) + " direct_discrimination"): text})
    index +=1
    

#writing out the matches to a CSV
with open("direct_discrimination.csv", "w") as f:
    f.write("text")
    for index, job_text in direct.items():
        for item in job_text:
            f.write((item + "\n"))


        
