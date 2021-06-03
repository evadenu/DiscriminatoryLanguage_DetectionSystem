import pickle
import re 
from re import search
import pandas as pd
import numpy as np

data = pd.read_pickle("../data/data_no_duplicate.pkl")
sentences = data['text']
discrimination_label = []

direct = ["(?P<relstr>integer)",
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
          "(?P<relstr>de\skroeg\sin)",
           "(?P<relstr>zwart\s)",
          "(?P<relstr>\swit\s)",
          "(?P<relstr>black)"]

                    #Low powerdistance terms:
indirect_regex = ["(?P<relstr>start-up\smentaliteit)",
                  "(?P<relstr>opstartmentaliteit)",
                  "(?P<relstr>platte\sorganisatiestructuur)",
                  "(?P<relstr>verantwoording)",
                  "(?P<relstr>betrouwbaarheid)",
                  "(?P<relstr>verantwoordelijkheid)",
                  "(?P<relstr>verantwoordelijk)",
                  "(?P<relstr>plezier)",
                  "(?P<relstr>gevarieerd\sleven)",
                  '(?P<relstr>zelfredzaamheid)',
                  "(?P<relstr>erkenning)",
                  "(?P<relstr>vrijheid)",
                  "(?P<relstr>gelijkheid)",
                  "(?P<relstr>ruimdenkend)",
                  "(?P<relstr>verbeeldingskracht)",
                  #feminine culture terms
                  "(?P<relstr>kind)",
                  "(?P<relstr>verbeeldingskracht)",
                  "(?P<relstr>opgewekt)",
                  "(?P<relstr>toegewijd)",
                  "(?P<relstr>gemeenschappelijk)",
                  "(?P<relstr>medelevend)",
                  "(?P<relstr>verbonden)",
                  "(?P<relstr>vertrouwe.*)",
                  "(?P<relstr>attent\s)",
                  "(?P<relstr>coöperatie.*)",
                  "(?P<relstr>afhankelijk)",
                  "(?P<relstr>vrouwelijk)",
                  "(?P<relstr>vriende.*)",
                  "(?P<relstr>emotioneel)",
                  "(?P<relstr>empathi.*)",
                  "(?P<relstr>zachtaardig)",
                  "(?P<relstr>interpersoonlijk)",
                  "(?P<relstr>interdependent)",
                  "(?P<relstr>onderdanig)",
                  "(?P<relstr>loyaal)",
                  "(?P<relstr>loyaliteit)",
                  "(?P<relstr>bescheiden.*)",
                  "(?P<relstr>beleefd.*)",
                  "(?P<relstr>koester.*)",
                  "(?P<relstr>gevoelig.*)",
                  "(?P<relstr>rustig)",
                  "(?P<relstr>sympathi.*)",
                  "(?P<relstr>samen)",
                  "(?P<relstr>begripvol)",
                  #indulgence terms:
                  "(?P<relstr>welwillendheid)",
                  "(?P<relstr>hoffelijkheid)",
                  "(?P<relstr>vriendelijk)",
                  "(?P<relstr>dienst.*)",
                  "(?P<relstr>servicegericht/s)",
                  #LTO terms:
                  "(?P<relstr>doorzettingsvermogen)",
                  "(?P<relstr>zuinigheid)",
                  "(?P<relstr>veranderende\s.{0,30}omstandigheden)",
                  "(?P<relstr>aanpassingsvermogen)"
                 ]

def is_direct_regex(text):
    for reg in direct:
        if re.search(reg, text)!=None:
            return True

def is_indirect_regex(text):
    for in_reg in indirect_regex:
        if re.search(in_reg, text) !=None:
            return True


for chunk in data['text']:
    if is_direct_regex(chunk):
        discrimination_label.append("2")
    elif is_indirect_regex(chunk):
        discrimination_label.append("1")
    else:
        discrimination_label.append("0")


new_df = pd.DataFrame()
new_df['sentence'] = sentences
new_df['discrimination_label'] = discrimination_label        


with open('data/discrimination_labels.pkl', 'wb') as outfile:
    pickle.dump(new_df, outfile)

    
