from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pickle
import csv
import re

sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
#model = SentenceTransformer('paraphrase-distilroberta-base-v1')

data = pd.read_pickle("../data/data_no_duplicate.pkl")
sentences = [] 

#20% of the data:
for index, row in data.sample(100000).iterrows():
    sentences.append(row.text)

#paraphrases = util.paraphrase_mining(model, sentences)
#Top 15 for memory reasons
paraphrases = util.paraphrase_mining(sbert_model, sentences, show_progress_bar=True,top_k=15)

print(f'{len(paraphrases)} sentence pairs were analysed')


header = ["sentence_1", "sentence_2", "cosine_similarity"]

with open ('data/sentence_similarity.tsv','wt') as csvfile:
    writer = csv.writer(csvfile, delimiter ="\t" )
    writer.writerow(header)

    for paraphrase in paraphrases:
        score, i, j = paraphrase
        
        sentence_1 = sentences[i]
        sentence_2 = sentences[j]
        result = score
        
        rows = [sentence_1,sentence_2,result]
        
        if score > 0.7:
            writer.writerow(rows)
        
tsv_file = csv.reader(open('data/sentence_similarity.tsv', "r"), delimiter="\t")
possible_discrimination = []

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
          "(?P<relstr>\blonde\s)",
          "(?P<relstr>\sblond\s)",
          "(?P<relstr>black)"]

def is_direct_regex(text):
    for reg in direct:
        if re.search(reg, text)!=None:
            return True

for row in tsv_file:
    sentence_1 = row[0]

    if is_direct_regex(sentence_1):
        possible_sentence = row[1]

        if not is_direct_regex(possible_sentence):
            possible_discrimination.append(possible_sentence)


with open('data/to_be_annotated.tsv', 'w') as csvfile:
    for line in possible_discrimination:
        csvfile.write(line + '\n')

