from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pickle
import csv
import re

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
data = pd.read_pickle("../data/data_no_duplicate.pkl")
sentences = [] 

for index, row in data.sample(90000).iterrows():
    sentences.append(row.text)

#The following piece of code to define the sentences in the data would be more efficient
#but raises an error for random indexes in the text

# sentences = data['text']

paraphrases = util.paraphrase_mining(model, sentences)

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

for row in tsv_file:
    sentence_1 = row[0]

    if is_direct_regex(sentence_1):
        possible_sentence = row[1]

        if not is_direct_regex(possible_sentence):
            possible_discrimination.append(possible_sentence)


with open('data/to_be_annotated.tsv', 'w') as csvfile:
    for line in possible_discrimination:
        csvfile.write(line + '\n')

