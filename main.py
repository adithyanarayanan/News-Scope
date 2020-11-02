# IMPORTS NECESSARY PACKAGES AND BUILDS SOME FROM LOCAL SOURCES

import spacy
import spacy.cli
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import neuralcoref
import file_read

# Remove comment if "en_core_web_sm" is not already downloaded - refer spaCy docs to find other ways to install
# spacy.cli.download("en_core_web_sm")

# Need punkt for paragraph level tokenization - remove comment if not downloaded
# nltk.download('punkt')

# Need Lexicon for Polarity analysis- remove comment if not downloaded
# nltk.download('vader_lexicon')


nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)




# The Master function that chains all the upcoming methods in sequence
def classify(input_str):
    subject_entity, doc_clusters, count_subject_entity, doc = breakdown(input_str)
    if subject_entity != None:
        map = make_maps(doc)
        coordinates = generate_sentence_indices(4, len(sent_tokenize(input_str)))
        sentences = sent_tokenize(input_str)
        print("____Beginning Document Analysis____")
        print("SUBJECT ENTITY:", subject_entity)
        scores = score(coordinates, sentences, subject_entity, doc, map)

        if scores['compound'] > 0:
            print("\n POLARITY: Positive")
        else:
            print("\n POLARITY: Negative")

        print("SCORES \n", scores)


# Maps each sentence in a document to its starting index-
# useful when sentences have to be parsed by word to analyze sentiments
# Used when a set of sentences don't have a single dominant entity
def make_maps(doc):
    map = {}
    map[0] = 0
    sentence = 0
    for i in range(1, len(doc)):
        if doc[i].text == ".":
            sentence += 1
            map[sentence] = i + 1

    return map



# Generates combinations of total number of sentences in a document that are in sequence- returns a dictionary mapping them to zeros
# (0123, 1234, 2345, 4567 etc. till the end of the document is the last value- makes a list of lists)
def generate_sentence_indices(gram, size):
    coordinates = {}
    opening = 0
    closing = gram
    while closing <= size-1:
        closing = opening + gram
        # print(opening, closing)
        coordinates[(range(opening, closing))] = 0
        opening += 1

    return coordinates





# Checks if there is a singular unambigous most referred entity in a given list of sentences
# Returns the most subject_entity, coref clusters of document, count of subject entity refs
# If not distinct entity exists, returns None (distinction definition is arbitrary- can be modified to make more robust)
def breakdown(input_str):
    doc = nlp(input_str)
    appearances = {}
    current_max = 0
    second_max = 0
    last_max = 0
    clusters = doc._.coref_clusters
    for cluster in clusters:
        appearances[cluster[0]] = len(cluster)
        if len(cluster) > current_max:
            current_max = len(cluster)
            current_cluster = cluster
        elif len(cluster) > second_max:
            second_max = len(cluster)
            second_cluster = cluster
        elif len(cluster) > last_max:
            last_max = len(cluster)
        else:
            continue


    if (current_max - second_max) > 0:
        # print(current_max, current_cluster)
        return current_cluster[0], clusters, current_max, doc

    print("SUBJECT ENTITY: No distinguishable dominant entity identified")

    print("Comment:")
    print(current_max ,"and",second_max, "are the number of times the two highest mentioned entities are referenced. Due to this number being very close, a clear polarity cannot be allocated")

    print(current_cluster, "and", second_cluster, " are the most referenced entities that we were able to identify")
    return None, None, None, None




# Given the parent document, and the subject entity that is dealt with in the parent document
# This function returns True if the most referenced entity between start and ending indices supplied
# is also the subject entity- helpful in judging if the sentence can be stratightaway analyzed for emotional value
# or if it needs to be broken down at the word level to analyze: read accompanying blog post
def max_referred_entity(doc, start_idx, end_idx, subject_entity):
    clusters = doc._.coref_clusters
    sub_count = 0
    non_sub_count = 0
    for i in range(start_idx, end_idx):
        token = doc[i]
        token_cluster = (token._.coref_clusters)

        if len(token_cluster) > 0:
            nbor_token = doc[i-1]
            nbor_cluster = nbor_token._.coref_clusters
            if len(nbor_cluster) > 0:
                if nbor_cluster[0][0] == token_cluster[0][0]:
                    continue

            if token_cluster[0][0] == subject_entity:
                sub_count += 1
            else:
                non_sub_count += 1
        else:
            continue

    if sub_count > non_sub_count:
        return True
    return False



# Identifies the closest succeeding emotive words to the parent document's subject entity to identify polarity
# Used when no individual entity dominates a passage to map emotions to subject entity
def individualize(subject_entity, doc, start_idx, end_idx):
    analyzer = SentimentIntensityAnalyzer()

    positive = 0
    negative = 0
    compound = 0
    neutral = 0
    scale = 0

    for i in range(start_idx, end_idx):
        token = doc[i]
        token_cluster = token._.coref_clusters

        if len(token_cluster) == 0:
            continue
        if token_cluster[0][0] == subject_entity:
            tempstr = token.text + " "
            for j in range(i + 1, end_idx):
                subtoken = doc[j]
                subcluster = subtoken._.coref_clusters
                if len(subcluster) == 0:
                    tempstr += subtoken.text + " "
                if len(subcluster) != 0:
                    score = analyzer.polarity_scores(tempstr)

                    positive += score['pos']
                    negative += score['neg']
                    compound += score['compound']
                    neutral += score['neu']
                    scale += 1
                    tempstr = ""
                    break

    if scale == 0:
        return {'pos': positive, 'neu': neutral , 'neg': negative, 'compound': compound}
    return {'pos': positive/scale, 'neu': neutral/scale, 'neg': negative/scale, 'compound': compound/scale}



# Given a set of sentences, this function decides if the sentences can be analyzed straightaway together
# by checking if the max referred entity is the subject max_referred_entity. If yes, notes the scores. If no,
# sends the sentences to be analyzed at a token level to individualize() and recieves and aggregates the score from there
def score(coordinates, sentences, subject_entity, doc, map):
    positive = 0
    negative = 0
    compound = 0
    neutral = 0
    sia = SentimentIntensityAnalyzer()


    for coordinate in coordinates.keys():
        indices = list(coordinate)
        # print(indices)
        tempstr = ""
        for index in indices:
            tempstr += sentences[index]
        # print(tempstr)
        start_idx = map[indices[0]]
        if indices[-1] == len(sentences) - 1:
            end_idx = len(doc) - 1
        else:
            end_idx = map[indices[-1] + 1] - 1

        # print(start_idx, end_idx)
        # print(doc[start_idx], doc[end_idx])

        # print(max_referred_entity(doc, start_idx, end_idx, subject_entity))

        if max_referred_entity(doc, start_idx, end_idx, subject_entity):
            score = sia.polarity_scores(tempstr)

        else:
            score = individualize(subject_entity, doc, start_idx, end_idx)

        positive += score['pos']
        negative += score['neg']
        compound += score['compound']
        neutral += score['neu']

    # print({'pos': positive, 'neu': neutral, 'neg': negative, 'compound': compound})

    return {'pos': positive/len(coordinates), 'neu': neutral/len(coordinates), 'neg': negative/len(coordinates), 'compound': compound/len(coordinates)}






""" Reads and Classifies a news artice by polarity"""

string = file_read.create_string('articles/trump_neg_1.txt') # supply any file name from the articles package
classify(string)

doc = nlp(string)
print("The clusters of entities after resolving coreference, in this article, are \n", doc._.coref_clusters)


print("______________Document Analysis Complete_________")

