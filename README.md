# News-Scope

Tool that analyzes news articles that are fed to it by resolving coreference for the entire document, identifying the main
subject entity being discussed, and and scoring all subsets of sequential sentences of a given size in a document using spaCy
and NLTK Vader to compute an overall polarity for the document w.r.t subject entity

To know more, read draft of article about this here: https://medium.com/@adithya-narayanan/macgyvering-a-large-scale-text-polarity-analyzer-studying-news-article-polarity-e0ab5c2fb209  [article discusses logic and preliminary results - will be published end of March 2020 after I test this tool more robusly]

Check out neuralcoref here: https://github.com/huggingface/neuralcoref

A good and clean documentation and a towards data science (hopefully!) coming soon. 

Disclaimer: I do not claim that the tool is accurate in it's predictive role. It is built for just experimentation purposes. 
I also do not own any of the articles that I have uploaded here as a part of an experiment for just the purposes of exemplification. 

I will soon curate an ownership list for each of the articles. Until then, if you are curious about the origin of the article, 
do email me at adithyan@buffalo.edu and I will gladly track the link down for you. 

TODO:

- Fix scoring mechanism to normalize it by size. 
- Annotate data to check its effectiveness across more articles.
