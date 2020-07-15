#%%
# Load Packages
import sumy
import pandas as pd
from rouge import Rouge 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer

#%%
### Extractive Summarizers ###

#%%
document1 ="""Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics."""
# %%
# For Strings
parser = PlaintextParser.from_string(document1,Tokenizer("english"))

# %%
# Using LexRank
lex_summarizer = LexRankSummarizer()
summary = lex_summarizer(parser.document, 2) # Summarize the document with 2 sentences
for sentence in summary:
    print(sentence)

# %%
# Using Luhn
luhn_summarizer = LuhnSummarizer()
summary_1 = luhn_summarizer(parser.document,2)
for sentence in summary_1:
    print(sentence)

# %%
# Using LSA
lsa_summarizer = LsaSummarizer()
summary_2 = lsa_summarizer(parser.document,2)
for sentence in summary_2:
    print(sentence)




#%%
### ROUGE Evaluation ###

# %%

hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)

# %%
print(scores)
print(scores[0])
print(scores[0]['rouge-1'])
print(scores[0]['rouge-2'])
print(scores[0]['rouge-l'])
print(scores[0]['rouge-1']['p'])
print(scores[0]['rouge-1']['r'])
print(scores[0]['rouge-1']['f'])
