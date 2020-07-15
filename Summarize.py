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
def append_summaries(sentences):
    summary = ''
    for sentence in sentences:
        summary += str(sentence)

    return summary
    

# %%
df = pd.read_pickle('cnn_dataset_10k.pkl')

# %%
df['summary_LexRank'] = ''
df['summary_Luhn'] =  ''
df['summary_LSA'] = ''

# %%
lex_summarizer = LexRankSummarizer()
luhn_summarizer = LuhnSummarizer()
lsa_summarizer = LsaSummarizer()
rouge = Rouge()

for i, r in df.iterrows():
    # print(df['text'].iloc[i])
    parser = PlaintextParser.from_string(df['text'].iloc[i], Tokenizer("english"))
    sentence_amount = 5 

    sentences = lex_summarizer(parser.document, sentence_amount) 
    df['summary_LexRank'].iloc[i] = append_summaries(sentences)
    # print(append_summaries(sentences))
    # print(sentences)

    sentences = luhn_summarizer(parser.document, sentence_amount) 
    df['summary_Luhn'].iloc[i] = append_summaries(sentences)
    # print(append_summaries(sentences))
    # print(sentences)

    sentences = lsa_summarizer(parser.document, sentence_amount) 
    df['summary_LSA'].iloc[i] = append_summaries(sentences)
    # print(append_summaries(sentences))
    # print(sentences)

   

