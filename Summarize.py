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

   

#%%
df['LexRank_rouge-1_precision'] = ''
df['LexRank_rouge-1_recall'] = ''
df['LexRank_rouge-1_f1'] = ''
df['LexRank_rouge-2_precision'] = ''
df['LexRank_rouge-2_recall'] = ''
df['LexRank_rouge-2_f1'] = ''
df['LexRank_rouge-l_precision'] = ''
df['LexRank_rouge-l_recall'] = ''
df['LexRank_rouge-l_f1'] = ''

df['Luhn_rouge-1_precision'] = ''
df['Luhn_rouge-1_recall'] = ''
df['Luhn_rouge-1_f1'] = ''
df['Luhn_rouge-2_precision'] = ''
df['Luhn_rouge-2_recall'] = ''
df['Luhn_rouge-2_f1'] = ''
df['Luhn_rouge-l_precision'] = ''
df['Luhn_rouge-l_recall'] = ''
df['Luhn_rouge-l_f1'] = ''

df['LSA_rouge-1_precision'] = ''
df['LSA_rouge-1_recall'] = ''
df['LSA_rouge-1_f1'] = ''
df['LSA_rouge-2_precision'] = ''
df['LSA_rouge-2_recall'] = ''
df['LSA_rouge-2_f1'] = ''
df['LSA_rouge-l_precision'] = ''
df['LSA_rouge-l_recall'] = ''
df['LSA_rouge-l_f1'] = ''

# %%
for i, r in df.iterrows():
    try:
        scores = rouge.get_scores(df['summary_LexRank'].iloc[i], df['summary'].iloc[i])[0]

        df['LexRank_rouge-1_precision'] = scores['rouge-1']['p']
        df['LexRank_rouge-1_recall'] = scores['rouge-1']['r']
        df['LexRank_rouge-1_f1'] = scores['rouge-1']['f']
        df['LexRank_rouge-2_precision'] = scores['rouge-2']['p']
        df['LexRank_rouge-2_recall'] = scores['rouge-2']['r']
        df['LexRank_rouge-2_f1'] = scores['rouge-2']['f']
        df['LexRank_rouge-l_precision'] = scores['rouge-l']['p']
        df['LexRank_rouge-l_recall'] = scores['rouge-l']['r']
        df['LexRank_rouge-l_f1'] = scores['rouge-l']['f']

    except ValueError:
        pass

# %%
df

# %%
