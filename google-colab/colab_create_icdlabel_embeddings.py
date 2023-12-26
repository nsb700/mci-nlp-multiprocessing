# !pip install sentence-transformers

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import re

model_name = 'sentence-transformers/all-MiniLM-L6-v2'

def getIcdCodesDf(filename: str):
    df = pd.read_csv(filename)
    df = df[['icdcode', 'icdlabel']]
    df['icdcode'] = df.apply(lambda x: re.sub(r'[^\w\s]', '', x['icdcode'].lower().strip()), axis=1)
    df['icdlabel'] = df.apply(lambda x: re.sub(r'[^\w\s]', '', x['icdlabel'].lower().strip()), axis=1)
    df['diagcodelvl1'] = df.apply(lambda x: x['icdcode'] if x['icdcode'].startswith('hcc') else x['icdcode'][:3], axis=1)
    return df

def getDiagLvl1Df(filename: str):
    df = pd.read_csv(filename)
    df['diagdesclvl1'] = df.apply(lambda x: re.sub(r'[^\w\s]', '', x['diagdesclvl1'].lower().strip()), axis=1)
    return df

df = getIcdCodesDf('codesfile.csv')
diaglvl1df = getDiagLvl1Df('diag_lvl1.csv')
df = df.merge(diaglvl1df, how='left', on='diagcodelvl1')
df['diagdesclvl1'] = df.apply(lambda x: x['icdlabel'] if pd.isna(x['diagdesclvl1']) else x['diagdesclvl1'], axis=1) 

model = SentenceTransformer(model_name_or_path = model_name)

emb = model.encode(
                  df['icdlabel'].to_list(),
                  convert_to_tensor=True,
                  normalize_embeddings = True
                )

torch.save(emb, 'tensor.pt')
df.to_csv('transformed_codesfile.csv', header=True, index=False)