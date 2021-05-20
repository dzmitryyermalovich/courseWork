import pandas as pd
import numpy as np
import json


with open("C:\\CourseWork\\GeneratePoetsText\\data\\\TextData\\stixi.json",encoding="UTF-8") as datafile:
    data=json.load(datafile)
df=pd.DataFrame(data)

df_pushkin=df[(df['poet_id']=="pushkin")]
df_esenin=df[(df['poet_id']=="esenin")]
df_blok=df[(df['poet_id']=="blok")]
df_tyutchev=df[(df['poet_id']=="tyutchev")]


def distinguish_poem(df):
    Poems=''
    for text in df['content']:
        Poems+=text
    return Poems

PushkinPoems=distinguish_poem(df_pushkin)
EseninPoems=distinguish_poem(df_esenin)
BlokPoems=distinguish_poem(df_blok)
TyutchevPoems=distinguish_poem(df_tyutchev)