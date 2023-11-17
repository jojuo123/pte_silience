import pandas as pd
import pickle

def compare():

    with open('./tmp.pkl', 'rb') as f:
        t = pickle.load(f)

    df = pd.read_csv('./data/ra-268.csv')

    t2 = []
    for i in range(len(df.index)):
        f = df['audio'].iloc[i].split('/')[-1]
        n = int(df['manual_pauses'].iloc[i])
        t2.append((f, n))

    t3 = []
    for i in t:
        for j in t2:
            f1, n1 = i
            n1 -= 1
            f2, n2 = j
            if f1 == f2:
                t3.append((f1, n1, n2))

    cnt = 0
    for i in t3:
        if i[1] == i[2]:
            cnt += 1

    return cnt * 100.0 / len(t3)
