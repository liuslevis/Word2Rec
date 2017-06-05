# Word2vecRec

## Setup

```
brew install pip3
pip3 install gensim
git clone https://github.com/liuslevis/Word2vecRec
cd Word2vecRec
```

## Data

```
SELECT `timestamp`,vid,bookid FROM wxg_tdbank::weread_dsl_wruseroper_fdt0
WHERE tdbank_imp_date >= '20170401' AND tdbank_imp_date <= '20170430' AND vid % 100 = 7 AND optype = 'SHELFADD'
```

## Usage

```
➜  Word2vecRec git:(master) ✗ ipython3 main.py
choose:
    index   item
    0
    1   巨人的陨落:世纪三部曲
    2   寻路中国:从乡村到工厂的自驾之旅
    3   百年孤独
    4   霍乱时期的爱情
    5   失控
    6   迟到的间隔年
    7   枪炮、病菌与钢铁:人类社会的命运
    8   中国历代政治得失
    9   活着
    10  从0到1
    11  背包十年:我的职业是旅行
like:9
     + 活着 = {'活着'}
hate:5
     + 失控 = {'失控'}
recommend:
    index   score   item
    13  0.17    人类简史:从动物到上帝
    8   0.15    中国历代政治得失
    7   0.10    枪炮、病菌与钢铁:人类社会的命运
    0   0.03
    12  0.03    创业维艰
    1   -0.01   巨人的陨落:世纪三部曲
    2   -0.02   寻路中国:从乡村到工厂的自驾之旅
    11  -0.04   背包十年:我的职业是旅行
    3   -0.06   百年孤独
    10  -0.08   从0到1
```
