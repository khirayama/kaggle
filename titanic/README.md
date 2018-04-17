## Step

- Download data
- Read our goal
- Read data
- Read submission sample
- Read our goal again

- Except which vars I should use
- Prediction using some approch
- Make a submission file

## Memo

https://www.kaggle.com/c/titanic

```
$ kaggle competitions download -c titanic -p ./input
```

```
$ ipython notebook
```

```
$ kaggle competitions submit --competition titanic -f ./predict_result_data.csv -m "First submit"
```
## Questions

- 欠損値の補完(中央値？test_dfに対してもdfで補完する？)

## Research

- [pandas.DataFrame.dropna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)
- [pandas.DataFrame.hist](pandas.DataFrame.histndas.DataFrame.hist)
- [pandas.DataFrame.fillna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)
- [pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html)
- [pandas.Series.drop](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.drop.html)
- [sklearn.ensemble.RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Refs

- [【Kaggle初心者入門編】タイタニック号で生き残るのは誰？](https://www.codexa.net/kaggle-titanic-beginner/)
- [Kaggle事始め](https://qiita.com/taka4sato/items/802c494fdebeaa7f43b7)
- [pandas で年齢階級をつくる](https://qiita.com/kshigeru/items/bfa8c11d1e6487c791d3)
- [代表的な機械学習手法一覧](https://qiita.com/tomomoto/items/b3fd1ec7f9b68ab6dfe2)
- [タイタニック号乗客の生存予測モデルを立ててみる](https://qiita.com/suzumi/items/8ce18bc90c942663d1e6)
- [Kaggleのtitanic問題で上位10%に入るまでのデータ解析と所感](http://www.mirandora.com/?p=1804)
- [ダミー変数で重回帰分析を応用しよう！](http://xica.net/magellan/marketing-idea/stats/abou-dummy-variable/)
