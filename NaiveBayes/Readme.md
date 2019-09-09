# Gender Recognition - NaiveBayes

This is the realization of a Naive Bayes Calssification for a [voice gender recognition competition](https://www.kaggle.com/primaryobjects/voicegender/home) in kaggle —— a data competition website —— with code and description. This project is a small classwork  in the Speech Signal Processing class during my 6th semester in Wuhan University in Autumn 2017. Here is the file structure:

```
NaiveBayes
    |-- src
        |-- data
            |-- TrainingSets.mat
            |-- ValidationSets.mat
            |-- voice.xls
            |-- voive_data.mat
            |-- voive_data_init.mat
        |-- functions
            |-- mycfsmtx.m
            |-- mydiscretization.m
            |-- myhowmany.m
            |-- myisinterger.m
            |-- mynumstatistic.m
            |-- myrowcheck.m
        |-- training.m
        |-- validation.m
        |-- xls2mat.m
    |-- LICENSE
    |-- kaggle _ 基于朴素贝叶斯分类器的语音性别识别.md
    |-- Readme.md
```
Among the files above:
- in folder 'src':
  - in folder 'data':
    - file 'voice.xls' is the rew data file containing all training and validation data which can be downloaded from the [kaggle](https://www.kaggle.com/primaryobjects/voicegender/home) website;
    - file 'voive_data_init.mat' is a .MAT file converted from file 'voice.xls';
    - file 'voive_data.mat' is the discretized data from file 'voive_data_init.mat' to adapt the following Naive Bayes algorithm;
    - file 'TrainingSets.mat' is the training dataset file splited from 'voive_data.mat';
    - file 'ValidationSets.mat' is the validation dataset file splited from 'voive_data.mat';
  - in folder 'functions':
    - several auxiliary functions for the project;
  - file 'training.m' is a function to **train** a Naive Bayes classification;
  - file 'validation.m' is function to **validate** the trained Naive Bayes classification;
  - file 'xls2mat.m' is is function to convert data from voice.xls to voive_data.mat in folder 'data';
- file 'LICENSE' is the license file produced by github;
- file 'kaggle _ 基于朴素贝叶斯分类器的语音性别识别.md' is a detailed introduction document for this project. 

For more detailed information, refer to article [kaggle _ 基于朴素贝叶斯分类器的语音性别识别.md](https://github.com/chentianyangWHU/Gender-Recognition/blob/master/NaiveBayes/kaggle%20_%20%E5%9F%BA%E4%BA%8E%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8%E7%9A%84%E8%AF%AD%E9%9F%B3%E6%80%A7%E5%88%AB%E8%AF%86%E5%88%AB.md).
