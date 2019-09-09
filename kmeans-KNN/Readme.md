# Gender Recognition - kmeans-KNN

This is the realization of a kmeans-KNN Calssification for a [voice gender recognition competition](https://www.kaggle.com/primaryobjects/voicegender/home) in kaggle —— a data competition website —— with code and description. This project is a small classwork  in the Speech Signal Processing class during my 6th semester in Wuhan University in Autumn 2017. Here is the file structure:

```
kmeans-KNN
    |-- src
        |-- data
            |-- voice_dedimen.mat
            |-- voive_data.mat
            |-- voive_data_init.mat
            |-- voive_dedimen_discreted.mat
        |-- functions
            |-- mycluster_plus.m
            |-- mydiscretization.m
            |-- mydist.m
            |-- mydist_corre.m
            |-- mydist_cosine.m
            |-- mydist_hamm.m
            |-- mydrawkmeans.m
            |-- myerrcal.m
            |-- mykmeans.m
            |-- mynumstatistic.m
        |-- gender_recog.m
        |-- preproces.m
    |-- LICENSE
    |-- kaggle _ 基于k-means和KNN的语音性别识别.md
    |-- Readme.md
```
Among the files above:
- in folder 'src':
  - in folder 'data':
    - file 'voive_data_init.mat' contains 20-dimension float data of the voice dataset;
    - file 'voive_data.mat' is the discretized data of 'voive_data_init.mat';
    - file 'voice_dedimen.mat' is the voice dataset after dimensionality reduction;
    - file 'voive_dedimen_discreted' is the discretized data of 'voice_dedimen.mat';
  - in folder 'functions':
    - functions to realize the kmeans algorithm, for detailed information, see project [KmeansCluster](https://github.com/chentianyangWHU/KmeansCluster);
  - file 'gender_recog.m' is a file to **train** and **validate** a kmeans-KNN classification;
  - file 'preproces.m' is file to do some pre-process for the voice dataset;
- file 'LICENSE' is the license file produced by github;
- file 'kaggle _ 基于k-means和KNN的语音性别识别.md' is a detailed introduction document for this project. 

For more detailed information, refer to article [kaggle _ 基于k-means和KNN的语音性别识别.md]().