# Structural Analyses

Analyze the contribution of structural information vs. BoW
- Start with classic techniques, then work up to positional-free vs position significant transformers


# LazyPredictor Unigram
                             Accuracy  Balanced Accuracy ROC AUC  F1 Score  Time Taken
Model
XGBClassifier                    0.36               0.26    None      0.35       75.29
BaggingClassifier                0.34               0.24    None      0.33        9.62
RandomForestClassifier           0.36               0.23    None      0.34       11.71
LinearDiscriminantAnalysis       0.23               0.22    None      0.21        0.41
NearestCentroid                  0.11               0.22    None      0.13        0.12
LogisticRegression               0.31               0.21    None      0.28       10.65
ExtraTreesClassifier             0.34               0.21    None      0.32        7.15
DecisionTreeClassifier           0.29               0.20    None      0.29        1.68
LinearSVC                        0.30               0.19    None      0.26      323.32
BernoulliNB                      0.21               0.18    None      0.19        0.14
LabelPropagation                 0.26               0.17    None      0.26       12.92
LabelSpreading                   0.26               0.17    None      0.26       21.02
CalibratedClassifierCV           0.29               0.15    None      0.24      976.13
KNeighborsClassifier             0.25               0.15    None      0.24        0.85
SVC                              0.29               0.15    None      0.25       70.13
SGDClassifier                    0.20               0.14    None      0.19       13.19
Perceptron                       0.19               0.14    None      0.18        3.23
ExtraTreeClassifier              0.20               0.14    None      0.20        0.15
RidgeClassifier                  0.27               0.13    None      0.21        0.17
RidgeClassifierCV                0.27               0.13    None      0.21        0.36
PassiveAggressiveClassifier      0.20               0.11    None      0.19        4.26
GaussianNB                       0.02               0.10    None      0.02        0.22
LGBMClassifier                   0.12               0.02    None      0.05        9.46
DummyClassifier                  0.13               0.02    None      0.03        0.09
AdaBoostClassifier               0.13               0.02    None      0.03        8.66
>>> predictions
                             Accuracy  Balanced Accuracy ROC AUC  F1 Score  Time Taken
Model
XGBClassifier                    0.36               0.26    None      0.35       75.29
BaggingClassifier                0.34               0.24    None      0.33        9.62
RandomForestClassifier           0.36               0.23    None      0.34       11.71
LinearDiscriminantAnalysis       0.23               0.22    None      0.21        0.41
NearestCentroid                  0.11               0.22    None      0.13        0.12
LogisticRegression               0.31               0.21    None      0.28       10.65
ExtraTreesClassifier             0.34               0.21    None      0.32        7.15
DecisionTreeClassifier           0.29               0.20    None      0.29        1.68
LinearSVC                        0.30               0.19    None      0.26      323.32
BernoulliNB                      0.21               0.18    None      0.19        0.14
LabelPropagation                 0.26               0.17    None      0.26       12.92
LabelSpreading                   0.26               0.17    None      0.26       21.02
CalibratedClassifierCV           0.29               0.15    None      0.24      976.13
KNeighborsClassifier             0.25               0.15    None      0.24        0.85
SVC                              0.29               0.15    None      0.25       70.13
SGDClassifier                    0.20               0.14    None      0.19       13.19
Perceptron                       0.19               0.14    None      0.18        3.23
ExtraTreeClassifier              0.20               0.14    None      0.20        0.15
RidgeClassifier                  0.27               0.13    None      0.21        0.17
RidgeClassifierCV                0.27               0.13    None      0.21        0.36
PassiveAggressiveClassifier      0.20               0.11    None      0.19        4.26
GaussianNB                       0.02               0.10    None      0.02        0.22
LGBMClassifier                   0.12               0.02    None      0.05        9.46
DummyClassifier                  0.13               0.02    None      0.03        0.09
AdaBoostClassifier               0.13               0.02    None      0.03        8.66



# Curr exp code, uni+bigrams
- - - - -
 SVC Poly
Training
Acc = 0.28167
Test
Acc = 0.17900

- - - - -
 SVC RBF
Training
Acc = 0.45211
Test
Acc = 0.32100

- - - - -
 LinearSVC
Training
Acc = 0.51500
Test
Acc = 0.36350

- - - - -
kNN N=1
Training
0.99433
Test
0.29950
- - - - -
kNN N=2
Training
0.99433
Test
0.30050
- - - - -
kNN N=3
Training
0.99472
Test
0.31050
- - - - -
kNN N=4
Training
0.99494
Test
0.31750
- - - - -
kNN N=5
Training
0.99500
Test
0.32550
- - - - -
kNN N=6
Training
0.99500
Test
0.32900
- - - - -
kNN N=7
Training
0.99500
Test
0.33200
- - - - -
kNN N=8
Training
0.99500
Test
0.32500
- - - - -
kNN N=9
Training
0.99500
Test
0.33400
- - - - -
kNN N=10
Training
0.99500
Test
0.33150


# Curr Exp code, bigrams only

- - - - -
 SVC Poly
Training
Acc = 0.28061
Test
Acc = 0.17850

- - - - -
 SVC RBF
Training
Acc = 0.45700
Test
Acc = 0.31800

- - - - -
 LinearSVC
Training
Acc = 0.51683
Test
Acc = 0.37100

- - - - -
kNN N=1
Training
0.99433
Test
0.30250
- - - - -
kNN N=2
Training
0.99433
Test
0.30250
- - - - -
kNN N=3
Training
0.99472
Test
0.30950
- - - - -
kNN N=4
Training
0.99500
Test
0.31950
- - - - -
kNN N=5
Training
0.99500
Test
0.33000
- - - - -
kNN N=6
Training
0.99500
Test
0.32650
- - - - -
kNN N=7
Training
0.99500
Test
0.33500
- - - - -
kNN N=8
Training
0.99500
Test
0.33800
- - - - -
kNN N=9
Training
0.99500
Test
0.34200
- - - - -
kNN N=10
Training
0.99500
Test
0.34000

# Trigrams
- - - - -
 SVC Poly
Training
Acc = 0.34789
Test
Acc = 0.19100

- - - - -
 SVC RBF
Training
Acc = 0.53694
Test
Acc = 0.32950

- - - - -
 LinearSVC
Training
Acc = 0.65078
Test
Acc = 0.30350

- - - - -
kNN N=1
Training
0.99478
Test
0.32450
- - - - -
kNN N=2
Training
0.99472
Test
0.32450
- - - - -
kNN N=3
Training
0.99511
Test
0.33000
- - - - -
kNN N=4
Training
0.99528
Test
0.34100
- - - - -
kNN N=5
Training
0.99533
Test
0.34650
- - - - -
kNN N=6
Training
0.99539
Test
0.35150
- - - - -
kNN N=7
Training
0.99539
Test
0.34950
- - - - -
kNN N=8
Training
0.99539
Test
0.34550
- - - - -
kNN N=9
Training
0.99539
Test
0.34950
- - - - -
kNN N=10
Training
0.99539
Test
0.34650

# Conducting analyses with ngram_range=(1, 3)

- - - - -
 SVC Poly
Training
Acc = 0.34483
Test
Acc = 0.18950

- - - - -
 SVC RBF
Training
Acc = 0.52967
Test
Acc = 0.33150

- - - - -
 LinearSVC
Training
Acc = 0.65028
Test
Acc = 0.30900
- - - - -
kNN N=1
Training
0.99478
Test
0.31800
- - - - -
kNN N=2
Training
0.99472
Test
0.31850
- - - - -
kNN N=3
Training
0.99511
Test
0.32600
- - - - -
kNN N=4
Training
0.99528
Test
0.34200
- - - - -
kNN N=5
Training
0.99539
Test
0.34650
- - - - -
kNN N=6
Training
0.99539
Test
0.34600
- - - - -
kNN N=7
Training
0.99539
Test
0.34800
- - - - -
kNN N=8
Training
0.99539
Test
0.34500
- - - - -
kNN N=9
Training
0.99539
Test
0.34400
- - - - -
kNN N=10
Training
0.99539
Test
0.34550
>>> 
