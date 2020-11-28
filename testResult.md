### TextCNN
#### sougo Embedding
```
Test Loss:  0.38,  Test Acc: 89.16%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9045    0.8900    0.8972      1000
       realty     0.9069    0.9160    0.9114      1000
       stocks     0.8497    0.8310    0.8402      1000
    education     0.9316    0.9400    0.9358      1000
      science     0.8500    0.8670    0.8584      1000
      society     0.8107    0.9080    0.8566      1000
     politics     0.8914    0.8370    0.8633      1000
       sports     0.9413    0.9460    0.9436      1000
         game     0.9516    0.8840    0.9165      1000
entertainment     0.8917    0.8970    0.8943      1000

     accuracy                         0.8916     10000
    macro avg     0.8929    0.8916    0.8917     10000
 weighted avg     0.8929    0.8916    0.8917     10000

Confusion Matrix...
[[890  13  50   4   9  17   9   2   3   3]
 [ 17 916  16   5   6  14   6   2   3  15]
 [ 54  25 831   3  33   7  34   6   4   3]
 [  1   5   5 940   5  21   5   4   1  13]
 [  5   6  26  12 867  23  19   6  23  13]
 [  2  14   4  17  13 908  16   6   0  20]
 [  8  12  34  10  10  76 837   4   0   9]
 [  0   5   3   3   5  15   2 946   3  18]
 [  5   4   5   3  57   7   6  14 884  15]
 [  2  10   4  12  15  32   5  15   8 897]]
Time usage: 0:00:00
```

#### wiki w2c
```
No optimization for a long time, auto-stopping...
Test Loss:  0.38,  Test Acc: 89.16%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9045    0.8900    0.8972      1000
       realty     0.9069    0.9160    0.9114      1000
       stocks     0.8497    0.8310    0.8402      1000
    education     0.9316    0.9400    0.9358      1000
      science     0.8500    0.8670    0.8584      1000
      society     0.8107    0.9080    0.8566      1000
     politics     0.8914    0.8370    0.8633      1000
       sports     0.9413    0.9460    0.9436      1000
         game     0.9516    0.8840    0.9165      1000
entertainment     0.8917    0.8970    0.8943      1000

     accuracy                         0.8916     10000
    macro avg     0.8929    0.8916    0.8917     10000
 weighted avg     0.8929    0.8916    0.8917     10000

Confusion Matrix...
[[890  13  50   4   9  17   9   2   3   3]
 [ 17 916  16   5   6  14   6   2   3  15]
 [ 54  25 831   3  33   7  34   6   4   3]
 [  1   5   5 940   5  21   5   4   1  13]
 [  5   6  26  12 867  23  19   6  23  13]
 [  2  14   4  17  13 908  16   6   0  20]
 [  8  12  34  10  10  76 837   4   0   9]
 [  0   5   3   3   5  15   2 946   3  18]
 [  5   4   5   3  57   7   6  14 884  15]
 [  2  10   4  12  15  32   5  15   8 897]]

```
