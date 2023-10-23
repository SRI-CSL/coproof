# Proof Completion Measure work

Development area for work on proof completion measures.


V2 of data prefixes start sequent tokens (src/notebooks/230209_completion_measure/230212_data_setup_v2.ipynb)

## 230209 Last Command prediction
- Use existing featurized tokens to determine when the next to last command is.
- Featurize using trained vocab from Lemma Retrieval



### Traditional Methods on V2 of data
combined2_summary.csv

knn & 0.65 & 0.65 \\ 
randforest & 0.65 & 0.65 \\ 
svc linear & 0.66 & 0.66 \\ 
svc poly & 0.61 & 0.63 \\ 
svc rbf & 0.67 & 0.68 \\ 


tfidf_summary.csv

knn & 0.55 & 0.55 \\ 
randforest & 0.58 & 0.58 \\ 
svc linear & 0.61 & 0.61 \\ 
svc poly & 0.55 & 0.57 \\ 
svc rbf & 0.59 & 0.60 \\ 


count_summary.csv

knn & 0.53 & 0.53 \\ 
randforest & 0.54 & 0.54 \\ 
svc linear & 0.60 & 0.60 \\ 
svc poly & 0.40 & 0.49 \\ 
svc rbf & 0.58 & 0.58 \\ 


combined_summary.csv

knn & 0.67 & 0.67 \\ 
randforest & 0.63 & 0.63 \\ 
svc linear & 0.66 & 0.66 \\ 
svc poly & 0.60 & 0.63 \\ 
svc rbf & 0.67 & 0.68 \\ 


cmd_std_summary.csv

knn & 0.65 & 0.65 \\ 
randforest & 0.69 & 0.69 \\ 
svc linear & 0.66 & 0.66 \\ 
svc poly & 0.61 & 0.63 \\ 
svc rbf & 0.66 & 0.67 \\ 