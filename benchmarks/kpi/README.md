## KPI Anomaly dataset

KPI anomaly dataset consists of KPI (key performace index) time series data from
many real scenarios of Internet companies with ground truth label.
The dataset can be found (here)[https://github.com/NetManAIOps/KPI-Anomaly-Detection]

The full dataset contains multiple KPI IDs. Different KPI time series have different structures
and patterns.
For our purpose, we are running anomaly detection for some of these KPI indices.

The performance table is shown below, although note that the hyperparameters have not been tuned.
The hyperparams used are available inside the results directory under each algorithm.


| KPI ID                               | KPI index | Algorithm       | ROC-AUC (test set) |
|--------------------------------------|-----------|-----------------|--------------------|
| 431a8542-c468-3988-a508-3afd06a218da | 14        | VanillaAE       | 0.89               |
| 431a8542-c468-3988-a508-3afd06a218da | 14        | Conv1dAE        | 0.88               |
| 431a8542-c468-3988-a508-3afd06a218da | 14        | LSTMAE          | 0.86               |
| 431a8542-c468-3988-a508-3afd06a218da | 14        | TransformerAE   | 0.82               |
| 431a8542-c468-3988-a508-3afd06a218da | 14        | SparseVanillaAE | 0.93               |
| 431a8542-c468-3988-a508-3afd06a218da | 14        | SparseConv1dAE  | 0.77               |


Full credit to Zeyan Li et al. for constructing large-scale real world benchmark datasets for AIOps.

@misc{2208.03938,
Author = {Zeyan Li and Nengwen Zhao and Shenglin Zhang and Yongqian Sun and Pengfei Chen and Xidao Wen and Minghua Ma and Dan Pei},
Title = {Constructing Large-Scale Real-World Benchmark Datasets for AIOps},
Year = {2022},
Eprint = {arXiv:2208.03938},
