Heres a mock Quality Dashboard I built with fictional MES/QMS/ERP data to practice and learn more about SQL/Python/Pandas and high level ML concepts

I learned most about building pipelines using SQL in python/pandas using multi-table joins from different databases
..and how to make ML-ready tables from qualitative data. (i.e assigning attributes like day/night shifts to 1/0)

I used Scikit/xgboost for the ML aspects (random forest/gradient boost)
When testing I initially got a ROC-AUC score of 1.0 (a perfect model), so i had to adjust with which attributes were being input into the model to end up with a more realistic accuracy of around ~0.6
..this felt like a real problem I could encounter and was rewarding to debug
