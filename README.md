Heres a mock Quality Dashboard with fictional MES/QMS/ERP data to practice and learn more about SQL/Python/Pandas and high level ML concepts


aerospace_quality.db = Mock database

generate_data.py = generates data random outputs from database

**pipeline.py = importing from generated data, using pandas/sql to aggregate/transform data, prep data for ML training**

ml_models.py = Scikit/xgboost for ML insights

dashboard.py = mostly CSS and formatting for hosting dashboard on streamlit


dashboard url: https://aerospace-quality-dashboard-wwh5hop3s3uvj2f89259sy.streamlit.app/

*Thoughts*

I learned most about building pipelines using SQL in python/pandas using multi-table joins from different databases
..and how to make ML-ready tables from qualitative data. (i.e. assigning attributes like day/night shifts to 1/0 for ML training)

I used Scikit/xgboost for the ML aspects (random forest/gradient boost)
When testing I initially got a ROC-AUC score of 1.0 (a perfect model), so i had to adjust with which attributes were being input into the model to end up with a more realistic accuracy of around ~0.6
..this felt like a real problem I could encounter and was rewarding to understand/debug

Code written by Claude with #learning notes, I mostly edited the SQL queries/commands as trial and error
