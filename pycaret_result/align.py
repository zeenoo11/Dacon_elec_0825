import pandas as pd

data = pd.read_csv('./output/xgboost_tuned_submission.csv')

print(data.head())

data_aligned = pd.DataFrame()

data_aligned['answer'] = data['answer']

data_aligned.to_csv('./output/xgboost_tuned_submission_aligned.csv', index=True)







