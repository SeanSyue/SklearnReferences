import pandas as pd
from MachineLearning.project.generate_dataset import project_dummify as dum

# Make sure :
# -- 1. Social features have dropped;
# -- 2. 'duration' have dropped;
# -- 3. Redundant samples have removed.
# For better results, please reshuffle and up-sampling data.

df_in = pd.read_csv('C:/bank/data_set/dataset_v2/for_client_data.csv')

age_dum = dum.dummify_age(df_in)
others_dum = pd.get_dummies(df_in.iloc[:, 1:10])
campaign_dum = dum.dummify_campaign(df_in)
pdays_dum = dum.dummify_pdays(df_in)
previous_dum = dum.dummify_previous(df_in)
df_in['y'] = df_in['y'].replace(('yes', 'no'), (1, 0))

df_out = pd.concat([age_dum, others_dum, campaign_dum, pdays_dum, previous_dum, df_in['y']], axis=1)


# df_out.to_csv('C:/bank/data_set/dataset_v2/client_train.csv', index=False)
