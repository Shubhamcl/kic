import statsmodels.formula.api as sm
from statsmodels.api import datasets
import pandas as pd

df = datasets.get_rdataset('Duncan', 'carData', cache=True).data

ar2 = dict()
candidates = []
last_max = -1

print(df)

type = df['type'].str.get_dummies().astype(bool)
df= pd.concat([df, type], axis=1)
df.drop('type', axis=1, inplace=True)

y = 'income'

print(df)

while(True):
    # Drop each column
    for x in df.drop([y] + candidates, axis=1).columns:
        # Add this column to all
        if len(candidates) == 0:
            features = x
        else:
            features = x + ' + '
            features += ' + '.join(candidates)
        
        model = sm.ols(y + ' ~ ' + features, df).fit()
        ar2[x] = model.rsquared

    max_ar2 =  max(ar2.values())
    max_ar2_key = max(ar2, key=ar2.get)

    if max_ar2 > last_max:
        candidates.append(max_ar2_key)
        last_max = max_ar2

        print('step: ' + str(len(candidates)))
        print(candidates)
        print('Adjusted R2: ' + str(max_ar2))
        print('===============')
    else:
        print(model.summary())
        break

print('\n\n')
print('elminated variables: ')
print(set(df.drop(y, axis=1).columns).difference(candidates))