#==============================================================================
# Developed by Monu18
#==============================================================================

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from math import sin, cos, sqrt, atan2, radians

def distance(lt1,lo1,lt2,lo2):
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(lt1)
    lon1 = radians(lo1)
    lat2 = radians(lt2)
    lon2 = radians(lo2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return(R*c*1000)

#%%
df_train=pd.read_csv('training2.csv',header=0)

df_train['timeinmilis'].fillna(0,inplace=True)
df_train.head()

#%%
vars_to_del = ['ID','subusername','accuracy','timeinmilis']
df_train1 = df_train.drop(vars_to_del,axis=1)

clustering = DBSCAN(eps=0.1, min_samples=10).fit(df_train1)

df_train['cluster'] = pd.DataFrame(clustering.labels_)
df_train.head()
result1 = df_train.groupby(['cluster']).count()
result1
#%%
df_train.to_csv('predicted_0.1_10.csv',index=False, header=True)

#%%
final_results = pd.DataFrame(df_train['cluster'].unique())
final_results=final_results.rename(columns={0: 'cluster'})
final_results = final_results.set_index('cluster', drop=False)
count = df_train.groupby(['cluster']).count()
lat_series = df_train.groupby(['cluster']).mean()['latitude']
lon_series = df_train.groupby(['cluster']).mean()['longitude']
final_sheet = pd.concat([final_results,count['ID'], lat_series,lon_series],axis=1,join='inner')

final_sheet

pd.options.mode.chained_assignment = None  # default='warn'
final_sheet1 = final_sheet[final_sheet['cluster']>=0]
home_location = final_sheet1[final_sheet1['ID']==final_sheet1['ID'].max()]
home_location = home_location.reset_index(drop=True)

final_sheet['distance'] = final_sheet['latitude'] 
for index,row in final_sheet.iterrows():
    lat=final_sheet['latitude'][index]
    lon=final_sheet['longitude'][index]
    final_sheet['distance'][index] = distance(home_location['latitude'][0], home_location['longitude'][0],lat,lon)
    
final_sheet
office_wip1 = final_sheet[final_sheet['cluster']>=0]
office_wip = office_wip1[(office_wip1['distance']>100) & (office_wip1['distance']<5000)]
office_location = office_wip[office_wip['ID'] == office_wip['ID'].max()]

home_location
office_location

#%%
