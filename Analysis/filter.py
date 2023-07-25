import pandas as pd

#Takes every driver out
def filter_passengers(df):
    #Delete all rows whose name ends with "driver.jpg"
    for index, row in df.iterrows():
        if row['name'].endswith('driver.jpg'):
            df.drop(index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    #For every row, delete the second instance of the same name - this is used for object detection where the images are not split up
    df['Drop'] = False
    for index, row in df.iterrows():
        if index == 0:
            continue
        if row['name'] == df.iloc[index-1]['name']:
            df.loc[index, 'Drop'] = True
    df.drop(df[df['Drop'] == True].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    #remove the Drop column
    df.drop(columns=['Drop'], inplace=True)
