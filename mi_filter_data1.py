"""
About the code:
    Reading the given MI data, checking the columns,
    Filtering the data using the bounding box
    Elimination of columns that are not useful like Salinity
    Elimination of null values
    Input:UW_data.csv
    Output: UW_data_r_Ireland.csv
    
    Using the output file to further analyse the data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#This is the original file having all the parameters no filtering
df=pd.read_csv("MI_data/original_UW_data/UW_data.csv",names=["Datetime","Lon","Lat","Temperature","Salinity"])

#Check the shape of data
print('Shape of the original Underway Data:',df.shape)#Shape of the original Underway Data: (54585495, 5)

#Check the data types of each columns
print("Datatype of each column:",df.dtypes)
#Datatype of each column: Datetime        object
#Lon            float64
#Lat            float64
#Temperature    float64
#Salinity       float64
#dtype: object

#Filtering of data for the selected bounding box

df=df[df['Lon']>=-14]
df=df[df['Lon']<=-5]
df=df[df['Lat']<=56]
df=df[df['Lat']>=49]
#After filtering with bounding box shape of the file is
print("After Filtering with bounding box, shape of the file is:",df.shape)#(45712165, 5)
#Dropping of Salinity Column, as we are focusing on SST
df=df.drop(columns="Salinity")
#Checking missing data in Temperature column
m=df[df["Temperature"]== -999.0]
print("Shape of NaN data:",m.shape)#Shape of NaN data: (1232, 4)

"""
Missing data is -999.0, there is 1232 missing data,
Original data (54585495,4)
Remove the -999.0 data
"""
#Dropping the rows having -999.0
df=df[df["Temperature"]!= -999.0]
print("After dropping, shape of of the file is:",df.shape)#(45710933, 4)

#Saving the dataframe into a new file
df.to_csv("MI_data/original_UW_data/UW_data_r_Ireland.csv",index=None)



