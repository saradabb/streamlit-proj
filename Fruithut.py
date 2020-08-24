#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[1]:

#importing neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import streamlit as st
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import time


def main ():
    st.title ("Fruithut Sales Analysis")
    st.subheader ("Streamlit Project by Sara Dabbous")
    #creating side menu
    menu = ["Home","Dataset", "Data Cleaning", "Data Insights", "Models and Results", "About This App"]
    choice = st.sidebar.selectbox ("Choose one of the below to explore", menu)
    #File uploader to drop the dataset
    data = st.file_uploader("Upload file here (only excel or csv acceptable):", type=['csv','xlsx'])
    #Reading the data and analyzing it
    if data is not None:
        data=pd.read_csv(data)
    else:
        data=pd.read_csv('C:/Users/sd69/fruithut.csv')

    #Adding Total Sales column
    data['TOTALSALES'] = data.UNITS * data.PRICE
    if choice == "Home":
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.1)
        st.success("Data Upload Successful")

    elif choice == "Dataset":
        st.header ("Fruithut Sales Dataset Exploration")
        st.dataframe(data.head(10))
        if st.checkbox ("Column Data Types"):
            st.subheader ("Column Data Types")
            st.table(data.dtypes)
        if st.checkbox ("Shape of Dataset"):
            st.subheader ("Shape of Dataset")
            st.table (data.describe())
        if st.checkbox ("Filter Multiple Columns"):
            columns_select=st.multiselect ("Which columns would you like to see?",data.columns)
            dataselect=data[columns_select]
            st.dataframe(dataselect)
            if st.checkbox("Summary of Selected Dataframe"):
                st.write(dataselect.describe())

    elif choice == "Data Cleaning":
        st.header ("Data Pre-Processing")
        #Transferring datatype from "object" to string
        data.DATENEW =  data.DATENEW.astype('datetime64')
        data['TOTALSALES'] = data.UNITS * data.PRICE
        if st.checkbox ('Data Type & Shape'):
            st.write(data.dtypes)
            st.write(data.shape)
        #specifiying and removing NA columns
        data.apply(lambda x: len(x.unique()))
        if st.checkbox ('Display null values'):
            st.write("The number of null values for each variable is found below:")
            st.write(data.isnull().sum())
            st.write("This shows that there are 55 rows that seem to be empty...")
            if st.button ("Let's Fix it!"):
                data.dropna(inplace=True)
                st.write(data.isnull().sum())
                st.success ("We remove all empty rows and we now have 0 null values")
        if st.checkbox ('Final Dataset after cleaning'):
            st.table(data.head(10))

    elif choice == "Data Insights":
        data['TOTALSALES'] = data.UNITS * data.PRICE
        #Dropping unneccesary columns
        data1 = data.drop(['TICKET','REFERENCE', 'CODE', 'TOTAL', 'TRANSID'], axis='columns')
        #Adjusting date format
        data1.DATENEW =  data.DATENEW.astype('datetime64')
        #Replacing null values
        data1.apply(lambda x: len(x.unique()))
        data1.dropna(inplace=True)
        if st.checkbox ("Units Sold per Fruit"):
            st.subheader ("Sales Units Distribution per Fruit")
            #Plotting sales per fruit by creating a pivot table first
            CATEGORY_pivot = data1.pivot_table (index = "CATEGORY", values= "UNITS", aggfunc=np.sum)
            CATEGORY_pivot_sorted= CATEGORY_pivot.sort_values ('UNITS', ascending= True)
            #setting blue and black as my_colors
            my_colors = 'bk'
            CATEGORY_pivot_sorted.plot(kind="barh", figsize= (15,10), color=my_colors, alpha=0.9)
            #changing axis titles and rotation
            plt.xlabel ("Fruit Category")
            plt.ylabel ("Quantity")
            plt.xticks (rotation=90)
            #Adjusting style of ticks
            sns.set_style('ticks')
            #choosing a style from matplotlib
            plt.style.use ('fast')
            st.pyplot(plt.show())
        if st.checkbox ("Total Sales per Fruit"):
           st.subheader ("Sales per Fruit in $")
           CATEGORY_pivot = data1.pivot_table (index = "CATEGORY", values= "TOTALSALES", aggfunc=np.sum)
           CATEGORY_pivot_sorted= CATEGORY_pivot.sort_values ('TOTALSALES')
           CATEGORY_pivot_sorted.plot.barh(figsize= (15,10), alpha=0.9)
           sns.set_style('ticks')
           plt.style.use('tableau-colorblind10')
           plt.ylabel ("Fruit")
           plt.xlabel ("Sales")
           st.pyplot(plt.show())
        if st.checkbox ("Payment Methods"):
            st.subheader ("Payment Methods")
            data1['PAYMENT'].value_counts().plot(kind='bar', colormap= "tab20c")
            plt.ylabel ("Number of Transactions")
            plt.xlabel ("Payment Method")
            st.pyplot(plt.show())
        if st.checkbox ("Correlation Matrix"):
            st.subheader ("Correlation Matrix")
            numeric_features = data1.select_dtypes (include=[np.number])
            df = numeric_features
            #Correlation between numeric features
            corr = numeric_features.corr()
            ##Correlation Matrix Generation
            f, ax=plt.subplots (figsize=(12,9))
            sns.heatmap (corr, vmax=0.9, square = True)
            st.pyplot (plt.show())
        if st.checkbox ('Sales Over Time'):
            st.subheader ('Chart Showing Sales over Time')
            ax = plt.gca().get_xaxis()
            #Setting x-axis format to be Y-m-d
            ax.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
            plt.plot(data1['DATENEW'], data1['TOTALSALES'])
            plt.xticks (rotation=90)
            plt.rcParams['figure.figsize'] = 80,20
            st.pyplot(plt.show())

    elif choice =="Models and Results":
        data['TOTALSALES'] = data.UNITS * data.PRICE
        #Adjusting date format
        data.DATENEW =  data.DATENEW.astype('datetime64')
        #Replacing null values
        data.apply(lambda x: len(x.unique()))
        data.dropna(inplace=True)
        #Allowing manipulation of sample (using all 600K observations will take time)
        sample = st.sidebar.slider ("Select Sample number of observations", 20000,300000,200000)
        datasample = data.sample(n=sample)
        st.table(datasample.head(20))

        if st.checkbox ("Select multiple columns"):
            columns_select=st.multiselect ("Which columns would you like to see?",datasample.columns)
            dataselect=datasample[columns_select]
            st.table(dataselect.head(20))

        #transforming data through label encoder to turn non-numerical labels to numeric
        le = LabelEncoder()
        datasample = datasample.apply(le.fit_transform)
        x = datasample.drop(['TOTALSALES', 'DATENEW', "TICKET"], axis = 1)
        y = datasample['TOTALSALES']
        #allowing for manipulation of set seed for random state
        SetSeed=st.sidebar.slider("Set Seed for Random State",1,200,100)
        st.write (data.shape)

        # splitting the datatest ot x_train,x_test, y_tain, y_test with 30% test
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=SetSeed)

        #Creating the option to choose between models
        classifier_name=st.sidebar.selectbox("Choose a Model:", ('Linear Regression', 'KNN','Decision Tree'))

        #defining parameters of KNN to allow filtering
        def add_parameter(name_of_clf):
            param=dict()
            if name_of_clf=='KNN':
                K=st.sidebar.slider('K',1,20)
                param['K']=K
                return param
        #calling the function
        param=add_parameter(classifier_name)

        #model creation upon choice of model above
        def get_classifier(name_of_clf,param):
            mdl= dict()
            if name_of_clf == 'KNN':
                mdl= KNeighborsClassifier(n_neighbors=param['K'])
            elif name_of_clf == 'Linear Regression':
                mdl=LinearRegression()
            elif name_of_clf == 'Decision Tree':
                mdl=DecisionTreeRegressor()
            else:
                st.warning ("Please choose a model")

            return mdl

        #calling function and fitting it to training
        mdl= get_classifier(classifier_name, param)
        mdl.fit (x_train, y_train)
        y_pred=mdl.predict(x_test)
        st.write(y_pred)

        #Measuring results of model
        if classifier_name == 'Linear Regression':
            mse = mean_squared_error(y_test, y_pred)
            st.write("Model Name:", classifier_name)
            st.write('RMSE :', np.sqrt(mse))
            st.write("Result :",mdl.score(x_train, y_train))


        else:
            accuracy=accuracy_score(y_test,y_pred)
            st.write("Model Name:", classifier_name)
            st.write ('Accuracy:', accuracy)

    elif choice == 'About This App':
        from PIL import Image
        img = Image.open ("image.png")
        if img.mode != 'RGB':
            img = img.convert('RGB')
            st.image(img, width=500)
        st.write("An Interactive Beginner-level Web-App analyzing sales data for a fruit grocery shop")
        st.write('To view/download dataset: https://www.kaggle.com/luckysan/retail-grocery-store-sales-data-from-20162019')
        st.write("To know more about this dashboard and how to explore it: https://drive.google.com/drive/folders/1_2gXyRNKmrSwx04IYRjfF4Dpi65MmD3S?usp=sharing")
        st.write ("To reach the developer: https://www.linkedin.com/in/sara-dabbous-1989ba65/")


if __name__ == '__main__':
    main ()
