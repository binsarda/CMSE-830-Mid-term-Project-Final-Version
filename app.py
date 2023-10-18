import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from PIL import Image

import hiplot as hip


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




stroke=pd.read_csv("stroke.csv")
heartdisease=pd.read_csv("heartdisease.csv")
lungcancer=pd.read_csv("lungcancer.csv")


st.markdown("# !!!! Welcome to my website !!!!")
col1,col2,col3=st.columns([1,2,1])

logo = Image.open('msu_logo.png')
col1.image(logo, caption='MSU')

egr = Image.open('msu.png')
egr= egr.resize((1000, 1000))
col3.image(egr, caption='Spartans')

st.sidebar.write("My github link: [link](https://github.com/binsarda?tab=repositories)")
cmse= Image.open('cmse.jfif')

st.sidebar.image(cmse, caption='CMSE')

col1,col2,col3=st.columns([1,2,1])
col1.write("Owner Name: Sardar Nafis Bin Ali")
col3.write("Institution: Michigan State University")
with st.expander(  "# Click here to know about author of this site!"):
    nafispic = Image.open('nafis.jpg')
    st.image(nafispic, caption='Sardar Nafis Bin Ali')
    st.write("Sardar is driven by a thirst for knowledge and has a spirit of exploration. "
             "He focused on the captivating domain of high-speed aerodynamics during his undergraduate "
             "studies in mechanical engineering. Currently, he is pursuing "
             "a Ph.D. in mechanical engineering and planning to do a dual"
             " degree with the department of communicative sciences and "
             "disorders. He aims to learn about the intricate aspects of"
             "human communication and contribute to advancements in voice science."
             " Apart from his academic pursuits, he finds solace in traveling, "
             "embracing diverse cultures, and capturing the world's beauty through "
             "his experiences. Sardar actively engages in initiatives "
             "promoting sustainability and environmental conservation. "
             "With an unquenchable curiosity and unwavering dedication, "
             "he continues to make a meaningful impact in his chosen fields and beyond.")

visitor_name=st.text_input("Enter your name:"," ")
st.write("Thank you",visitor_name,", for visiting my website.")

st.header("!!!Instructions!!!")
st.subheader("Please select the default 'Stroke' dataset. This app works for all kind of dataesets but you will get most options when you select 'Stroke' dataset.  Other options are "
         " included for future developement!!! And do not click on anything when you see running icon!!!! ")


st.markdown("Here you can find datasets related to serious diseases.")
st.write("There are 3 datsets-")
st.markdown("*")
st.write("1. Heart Disease")
st.markdown("*")
st.write("2.Lungs Cancer")
st.markdown("*")
st.write("3.Brain Stroke")
st.markdown("*")



stroke_read_status=False

data_button=st.selectbox('Please select one dataset from the following:',['Brain Stroke','Lungs Cancer','Heart Disease'])

if data_button == 'Brain Stroke':
    df1 = stroke
    st.write("You have selected 'Brain Stroke' Dataset")
    stroke_read_status = True
elif data_button=='Heart Disease':
    df1=heartdisease
    st.write("You have selected 'Heart Disease' Dataset")
elif data_button=='Lungs Cancer':
    df1 = lungcancer
    st.write("You have selected 'Lungs Cancer' Dataset")


status=False
file_button=st.radio('Do you want your own dataset to upload', ['No', 'Yes'])
if file_button=='Yes':
    uploaded_file = st.file_uploader("Choose a CSV file")
    try:
        pd.read_csv(uploaded_file.name)
        status=True
    except:
        status=False
        st.write("File reading not successful")
    if status:
        st.write(f"You uploaded your file successfully and name of your uploaded file is: {uploaded_file.name}")
        df1=pd.read_csv(uploaded_file.name)
        stroke_read_status = False


button=st.radio('Do you want to delete any row having NaN in at least one of the fields', ['Yes', 'No'])
if button=='Yes':
    df=df1.dropna()
    st.write("You deleted rows having NaN in at least one of the fields")
elif button=='No':
    df = df1
st.write(df.head(12))

if stroke_read_status:
    with st.expander("# Click here to learn about ' Brain Stroke' dataset. "):
        strokepic = Image.open('strokepic.jpg')
        st.image(strokepic, caption='Brain-Stroke')
        st.write("Finding significant insights and patterns within"
                 " the variables of this 'Brain Stroke' dataset is the main objective of "
                 "data visualization in the context of exploratory data "
                 "analysis (EDA). We want to learn more about the potential "
                 "connections between the prevalence of strokes and various "
                 "variables like age, gender, particular health conditions, lifestyle"
                 " preferences, and occupation. Our goal is to lay the groundwork "
                 "for deeper analyses and the creation of predictive models, ultimately "
                 "facilitating a more thorough assessment of stroke risk. We do this by "
                 "using visual representations that highlight distributions, associations, and correlations.")









col1,col2,col3=st.columns([2,1,2])
button1=col1.button("Show Statistics")
if button1:
    st.sidebar.write(df.describe())

if col3.button("Hide Statistics"):
    button1=False










cols=df.columns
numcols= df.select_dtypes(include=[np.number]).columns
strcols=df.select_dtypes(include=['object']).columns
numcoldf=df[numcols]
strcoldf=df[strcols]
####start

col1,col2,col3=st.columns([2,1,2])
button2=col1.button("Show Columns")
if button2:
    st.sidebar.write("No. of columns are ",len(cols))
    st.sidebar.write("The columns are following-")
    st.sidebar.write(df.columns)
    st.sidebar.write("Name of columns containing numerical values: ")
    st.sidebar.write(numcols)
    st.sidebar.write("Name of columns containing string or non-numerical values: ")
    st.sidebar.write(strcols)
if col3.button("Hide Columns"):
    button2=False

col1,col2,col3=st.columns([2,2,3])
col3.header("Interactive Plot")
exp = hip.Experiment.from_dataframe(df)
htmlcomp=exp.to_html()
st.components.v1.html(htmlcomp,width=1000, height=700, scrolling=True)


#st.write("Please select following variables for  plotting")
#xv=st.selectbox('Please select x or first variable:',numcols)
#yv=st.selectbox('Please select y or second variiable:',numcols)
#zv=st.selectbox('Please select hue or third variiable:',strcols)


st.header("Different kinds of plots")

graph_button=st.selectbox('Please select one kind of graph from the following options:',['Bar Plot','HeatMap','Violin Plot','Box Plot','2-D Scatter Plot','3-D Scatter Plot'])



if graph_button=='Bar Plot':
    st.write("Please select following variables for bar plot")
    xv = st.selectbox('Please select x or first variable for bar plot:', cols)
    yv = st.selectbox('Please select y or second variiable for bar plot:', cols)
    st.bar_chart(data=df, x=xv, y=yv)
    st.pyplot(plt.gcf())



elif graph_button=='HeatMap':
    sns.heatmap(numcoldf.corr(), annot=True)
    st.pyplot(plt.gcf())





elif graph_button=='Violin Plot':
    st.write("Please select following variables for violin plot")
    xv = st.selectbox('Please select x or first variable for violin plot:', strcols)
    yv = st.selectbox('Please select y or second variiable for violin plot:', numcols)
    zv = st.selectbox('Please select hue or third variiable for violin plot:', strcols)
    sns.violinplot(data=df, x=xv, y=yv, hue=zv)
    st.pyplot(plt.gcf())




elif graph_button=='Box Plot':
    st.write("Please select following variables for  plot")
    xv = st.selectbox('Please select x or first variable for  plot:', strcols)
    yv = st.selectbox('Please select y or second variiable for  plot:', numcols)
    zv = st.selectbox('Please select hue or third variiable for   plot:', strcols)
    sns.boxplot(x=xv, y=yv,hue=zv,  data=df)
    st.pyplot(plt.gcf())

elif graph_button=='2-D Scatter Plot':
    st.write("Please select following variables for  plot")
    xv = st.selectbox('Please select x or first variable for  plot:', numcols)
    yv = st.selectbox('Please select y or second variiable for  plot:', numcols)
    zv= st.selectbox('Please select z or hue or third variiable for  plot:', strcols)

    a=sns.scatterplot(data=df, x=xv, y=yv, hue=zv)
    st.pyplot(plt.gcf())
elif graph_button=='3-D Scatter Plot':
    st.write("Please select following variables for  plot")
    xv = st.selectbox('Please select x or first variable for plot:', numcols)
    yv = st.selectbox('Please select y or second variiable for plot:', numcols)
    zv= st.selectbox('Please select z or hue or third variiable for  plot:', numcols)
    fig3d = plt.figure(figsize=(15,15))
    ax3d = fig3d.add_subplot( projection='3d')



    ax3d.set_xlabel(xv)
    ax3d.set_ylabel(yv)
    ax3d.set_zlabel(zv)

    ax3d.scatter(df[xv],df[yv], df[zv])



    st.pyplot(fig3d)



### finish
st.header("Please select reduced number of columns for Reduced Dataset (Select at least 3 variables (at least 2  of numerical type"
         " and at least one  of string or non-numerical type))")
red_cols=st.multiselect('Pick the columns', cols)



if len(red_cols)>0:
    red_df = df[red_cols]
    st.write(
        f"You have choosen {len(red_cols)} number of columns in datatset and number of different column is {len(red_cols)} ")
    st.write("Reduced Dataset")
    st.write(red_df.head(10))
    red_numcols = red_df.select_dtypes(include=[np.number]).columns
    red_strcols = red_df.select_dtypes(include=['object']).columns
    red_ndf=df[red_numcols]
    red_sdf=df[red_strcols]
    st.sidebar.write("For reduced dataset")
    st.sidebar.write("No. of columns are ", len(red_cols))
    st.sidebar.write("The columns are following-")
    st.sidebar.write(red_df.columns)
    st.sidebar.write("Name of columns containing numerical values: ")
    st.sidebar.write(red_numcols)
    st.sidebar.write("Name of columns containing string or non-numerical values: ")
    st.sidebar.write(red_strcols)
    if len(red_numcols) == 1:
        st.write("Please select following variables for different plotting (for reduced dataset)")
        rxv = st.selectbox('(For reduced dataset) Please select x or first variable:', red_numcols)


    if len(red_numcols) >= 2:
        st.write("Please select following variables for different plotting (for reduced dataset)")
        rxv = st.selectbox('(For reduced dataset) Please select x or first variable:', red_numcols)
        ryv = st.selectbox('(For reduced dataset) Please select y or second variiable:', red_numcols)
        if len(red_strcols) >= 1:
            rzv = st.selectbox('(For reduced dataset) Please select hue or third variiable:', red_strcols)


        plot1 = plt.figure(figsize=(10, 4))
        sns.lineplot(x=rxv, y=ryv, data=red_df)
        st.pyplot(plot1)

        plot2 = sns.pairplot(red_df)
        st.pyplot(plot2.fig)

        plot3 = sns.heatmap(red_ndf.corr(), annot=True)
        st.pyplot(plot3.get_figure())

        fig4, ax4 = plt.subplots()
        sns.heatmap(red_ndf.corr(), ax=ax4, annot=True)
        st.write(fig4)

if len(red_cols)>0:
    if len(red_numcols) >= 2:
        st.write("Linear Regression")
        rrxv1 = st.selectbox(' Please select x or independent variable for linear regression:', red_numcols)
        rryv1 = st.selectbox('(For reduced dataset) Please select y or dependent variable for linear regression:',
                             red_numcols)



        rrxv = red_df[rrxv1]
        rryv = red_df[rryv1]

        regressor = LinearRegression()
        regressor.fit(rrxv.values.reshape(-1, 1), rryv)
        interce = regressor.intercept_
        coeff = regressor.coef_


        y_pred = rrxv*coeff+interce

        fig, ax = plt.subplots()
        ax.scatter(rrxv, rryv, color='red')
        ax.plot(rrxv, y_pred, color='blue')
        plt.xlabel(f'{rrxv1}', fontsize=18)
        plt.ylabel(f'{rryv1}', fontsize=16)
        plt.title("Linear Regression Plot")

        fig.show()
        st.pyplot(fig)

        buttonreg = st.radio(
            f' Select yes to predict your {rryv1} based on your {rrxv1}',
            ["No","Yes"])



        if buttonreg=="Yes":
            rinp = st.number_input(f"Insert value to predict your {rryv1}", value=0)
            if rinp != None:
                rpredict = rinp*coeff+interce

                st.write(f"Your {rryv1} is {round(rpredict[0],3)} for {rrxv1} value of {rinp}")



if stroke_read_status:
    st.header("Evaluation of Stroke based on all other parameters")
    st.header("Assigning discrete integer values to catagorical data(Label Encoding) ")
    dflast = df.copy(deep=True)

    st.subheader("For Gender")
    a = df["gender"].unique()
    b = [1, 2]
    dflast['gender'].replace(a, b, inplace=True)
    j = len(b)
    for i in range(j):
        st.write(f"{a[i]} is replaced with number {b[i]} ")

    st.subheader("For Ever Married")
    a = df["ever_married"].unique()
    b = [1, 0]
    dflast['ever_married'].replace(a, b, inplace=True)
    j = len(b)
    for i in range(j):
        st.write(f"{a[i]} is replaced with number {b[i]} ")

    st.subheader("For Work Type")
    a = df["work_type"].unique()
    b = [3, 2, 1, 0]
    dflast['work_type'].replace(a, b, inplace=True)
    j = len(b)
    for i in range(j):
        st.write(f"{a[i]} is replaced with number {b[i]} ")

    st.subheader("For Residence Type")
    a = df["Residence_type"].unique()
    b = [1, 2]
    dflast['Residence_type'].replace(a, b, inplace=True)
    j = len(b)
    for i in range(j):
        st.write(f"{a[i]} is replaced with number {b[i]} ")


    st.subheader("For Smoking Status")
    a = df["smoking_status"].unique()
    b = [2, 1, 3, 0]
    dflast['smoking_status'].replace(a, b, inplace=True)
    j = len(b)
    for i in range(j):
        st.write(f"{a[i]} is replaced with number {b[i]} ")

    st.sidebar.header("Numerical Values used for Catagorical Data")
    st.sidebar.subheader("For Gender")
    a = df["gender"].unique()
    b = [1, 2]
    j = len(b)
    for i in range(j):
        st.sidebar.write(f"{a[i]} is replaced with number {b[i]} ")

    st.sidebar.subheader("For Ever Married")
    a = df["ever_married"].unique()
    b = [1, 0]
    j = len(b)
    for i in range(j):
        st.sidebar.write(f"{a[i]} is replaced with number {b[i]} ")

    st.sidebar.subheader("For Work Type")
    a = df["work_type"].unique()
    b = [3, 2, 1, 0]
    j = len(b)
    for i in range(j):
        st.sidebar.write(f"{a[i]} is replaced with number {b[i]} ")

    st.sidebar.subheader("For Residence Type")
    a = df["Residence_type"].unique()
    b = [1, 2]
    j = len(b)
    for i in range(j):
        st.sidebar.write(f"{a[i]} is replaced with number {b[i]} ")

    st.sidebar.subheader("For Smoking Status")
    a = df["smoking_status"].unique()
    b = [2, 1, 3, 0]
    j = len(b)
    for i in range(j):
        st.sidebar.write(f"{a[i]} is replaced with number {b[i]} ")









    st.header("Now, predict your percentage probability of stroke based on all other parameters: ")
    numofcols = len(dflast.columns)
    x_train = dflast.iloc[:, 0:numofcols - 1]
    y_train = dflast.iloc[:, -1]
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    Intercept = reg.intercept_
    Coefficients = reg.coef_


    colnames=dflast.columns


    dict={}

    dummy=[];


    for i in range(numofcols-1):

        dummy.append( st.number_input(f"Insert the value of your {colnames[i]}", value=0 ) )

        #dict.update({ f'k{i}' ,    dummy[i]      })

    #for j in range(numofcols - 1):
        #if dict[f'k{j}'] == None:
            #dict[f'k{j}'] == 0

    #dict.update({f'key{j}': 'geeks'})

    for i in range(numofcols-1):

        if dummy[i] is None:
            dummy[i]=0

    for i in range(numofcols - 1):
        st.write(f"You selected value: {dummy[i]} for the property '{colnames[i]}' ")

    y_predicted = 0

    for i in range(numofcols-1):
        y_predicted=y_predicted+dummy[i]*Coefficients[i]

    y_predicted = y_predicted + Intercept

    #for i in range(len(Coefficients)):
        #y_predicted = y_predicted + x_inps[i] * Coefficients[i]
    #y_predicted = y_predicted + Intercept

    if y_predicted <= 0:
        y_predicted = 0
    if y_predicted >= 1:
        y_predicted = 1

    y_predicted = y_predicted * 100

    st.write(f"Your chance of getting stroke is {round(y_predicted, 3)} %.")

    st.header("My own analysis for Stroke dataset")




    st.subheader("Relationship-Plot")
    sns.relplot(data=df, x="age", y="avg_glucose_level", hue="smoking_status", style="stroke", sizes=(400, 400),alpha=1, palette="muted",height=6)
    st.pyplot(plt.gcf())

    sns.relplot(data=df, x="age", y="bmi", hue="smoking_status", style="stroke", sizes=(400, 400), alpha=1,palette="muted",height=6)
    st.pyplot(plt.gcf())

    st.subheader("Data-Distribution-Plot")
    sns.displot(df, x="stroke", col="smoking_status", row="heart_disease",binwidth=0.2, height=3, facet_kws={'margin_titles':True, 'sharex':False, 'sharey':False}   )
    st.pyplot(plt.gcf())

    st.subheader("Linear Regression Plot")
    sns.lmplot(data=df, x="age", y="avg_glucose_level",col="stroke", row="heart_disease", height=3, facet_kws={'margin_titles':True, 'sharex':False, 'sharey':False} )
    st.pyplot(plt.gcf())

    sns.lmplot(data=df, x="age", y="bmi", col="stroke", row="heart_disease", height=3,
               facet_kws={'margin_titles': True, 'sharex': False, 'sharey': False})
    st.pyplot(plt.gcf())

    sns.lmplot(data=df, x="bmi", y="avg_glucose_level", col="stroke", row="heart_disease", height=3,
               facet_kws={'margin_titles': True, 'sharex': False, 'sharey': False})
    st.pyplot(plt.gcf())

    col1, col2, col3 = st.columns([1, 2, 1])
    buttonlr = col1.button("Show Logistic Regressions")
    if buttonlr:
        st.subheader("Logistics Regression Plot")
        plt.figure(figsize=(5, 5))
        sns.lmplot(x="age", y="stroke", row="smoking_status", hue="gender", data=df,logistic = True, truncate = True)
        #aaaa.set(xlim=(0, 100), ylim=(-.05, 1.05))
        st.pyplot(plt.gcf())
    if col3.button("Hide Logistic Regressions"):
        buttonlr = False



    st.subheader("Summarize")
    st.write("From above plots we can notice:")
    smoke = 100 * len(df.loc[(df["smoking_status"] == 'smokes') & (df["stroke"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'smokes'])
    str = 'smoke'
    st.write(
        f"Percentage of person who {str} and having stroke compared to total number of person who {str} is {round(smoke, 3)} % ")

    formerlysmoke = 100 * len(df.loc[(df["smoking_status"] == 'formerly smoked') & (df["stroke"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'formerly smoked'])
    str = 'formerly smoke'
    st.write(
        f"Percentage of person who {str} and having stroke compared to total number of person who {str} is {round(formerlysmoke, 3)} % ")

    neversmoke = 100 * len(df.loc[(df["smoking_status"] == 'never smoked') & (df["stroke"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'never smoked'])
    str = 'never smoke'
    st.write(
        f"Percentage of person who {str} and having stroke compared to total number of person who {str} is {round(neversmoke, 3)} % ")

    unknown = 100 * len(df.loc[(df["smoking_status"] == 'Unknown') & (df["stroke"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'Unknown'])
    str = 'smoking condition is unknown'
    st.write(
        f"Percentage of person whose {str} and having stroke compared to total number of person whose {str} is {round(unknown, 3)} % ")

    smoke2 = 100 * len(df.loc[(df["smoking_status"] == 'smokes') & (df["heart_disease"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'smokes'])
    str = 'smoke'
    st.write(
        f"Percentage of person who {str} and having heart disease compared to total number of person who {str} is {round(smoke2, 3)} % ")

    formerlysmoke2 = 100 * len(df.loc[(df["smoking_status"] == 'formerly smoked') & (df["heart_disease"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'formerly smoked'])
    str = 'formerly smoke'
    st.write(
        f"Percentage of person who {str} and having heart disease compared to total number of person who {str} is {round(formerlysmoke2, 3)} % ")

    neversmoke2 = 100 * len(df.loc[(df["smoking_status"] == 'never smoked') & (df["heart_disease"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'never smoked'])
    str = 'never smoke'
    st.write(
        f"Percentage of person who {str} and having heart disease compared to total number of person who {str} is {round(neversmoke2, 3)} % ")

    unknown2 = 100 * len(df.loc[(df["smoking_status"] == 'Unknown') & (df["heart_disease"] == 1)]) / len(
        df.loc[df["smoking_status"] == 'Unknown'])
    str = 'smoking condition is unknown'
    st.write(
        f"Percentage of person whose {str} and having heart disease compared to total number of person whose {str} is {round(unknown2, 3)} % ")
