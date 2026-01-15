# data acess and understanding 

from json import encoder
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
df=pd.read_csv("cars.csv")
# checking the the first five rows 
print(df.head())
# checking the last five row 
print(df.tail())
# checking the basic  rows and columns info 
print("\nShape:", df.shape)
# checking the columns name 
print("\nColumns:", df.columns)
# checking the non null values 
print("\ninfo:",df.info())
# checking the null values in each columns 
print("\nNull values:",df.isnull().sum().sort_values(ascending=False))
# checking the duplicate rows 
print("\nDuplicated rows:", df.duplicated().sum())
# TARGET COLUMNS ANALYSIS 
# show the first 5 body types 
print("\nBody types:",df["BodyType"].head())
# showing how many the unique body types exit in this dataset 
print("Body type nunique:",df["BodyType"].nunique())
# showing the body types top with how many types it appaer in the dataset 
print("Body type count values:")
print(df["BodyType"].value_counts())
# statistical analysis 
print("\nDescribe:",df.describe())
# renaming columns that have underscores
df.rename(columns={
    'Power_kW_HP': 'PowerHP',
    'EngineCapacity_Corrected': 'EngineCapacityCorr',
    'APK_month_diff': 'APKmonth'
}, inplace=True)
# checking columns after renaming
print("\nColumns after renaming:",df.columns)
 # DATA CLEANING 
 # removing the duplicate 
df = df.drop_duplicates()
print("duplicates removed New shape of the dataset :", df.shape)
# dropping the high missing values columns 
df=df.drop(columns=["FuelConsumption",'PreviousOwners',"APK"])
print("after dropping high missing value new shape of the data set ",df.shape)
# apply the mapping on the body type bcz of the fragmented data and fix rare noisy data 
df["BodyType"] = df["BodyType"].replace ({
    "SUV/Off-Road/Pick-Up":"SUV",
    "Stationwagen":"Wagon",
    "Coupé":"Coupe",
    "Cabrio":"Convertible",
    "Bedrijfswagen":"Van",
    "Gesloten bestelwagen":"Van",
    "Combi/Van":"Van",
    "Koel/geisoleerde":"Van"
})
print("body type after mapping:")
print(df["BodyType"].value_counts())
# EDA (exploratory data analysis)

print("\nBody Type Distribution:")
print(df["BodyType"].value_counts())

# Visualization 
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="BodyType", hue="BodyType", palette="viridis", legend=False)
plt.xticks(rotation=45, ha='right')
plt.title("Body Type Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Body Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()
# insights:
# hatchback and SUV having the long bar indicating that both body types are more comman as well showing the data is not balaance 
# the cope convertible and overig having the small bars showing long tail classes in the data 
# the data will make the model biased toward the more comman data type 

#2)engine capicity vs Body type 
median_engine_capacity_by_bodytype = df.groupby("BodyType")["EngineCapacity"].median()
print("\nMedian Engine Capacity by Body Type:")
print(median_engine_capacity_by_bodytype)
#visualization
sns.boxplot(data=df,x="BodyType",y="EngineCapacity") 
plt.xticks(rotation=45,ha="right")
plt.title("Engine Capacity vs Body Type",fontsize=14,fontweight="bold")
plt.xlabel("Body Type",fontsize=12)
plt.ylabel("Engine Capacity",fontsize=12)
plt.tight_layout()
plt.show()

# insights 
# hatchback and mpv and overig having the smaller engine 
#suv and van and couple having the large engine capaicity
# coupe and convertible  types show that they are powerful having the high engine outlieers 
# median provide the robust comparison between the engine sizes 

#3) power vs body type 
max_power = df.groupby("BodyType")["PowerHP"].max()
print("\nmax power by body type:")
print(max_power)

# visualization
sns.stripplot(data=df,x="BodyType",y="PowerHP",jitter=True,alpha=0.5)
plt.xticks(rotation=45,ha="right")
plt.title("Power vs Body Type",fontsize=14,fontweight="bold")
plt.xlabel("Body Type",fontsize=12)
plt.ylabel("power distribution by body",fontsize=12)
plt.tight_layout()
plt.show()
#insights 
# different body type have the different power distribution
# wagon ,convertible and couple having the high power car 
# hatchback ,mpv and overig,van having the low power cars
# suv having mid power cars

#4) weight vs body type  
normal_heavy_car=df.groupby("BodyType")["EmptyWeight"].quantile(0.75)
print("\n75th percentile weight by body type:")
print(normal_heavy_car)

# vislualization
sns.violinplot(data=df,x="BodyType",y="EmptyWeight")
plt.xticks(rotation=45)
plt.title("weight vs bodytype",fontsize=14,fontweight="bold")
plt.xlabel("body type",fontsize=12)
plt.ylabel("weight distribution by body",fontsize=12)
plt.tight_layout()
plt.show()
# insights
# the weight increase the body type 
# while the hatchback having the lightst weight car 
# while the vans are the heaviers
# suv ,seden overrig occupy the higher weight cars
# mpv wagon copes lie in the mid range 

# 5)km vs body type
median_km_by_bodytype = df.groupby("BodyType")["Km"].median()
print("\nMedian Km by Body Type:")
print(median_km_by_bodytype)
# visulization
sns.boxplot(data=df, x="BodyType", y="Km")
plt.xticks(rotation=45)
plt.title("Mileage vs Body Type")
plt.show()
# insights 
# hatchback and sedans moderate mileage reflect daily uses
# coupes and convertible have lower mileage suggest limitied or leisure use 
# suv and van having higher mileage indicating heavy and commercial use

#6) equipment embedding vs body type 
df["Equipscore"]=df[["A1","A2","A3","A4"]].sum(axis=1)
avg_equipscore_by_bodytype=df.groupby("BodyType")["Equipscore"].mean()
print("\nAverage equipment score by Body Type:")
print(avg_equipscore_by_bodytype)
# visuaization
sns.barplot(data=df, x="BodyType", y="Equipscore", errorbar=None)
plt.xticks(rotation=45)
plt.title("Average Equipment Score by Body Type")
plt.show()
# insights
# suv highest equipment score showing they are more comfotable
#hatchback and mpvs are moderately equipped
# van lowest equipment reflect the commerical oritented design

# MARKET SEGNMENT 
# brand domination by body type 
df.groupby(["BodyType","Brand"]).size() \
  .sort_values(ascending=False) \
  .groupby(level=0).head(2)
# visualization
top_brands = (
    df.groupby(["BodyType","Brand"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
      .groupby("BodyType")
      .head(3)
)

sns.barplot(data=top_brands, y="Brand", x="count", hue="BodyType", errorbar=None)
plt.title("Top Brands per Body Type")
plt.show()
# insights 
#volkswagon is most dominent brand overall
# while luxury dominent sedan ,couple and convertible 

# model dominent by body type 
df.groupby(["BodyType","Model"]).size() \
  .sort_values(ascending=False) \
  .groupby(level=0).head(3)
 # visualization
top_models = (
    df.groupby(["BodyType","Model"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
      .groupby("BodyType")
      .head(3)
)

sns.barplot(data=top_models, y="Model", x="count", errorbar=None)
plt.title("Top Models per Body Type")
plt.show()
# INSIHTS
# Model like polo 500 aygo have long bar showing widely used daily car
# A5 appaer short because of luxuary model
# kangoo and ductao are rare commmerical vehicales
#the graph shows that everyday economy cars 
# dominate the market, while luxury (A5) and commercial vehicles (Kangoo, Ducato)  appear far less frequently

# YEAR TREND 
df.groupby("YearBuilt").size()
# visualization
year_trend = df.groupby("YearBuilt").size().reset_index(name="count")

sns.lineplot(data=year_trend, x="YearBuilt", y="count")
plt.title("Car Listings Over Years")
plt.show()
# insights
# the graph shows an upward trend in car listings from 2015 to 2018
# this indicates a growing market for used cars over the past two decades
# older cars are less available.

# PREPROCESSING AFTER EDA
# 1) filling the null values of numerical columns with the median 
numerical_col=["Cylinders","EmptyWeight","Gears","PowerHP","EngineCapacity","CO2Emission"]
for col in numerical_col:
# Fill with group median first, then global median for remaining NaNs
    df[col] = df.groupby(["Brand","Model","YearBuilt"])[col] \
                 .transform(lambda x: x.fillna(x.median()))
# Fill any remaining NaNs with global median (for groups with all NaN values)
    df[col] = df[col].fillna(df[col].median())
# checking after filling the null values 
print("\n null values after filling :",df.isnull().sum())
#2) filling the categorical columns with mode(most occuring value)
categorical_col=["FuelType","EmissionClass","Drive"]
for col in categorical_col:
    mode=df[col].mode()[0]
    df[col]=df[col].fillna(mode)
# checking after filling the categorical columns
print("\n null values after filling the categorical columns :",df.isnull().sum())

# FEATURE ENGINEERING 
#1) power/weight ratio 
df['Power_Weight_Ratio']=df["PowerHP"]/(df["EmptyWeight"]+1)
#2) mileage log transform
df["LogKm"]=np.log(df["Km"] + 1)
#3) age of the car 
df["CarAge"]=2018-df["YearBuilt"]
print("\n Created engineered features: Power_Weight_Ratio, LogKm, CarAge")

# dropping the irrelevant columns BEFORE split
df = df.drop(columns=["Color"])
print(" Dropped Color column")

# apply the one hot encoding onthe categorical columns 
y=df["BodyType"]
x=df.drop(columns=["BodyType"])
low_cardinality=["Drive","Transmission","Category","EmissionClass","Warranty"]
x=pd.get_dummies(x,columns=low_cardinality,drop_first=True)
# checking the final features set
print("\nFinal features set columns:",x.columns)

# apply the categorical_encoder on the high cardinality 
import category_encoders as ce
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
encoder = ce.TargetEncoder(cols=["Brand","Model","FuelType"])
x_train[["Brand","Model","FuelType"]]=encoder.fit_transform(x_train[["Brand","Model","FuelType"]],y_train)
x_test[["Brand","Model","FuelType"]]=encoder.transform(x_test[["Brand","Model","FuelType"]])
# checking the result after encoding 
print(x_train[["Brand","Model","FuelType"]].head())
print(x_train.dtypes.head(10))
print("\nFinal features in training set:", x_train.shape[1])

# scaling the numerical columns with the enginerried features 
numerical_col=["Km","YearBuilt","EmptyWeight","PowerHP","EngineCapacity","CO2Emission","Price","Equipscore","APKmonth","EngineCapacityCorr","Power_Weight_Ratio","LogKm","CarAge"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train[numerical_col]=scaler.fit_transform(x_train[numerical_col])
x_test[numerical_col]=scaler.transform(x_test[numerical_col])
# checking after scaling
print("\n Scaled features")
print(x_train.head())

# HANDLE CLASS IMBALANCE WITH SMOTE (limited sampling for faster training)
from imblearn.over_sampling import SMOTE
from collections import Counter
print("\nHandling Class Imbalance ")
print(f"Before SMOTE: {Counter(y_train)}")
# Set target to balance minority classes up to 50000 samples
class_counts = Counter(y_train)
max_class_count = max(class_counts.values())
target_count = min(max_class_count, 50000)
sampling_strategy = {cls: target_count for cls, count in class_counts.items() if count < target_count}
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(f"After SMOTE: {Counter(y_train)}")
print(f" Balanced training set: {x_train.shape[0]} samples")

# Modeling 
#print("\n TRAINING MODELS ")
#1) logistic regression 
#print("Training Logistic Regression")
#from sklearn.linear_model import LogisticRegression
#lm=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)
#lm.fit(x_train,y_train)
#print(" Logistic Regression trained")

#5) applying the second classifier random forest
print("Training Random Forest")
from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,max_depth=20)
rc.fit(x_train,y_train)
print(" Random Forest trained")



#print(" Training Linear SVM...")
#from sklearn.svm import SVC
#svm=SVC(kernel="linear",class_weight="balanced",probability=True,random_state=42)
#svm.fit(x_train,y_train)
#print(" Linear SVM trained")
#print("\n=== ALL MODELS TRAINED ===\n")

# evaluation matrix 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
def evaluated_model(model,x_test,y_test,name):
    y_pred=model.predict(x_test)
    print(f"\n=={name}===")
    print("accuracy:",accuracy_score(y_test,y_pred))
    print("macrof1:",f1_score(y_test,y_pred,average="macro"))
    print("precision:",precision_score(y_test,y_pred,average="weighted"))
    print("recall:",recall_score(y_test,y_pred,average="weighted"))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    # evaluation 
print(" EVALUATING MODELS ")
#evaluated_model(lm,  x_test, y_test, "Logistic Regression (Baseline)")
evaluated_model(rc,  x_test, y_test, "Random Forest")
#evaluated_model(svm, x_test, y_test, "Linear SVM")
#print("\n ALL EVALUATIONS COMPLETE!")
    
    