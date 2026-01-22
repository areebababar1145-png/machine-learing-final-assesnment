import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("cars.csv")
print(df.head())
print(df.tail())
print("\nshape:", df.shape)
print("columns:", df.columns)
df.info()
print("\nnulls:", df.isnull().sum().sum())
print("duplicates:", df.duplicated().sum())
# target variable
print("\nbody types:")
print(df["BodyType"].value_counts())
print(df.describe())
# clean data
df.rename(columns={'Power_kW_HP':'PowerHP', 'EngineCapacity_Corrected':'EngineCapacityCorr', 'APK_month_diff':'APKmonth'}, inplace=True)
df = df.drop_duplicates()
df = df.drop(columns=["FuelConsumption",'PreviousOwners',"APK"])
print("\nshape after cleaning:", df.shape)
# fix body types
df["BodyType"] = df["BodyType"].replace({"SUV/Off-Road/Pick-Up":"SUV", "Stationwagen":"Wagon", 
    "Coupé":"Coupe", "Cabrio":"Convertible", "Bedrijfswagen":"Van", 
    "Gesloten bestelwagen":"Van", "Combi/Van":"Van", "Koel/geisoleerde":"Van"})
print(df["BodyType"].value_counts())
# eda
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="BodyType", hue="BodyType", legend=False)
plt.xticks(rotation=45)
plt.title("Body Type Distribution")
plt.show()
# class imbalance - hatchback and suv have way more data 

# engine capacity
print(df.groupby("BodyType")["EngineCapacity"].median())
sns.boxplot(data=df, x="BodyType", y="EngineCapacity") 
plt.xticks(rotation=45)
plt.title("Engine Capacity")
plt.show() 

# power
sns.stripplot(data=df, x="BodyType", y="PowerHP", alpha=0.3)
plt.xticks(rotation=45)
plt.title("Power")
plt.show()

# weight
sns.violinplot(data=df, x="BodyType", y="EmptyWeight")
plt.xticks(rotation=45)
plt.title("Weight")
plt.show()
plt.tight_layout()
plt.show()
# vans heaviest, hatchbacks lightest 

# mileage
sns.boxplot(data=df, x="BodyType", y="Km")
plt.xticks(rotation=45)
plt.title("Mileage")
plt.show()

# equipment features
df["Equipscore"] = df["A1"] + df["A2"] + df["A3"] + df["A4"]
sns.barplot(data=df, x="BodyType", y="Equipscore", errorbar=None)
plt.xticks(rotation=45)
plt.title("Equipment Score")
plt.show()

# top brands
top_brands = df.groupby(["BodyType","Brand"]).size().reset_index(name="count").sort_values("count", ascending=False).groupby("BodyType").head(2)
sns.barplot(data=top_brands, y="Brand", x="count", hue="BodyType")
plt.title("Top Brands")
plt.show()

# top models
top_models = df.groupby(["BodyType","Model"]).size().reset_index(name="count").sort_values("count", ascending=False).groupby("BodyType").head(2)
sns.barplot(data=top_models, y="Model", x="count")
plt.title("Top Models")
plt.show()

# year trend
year_trend = df.groupby("YearBuilt").size().reset_index(name="count")
sns.lineplot(data=year_trend, x="YearBuilt", y="count")
plt.title("Listings by Year")
plt.show()

# preprocessing
# filling missing values - using median for numerical
numerical_col=["Cylinders","EmptyWeight","Gears","PowerHP","EngineCapacity","CO2Emission"]
for col in numerical_col:
    df[col] = df.groupby(["Brand","Model","YearBuilt"])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())  
print("\nnulls after filling:",df.isnull().sum())
# categorical - use mode
categorical_col=["FuelType","EmissionClass","Drive"]
for col in categorical_col:
    df[col]=df[col].fillna(df[col].mode()[0])
print("\nnulls after categorical:",df.isnull().sum())

# new features
df['PowerWeightRatio'] = df["PowerHP"] / (df["EmptyWeight"] + 1)
df["LogKm"] = np.log(df["Km"] + 1)
df["Age"] = 2018 - df["YearBuilt"]
df = df.drop(columns=["Color"])
print("features created")

# prepare for modeling
y = df["BodyType"]
X = df.drop(columns=["BodyType"])

# one hot encoding
low_card = ["Drive","Transmission","Category","EmissionClass","Warranty"]
X = pd.get_dummies(X, columns=low_card, drop_first=True)
print("features:", X.shape[1])

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# target encoding for brand, model, fuel
import category_encoders as ce
encoder = ce.TargetEncoder(cols=["Brand","Model","FuelType"])
X_train[["Brand","Model","FuelType"]] = encoder.fit_transform(X_train[["Brand","Model","FuelType"]], y_train)
X_test[["Brand","Model","FuelType"]] = encoder.transform(X_test[["Brand","Model","FuelType"]])
print("encoded")

# scaling the numerical columns with the enginerried features 
numerical_col=["Km","YearBuilt","EmptyWeight","PowerHP","EngineCapacity","CO2Emission","Price","Equipscore","APKmonth","EngineCapacityCorr","PowerWeightRatio","LogKm","Age"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train[numerical_col]=scaler.fit_transform(X_train[numerical_col])
X_test[numerical_col]=scaler.transform(X_test[numerical_col])
# checking after scaling
print("\n Scaled features")
print(X_train.head())

# handle imbalance with smote
from imblearn.over_sampling import SMOTE
from collections import Counter
print("\nbefore:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("after:", Counter(y_train))
print(f" Balanced training set: {X_train.shape[0]} samples")

# Modeling 
#1) logistic regression 
print("Training Logistic Regression")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)
lr.fit(X_train,y_train)
print(" Logistic Regression trained")

#5) applying the second classifier random forest
print("Training Random Forest")
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,max_depth=20)
rf.fit(X_train,y_train)
print(" Random Forest trained")



print(" Training Linear SVM...")
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
# SGDClassifier is much faster than LinearSVC for large datasets
# Using 50% sample for ultra-fast training
sample_size = int(len(X_train) * 0.5)
X_svm = X_train.sample(n=sample_size, random_state=42)
y_svm = y_train[X_svm.index]
# SGD with hinge loss = linear SVM, very fast
svm_base = SGDClassifier(loss='hinge', class_weight='balanced', random_state=42, 
                         max_iter=300, tol=1e-3, n_jobs=-1)
svm = CalibratedClassifierCV(svm_base, cv=2)
svm.fit(X_svm, y_svm)
print(" Linear SVM trained")
print("\n=== ALL MODELS TRAINED ===\n")

# evaluate
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def eval_model(model, X_test, y_test, name):
    pred = model.predict(X_test)
    print(f"\n{name}:")
    print("accuracy:", accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred, average="macro"))
    print(classification_report(y_test, pred))
    
eval_model(lr, X_test, y_test, "Logistic Regression")
eval_model(rf, X_test, y_test, "Random Forest")
eval_model(svm, X_test, y_test, "SVM")

# cross validation
from sklearn.model_selection import cross_val_score
print("\ncv scores:")
print("LR:", cross_val_score(lr, X_train, y_train, cv=5, scoring='f1_macro').mean())
print("RF:", cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro').mean())

# feature importance
importances = pd.DataFrame({'feature': X_train.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print("\ntop features:")
print(importances.head(10))
importances.to_csv('feature_importance.csv', index=False)

# check minority classes
minority = y.value_counts()[y.value_counts() < y.value_counts().median()].index
print("\nminority classes:", list(minority))

# temporal analysis
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['pred'] = rf.predict(X_test)
test_df['year'] = df.loc[y_test.index, 'YearBuilt'].values

year_acc = test_df.groupby('year').apply(lambda x: accuracy_score(x['actual'], x['pred']))
print("\naccuracy by year:")
print(year_acc.tail())

print("\nall done!")
print("="*80)

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Use original data before SMOTE for CV (to avoid data leakage)
# Recreate the split
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Re-encode and scale for CV
encoder_cv = ce.TargetEncoder(cols=["Brand","Model","FuelType"])
X_train_cv = X_train_orig.copy()
X_train_cv[["Brand","Model","FuelType"]] = encoder_cv.fit_transform(
    X_train_cv[["Brand","Model","FuelType"]], y_train_orig
)

scaler_cv = StandardScaler()
X_train_cv[numerical_col] = scaler_cv.fit_transform(X_train_cv[numerical_col])

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.svm import LinearSVC
models_cv = {
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    'Linear SVM': LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
}

cv_results = {}
for name, model in models_cv.items():
    print(f"\nPerforming 5-Fold CV for {name}...")
    scores = cross_val_score(model, X_train_cv, y_train_orig, cv=cv, 
                             scoring='f1_macro', n_jobs=-1)
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"  Macro F1 Scores: {scores}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Save CV results
cv_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'Mean_F1': [cv_results[m]['mean'] for m in cv_results],
    'Std_F1': [cv_results[m]['std'] for m in cv_results]
})
cv_df = cv_df.sort_values('Mean_F1', ascending=False)
print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY")
print("="*80)
print(cv_df.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(cv_df['Model'], cv_df['Mean_F1'], xerr=cv_df['Std_F1'], 
         capsize=5, alpha=0.7, color='steelblue')
plt.xlabel('Macro F1 Score', fontsize=12)
plt.title('Cross-Validation Results (5-Fold)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('cv_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: cv_results.png")


# MINORITY CLASS ANALYSIS

print("\n" + "="*80)
print("MINORITY CLASS PERFORMANCE ANALYSIS")
print("="*80)

# Identify minority classes 
class_distribution = y_test.value_counts()
total_samples = len(y_test)
minority_threshold = 0.10
minority_classes = class_distribution[class_distribution / total_samples < minority_threshold].index.tolist()

print(f"\nMinority classes (< {minority_threshold*100}% of data): {minority_classes}")
print(f"Class distribution in test set:")
print(class_distribution)

# Detailed per-class metrics for all models
from sklearn.metrics import precision_recall_fscore_support

minority_results = []
for model_name, model in [('Logistic Regression', lr), 
                          ('Random Forest', rf), 
                          ('Linear SVM', svm)]:
    y_pred = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=minority_classes, average=None, zero_division=0
    )
    
    for i, cls in enumerate(minority_classes):
        minority_results.append({
            'Model': model_name,
            'Class': cls,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-Score': f1[i],
            'Support': support[i]
        })

minority_df = pd.DataFrame(minority_results)
print("\n" + "="*80)
print("MINORITY CLASS DETAILED METRICS")
print("="*80)
print(minority_df.to_string(index=False))

# Save minority class results
minority_df.to_csv('minority_class_performance.csv', index=False)
print("\n Saved: minority_class_performance.csv")

# Visualization
plt.figure(figsize=(12, 6))
for metric in ['Precision', 'Recall', 'F1-Score']:
    plt.figure(figsize=(10, 6))
    pivot = minority_df.pivot(index='Class', columns='Model', values=metric)
    pivot.plot(kind='bar', width=0.8)
    plt.title(f'{metric} for Minority Classes', fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Body Type', fontsize=12)
    plt.legend(title='Model')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'minority_{metric.lower().replace("-", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Saved: minority_{metric.lower().replace('-', '_')}.png")


# FEATURE IMPORTANCE ANALYSIS

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features (Random Forest):")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("\n Saved: feature_importance.csv")

# Visualization
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['Importance'], color='coral', alpha=0.8)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: feature_importance.png")

# SHAP ANALYSIS FOR MODEL INTERPRETABIL
print("\n" + "="*80)
print("SHAP ANALYSIS - MODEL INTERPRETABILITY")
print("="*80)

try:
    import shap
    print("\nCalculating SHAP values for Random Forest (this may take a few minutes)...")
    
    # Use a sample for SHAP to speed up computation
    sample_size = min(500, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Get class names
    class_names = rf.classes_
    
    # Summary plot for first class (or dominant class)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[0], X_test_sample, plot_type="bar", 
                      show=False, max_display=20)
    plt.title(f'SHAP Feature Importance - {class_names[0]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: shap_summary.png")
    
    # Detailed SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[0], X_test_sample, show=False, max_display=20)
    plt.title(f'SHAP Value Distribution - {class_names[0]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: shap_detailed.png")
    
    print("\n SHAP analysis complete!")
except Exception as e:
    print(f"SHAP analysis skipped: {e}")

# ROBUSTNESS: PERFORMANCE BY MODEL YEAR (TEMPORAL ANALYSIS)
print("\n" + "="*80)
print("TEMPORAL ANALYSIS - PERFORMANCE BY MODEL YEAR")
print("="*80)

# Group years into decades/periods
def year_to_period(year):
    if year < 2000:
        return "Pre-2000"
    elif year < 2005:
        return "2000-2004"
    elif year < 2010:
        return "2005-2009"
    elif year < 2015:
        return "2010-2014"
    else:
        return "2015+"

# Add period to test data
X_test_with_year = X_test.copy()
X_test_with_year['Period'] = df.loc[X_test.index, 'YearBuilt'].apply(year_to_period)
y_test_with_period = y_test.copy()

# Analyze performance by period
temporal_results = []
for period in ["Pre-2000", "2000-2004", "2005-2009", "2010-2014", "2015+"]:
    period_mask = X_test_with_year['Period'] == period
    if period_mask.sum() == 0:
        continue
    
    X_period = X_test[period_mask]
    y_period = y_test_with_period[period_mask]
    
    for model_name, model in [('Logistic Regression', lr), 
                              ('Random Forest', rf), 
                              ('Linear SVM', svm)]:
        y_pred = model.predict(X_period)
        macro_f1 = f1_score(y_period, y_pred, average='macro')
        accuracy = accuracy_score(y_period, y_pred)
        
        temporal_results.append({
            'Period': period,
            'Model': model_name,
            'Macro_F1': macro_f1,
            'Accuracy': accuracy,
            'Sample_Size': len(y_period)
        })

temporal_df = pd.DataFrame(temporal_results)
print("\n" + "="*80)
print("PERFORMANCE BY TIME PERIOD")
print("="*80)
print(temporal_df.to_string(index=False))

# Save temporal results
temporal_df.to_csv('temporal_performance.csv', index=False)
print("\n Saved: temporal_performance.csv")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Macro F1 by period
pivot_f1 = temporal_df.pivot(index='Period', columns='Model', values='Macro_F1')
pivot_f1.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_title('Macro F1 Score by Time Period', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Macro F1', fontsize=12)
axes[0].set_xlabel('Time Period', fontsize=12)
axes[0].legend(title='Model')
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Accuracy by period
pivot_acc = temporal_df.pivot(index='Period', columns='Model', values='Accuracy')
pivot_acc.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('Accuracy by Time Period', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_xlabel('Time Period', fontsize=12)
axes[1].legend(title='Model')
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('temporal_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: temporal_performance.png")


# ROBUSTNESS: PERFORMANCE BY BRAND SEGMENT

print("\n" + "="*80)
print("BRAND SEGMENT ANALYSIS")
print("="*80)

# Define brand segments
luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Jaguar', 'Lexus', 
                 'Volvo', 'Land Rover', 'Maserati', 'Bentley']
economy_brands = ['Volkswagen', 'Ford', 'Opel', 'Renault', 'Peugeot', 'Citroën', 
                  'Toyota', 'Hyundai', 'Kia', 'Skoda', 'Seat', 'Fiat', 'Suzuki']

def brand_to_segment(brand):
    if brand in luxury_brands:
        return "Luxury"
    elif brand in economy_brands:
        return "Economy"
    else:
        return "Mid-Range"

# Add segment to test data
X_test_with_brand = X_test.copy()
original_brands = df.loc[X_test.index, 'Brand']
X_test_with_brand['Segment'] = original_brands.apply(brand_to_segment)

# Analyze performance by brand segment
segment_results = []
for segment in ["Luxury", "Economy", "Mid-Range"]:
    segment_mask = X_test_with_brand['Segment'] == segment
    if segment_mask.sum() == 0:
        continue
    
    X_segment = X_test[segment_mask]
    y_segment = y_test[segment_mask]
    
    for model_name, model in [('Logistic Regression', lr), 
                              ('Random Forest', rf), 
                              ('Linear SVM', svm)]:
        y_pred = model.predict(X_segment)
        macro_f1 = f1_score(y_segment, y_pred, average='macro')
        accuracy = accuracy_score(y_segment, y_pred)
        
        segment_results.append({
            'Segment': segment,
            'Model': model_name,
            'Macro_F1': macro_f1,
            'Accuracy': accuracy,
            'Sample_Size': len(y_segment)
        })

segment_df = pd.DataFrame(segment_results)
print("\n" + "="*80)
print("PERFORMANCE BY BRAND SEGMENT")
print("="*80)
print(segment_df.to_string(index=False))

# Save segment results
segment_df.to_csv('segment_performance.csv', index=False)
print("\n Saved: segment_performance.csv")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Macro F1 by segment
pivot_f1_seg = segment_df.pivot(index='Segment', columns='Model', values='Macro_F1')
pivot_f1_seg.plot(kind='bar', ax=axes[0], width=0.8, color=['steelblue', 'coral', 'lightgreen'])
axes[0].set_title('Macro F1 Score by Brand Segment', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Macro F1', fontsize=12)
axes[0].set_xlabel('Brand Segment', fontsize=12)
axes[0].legend(title='Model')
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=0)

# Accuracy by segment
pivot_acc_seg = segment_df.pivot(index='Segment', columns='Model', values='Accuracy')
pivot_acc_seg.plot(kind='bar', ax=axes[1], width=0.8, color=['steelblue', 'coral', 'lightgreen'])
axes[1].set_title('Accuracy by Brand Segment', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_xlabel('Brand Segment', fontsize=12)
axes[1].legend(title='Model')
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('segment_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: segment_performance.png")
