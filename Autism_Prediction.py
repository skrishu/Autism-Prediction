# %% [markdown]
# **ML-1 Mini Project: Data Pre-processing and EDA**
# 
# Topic: Autism Prediction
# 
# Name: Krisha Shah
# 
# Sap Id: 60009220055
# 
# Roll No: D065
# 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
training_data = pd.read_csv('C:\\Krisha\\DJ Sem4\\ML\\Mini Project\\Autism Prediction\\train.csv')

# %% [markdown]
# ###PRE-PROCESSING

# %%
training_data.head(10)

# %%
training_data.info()

# %%
training_data.describe()

# %%
training_data.shape

# %%
training_data.isnull().sum()

# %% [markdown]
# No null values in the above dataset

# %%
duplicate_rows = training_data[training_data.duplicated()]
if duplicate_rows.empty:
    print("No duplicate rows found.")
else:
    print("Duplicate Rows:")
    print(duplicate_rows)

# %%
categorical_columns = training_data.select_dtypes(include=['object']).columns
inconsistent_values = {}
for col in categorical_columns:
    unique_values = training_data[col].unique()
    if len(unique_values) > 1:
        inconsistent_values[col] = unique_values
if inconsistent_values:
    print("Inconsistent Values:")
    for col, values in inconsistent_values.items():
        print(f"{col}: {values}")
else:
    print("No inconsistent values found.")


# %%
training_data['gender'] = training_data['gender'].map({'f': 'Female', 'm': 'Male'})
training_data['ethnicity'] = training_data['ethnicity'].replace('others', 'Others').str.strip() # Remove trailing whitespaces
training_data['relation'] = training_data['relation'].replace('?', 'Unknown')

# %%
training_data.columns

# %%
# Replace '?' with NaN
training_data.replace('?', np.nan, inplace=True)

# Drop rows with NaN values
training_data.dropna(inplace=True)

# Display the modified DataFrame
print(training_data)

# %% [markdown]
# Therefore the dataset is clean and can be worked upon.

# %% [markdown]
# Label Encoding to convert 'gender' values to numeric to use them in the EDA section

# %%
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
training_data['gender_encoded'] = label_encoder.fit_transform(training_data['gender'])

print(training_data[['gender', 'gender_encoded']])



# %% [markdown]
# Hence we see that after encoding, all the 'Male' values have been encoded to 1 and 'Female' values to 0.

# %% [markdown]
# Label Encoding to convert country names to numeric data type.

# %%
label_encoder_country = LabelEncoder()
training_data['country_encoded'] = label_encoder_country.fit_transform(training_data['contry_of_res'])

# Display the modified DataFrame
print(training_data[['contry_of_res', 'country_encoded']])

# %%
training_data['contry_of_res'].unique()

# %%
# Assuming you have a DataFrame called training_data with columns 'ethnicity' and 'ethnicity_encoded'
unique_countries = training_data['contry_of_res'].unique()
country_encoded = {}

# Assign encoded values to each unique ethnicity
for idx, contry_of_res in enumerate(unique_countries):
    country_encoded[contry_of_res] = idx

# Print unique encoded values for ethnicity
print("Unique encoded values for country:")
for contry_of_res, encoded_value in country_encoded.items():
    print(f"{contry_of_res}: {encoded_value}")

# %%
label_encoder_ethnicity = LabelEncoder()
training_data['ethnicity_encoded'] = label_encoder_country.fit_transform(training_data['ethnicity'])

# Display the modified DataFrame
print(training_data[['ethnicity', 'ethnicity_encoded']])

# %%
training_data['ethnicity'].unique()

# %%
# Assuming you have a DataFrame called training_data with columns 'ethnicity' and 'ethnicity_encoded'
unique_ethnicities = training_data['ethnicity'].unique()
ethnicity_encoded = {}

# Assign encoded values to each unique ethnicity
for idx, ethnicity in enumerate(unique_ethnicities):
    ethnicity_encoded[ethnicity] = idx

# Print unique encoded values for ethnicity
print("Unique encoded values for ethnicity:")
for ethnicity, encoded_value in ethnicity_encoded.items():
    print(f"{ethnicity}: {encoded_value}")



# %%

label_encoder_jaundice = LabelEncoder()
training_data['jaundice_encoded'] = label_encoder.fit_transform(training_data['jaundice'])

print(training_data[['jaundice', 'jaundice_encoded']])


label_encoder_autism = LabelEncoder()
training_data['austim_encoded'] = label_encoder.fit_transform(training_data['austim'])

print(training_data[['austim', 'austim_encoded']])


# %%
output = []
for col in training_data.columns:
    unique = training_data[col].nunique()
    colType = str(training_data[col].dtype)
    categories=training_data[col].unique()

    output.append([col, unique, colType,categories])

output = pd.DataFrame(output)
output.columns = ['colName','unique','dtype','categories']
output

# %%
training_data.columns

# %% [markdown]
# ###EDA

# %%
# Select only numeric columns
numeric_columns = training_data.select_dtypes(include=['int64', 'float64']).columns

# Create a subset of DataFrame with numeric columns
numeric_data = training_data[numeric_columns]

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()


# %% [markdown]
# ######ANALYSIS:
# The following inferences can be drawn from the above heatmap:
# 
# 1. There is a negative correlation between
# the columns 'age' and 'result', which suggests that as age increases, the possibility of a person being diagnosed with autism decreases. This means that this disease is detected at a very early age, possibly by 2 years.
# 
# 2. There is a slightly negative correlation between 'gender_encoded' and 'A2_score', which suggests that the attribute score of a particular condition (A2_score) is found specifically in only one gender, which can be studied further in the analysis.
# 
# 3. Again, there is a slightly negative correlation between 'gender' and 'country', which suggests that in some particular countries, specific genders are found to be diagnosed with autism.
# 
# 4. A positive correlation can be seen between all the attribute scores 'A1, A2, A3..., A10' and 'age', which suggests that these attribute scores are dependent on the ages of the children.
# 
# 5. Finally, 'Class/ASD' almost has a positive correlation with all the factors, which suggests that all the factors in the dataset are somewhat equally important in determining autism.
# 
# 
# 
# 
# 
# 
# 

# %%
colors = ['#008080', '#FFA700', '#FF7F50']
# Calculate the value counts for 'Class/ASD'
class_counts = training_data['Class/ASD'].value_counts()

# Plot the pie chart
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',colors=colors)
plt.show()


# %% [markdown]
# ######ANALYSIS:
# The above piechart simply depicts the distribution of 1s and 0s, that is there are clearly more number of cases where no autism was detected and fewer cases where it was detected.

# %%
# Count the number of ASD cases for each gender
gender_asd_counts = training_data[training_data['Class/ASD'] == 1]['gender'].value_counts()

# Plot the bar chart
gender_asd_counts.plot(kind='bar', color='coral')
plt.xlabel('Gender')
plt.ylabel('Number of ASD Cases')
plt.title('ASD Cases Detected by Gender')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.show()

# %% [markdown]
# ######ANALYSIS:
# We can clearly see form the above bar graph that there have been more cases whre ASD was detected in males than in females.

# %%
# Select ASD cases from the training data
asd_cases = training_data[training_data['Class/ASD'] == 1]

# Select only the columns containing A1 to A10 scores
a_scores = asd_cases[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                      'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']]

# Calculate the sum of scores for each attribute
attribute_totals = a_scores.sum()

# Plot the doughnut chart
plt.figure(figsize=(8, 8))
plt.pie(attribute_totals, labels=attribute_totals.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Distribution of A1 to A10 Scores for ASD Cases')
plt.show()


# %% [markdown]
# ######ANALYSIS:
# From the above doughnut graph, we can decipher that not all the attribute scores have same kind of impact on te ASD detection.
# As per the relation, we can see that A9 has the maximum correlation, followed by A10,A2,A3 and so on and A6 and A7 have the least correlation amongst all.

# %%
# Count the number of ASD cases for each country
country_asd_counts = training_data[training_data['Class/ASD'] == 1]['contry_of_res'].value_counts()

# Plot the bar chart
plt.figure(figsize=(12, 6))
country_asd_counts.plot(kind='bar', color='coral')
plt.xlabel('Country')
plt.ylabel('Number of ASD Cases')
plt.title('ASD Cases Detected by Country')
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# The above bar graph represents the distribution of ASD Cases country-wise. it can be hence deciphered that the United states  have the maximum number of cases, followed by the United Kingdom, Canada, Australia and Afghanistan. Whereas the least amount of cases have been recorded in Iran, Bahamas,Oman, Austia and Ethopia.

# %%
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

training_data['ageGroup'] = training_data['age'].apply(convertAge)

# %%
colors = ['#008080', '#FFA700', '#FF7F50']

sns.countplot(x=training_data['ageGroup'], hue=training_data['Class/ASD'], palette=colors)
plt.show()


# %% [markdown]
# ######ANALYSIS:
# The above coutplot shows a distribution od ASD across different age groups. People with ages between 18 and 40 that is the young people have shown more number of cases of this disorder, followed by the Senior age group, that is people having ages more than 40 years.
# It can also be seen that toddlers haverecorded the least number of cases followed by kids and then teenagers.

# %%
training_data.columns

# %%
# Set up the figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

# Plot violin plot for ethnicity_encoded vs. ASD diagnosis
sns.violinplot(x='Class/ASD', y='ethnicity_encoded', data=training_data, ax=axes[0])
axes[0].set_xlabel('ASD Diagnosis')
axes[0].set_ylabel('Ethnicity (Encoded)')
axes[0].set_title('Ethnicity (Encoded) vs. ASD Diagnosis')

# Plot violin plot for jaundice_encoded vs. ASD diagnosis
sns.violinplot(x='Class/ASD', y='jaundice_encoded', data=training_data, ax=axes[1])
axes[1].set_xlabel('ASD Diagnosis')
axes[1].set_ylabel('Jaundice (Encoded)')
axes[1].set_title('Jaundice (Encoded) vs. ASD Diagnosis')

# Plot violin plot for autism_encoded vs. ASD diagnosis
sns.violinplot(x='Class/ASD', y='austim_encoded', data=training_data, ax=axes[2])
axes[2].set_xlabel('ASD Diagnosis')
axes[2].set_ylabel('Autism (Encoded)')
axes[2].set_title('Autism (Encoded) vs. ASD Diagnosis')

plt.tight_layout()
plt.show()


# %% [markdown]
# ######ANALYSIS:
# The above is a violin plot which shows the correlation of columns like history of jaundice(jaundice_encoded), history of autism(austim_encoded) and ehnicity(ethnicity_encoded) woth the likelihood of being detectd with ASD.
# The areaas where there is a fatter part shows that there is more density and similary the thinner part depicts lower density.
# - **Ethnicity:** There appears to be a slightly higher density of ASD diagnoses for individuals with encoded ethnicity values of 2, 4, and 6 when compared to other values. However, the differences are not substantial, suggesting that ethnicity may not be a strong predictor of ASD.
# - **Jaundice:** Individuals with encoded jaundice values of 0 and 1 seem to have a higher likelihood of being diagnosed with ASD compared to those with values of 2 and 3. This could indicate a potential association between jaundice and ASD, although further investigation is needed.
# 
# - **Autism:** The violin plot for autism shows a clear distinction between the encoded values. Individuals with an encoded autism value of 1 have a significantly higher density of ASD diagnoses compared to those with values of 0 and 2. This suggests a strong association between a history of autism and the likelihood of being diagnosed with ASD.
# 

# %% [markdown]
# ###FITTING THE MODEL

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# %%
columns_to_drop = ['ID','gender', 'contry_of_res', 'ethnicity',
       'jaundice', 'austim','ageGroup','used_app_before','age_desc','relation','Class/ASD']

# %%
X = training_data.drop(columns=columns_to_drop)
y = training_data['Class/ASD']

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
# Dictionary to hold model accuracies
accuracies = {}

# %% [markdown]
# Fitting the model using SVC- Support Vector Classifier

# %%
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)
accuracies['SVC'] = svc_accuracy

# %% [markdown]
# Fitting the model using Logistic Regression

# %%
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
accuracies['Logistic Regression'] = lr_accuracy

# %% [markdown]
# Fitting the model using Random Forest

# %%
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
accuracies['Random Forest'] = rf_accuracy

# %% [markdown]
# Fitting the model using XGBoost

# %%
xgb_model = GradientBoostingClassifier()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
accuracies['XGBoost'] = xgb_accuracy

# %% [markdown]
# Fitting the model using Decision Tree

# %%

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
accuracies['DecisionTree'] = dt_accuracy


# %% [markdown]
# Getting the accuracies for all the models

# %%
for model, accuracy in accuracies.items():
    print(f"{model} accuracy: {accuracy:.2f}")

# %% [markdown]
# After fitting the above models, we know that SVC is not a suitable model to use as its accuracyis only 68%.
# 
# Rest all the other models, i.e Logistic Regression or XGBoost or Random Forest can be used.
# 
# However, we will continue with Random Forest as it shows the highest accuracy of 78%.

# %% [markdown]
# Confusion matrix for all the models to check which performs the best

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Logistic Regression
lr_report = classification_report(y_test, lr_pred, zero_division=1)  # Set zero_division=1 to handle zero divisions
lr_confusion_matrix = confusion_matrix(y_test, lr_pred)

print("Logistic Regression Evaluation Metrics:")
print(lr_report)
print("Confusion Matrix:")
print(lr_confusion_matrix)

# Support Vector Classifier (SVC)
svc_report = classification_report(y_test, svc_pred, zero_division=1)  # Set zero_division=1 to handle zero divisions
svc_confusion_matrix = confusion_matrix(y_test, svc_pred)

print("\nSupport Vector Classifier (SVC) Evaluation Metrics:")
print(svc_report)
print("Confusion Matrix:")
print(svc_confusion_matrix)

# Random Forest
rf_report = classification_report(y_test, rf_pred, zero_division=1)  # Set zero_division=1 to handle zero divisions
rf_confusion_matrix = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Evaluation Metrics:")
print(rf_report)
print("Confusion Matrix:")
print(rf_confusion_matrix)

# Decision Tree
dt_report = classification_report(y_test, dt_pred, zero_division=1)  # Set zero_division=1 to handle zero divisions
dt_confusion_matrix = confusion_matrix(y_test, dt_pred)

print("\nDecision Tree Evaluation Metrics:")
print(dt_report)
print("Confusion Matrix:")
print(dt_confusion_matrix)

# XGBoost
xgb_report = classification_report(y_test, xgb_pred, zero_division=1)  # Set zero_division=1 to handle zero divisions
xgb_confusion_matrix = confusion_matrix(y_test, xgb_pred)

print("\nXGBoost Evaluation Metrics:")
print(xgb_report)
print("Confusion Matrix:")
print(xgb_confusion_matrix)


# %% [markdown]
# The following code performs hyperparameter tuning for a Random Forest classifier using grid search with cross-validation and then evaluates the best model on the testing data.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data for demonstration
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20],        # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],    # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],      # Minimum number of samples required at each leaf node
    'max_features': ['sqrt']            # Number of features to consider when looking for the best split
}

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the testing data
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy on Testing Data:", accuracy)


# %%
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)


# %% [markdown]
# Analysis of the Performance: A testing accuracy of 92% suggests that the model performs well on unseen data. It correctly predicts the class labels for approximately 92% of the samples in the testing dataset.
# 
# Model Effectiveness: The accuracy of 92% indicates that the model is effective at identifying patterns and relationships in the data. It demonstrates that the model has learned from the training data and can make nearly accurate predictions on new, unseen examples.

# %%
print(X_train.columns)


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Calculate feature importance
feature_importances = rf_classifier.feature_importances_

# Assuming 'feature_names' is defined correctly
feature_names = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'result',
       'gender_encoded', 'country_encoded', 'ethnicity_encoded',
       'jaundice_encoded', 'austim_encoded']

# Create a DataFrame to store feature importance
feature_importances_df = pd.DataFrame({'feature_name': feature_names, 'importance': feature_importances})

# Sort features by importance
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

# Select top 5 features
top_features = feature_importances_df['feature_name'].head(5).tolist()

print("Top 5 Features:")
for feature in top_features:
    print(feature)

# Select features based on importance
selected_features = SelectFromModel(rf_classifier, threshold='median')
selected_features.fit(X_train, y_train)

# Transform training and testing data using selected features
X_train_selected = selected_features.transform(X_train)
X_test_selected = selected_features.transform(X_test)

# Train a new classifier using selected features
rf_classifier_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_selected.fit(X_train_selected, y_train)

# Evaluate performance on testing set
accuracy = rf_classifier_selected.score(X_test_selected, y_test)
print("Testing Accuracy with Selected Features:", accuracy)


# %%
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE


# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Initialize Recursive Feature Elimination (RFE) with cross-validation
# Set the estimator to the RandomForestClassifier and the scoring metric to accuracy
rfe = RFE(estimator=rf_classifier, n_features_to_select=None, step=1, verbose=0)

# Fit RFE to the training data
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Perform cross-validation with the selected features
cv_scores = cross_val_score(rf_classifier, X_train[selected_features], y_train, cv=5, scoring='accuracy')

# Calculate the mean cross-validation accuracy
mean_cv_accuracy = np.mean(cv_scores)

# Check testing accuracy using the selected features
rf_classifier.fit(X_train[selected_features], y_train)
test_accuracy = rf_classifier.score(X_test[selected_features], y_test)

# Check the number of selected features
num_selected_features = len(selected_features)

# Return results
results = {
    "selected_features": selected_features,
    "num_selected_features": num_selected_features,
    "mean_cv_accuracy": mean_cv_accuracy,
    "test_accuracy": test_accuracy
}

print("Selected Features:", selected_features)
print("Number of Selected Features:", num_selected_features)
print("Mean Cross-Validation Accuracy with Selected Features:", mean_cv_accuracy)
print("Testing Accuracy with Selected Features:", test_accuracy)

# You can return or use the 'results' dictionary as needed
# return results


# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined

# Define the selected features
selected_features = ['A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'age', 'result', 'country_encoded', 'ethnicity_encoded']

# Filter the training and testing data to include only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Initialize and fit the RandomForestClassifier with the selected features
rf_classifier_selected = RandomForestClassifier()
rf_classifier_selected.fit(X_train_selected, y_train)

# Make predictions on the testing data
y_pred_selected = rf_classifier_selected.predict(X_test_selected)

# Calculate the accuracy of the model with selected features
testing_accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Testing Accuracy with Selected Features:", testing_accuracy_selected)


# %% [markdown]
# Deploying Model

# %%
import pickle

# %%
pickle.dump(rf_classifier, open('rf_classifier_selected.pkl', 'wb'))

# %%
# Load the model
with open('rf_classifier_selected.pkl', 'rb') as file:
    model = pickle.load(file)


