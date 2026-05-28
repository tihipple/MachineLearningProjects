# MachineLearningProjects
# Gym Injury Risk Predictor

This machine learning project predicts injury risk based on:
- workout frequency
- sleep
- recovery time
- training intensity

## Tools Used
- Python
- Pandas
- Scikit-Learn
- Matplotlib

## Goal
Practice beginner machine learning workflows and data visualization.

## Phase 1
# Identifying Source Data
The source data for my project is going to be generated using Python. Since real injury data is
hard to collect and could involve some privacy issues, this will allow me to simulate realistic gym
user behaviors and injury risk factors. The dataset will include workout and recovery habits,
health indicators, injury history, and lifestyle behaviors. I’ll then store it in a CSV so it can be
reused for future steps.
# Features
As previously mentioned, these are the variables that I will use.
Feature Description
Age User age
Gender Male/Female
Workout Type Strength, Cardio, HIIT, etc
Workouts Per Week # of workouts weekly
Workout Duration Minutes per workout
Intensity Level Low, Medium, High
Sleep Hours Hours of sleep
Recovery Days # of rest days per week
Warm Up Completed Yes/No
Previous Injury Yes/No
Pain Level After Workout Scale from 0-10
Hydration Level Low, Medium, High
# Label (Prediction Target)
Label Description
Injury Risk Low, Medium, High
The main goal is to have the model predict the injury risk category.
# Features That Carry the Most Weight
High Importance Features Why
Previous Injury Past injuries typically increase future injury
risk
Pain Level After Workout High pain could indicate overtraining
Sleep Hours Poor sleep reduces recovery
Recovery Days Lack of recovery increases physical stress
Intensity Level Higher intensity increases physical stress
Warm-Up Completed Skipping warm-ups may increase injury risk
# Features That May Be Noise
Possible Noise Features Reason
Gender May not directly predict injury risk
Workout Type Some overlap between categories
Age Might matter less in a younger dataset
# My Script to Draw Data
My dataset is going to be generated using Python with the pandas, numpy, and random
libraries. The script I create is going to generate realistic workout data, randomly assign fitness
habits, apply rules for injury risk labels, and export the data into a CSV file.
# Storing Data in a Working Form
The data will be stored in a CSV format, but as the project expands, it could move into a fitness
tracking app and a SQLite database
# Hypothesis/Value of the Data
My hypothesis is that workout recovery habits and training intensity can help predict injury risk. I
believe that the model is going to show that there is a higher injury risk with poor recovery and
higher workout intensity, but a lower risk with proper sleep and hydration. I’m hoping that this
could be used in fitness apps for personal trainers, gyms, and injury prevention systems.

## Phase 2
# Identifying Machine Learning Candidates
For this phase of the project, I tested multiple machine learning models to determine which
model would best predict gym injury risk based on the workout, recovery, and health-related
features. The models I chose were Decision Tree Classifier, Random Forest Classifier,
K-Nearest Neighbors (KNN), and Logistic Regression. Each model was tested using the same
dataset, so I could compare their overall performance and accuracy.
# Splitting the Data
Before training the models, I split the dataset into training and testing groups using
train_test_split() from sklearn. I used 80% of the data for training and 20% for testing. This
allowed the models to learn from one portion of the data while being tested on new, unseen data
to evaluate performance. I also used stratification of the injury risk categories, which remained
balanced between the training and testing datasets.
# Cleaning and Encoding the Data
Before processing the data through the models, I cleaned the dataset by removing duplicate
rows, validating realistic values, and checking for invalid ranges. I then encoded categorical text
values into numbers so the machine learning models could process them.
For example, Yes/No values became 1 and 0. Low/Medium/High became 0, 1, and 2. I also
used one-hot encoding for columns like Gender and Workout Type.
# Should the Data Be Scaled?
Some models required scaling while others did not. I used StandardScaler() for KNN and
Logistic Regression since these models are more sensitive to differences in feature sizes.
Meaning that the larger number ranges could affect the predictions. I did not use scaling for
Decision Tree or Random Forest because tree-based models don’t rely on distance
calculations, so I didn’t find it necessary.
# Model Parameters and Testing
To test multiple parameter combinations, I used GridSearchCV() for each of the machine
learning models.
Decision Tree Parameters
● Max_depth - Can affect how many levels the tree can grow
● Min_samples_leaf - Can affect how many data points are required.
Random Forest Parameters
● N_estimators - Can affect the number of trees used
● Max_depth - Can affect the overall model complexity
● Min_samples_leaf - Can prevent overfitting
KNN Parameters
● N_neighbors - Can affect how many nearby data points are used.
● Weights - Can affect whether closer neighbors have more influence
Logistic Regression Parameters
● C - Can control the regularization strength and prevent overfitting.
# Scoring the Models
After testing each model, I compared the cross-validation scores and testing accuracy scores.
The results showed that Logistic Regression performed the best overall for my dataset with an
accuracy score of approximately 82%. The model comparison process helped determine which
machine learning algorithm was most effective for predicting injury risk.
# Best Model Selection
The best-performing model for my project is Logistic Regression. I chose this model because it
achieved the highest testing accuracy, it generalized well to new data, and it handled the
dataset effectively after scaling. Random Forest and Decision Tree also did well, but it seems
that Logistic Regression produced the most consistent results overall.
# Saving the Trained Model
After I picked the model I wanted to use, I saved the trained model as a .pkl file using joblib
because this allows the model to be reused later. I can then reuse it for new injury risk
predictions, future project phases, or integration into a fitness app.

## Phase 3
# Generating Plots and Graphs
For this phase of the project, I generated visualizations so I could better understand how each
machine learning model performed. The graphs helped me compare the accuracy scores
between the models and made it easier to determine which model was the best choice for
predicting injury risk.
The visualizations I included were:
● The model accuracy chart allowed me to visually compare the performance of Decision
Tree, Random Forest, K-Nearest Neighbors, and Logistic Regression.
● The confusion matrix helped show how accurately the best model predicted Low,
Medium, and High injury risk categories.
# Selecting the Best Model
After comparing all of the models again, Logistic Regression stayed as the best-performing
model for my dataset. It produced the highest accuracy score and handled the scaled data
consistently.
Even though Random Forest and Decision Tree also did well, Logistic Regression gave the
most reliable results when predicting injury risk categories.
# Creating a User Evaluation Tool
For this phase, I also created a simple evaluation tool that allows a user to enter new workout
and recovery information. The trained machine learning model then predicts whether the user
has a Low, Medium, or High injury risk.
The program asks the user to enter information such as:
● Age
● Workout Frequency
● Workout Duration
● Intensity Level
● Sleep hours
● Recovery Days
● Warm-Up Completion
● Previous Injuries
● Pain Level After Workouts
● Hydration Level
● Gender
● Workout Type
Once the information is entered, the trained model processes the data and predicts the injury
risk category.
# Example Prediction
One example user (aka me) entered:
● Age: 18
● Workout Frequency: 2 days per week
● Workout Duration: 60 minutes
● Intensity Level: High
● Sleep hours: 6
● Recovery Days: 4
● Warm-Up Completion: No
● Previous Injuries: Yes
● Pain Level After Workouts: 4/10
● Hydration Level: Medium
● Gender: Female
● Workout Type: Strength Training
The model predicted that this user had a High injury risk. This makes sense because the user
reported high workout intensity, a previous injury, no warm-up, and pain after the workout. Even
though the user had multiple recovery days, the model still recognized the overall risk as high
because several major injury-related risk factors were present.
# Saving and Reusing the Model
The trained model was saved as a .pkl file using joblib. This allows the model to be reused later
without needing to retrain it every time the program runs.
