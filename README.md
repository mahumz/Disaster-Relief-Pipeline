# Disaster Response Pipeline Project

### Introduction:
The focus of this project was to filter disaster messages sent during a natural disaster to emergency personel. Once messages are filtered they can
then be tranferred to the appropriate department. This way all emergencies can be handled in time and accurately. The app is built to contain a 
Machine Learning model to categorize messages accurately.

### Files Breakdown:
1. data: process_data.py: This python excutuble code takes as its input csv files and creates a SQL database
2. models: train_classifier.py: This code trains the ML model with the SQL data base
3. data: This folder contains sample messages and categories datasets in csv format.
4. app: cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves run the following command in the models directory.
        `python train_classifier.py ../data/DisasterResponse.db test_classifier.pkl vocab.pkl category_pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
