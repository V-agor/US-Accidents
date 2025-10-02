🚦 City Traffic and Accident Analysis
📌 Project Overview

This project analyzes traffic accident data from the US Accidents (March 2023) dataset.
The objectives are:

Data Exploration – Understand the dataset, handle missing values, and identify key patterns.

Machine Learning Application – Apply a predictive model to explore accident severity and contributing factors.

⚠️ Note: This dataset does not contain information for New York.

🛠️ Step 1: Data Preparation & Cleaning

Before analysis, the dataset must be cleaned and explored.

Import libraries and load data
import pandas as pd

# Load dataset with low_memory disabled
data = pd.read_csv("US_Accidents_March23.csv", low_memory=False)
data.head()

Dataset Overview

Shape: 280,557 rows × 46 columns

Columns include: Accident ID, Source, Severity, Start/End time, Location (lat/lng), Distance, Weather conditions, Traffic features, etc.

print(len(data))         # Number of records
print(data.columns)      # Available columns
data.info()              # Summary of columns
data.describe()          # Statistics for numeric columns

Missing Data Analysis

We analyze missing values and visualize them:

missing_percentages = (data.isna().sum().sort_values(ascending=False)) / len(data)
missing_percentages[missing_percentages != 0].plot(kind='barh')


🔎 Key insights:

End_Lat and End_Lng are completely missing.

Weather-related columns (e.g., Precipitation, Wind_Chill) have high missing percentages.

Location & time data are mostly complete.

📊 Step 2: Exploratory Data Analysis (EDA)

We focus on cities, start time, and weather conditions to understand accident patterns.

1️⃣ City-wise Analysis
cities_by_accident = data.City.value_counts()
cities_by_accident[:20].plot(kind='barh')


Los Angeles, Houston, and Atlanta record the highest number of accidents.

Most cities have < 1000 accidents.

We classify into buckets:

high_accident_cities = cities_by_accident[cities_by_accident >= 1000]
low_accident_cities  = cities_by_accident[cities_by_accident < 1000]


✅ Only ~31 cities fall into the high-accident group.

2️⃣ Start Time Analysis

Convert accident time into a datetime format:

data.Start_Time = pd.to_datetime(data.Start_Time)


Hourly trend:

import seaborn as sns
sns.histplot(data.Start_Time.dt.hour, bins=24)


🚗 Accidents peak during commute hours (6–10 AM and 3–6 PM).

Day of the week trend:

sns.histplot(data.Start_Time.dt.dayofweek, bins=7)


📉 Fewer accidents occur during weekends compared to weekdays.

3️⃣ Weather Conditions
data['Weather_Condition'].value_counts().head(10)


Top contributors include: Fair, Mostly Cloudy, Light Rain, Clear, Fog.

Weather impacts visibility and accident severity.

🤖 Step 3: Machine Learning Model (Planned)

We will apply a classification model (e.g., Logistic Regression, Random Forest) to predict Severity of accidents based on:

Location features: Start_Lat, Start_Lng, City

Environmental features: Temperature, Visibility, Weather Condition

Time features: Hour, Day of Week

📂 Project Structure
📁 CityTrafficAccidents
 ┣ 📜 README.md            ← Project documentation
 ┣ 📜 US_Accidents_March23.csv  ← Dataset
 ┣ 📜 analysis.ipynb       ← Data cleaning & exploration
 ┣ 📜 model.ipynb          ← ML model (planned)
 ┗ 📜 visuals/             ← Plots & charts

📌 Key Insights So Far

Accident data is heavily skewed toward a few major cities.

Morning and evening commute hours see the most accidents.

Weather conditions play a significant role in accident severity.

Next step → Build and evaluate a predictive model. 🚀
