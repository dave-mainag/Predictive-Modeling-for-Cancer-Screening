# PREDICTIVE MODELLING FOR CANCER SCREENING
![Fight against Cancer](Cancer.jpeg)

# Flask Web App
The Flask WebApp delivers an intuitive interface for interacting with our Capstone project's machine learning model. Users can input data through a web form and receive instant predictions, making it easy to visualize and act on the model’s insights.
Explore the app, deployed on [Render](https://predictive-modeling-for-cancer-screening.onrender.com), to see the predictive model in action.
Make sure to fill in all fields and provide accurate data to obtain meaningful predictions.


# OVERVIEW

With cancer being the third leading cause of death in the country and late diagnosis contributing to high mortality rates, there is an urgent need to enhance screening programs and increase participation rates. This project aims at improving cancer screening efforts through data-driven strategies. By identifying high-risk populations and implementing targeted interventions, we anticipate increased screening participation rates, earlier detection of cancers, and improved treatment outcomes. Through these concerted efforts, we envision a future where cancer screening is widely accessible, culturally sensitive, and seamlessly integrated into routine healthcare services. Ultimately, this initiative has the potential to reduce cancer-related morbidity and mortality, alleviate the burden on healthcare systems and save lives in Kenya. 

## Business Problem

The project focuses on improving cancer screening efforts in Kenya by addressing barriers to early detection and management through targeted interventions and data-driven strategies. Cancer screening is critical for identifying cancer at its early stages when treatment is most effective. Despite advancements in medical technology and increased awareness of cancer risk factors, many individuals still receive diagnoses at advanced stages, when treatment options are limited, and prognosis is poor. This delay in diagnosis not only exacerbates the physical and emotional burden on patients and their families but also significantly impacts healthcare systems, leading to higher treatment costs and reduced effectiveness. In Kenya, several challenges hinder widespread access to and utilization of screening services, including limited awareness, inadequate healthcare infrastructure, and financial constraints.

## Business Objectives

+ Early Detection: Detect cancer at its earliest stages, using behavioral health factors, to improve treatment outcomes and reduce mortality rates.

+ Reduce cancer-related morbidity and mortality using predictive modelling

+ Effectiveness: Improve effectiveness of cancer screening through innovative methods and strategic partnerships.

+ Public Health Impact: Contribute to reducing the burden of cancer on public health by implementing evidence-based screening programs.

## Data Understanding

The project will involve collecting the Behavioral Risk Factor Surveillance System (BRFSS) data for the year 2020, which is compiled and maintained by the Centers for Disease Control and Prevention (CDC) in the United States. BRFSS is a nationwide survey that collects data on health-related behaviors, chronic health conditions, and use of preventive services among adults in the United States.
The raw data will be obtained directly from the CDC, which regularly releases BRFSS datasets to the public for research and analysis purposes.

This dataset has a total of 401,959 records with 50 features(variables). The features used in the analysis include but not limited to age, gender, general health, physical health, mental health, smoking status, alcohol consumption, physical activity, weight, height and history of cancer screening.

## Data Preparation

The BRFSS data for 2020 is stored in tabular format, as a csv file. Variables include demographic info, health behaviors, chronic conditions among other features, with a mix of categorical, numerical, and ordinal types.

## Data Preprocessing and Visualizations

Preprocessing steps involved handling missing values, feature engineering using columns with high missing data, dropping columns with high missing values, dropping the rows with misssing values in our target variable, engineering the target variable (y) from multiple columns, splitting data into predictor variables (X) and target variables (y), encoding categorical variables, scaling numerical features, and removing redundant variables.

Visualization techniques include barplots and histograms to understand data distributions.

# Modelling

The project employs various machine learning algorithms to model behavioral health factors for early cancer screening. Models exlpored include Logistic Regression, Random Forest, Linear Support Vector Machines (SVM), Naive Bayes, Neural Network and XGBoost.

## Model Performance

|No|Model|Test Accuracy
|-|-|-|
|1|Logistic Regression (Base) |	0.6956|
|2|Logistic Regression (Tuned) |0.6857|
|3|Random Forest (Base) |0.8329|
|4|Random Forest (Tuned) |0.8283|
|5|Linear Support Vector Machines (SVM) (Base) |0.6819|
|6|Linear Support Vector Machines (SVM) (Tuned) |0.6821|
|7|Naive Bayes |0.5921|
|8|Neural Network (Base) |0.1641|
|9|XGBoost (Base) |0.8364|


# Recommendations

XGBoost model was the best performing model 
XGBOost model woild be pickled and used at the back end of a flask app to predict cancer risk for new users using
their behaviroal health factors

# Conclusions

The project demonstrates the feasibility of leveraging behavioral health factors to construct predictive models, enhancing screening initiatives for improved treatment outcomes and lower mortality rates. 
Additionally, it showcases the potential for establishing strategic partnerships with healthcare providers, government agencies, and non-governmental organizations (NGOs) to effectively implement and scale machine learning-driven cancer screening programs across Kenya.

# Contributors
1. Branton Kieti
2. David Githaiga
3. Baker Otieno
4. Faith Gitau
5. David Kirianja
6. Linet Wangui
