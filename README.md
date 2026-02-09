# Cosmos Modeling
This repository contains Structured Query Language (SQL) scripts used to extract data marts from the _Expertly Determined De-Idnetified_ (EDDI) database hosted on the Cosmos server, as well as Python scripts for cleaning, preprocessing and modeling these data marts on the assigned project server. 

The entire workflow was conducted within the Cosmos _Data Science Virtual Machine_ (DSVM) environment, which prohibits the export of line-level data to ensure compliance with privacy and security standard. _VSCodium_, a freely licensed code editor developed by Mircosoft, serves as the primary Integrated Development Environment (IDE) for Python within the DSVM. Accordingly, most Python scripts in this repository are provided in Jupyter Notebook (.ipynb) format to facilitate transparency and reproducibility. Python 3.12.10 was used in the Cosmos DSVM environment (with pre-installed packages).   

# Data Extraction 
Scripts can be found in `01_SQL_Scripts/`. 

|Script name|Description|
|---|---|
|`01_CreateTables.sql`|Create tables that will hold extracted data.|
|`02_MainQueries.sql`|Main block of code to extract data.|
|`03_PageCompression.sql`|Reduce size of tables and meet the project storage limits imposed by Epic.|
|`04_AddSAVariables.sql`|Extract the additional binary variable suicide attempt occurred 30 days prior to the encounter.|

# Data Cleaning
Scripts can be found in `02_Python_Scripts/01_Cleaning_Scripts/`. (To be uploaded)

|Script name|Description|
|---|---|

# Data Preprocessing
Scripts can be found in `02_Python_Scripts/02_Preprocessing_Scripts/`.

|Script name|Description|
|---|---|
|`P01_Subject_Inclusion.ipynb`|Implement the subject inclusion criteria.|
|`P02_Stratified_Partitioning.ipynb`|Partition the dataset into training and held-out test sets.|
|`P03a_Feature_Extraction_Patient.ipynb`|Extract the cleaned patient-level features.|
|`P03b_Feature_Extraction_Encounter.ipynb`|Extract the cleaned encounter-level features.|
|`P04_Winsorizing_Scaling.ipynb`|Clip and scale the feature values to facilitate subsequent model training.|
|`P05_Imputation.ipynb`|Implement different imputation methods.|
|`P06_Point_Data_Preparation.ipynb`|Prepare organized dataset for subsequent point-prediction modeling.|
|`P07_Longitudinal_Data_Preparation.ipynb`|Prepare organized dataset for subsequent longitudinal-prediction modeling.|
|`Statistics_01_Demographics.ipynb`|Extract demographic characteristics of patients.|
|`Statistics_02_Mental_Health_Visits.ipynb`|Extract patients' information regarding their mental health visits.|

# Data Modeling
Scripts can be found in `02_Python_Scripts/03_Modeling_Scripts/`. 

|Script name|Description|
|---|---|
|`M01_Point_ANN.ipynb`|Run the artificial neural network (ANN) point-prediction modeling.|
|`M02_Point_ElasticNet.ipynb`|Run the elastic net point-prediction modeling.|
|`M03_Point_LogReg.ipynb`|Run the logistic regression point-prediction modeling.|
|`M04_Point_SVM.ipynb`|Run the support vector machine (SVM) point-prediction modeling.|
|`M05_Point_XGBoost.ipynb`|Run the extreme gradient boosting (XGBoost) point-prediction modeling.|
|`M06_Longitudinal_RiskPath.ipynb`|Run the RiskPath longitudinal-prediction modeling.|
|`O01_Point_Evaluation.ipynb`|Gather performance statistics for point-prediction models.|
|`O02_Point_Evaluation_Imputation_Sensitivity.ipynb`|Gather performance statistics for point-prediction models regarding the sensitivity analysis for different imputation methods.|
|`O03_Point_Decision_Curve_Analysis.ipynb`|Visualize the decision curve analysis for point-prediction models.|
|`O04_Longitudinal_Evaluation.ipynb`|Gather performance statistics for longitudinal-prediction models.|
|`O05_Longitudinal_Evaluation_Imputation_Sensitivity.ipynb`|Gather performance statistics for longitudinal-prediction models regarding the sensitivity analysis for different imputation methods.|
|`O06_Longitudinal_Decision_Curve_Analysis.ipynb`|Visualize the decision curve analysis for longitudinal-prediction models.|

# Data Simulation
Scripts can be found in `02_Python_Scripts/04_Simulation_Scripts/`. (To be uploaded)

|Script name|Description|
|---|---|

# Data Dictionaries
The data dictionaries can be found in `02_Python_Scripts/05_Data_Dictionaries/`.

|File name|Description|
|---|---|
|`Data_Dictionary.xlsx`|The data dictionary for the raw features extracted from Cosmos.|
|`Variable_Type_Categorization.xlsx`|A helper spreadsheet to catgorize the variable type of the cleaned features. (To be uploaded)|
<hr>
<div align="right">
  Last update: 2025 Feburary 6, 17:04 MT (by Wayne Lam)
</div>
<hr>
