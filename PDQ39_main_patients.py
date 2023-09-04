#!/usr/bin/env python
# coding: utf-8

# # PDQ-39 scores analysis
# 
# ## FIRST SECTION: PRE-DBS 
# 
#  Analysing patients'/ caregivers' PDQ-39 scores before DBS
#  
#  In this section I am interested in performing the scoring of PDQ-39 for each patient before DBS
#  and then plot total scores and single-scale scores
#  
#  
#  p.s. : I don't have enough data to perform statistics as Wilcoxon or t-test 
#  so I didn't add that lines of code.
#  

# In[ ]:


### Log csv files with patients' PDQ-39 scores PRE-DBS   ###

import pandas as pd
import logging

logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specify the path to your CSV file
csv_file_path = r'C:\Users\debora.spagnolo\Desktop\Deb_PhD\PD_DBS\patients_PRE_PDQ39_rawscores.csv'

# Log the CSV file

try:
    patients_pre_df = pd.read_csv(r'C:\Users\debora.spagnolo\Desktop\Deb_PhD\PD_DBS\patients_PRE_PDQ39_rawscores.csv')
    logging.info(f'CSV file was successfully logged.')
except FileNotFoundError:
    logging.error(f'CSV file not found.')
except Exception as e:
    logging.error(f'An error occurred while logging CSV: {str(e)}')

# Print CSV file    
patients_pre_df


# In[ ]:


# Create an empty dictionary to store patient scores in PDQ-39 specific scales
patient_pre_scores_by_scale = {}

# Define the item ranges for each scale
pdq_scales = {
    'Mobility': range(0, 10),             # Items 0 to 9
    'Activity_of_daily_living': range(10, 16),   # Items 10 to 15
    'Emotional_wellbeing': range(16, 22),        # Items 16 to 21
    'Stigma': range(22, 26),                    # Items 22 to 25
    'Social_support': range(26, 29),            # Items 26 to 28
    'Cognition': range(29, 33),                 # Items 29 to 32
    'Communication': range(33, 36),             # Items 33 to 35
    'Bodily_discomfort': range(36, 39)          # Items 36 to 38
}

# Loop through each patient's scores and organize them into scales
for patient_name in patients_pre_df.columns:
    patient_scores = patients_pre_df.loc[0:38, patient_name].tolist()  # PDQ-39 has 39 items
    patient_scale_scores = {}
    
    for scale, item_range in pdq_scales.items():
        pdq_scales_scores = [patient_scores[item] for item in item_range]
        patient_scale_scores[scale] = pdq_scales_scores

    patient_pre_scores_by_scale[patient_name] = patient_scale_scores

# Print the patient scores organized by scales
patient_pre_scores_by_scale_df = pd.DataFrame(patient_pre_scores_by_scale)
patient_pre_scores_by_scale_df = patient_pre_scores_by_scale_df.T
patient_pre_scores_by_scale_df


# In[ ]:


###### SCORE FOR SINGLE SCALE (DOMAIN) ######


# To calculate the domain-specific score we need to sum the raw scores from each domain (scale)
# then we divide this number by the maximum score for each question (4)
# then we divide this number by the total number of questions (items) for each domain (scale)
# then we multiply this last result for 100


def calculate_domain_score(questions):
    # Sum the scores for the questions within the domain
    domain_score = sum(questions)
    # And divide for the maximal score for each question (4)
    domain_score/= 4


    # Divide by the number of questions in the domain
    domain_score = (domain_score/len(questions)) * 100
    if isinstance(domain_score, float):
        domain_score = round(domain_score)
        return domain_score
    else:
        return domain_score


# In[ ]:


# Apply the calculate_domain_score function to each row (patient) in the DataFrame
patients_pre_dbs_scores = patient_pre_scores_by_scale_df.apply(lambda row: row.apply(calculate_domain_score), axis=1)

# Calculate the total score for each patient
patients_pre_dbs_scores['Total_score'] = patients_pre_dbs_scores.sum(axis=1) / 8

# Round the total scores
patients_pre_dbs_scores['Total_score'] = patients_pre_dbs_scores['Total_score'].round()

# Print the DataFrame with the total scores for each patient
patients_pre_dbs_scores = patients_pre_dbs_scores.drop("items", axis =0)
new_col = ['patient_1', 'patient_2', 'patient_3', 'patient_5', 'patient_6', 'patient_7', 'patient_8', 'patient_10', 
           'patient_11', 'patient_13' ]
patients_pre_dbs_scores.insert(loc=0, column='Patient_ID', value=new_col)

# Reset df index and print the resulting dataframe
patients_pre_dbs_scores.reset_index(drop=True, inplace=True)
patients_pre_dbs_scores


# In[ ]:


### PLOT PATIENTS' TOTAL SCORES PRE-DBS ###

import matplotlib.pyplot as plt

# Extract the patient IDs and total scores
patient_ids = patients_pre_dbs_scores["Patient_ID"]
total_scores_pre = patients_pre_dbs_scores["Total_score"]

# Create a list of patient labels (ID) for the x-axis
x_labels = [f'Patient {patient_id.split("_")[1]}' for patient_id in patient_ids]

# Plot the total scores
plt.figure(figsize=(12, 6))
plt.bar(range(len(patient_ids)), total_scores_pre, color="cadetblue")
plt.xlabel('Patients')
plt.ylabel('Score range')
plt.ylim(0, 100)
plt.title('Total Scores PRE-DBS for Each Patient')
plt.xticks(range(len(patient_ids)), x_labels, rotation=45)

# Add annotations to display scores on top of each bar
for i, score in enumerate(total_scores_pre):
    plt.annotate(str(score), xy=(i, score), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[ ]:


## SCALE-SPECIFIC PLOTS PRE-DBS ##

import matplotlib.pyplot as plt

# Create separate plots for each domain scale
domain_names = patients_pre_dbs_scores.columns[1:-1]  # Domain scales start from the second column

for domain in domain_names:
    plt.figure(figsize=(16, 8))  # Increase figure size
    
    # Get the patient IDs and scores for the current domain
    patient_ids = patients_pre_dbs_scores["Patient_ID"]
    scores = patients_pre_dbs_scores[domain]
    
    plt.bar(patient_ids, scores, color='cadetblue', label=domain)
    plt.xlabel('Patient ID')
    plt.ylabel('Score')
    plt.ylim(0, 100)
    plt.title(f'{domain} Scores for Each Patient PRE-DBS')


    # Add annotations to display scores on top of each bar
    for patient_id, score in zip(patient_ids, scores):
        plt.annotate(str(score), xy=(patient_id, score), ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.show()


# # PDQ-39 scores analysis
# 
# ## SECOND SECTION: POST-DBS
# 
#  Analysing patients'/ caregivers' PDQ-39 scores after DBS
#  
# In this section I am interested in performing the scoring of PDQ-39 for each patient after DBS
# and then plot total scores and single-scale scores comparing PDQ-39 PRE/POST results.
# 
# 
#  
#  p.s. : as above, I don't have enough data (and even less for the post-DBS) to perform statistics as Wilcoxon or t-test 
#  so I didn't add that lines of code.
#  

# In[ ]:


### Log csv files with patients' PDQ-39 scores POST-DBS   ###


import pandas as pd
import logging

logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specify the path to your CSV file
csv_file_path = r'C:\Users\debora.spagnolo\Desktop\Deb_PhD\PD_DBS\patients_POST_PDQ39_rawscores.csv'

# Log the CSV file
try:
    patients_post_df = pd.read_csv(r'C:\Users\debora.spagnolo\Desktop\Deb_PhD\PD_DBS\patients_POST_PDQ39_rawscores.csv')
    logging.info(f'CSV file was successfully logged.')
except FileNotFoundError:
    logging.error(f'CSV file not found.')
except Exception as e:
    logging.error(f'An error occurred while logging CSV: {str(e)}')
    
patients_post_df


# In[ ]:


# Create an empty dictionary to store patient scores in PDQ-39 specific scales
patient_post_scores_by_scale = {}

# Define the item ranges for each scale
pdq_scales = {
    'Mobility': range(0, 10),             # Items 0 to 9
    'Activity_of_daily_living': range(10, 16),   # Items 10 to 15
    'Emotional_wellbeing': range(16, 22),        # Items 16 to 21
    'Stigma': range(22, 26),                    # Items 22 to 25
    'Social_support': range(26, 29),            # Items 26 to 28
    'Cognition': range(29, 33),                 # Items 29 to 32
    'Communication': range(33, 36),             # Items 33 to 35
    'Bodily_discomfort': range(36, 39)          # Items 36 to 38
}

# Loop through each patient's scores and organize them into scales
for patient_name in patients_post_df.columns:
    patient_scores = patients_post_df.loc[0:38, patient_name].tolist()  # PDQ-39 has 39 items
    patient_scale_scores = {}
    
    for scale, item_range in pdq_scales.items():
        pdq_scales_scores = [patient_scores[item] for item in item_range]
        patient_scale_scores[scale] = pdq_scales_scores

    patient_post_scores_by_scale[patient_name] = patient_scale_scores

# Print the patient scores organized by scales
patient_pre_scores_by_scale

patient_post_scores_by_scale = pd.DataFrame(patient_post_scores_by_scale)
patient_post_scores_by_scale = patient_post_scores_by_scale.T
patient_post_scores_by_scale


# In[ ]:


# Apply the calculate_domain_score function (above in the first section) to each row (patient) in the DataFrame
patients_post_dbs_scores = patient_post_scores_by_scale.apply(lambda row: row.apply(calculate_domain_score), axis=1)

# Calculate the total score for each patient
patients_post_dbs_scores['Total_score'] = patients_post_dbs_scores.sum(axis=1) / 8

# Round the total scores
patients_post_dbs_scores['Total_score'] = patients_post_dbs_scores['Total_score'].round()

# Print the DataFrame with the total scores for each patient
patients_post_dbs_scores = patients_post_dbs_scores.drop("items", axis =0)
new_col = ['patient_1', 'patient_2', 'patient_3', 'patient_5', 'patient_6']
patients_post_dbs_scores.insert(loc=0, column='Patient_ID', value=new_col)

# Reset df index and print the dataframe
patients_post_dbs_scores.reset_index(drop=True, inplace=True)
patients_post_dbs_scores


# In[ ]:


## SCALE-SPECIFIC PLOTS POST-DBS ##

import matplotlib.pyplot as plt

# Create separate plots for each domain scale
domain_names = patients_post_dbs_scores.columns[1:-1]  # Domain scales start from the second column

for domain in domain_names:
    plt.figure(figsize=(16, 8))  # Increase figure size
    
    # Get the patient IDs and scores for the current domain
    patient_ids = patients_post_dbs_scores["Patient_ID"]
    scores = patients_post_dbs_scores[domain]
    
    plt.bar(patient_ids, scores, color='cadetblue', label=domain)
    plt.xlabel('Patient ID')
    plt.ylabel('Score')
    plt.ylim(0, 100)
    plt.title(f'{domain} Scores for Each Patient PRE-DBS')


    # Add annotations to display scores on top of each bar
    for patient_id, score in zip(patient_ids, scores):
        plt.annotate(str(score), xy=(patient_id, score), ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:


## SINGLE-PATIENT PLOT - PDQ-39 SCORES PRE/POST DBS ##


import matplotlib.pyplot as plt

# Specify the patient ID
patient_id = 'patient_2'

# Filter data for the specified patient
patient_data_pre = patients_pre_dbs_scores[patients_pre_dbs_scores["Patient_ID"] == patient_id]
patient_data_post = patients_post_dbs_scores[patients_post_dbs_scores["Patient_ID"] == patient_id]

# Get domain names for Y-axis
domain_names = patient_data_pre.columns[1:-1]

# Plot bar chart for Patient x at two time points
plt.figure(figsize=(10, 6))
width = 0.35
x = range(len(domain_names))

plt.bar(x, patient_data_pre.iloc[0, 1:-1], width=width, color="cadetblue", label= 'PRE DBS')
plt.bar([pos + width for pos in x], patient_data_post.iloc[0, 1:-1], width=width, color= 'navy', label='POST DBS')

plt.xlabel('Domain Scales')
plt.ylabel('Scores')
plt.title(f'{patient_id} PDQ-39 Scores PRE/POST DBS')

# Add annotations to display scores on top of each bar for Time Point 1
for i, score in enumerate(patient_data_pre.iloc[0, 1:-1]):
    plt.annotate(str(score), xy=(i, score), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

# Add annotations to display scores on top of each bar for Time Point 2
for i, score in enumerate(patient_data_post.iloc[0, 1:-1]):
    plt.annotate(str(score), xy=(i + width, score), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

plt.xticks([pos + width / 2 for pos in x], domain_names, rotation=45)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


### SELECT PATIENT(S) - PLOT TOTAL SCORE PRE/POST DBS ###


# List of patient IDs to plot
patients_to_plot = ['patient_1', 'patient_2', 'patient_3', 'patient_5', 'patient_6']

# Filter the DataFrames to include data for the selected patients
df_before_dbs_subset = pd.DataFrame(patients_pre_dbs_scores[patients_pre_dbs_scores['Patient_ID'].isin(patients_to_plot)])
df_after_dbs_subset = patients_post_dbs_scores[patients_post_dbs_scores['Patient_ID'].isin(patients_to_plot)]

# Total scores before DBS for the selected patients
scores_before_dbs = df_before_dbs_subset['Total_score']

# Total scores after DBS for the selected patients
scores_after_dbs = df_after_dbs_subset['Total_score']

# Set the positions and width for the bars
width = 0.35
x = np.arange(len(patients_to_plot))

# Create the grouped bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x, scores_before_dbs, width, color='cadetblue', label='PRE DBS')
bars2 = ax.bar(x + width, scores_after_dbs, width,color= 'navy', label='POST DBS')

# Add scores above the bars
for bar1, bar2 in zip(bars1, bars2):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.annotate(f'{height1}', xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    ax.annotate(f'{height2}', xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

# Set labels, title, and legend
ax.set_xlabel('Patient ID')
ax.set_ylabel('Total Scores')
plt.ylim(0, 100)
ax.set_title('PDQ-39 Total Scores Before and After DBS (Selected Patients)')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(patients_to_plot)
ax.legend()

plt.tight_layout()

