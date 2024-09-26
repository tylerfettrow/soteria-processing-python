import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
# Load the data

subprocess.call('gsutil -m cp "gs://soteria_study_data/Analysis/total_eventSmarteye_metric_dataframe.csv" .', shell=True)

data_file = 'total_eventSmarteye_metric_dataframe.csv'
data_dir = os.getcwd()
data_path = os.path.join(data_dir, data_file)
dat = pd.read_csv(data_path)

gazevar_dat = dat[dat.gaze_variance.notnull()]
# Fit the linear mixed-effects model
lm_model = sm.MixedLM.from_formula('gaze_variance ~ C(crew)*seat', data=gazevar_dat, groups=gazevar_dat['seat'])
lm_results = lm_model.fit(reml=False)
print(lm_results.summary())

grid = np.array(np.meshgrid(
  gazevar_dat["gaze_variance"].unique(), 
  gazevar_dat["crew"].unique(),
  gazevar_dat["seat"].unique()
)).reshape(3, 4).T

grid = pd.DataFrame(grid, columns = ['task','stimulus'])

print(grid)

mat = dmatrix(
  "C(task, Treatment(1))*C(stimulus, Treatment(1))", 
  grid, 
  return_type = "matrix"
)
print(mat)

betas = mod_fit.fe_params
print(betas)

emmeans = grid
emmeans['means'] = mat @ betas
print(emmeans)


# Perform ANOVA
anova_lm_model_gaze_variance_event1_delta = AnovaRM(data=gazevar_dat, depvar='gaze_variance', subject='seat',
                                                    within=['crew'], aggregate_func='mean').fit()

print(anova_lm_model_gaze_variance_event1_delta.summary())

# Get confidence intervals for emmeans
emmeans_results = emmeans.emmeans(lm_results, specs="crew", dtype='response')

# Extract confidence intervals
ci_df = emmeans_results.summary_frame()

# Plot the results
figure_name = "gaze_variance.tiff"

plt.figure()
plt.bar(ci_df['crew'], ci_df['emmean'], yerr=(ci_df['emmean'] - ci_df['lower.CL'], ci_df['upper.CL'] - ci_df['emmean']),
        align='center', alpha=0.5)
plt.ylim(0, 0.5)
plt.xlabel("")
plt.ylabel("Gaze Variance")
plt.title("Gaze Variance")
plt.xticks(rotation=45)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join("Figures", figure_name), format="tiff")
plt.show()
