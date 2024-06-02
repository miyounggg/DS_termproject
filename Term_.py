import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import init_notebook_mode
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

filepath=r'C:\HoeunSchool\3-1\데이터과학\pythonProject\healthcare-dataset-stroke-data.csv'
stroke = pd.read_csv(filepath)

# Distribution of stroke by gender
dst_st_gen = stroke.query('gender != "Other"').groupby(['gender', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
#Change the value of the stroke column
dst_st_gen.iloc[[0, 2], 1] = "Stroke X"
dst_st_gen.iloc[[1, 3], 1] = "Stroke"


fig = px.sunburst(dst_st_gen, path = ['gender', 'stroke'], values = 'count', color = 'gender',
                 color_discrete_map = {'Female': '#BE2E22', 'Male': '#0047AB'}, width = 700, height = 700)

#Update the chart layour
fig.update_layout(annotations = [dict(text = 'Distribution of stroke by gender',
x = 0.5, y = 1.1, font_size = 22, showarrow = False,
            font_family = 'Georzia', font_color = 'black')])


fig.update_traces(textinfo = 'label + percent parent')
fig.show()

# Create age groups
stroke['age_group'] = 0
for i in range(len(stroke.index)):
    if stroke.iloc[i, 2] < 2:
        stroke.iloc[i, 12] = 'Baby'
    elif stroke.iloc[i, 2] < 17 and stroke.iloc[i, 2] >= 2:
        stroke.iloc[i, 12] = 'Child'
    elif stroke.iloc[i, 2] < 30 and stroke.iloc[i, 2] >= 17:
        stroke.iloc[i, 12] = 'Young adults'
    elif stroke.iloc[i, 2] < 60 and stroke.iloc[i, 2] >= 30:
        stroke.iloc[i, 12] = 'Middle Age'
    elif stroke.iloc[i, 2] < 80 and stroke.iloc[i, 2] >= 60:
        stroke.iloc[i, 12] = 'Senior'
    else:
        stroke.iloc[i, 12] = 'Elderly'

# Create bmi groups
stroke['bmi_group'] = 0
for i in range(len(stroke.index)):
    if stroke.iloc[i, 9] < 18.5:
        stroke.iloc[i, 13] = 'Underweight'
    elif stroke.iloc[i, 9] < 25.0 and stroke.iloc[i, 9] >= 18.5:
        stroke.iloc[i, 13] = 'Normal weight'
    elif stroke.iloc[i, 9] < 30.0 and stroke.iloc[i, 9] >= 25.0:
        stroke.iloc[i, 13] = 'Overweight'
    else:
        stroke.iloc[i, 13] = 'Obese'

# Create glucose groups
stroke['glucose_group'] = 0
for i in range(len(stroke.index)):
    if stroke.iloc[i, 8] < 100:
        stroke.iloc[i, 14] = 'Normal'
    elif stroke.iloc[i, 8] >= 100 and stroke.iloc[i, 8] < 125:
        stroke.iloc[i, 14] = 'Prediabetes'
    else:
        stroke.iloc[i, 14] = 'Diabetes'

# Grouping by categorical features
dst_st_age = stroke.groupby(['age_group', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
hyper = stroke.groupby(['hypertension', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
heart = stroke.groupby(['heart_disease', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
marry = stroke.groupby(['ever_married', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
work = stroke.groupby(['work_type', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
residence = stroke.groupby(['Residence_type', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
glucose_group = stroke.groupby(['glucose_group', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
bmi_group = stroke.groupby(['bmi_group', 'stroke']).agg({'stroke': 'count'}).rename(
    columns={'stroke': 'count'}).reset_index()
smoking = stroke.query('smoking_status != "Unknown"').groupby(['smoking_status', 'stroke']).agg({'stroke': 'count'}) \
    .rename(columns={'stroke': 'count'}).reset_index()

#percentage
# Create percent column for data frames
def percent(data):
    data['percent'] = 0
    for i in range(len(data.index)):
        if i < len(data.index) - 1:
            if data.iloc[i, 0] == data.iloc[i + 1, 0]:
                data.iloc[i, 3] = round((data.iloc[i, 2] / (data.iloc[i, 2] + data.iloc[i + 1, 2])) * 100, 1)
            elif data.iloc[i, 0] == data.iloc[i - 1, 0]:
                data.iloc[i, 3] = 100 - data.iloc[i - 1, 3]
            else:
                data.iloc[i, 3] = 100.0
        else:
            if data.iloc[i, 0] == data.iloc[i - 1, 0]:
                data.iloc[i, 3] = 100 - data.iloc[i - 1, 3]
            else:
                data.iloc[i, 3] = 100.0


percent(dst_st_age)
percent(hyper)
percent(heart)
percent(marry)
percent(work)
percent(residence)
percent(glucose_group)
percent(bmi_group)
percent(smoking)

dst_st_age.iloc[[0, 2, 4, 6, 8, 10], 1] = "Stroke X"
dst_st_age.iloc[[1, 3, 5, 7, 9], 1] = "Stroke"

hyper.iloc[[0, 1], 0] = 'Hypertension X'
hyper.iloc[[2, 3], 0] = 'Hypertension'

heart.iloc[[0, 1], 0] = 'Heart diseases  X'
heart.iloc[[2, 3], 0] = 'Heart diseases'

fig = plt.figure(figsize=(18, 60))
fig.patch.set_facecolor('#fafafa')

#Histogram
plt.subplot(221)
sns.set_style('white')
plt.title('Age', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
a = sns.barplot(data = dst_st_age, x = dst_st_age['age_group'], y = dst_st_age['count'], hue = dst_st_age['stroke'], palette = ['#1092c9','#c91010'])
plt.xticks(rotation = 10)
plt.ylabel('')
plt.xlabel('')
plt.legend(loc = 'upper left')


plt.subplot(222)
plt.grid(color = '#495057', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
a2 = sns.barplot(data = dst_st_age, x = dst_st_age['age_group'], y = dst_st_age['percent'], hue = dst_st_age['stroke'], palette = ['#1092c9','#c91010'])
plt.xticks(rotation = 10)
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


plt.subplot(223)
sns.set_style('white')
plt.title('Hypertension', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
b = sns.barplot(data = hyper, x = hyper['hypertension'], y = hyper['count'], hue = hyper['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


plt.subplot(224)
plt.grid(color = '#495057', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
b2 = sns.barplot(data = hyper, x = hyper['hypertension'], y = hyper['percent'], hue = hyper['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


# add annotations
for i in [a,b]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')


for i in [a2,b2]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')


fig.subplots_adjust(hspace = 1)
plt.show()
##################
plt.subplot(221)
sns.set_style('white')
plt.title('Heart deseases', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
c = sns.barplot(data = heart, x = heart['heart_disease'], y = heart['count'], hue = heart['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)
fig.subplots_adjust

plt.subplot(222)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
c2 = sns.barplot(data = heart, x = heart['heart_disease'], y = heart['percent'], hue = heart['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


plt.subplot(223)
sns.set_style('white')
plt.title('A person who has been married', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
d = sns.barplot(data = marry, x = marry['ever_married'], y = marry['count'], hue = marry['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)

plt.subplot(224)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
d2 = sns.barplot(data = marry, x = marry['ever_married'], y = marry['percent'], hue = marry['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)

# add annotations
for i in [c,d]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

for i in [c2,d2]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

plt.show()
##

plt.subplot(221)
sns.set_style('white')
plt.title('Work', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
e = sns.barplot(data = work, x = work['work_type'], y = work['count'], hue = work['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)
fig.subplots_adjust(hspace = 2)

plt.subplot(222)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
e2 = sns.barplot(data = work, x = work['work_type'], y = work['percent'], hue = work['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)

plt.subplot(223)
sns.set_style('white')
plt.title('Residence', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
f = sns.barplot(data = residence, x = residence['Residence_type'], y = residence['count'], hue = residence['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)

plt.subplot(224)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
f2 = sns.barplot(data = residence, x = residence['Residence_type'], y = residence['percent'], hue = residence['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)

# add annotations
for i in [e,f]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

for i in [e2,f2]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

plt.show()

plt.subplot(221)
sns.set_style('white')
plt.title('Glucose', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
g = sns.barplot(data = glucose_group, x = glucose_group['glucose_group'], y = glucose_group['count'], hue = glucose_group['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)
fig.subplots_adjust(hspace = 2)

plt.subplot(222)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
g2 = sns.barplot(data = glucose_group, x = glucose_group['glucose_group'], y = glucose_group['percent'], hue = glucose_group['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


plt.subplot(223)
sns.set_style('white')
plt.title('BMI', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
h = sns.barplot(data = bmi_group, x = bmi_group['bmi_group'], y = bmi_group['count'], hue = bmi_group['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


plt.subplot(224)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
h2 = sns.barplot(data = bmi_group, x = bmi_group['bmi_group'], y = bmi_group['percent'], hue = bmi_group['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)



# add annotations
for i in [g,h]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

for i in [g2,h2]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

plt.show()

#

plt.subplot(121)
sns.set_style('white')
plt.title('Smoking', size = 15, x = 1.1, y = 1.03)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
j = sns.barplot(data = smoking, x = smoking['smoking_status'], y = smoking['count'], hue = smoking['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)


plt.subplot(122)
plt.grid(color = 'gray', linestyle = ':', axis = 'y', zorder = 0,  dashes = (1,7))
j2 = sns.barplot(data = smoking, x = smoking['smoking_status'], y = smoking['percent'], hue = smoking['stroke'], palette = ['#1092c9','#c91010'])
plt.ylabel('')
plt.xlabel('')
plt.legend('').set_visible(False)

# add annotations
for i in [j]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

for i in [j2]:
    for p in i.patches:
        height = p.get_height()
        i.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha = 'center', va = 'center',
                   size = 10,
                   xytext = (0, 5),
                   textcoords = 'offset points')

plt.show()

# Affect of age, BMI and average glucose level on risk of stroke
fig = plt.figure(figsize = (18, 24))
fig.patch.set_facecolor('#fafafa')

#Age
plt.subplot(221)
sns.set_style("dark")
plt.title('Age', size = 20)
sns.kdeplot(stroke.query('stroke == 1')['age'], color = '#960018', shade = True, label = 'Stroke', alpha = 0.5)
sns.kdeplot(stroke.query('stroke == 0')['age'], color = '#87CEEB', shade = True, label = "Stroke X", alpha = 0.5)
plt.grid(color = 'gray', linestyle = ':', axis = 'x', zorder = 0,  dashes = (1,7))
plt.ylabel('')
plt.xlabel('')
plt.yticks([])
plt.legend(loc = 'upper left')

#BMI
plt.subplot(222)
plt.title('BMI', size = 20)
sns.kdeplot(stroke.query('stroke == 1')['bmi'], color = '#960018', shade = True, label = 'Stroke', alpha = 0.5)
sns.kdeplot(stroke.query('stroke == 0')['bmi'], color = '#87CEEB', shade = True, label = "Stroke X", alpha = 0.5)
plt.grid(color = 'gray', linestyle = ':', axis = 'x', zorder = 0,  dashes = (1,7))
plt.ylabel('')
plt.xlabel('')
plt.yticks([])
plt.legend('').set_visible(False)

#Glucose
plt.subplot(223)
plt.title('Glucose', size = 20)
sns.kdeplot(stroke.query('stroke == 1')['avg_glucose_level'], color = '#960018', shade = True, label = 'Stroke', alpha = 0.5)
sns.kdeplot(stroke.query('stroke == 0')['avg_glucose_level'], color = '#87CEEB', shade = True, label = "Stroke X", alpha = 0.5)
plt.grid(color = 'gray', linestyle = ':', axis = 'x', zorder = 0,  dashes = (1,7))
plt.ylabel('')
plt.xlabel('')
plt.yticks([])
plt.legend('').set_visible(False)


plt.show()

# Smoking
plt.subplot(324)
plt.title('Smoking', size=20)
sns.countplot(data=stroke.query('smoking_status != "Unknown"'), x='smoking_status', hue='stroke', palette=['#87CEEB', '#960018'])
plt.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,7))
plt.ylabel('')
plt.xlabel('')
plt.legend(title='Stroke', loc='upper left')

# Heart disease
plt.subplot(325)
plt.title('Heart Disease', size=20)
sns.countplot(data=stroke, x='heart_disease', hue='stroke', palette=['#87CEEB', '#960018'])
plt.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,7))
plt.ylabel('')
plt.xlabel('')
plt.legend(title='Stroke', loc='upper left')

plt.show()