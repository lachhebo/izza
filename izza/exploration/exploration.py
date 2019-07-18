import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def missingData(data):
    """Return a pandas dataframe of the missing observations by variables (number and percentage) and plot 
    a graph of those missing values by variable.
    
    """
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    md = md[md["Percent"] > 0]
    sns.set(style = 'darkgrid')
    plt.figure(figsize = (8, 4))
    plt.xticks(rotation='90')
    sns.barplot(md.index, md["Percent"],color="g",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return md

def tabcategorial(data, variable, hue):
    "return a pandas dataframe containing data from the category by class (useful for binary classification)"
    pass

