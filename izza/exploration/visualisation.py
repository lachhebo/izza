import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import minisom
from sklearn.decomposition import PCA


def pca_visualisation(X, y, colordict, labeldict, markerdict, alphadict, figsize=(10, 10)):
    '''
    plot a visualisation of the data using the two main composant of a PCA. 

    Parameters
    ----------

    X : the explaining features, they should be normalised (numpy)

    y : the target feature (numpy)

    colordict : the color of each class present in y

    labeldict : the label of each class present in y

    markerdict : the marker of each class present in y

    alphadict :the alpha of each class present in y

    figsize : the size of the figure

    Return
    -------

    explained_variance_ratio : the ratio of the explained variance obtained with the PCA.

    Examples
    --------
    >>> from izza import pca_visualisation
    >>> cdict = {0 : "green", 1 : "red"}
    >>> labl  = {0: "healthy", 1 : "sick"}
    >>> marker= {0 : "o", 1: "*"}
    >>> alpha = {0: 0.5, 1: 0.5} 
    >>> pca_visualisation(X,y,cdict,labl,marker,alpha)
    '''

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    cdict = colordict
    labl = labeldict
    marker = markerdict
    alpha = alphadict

    _ = plt.figure(figsize=figsize)

    for l in np.unique(y):
        ix = np.where(y == l)
        plt.scatter(X_pca[ix, 0], X_pca[ix, 1], c=cdict[l],
                    s=40, label=labl[l], marker=marker[l], alpha=alpha[l])

    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    plt.legend()

    return pca.explained_variance_ratio_


def missingData(data):
    """Return a pandas dataframe of the missing observations by variables (number and percentage) and plot 
    a graph of those missing values by variable.
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()
               * 100).sort_values(ascending=False)
    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    md = md[md["Percent"] > 0]
    sns.set(style='darkgrid')
    plt.figure(figsize=(8, 4))
    plt.xticks(rotation='90')
    sns.barplot(md.index, md["Percent"], color="g", alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return md


def tabcategorial(data, variable, hue):
    "return a pandas dataframe containing data from the category by class (useful for binary classification)"
    pass


def camembert_plot(dataset, variable, target, cutoff, plot=True, figsize=(8, 8)):
    '''
    plot a camembert plot for a variable in the case of the non fraudster
    population and an other one for the fraudster population and print a
    table containing the data used to create the figures.

    Parameters
    ----------

    dataset : a pandas dataframe

    variable : the name of the variable to plot

    target : the name of the target variable to take into account

    cutoff : the minimum importance for a modality

    plot : a boolean representing if it is needed to plot the data

    figsize : the size of the figure

    Return
    -------

    table : the table containing the data used to create the figures

    '''
    data_nofrd = dataset[dataset[target] == 0]
    data_fraud = dataset[dataset[target] == 1]

    nofraudster = (data_nofrd[variable].value_counts() / len(data_nofrd)) * 100
    fraudsters = (data_fraud[variable].value_counts() / len(data_fraud)) * 100

    nofraudster = nofraudster.to_frame()
    fraudsters = fraudsters.to_frame()

    if plot:
        nofraudster[nofraudster[variable] > cutoff].plot.pie(
            variable, figsize=(5, 5))
        fraudsters[fraudsters[variable] > cutoff].plot.pie(
            variable, figsize=(5, 5))

    result = pd.concat([nofraudster, fraudsters], axis=1, join='inner')
    result.columns = ['not fraudsters', 'fraudsters']

    return result


def kohohen_maps(X, y, som, size_x, size_y):
    '''
    plot a kohohen maps, the number in red is the proportion of fraudster in
    each cell and the color represent the distance of each withe the
    neighbouring cells.

    Parameters
    ----------

    X : a numpy array with explaining features

    y : a numpy array with the target feature

    som : a minisom instance

    size_x : the size of the grid (width)

    size_y : the size of the grid (height)

    '''

    plt.figure(figsize=(size_x, size_y))

    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()

    fraudsters = {}
    total = {}
    peri = {}

    for cnt, xx in enumerate(X):
        w = som.winner(xx)  # getting the winner

        if str(w[0] + .3) + str(w[1] + .3) in fraudsters:

            calcul1 = fraudsters[str(w[0] + .3) + str(w[1] + .3)]
            calcul2 = total[str(w[0] + .3) + str(w[1] + .3)]

            fraudsters[str(w[0] + .3) + str(w[1] + .3)] = calcul1 + y[cnt]
            total[str(w[0] + .3) + str(w[1] + .3)] = calcul2 + 1

        else:
            fraudsters[str(w[0] + .3) + str(w[1] + .3)] = y[cnt]
            total[str(w[0] + .3) + str(w[1] + .3)] = 1
            peri[str(w[0] + .3) + str(w[1] + .3)] = (w[0] + .3, w[1] + .3)

    proportion = {}

    for i in total:
        proportion[i] = round(fraudsters[i] / total[i], 3)

    for i in total:
        prop = proportion[i]
        lat, long = peri[i]

        plt.text(lat, long, s=prop, color="red")


def activation_frequencies(X, som, size_x, size_y):
    '''
    plot a kohohen maps, the color in each cell represent how many observation
    end up in the cell

    Parameters
    ----------

    X : a numpy array with explaining features

    som : a minisom instance

    size_x : the size of the grid (width)

    size_y : the size of the grid (height)

    '''
    plt.figure(figsize=(size_x, size_y))
    frequencies = som.activation_response(X)
    plt.pcolor(frequencies.T, cmap='Blues')
    plt.colorbar()
    plt.show()
