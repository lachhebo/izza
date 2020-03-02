from sklearn.metrics.cluster import contingency_matrix


def f_macro_score(label_observed, label_pred, cut_CP, cut_CR):
    '''
    assess the f-macro score

    Parameters
    ----------

    label_observed : the true class of the observations
        numpy array

    label_pred : the class predicted by the clustering algorithm
        numpy array

    cut_P : the minimum percentage of precision by cluster
        float

    cut_R : the minimum percentage of recall by cluster
        float

    Return
    -------

    f-macro:  the f-macro score

    '''
    # calculate contengency matrix for clustering

    matrix_contengency = contingency_matrix(label_observed, label_pred)

    # calculate precsion and recall for each cluster

    nb_cluster = 0
    classes = []

    for i in range(0, len(label_pred)):
        if label_pred[i] in classes:
            pass
        else:
            classes.append(label_pred[i])
            nb_cluster = nb_cluster + 1

    Plist = []
    Rlist = []

    for i in range(nb_cluster):
        P = matrix_contengency[1, i] / \
            matrix_contengency.sum(axis=0, dtype='float')[i]
        Plist.append(P)

    for j in range(nb_cluster):
        R = matrix_contengency[1, j] / \
            matrix_contengency.sum(axis=1, dtype='float')[1]
        Rlist.append(R)

    # select only viable cluster

    viable_cluster = []
    for i in range(0, nb_cluster):
        if Plist[i] >= cut_CP and Rlist[i] >= cut_CR:
            viable_cluster.append({'P': Plist[i],
                                   'R': Rlist[i]})

    # calculate global precision and global recall

    if len(viable_cluster) == 0:
        return 0
    else:
        global_precision = float(
            sum(cluster['P'] for cluster in viable_cluster)) / len(viable_cluster)
        global_recall = sum(cluster['R'] for cluster in viable_cluster)

        # calculate global f_score

        global_f_score = 2 * (global_precision * global_recall) / \
            (global_precision + global_recall)

        return global_f_score


def precision_macro_score(label_observed, label_pred, cut_CP, cut_CR):
    '''
    evaluate the mean precision on the cluster respecting the constraint

    Parameters
    ----------

    label_observed : the true class of the observations
        numpy array

    label_pred : the class predicted by the clustering algorithm
        numpy array

    cut_P : the minimum percentage of precision by cluster
        float

    cut_R : the minimum percentage of recall by cluster
        float

    Return
    -------

    f-precision: the mean precision on the cluster respecting the constraint

    '''
    # calculate contengency matrix for clustering

    matrix_contengency = contingency_matrix(label_observed, label_pred)

    # calculate precsion and recall for each cluster

    nb_cluster = 0
    classes = []

    for i in range(0, len(label_pred)):
        if label_pred[i] in classes:
            pass
        else:
            classes.append(label_pred[i])
            nb_cluster = nb_cluster + 1

    Plist = []
    Rlist = []

    for i in range(nb_cluster):
        P = matrix_contengency[1, i] / \
            matrix_contengency.sum(axis=0, dtype='float')[i]
        Plist.append(P)

    for j in range(nb_cluster):
        R = matrix_contengency[1, j] / \
            matrix_contengency.sum(axis=1, dtype='float')[1]
        Rlist.append(R)

    # select only viable cluster

    viable_cluster = []
    for i in range(0, nb_cluster):
        if Plist[i] >= cut_CP and Rlist[i] >= cut_CR:
            viable_cluster.append({'P': Plist[i],
                                   'R': Rlist[i]})

    # calculate global precision

    if len(viable_cluster) == 0:
        return 0
    else:
        global_precision = float(
            sum(cluster['P'] for cluster in viable_cluster)) / len(viable_cluster)

        return global_precision


def viable_clusters(label_observed, label_pred, cut_CP, cut_CR):
    '''
    evaluate the cluster respecting the constraints

    Parameters
    ----------

    label_observed : the true class of the observations
        numpy array

    label_pred : the class predicted by the clustering algorithm
        numpy array

    cut_P : the minimum percentage of precision by cluster
        float

    cut_R : the minimum percentage of recall by cluster
        float

    Return
    -------

    clusters: the clusters respecting the constraints

    '''

    matrix_contengency = contingency_matrix(label_observed, label_pred)

    # calculate precsion and recall for each cluster

    nb_cluster = 0
    classes = []

    for i in range(0, len(label_pred)):
        if label_pred[i] in classes:
            pass
        else:
            classes.append(label_pred[i])
            nb_cluster = nb_cluster + 1

    Plist = []
    Rlist = []

    for i in range(nb_cluster):
        P = matrix_contengency[1, i] / \
            matrix_contengency.sum(axis=0, dtype='float')[i]
        Plist.append(P)

    for j in range(nb_cluster):
        R = matrix_contengency[1, j] / \
            matrix_contengency.sum(axis=1, dtype='float')[1]
        Rlist.append(R)

    # select only viable cluster

    viable_cluster = []
    for i in range(0, nb_cluster):
        if Plist[i] >= cut_CP and Rlist[i] >= cut_CR:
            viable_cluster.append(i)

    return viable_cluster


def recall_macro_score(label_observed, label_pred, cut_CP, cut_CR):
    '''
    evaluate the sum of the recall on the cluster respecting the constraints

    Parameters
    ----------

    label_observed : the true class of the observations
        numpy array

    label_pred : the class predicted by the clustering algorithm
        numpy array

    cut_P : the minimum percentage of precision by cluster
        float

    cut_R : the minimum percentage of recall by cluster
        float

    Return
    -------

    f-recall: the sum of the recall on the cluster respecting the constraints

    '''
    # calculate contengency matrix for clustering

    matrix_contengency = contingency_matrix(label_observed, label_pred)

    # calculate precsion and recall for each cluster

    nb_cluster = 0
    classes = []

    for i in range(0, len(label_pred)):
        if label_pred[i] in classes:
            pass
        else:
            classes.append(label_pred[i])
            nb_cluster = nb_cluster + 1

    Plist = []
    Rlist = []

    for i in range(nb_cluster):
        P = matrix_contengency[1, i] / \
            matrix_contengency.sum(axis=0, dtype='float')[i]
        Plist.append(P)

    for j in range(nb_cluster):
        R = matrix_contengency[1, j] / \
            matrix_contengency.sum(axis=1, dtype='float')[1]
        Rlist.append(R)

    viable_cluster = []
    for i in range(0, nb_cluster):
        if Plist[i] >= cut_CP and Rlist[i] >= cut_CR:
            viable_cluster.append({'P': Plist[i],
                                   'R': Rlist[i]})

    # calculate global recall

    if len(viable_cluster) == 0:
        return 0
    else:
        global_recall = sum(cluster['R'] for cluster in viable_cluster)

        # calculate global recall

        return global_recall
