from flask import Flask, render_template, request, Response
from flask_material import Material
from flask import send_file
from flask import *

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import io
import base64


import pandas as pd  # reading all required header files
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal     # for generating pdf
from sklearn.metrics import mutual_info_score

app = Flask(__name__)
Material(app)


# def accuracy(cluster_labels, class_labels, n):

#     correct_pred = 0
#     seto = max(set(cluster_labels[0:50]), key=cluster_labels[0:50].count)
#     vers = max(set(cluster_labels[50:100]), key=cluster_labels[50:100].count)
#     virg = max(set(cluster_labels[100:]), key=cluster_labels[100:].count)

#     for i in range(n):
#         if cluster_labels[i] == seto and class_labels[i] == 'Iris-setosa':
#             correct_pred = correct_pred + 1
#         if cluster_labels[i] == vers and class_labels[i] == 'Iris-versicolor' and vers != seto:
#             correct_pred = correct_pred + 1
#         if cluster_labels[i] == virg and class_labels[i] == 'Iris-virginica' and virg != seto and virg != vers:
#             correct_pred = correct_pred + 1

#     accuracy = (correct_pred / n)*100
#     return accuracy

def accuracy(cluster_labels, class_labels):
    return mutual_info_score(cluster_labels, class_labels)


def initializeMembershipMatrix(n, k):  # initializing the membership matrix
    membership_mat = []
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]

        flag = temp_list.index(max(temp_list))
        for j in range(0, len(temp_list)):
            if(j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0

        membership_mat.append(temp_list)
    return membership_mat


# calculating the cluster center
def calculateClusterCenter(membership_mat, n, k, m, df):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


# Updating the membership value
def updateMembershipValue(membership_mat, cluster_centers, n, k, m, df):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(
            np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p)
                       for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat


def getClusters(membership_mat, n):  # getting the clusters
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx)
                           for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


# Third iteration Random vectors from data
def fuzzyCMeansClustering(n, k, m, df, MAX_ITER):
    # Membership Matrix
    membership_mat = initializeMembershipMatrix(n, k)
    curr = 0
    acc = []
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat, n, k, m, df)
        membership_mat = updateMembershipValue(
            membership_mat, cluster_centers, n, k, m, df)
        cluster_labels = getClusters(membership_mat, n)

        acc.append(cluster_labels)

        if(curr == 0):
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    print(np.array(membership_mat))
    # return cluster_labels, cluster_centers
    return cluster_labels, cluster_centers, acc


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/uploadFile', methods=['GET', 'POST'])
def preview():
    if request.method == 'POST':
        f = request.files['csvfile']
        df_full = pd.read_csv(f)    # Read CSV file
        if request.form['submit_button'] == 'Plot_iris_result':
            columns = list(df_full.columns)
            features = columns[:len(columns)-1]
            class_labels = list(df_full[columns[-1]])
            df = df_full[features]

            img_sepal = io.BytesIO()
            img_petal = io.BytesIO()

            # Number of Clusters
            k = int(request.form['k_val'])
            # Maximum number of iterations
            MAX_ITER = 100
            # Number of data points
            n = len(df)
            # Fuzzy parameter
            # Select a value greater than 1 else it will be knn
            m = float(request.form['m_val'])

            membership_mat = initializeMembershipMatrix(n, k)

            # cluster_centers = calculateClusterCenter(membership_mat)
            calculateClusterCenter(membership_mat, n, k, m, df)

            labels, centers, acc = fuzzyCMeansClustering(n, k, m, df, MAX_ITER)
            a = accuracy(labels, class_labels)

            # Calculate the mean and deviation over 100 loops
            acc_list = []
            for i in range(0, len(acc)):
                # val = accuracy(acc[i], class_labels, n)
                val = accuracy(acc[i], class_labels)
                acc_list.append(val)
            acc_list = np.array(acc_list)
            mean = np.mean(acc_list)
            deviation = np.std(acc_list)

            # Plotting result
            seto = max(set(labels[0:50]), key=labels[0:50].count)
            vers = max(set(labels[50:100]), key=labels[50:100].count)
            virg = max(set(labels[100:]), key=labels[100:].count)

            # sepal
            s_mean_clus1 = np.array([centers[seto][0], centers[seto][1]])
            s_mean_clus2 = np.array([centers[vers][0], centers[vers][1]])
            s_mean_clus3 = np.array([centers[virg][0], centers[virg][1]])

            values = np.array(labels)  # label

            # search all 3 species
            searchval_seto = seto
            searchval_vers = vers
            searchval_virg = virg

            # index of all 3 species
            ii_seto = np.where(values == searchval_seto)[0]
            ii_vers = np.where(values == searchval_vers)[0]
            ii_virg = np.where(values == searchval_virg)[0]
            ind_seto = list(ii_seto)
            ind_vers = list(ii_vers)
            ind_virg = list(ii_virg)

            sepal_df = df_full.iloc[:, 0:2]
            seto_df = sepal_df[sepal_df.index.isin(ind_seto)]
            vers_df = sepal_df[sepal_df.index.isin(ind_vers)]
            virg_df = sepal_df[sepal_df.index.isin(ind_virg)]

            cov_seto = np.cov(np.transpose(np.array(seto_df)))
            cov_vers = np.cov(np.transpose(np.array(vers_df)))
            cov_virg = np.cov(np.transpose(np.array(virg_df)))

            sepal_df = np.array(sepal_df)

            x1 = np.linspace(4, 8, 150)
            x2 = np.linspace(1.5, 4.5, 150)
            X, Y = np.meshgrid(x1, x2)
            Z1 = multivariate_normal(s_mean_clus1, cov_seto)
            Z2 = multivariate_normal(s_mean_clus2, cov_vers)
            Z3 = multivariate_normal(s_mean_clus3, cov_virg)

            # a new array of given shape and type, without initializing entries
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y

            # creating the figure and assigning the size
            plt.figure(figsize=(10, 10))
            plt.scatter(sepal_df[:, 0], sepal_df[:, 1], marker='o')
            plt.contour(X, Y, Z1.pdf(pos), colors="r", alpha=0.5)
            plt.contour(X, Y, Z2.pdf(pos), colors="b", alpha=0.5)
            plt.contour(X, Y, Z3.pdf(pos), colors="g", alpha=0.5)
            # making both the axis equal
            plt.axis('equal')
            # X-Axis
            plt.xlabel('Sepal Length', fontsize=16)
            # Y-Axis
            plt.ylabel('Sepal Width', fontsize=16)
            plt.title('Final Clusters(Sepal)', fontsize=22)
            plt.savefig(img_sepal, format='png')
            img_sepal.seek(0)
            url_sepal_final = base64.b64encode(img_sepal.getvalue()).decode()

            # petal
            p_mean_clus1 = np.array([centers[seto][2], centers[seto][3]])
            p_mean_clus2 = np.array([centers[vers][2], centers[vers][3]])
            p_mean_clus3 = np.array([centers[virg][2], centers[virg][3]])
            petal_df = df_full.iloc[:, 2:4]
            seto_df = petal_df[petal_df.index.isin(ind_seto)]
            vers_df = petal_df[petal_df.index.isin(ind_vers)]
            virg_df = petal_df[petal_df.index.isin(ind_virg)]
            cov_seto = np.cov(np.transpose(np.array(seto_df)))
            cov_vers = np.cov(np.transpose(np.array(vers_df)))
            cov_virg = np.cov(np.transpose(np.array(virg_df)))
            petal_df = np.array(petal_df)

            x1 = np.linspace(0.5, 7, 150)
            x2 = np.linspace(-1, 4, 150)
            X, Y = np.meshgrid(x1, x2)

            Z1 = multivariate_normal(p_mean_clus1, cov_seto)
            Z2 = multivariate_normal(p_mean_clus2, cov_vers)
            Z3 = multivariate_normal(p_mean_clus3, cov_virg)

            # a new array of given shape and type, without initializing entries
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y

            # creating the figure and assigning the size
            plt.figure(figsize=(10, 10))
            plt.scatter(petal_df[:, 0], petal_df[:, 1], marker='o')
            plt.contour(X, Y, Z1.pdf(pos), colors="r", alpha=0.5)
            plt.contour(X, Y, Z2.pdf(pos), colors="b", alpha=0.5)
            plt.contour(X, Y, Z3.pdf(pos), colors="g", alpha=0.5)
            # making both the axis equal
            plt.axis('equal')
            # X-Axis
            plt.xlabel('Petal Length', fontsize=16)
            # Y-Axis
            plt.ylabel('Petal Width', fontsize=16)
            plt.title('Final Clusters(Petal)', fontsize=22)
            plt.savefig(img_petal, format='png')
            img_petal.seek(0)
            url_petal_final = base64.b64encode(img_petal.getvalue()).decode()

            return render_template("cluster.html", acc=a, cluster_centers=centers, mean_val=mean, dev=deviation,
                                   url3=url_sepal_final, url4=url_petal_final)

        elif request.form['submit_button'] == 'View_dataset':
            return render_template("preview.html", df_view=df_full)

        elif request.form['submit_button'] == 'Plotting_iris':
            # df_full = df_full.drop(['Id'], axis=1)

            columns = list(df_full.columns)
            features = columns[:len(columns)-1]
            class_labels = list(df_full[columns[-1]])
            df = df_full[features]

            img1 = io.BytesIO()
            img2 = io.BytesIO()
            # scatter plot of sepal length vs sepal width
            plt.figure(figsize=(10, 10))
            plt.scatter(list(df.iloc[:, 0]), list(df.iloc[:, 1]), marker='o')
            plt.axis('equal')
            plt.xlabel('Sepal length', fontsize=16)
            plt.ylabel('Sepal width', fontsize=16)
            plt.title('Sepal Plot', fontsize=22)
            plt.savefig(img1, format='png')
            img1.seek(0)
            url1 = base64.b64encode(img1.getvalue()).decode()

            # scatter plot of petal length vs sepal width
            plt.figure(figsize=(10, 10))
            plt.scatter(list(df.iloc[:, 2]), list(df.iloc[:, 3]), marker='o')
            plt.axis('equal')
            plt.xlabel('Petal length', fontsize=16)
            plt.ylabel('Petal_width', fontsize=16)
            plt.title('Petal Plot', fontsize=22)
            plt.savefig(img2, format='png')
            img2.seek(0)
            url2 = base64.b64encode(img2.getvalue()).decode()
            return render_template('plot.html', url_sepal=url1, url_petal=url2)

        elif request.form['submit_button'] == 'Describe_dataset':
            a = df_full.head()
            b = df_full.tail()
            des = df_full.describe()
            row_num = len(df_full)
            col_num = len(df_full.columns)
            return render_template('describe.html', df_head=a, df_tail=b, rows=row_num, cols=col_num, describe=des)

        elif request.form['submit_button'] == 'Result':
            columns = list(df_full.columns)
            features = columns[:len(columns)-1]
            class_labels = list(df_full[columns[-1]])
            df = df_full[features]

            # Number of Clusters
            k = int(request.form['k_val'])
            # Maximum number of iterations
            MAX_ITER = 100
            # Number of data points
            n = len(df)
            # Fuzzy parameter
            # Select a value greater than 1 else it will be knn
            m = float(request.form['m_val'])

            membership_mat = initializeMembershipMatrix(n, k)

            # cluster_centers = calculateClusterCenter(membership_mat)
            calculateClusterCenter(membership_mat, n, k, m, df)

            labels, centers, acc = fuzzyCMeansClustering(n, k, m, df, MAX_ITER)
            a = accuracy(labels, class_labels)

            seto = max(set(labels[0:50]), key=labels[0:50].count)
            vers = max(set(labels[50:100]), key=labels[50:100].count)
            virg = max(set(labels[100:]), key=labels[100:].count)

            labels_str = []
            for i in labels:
                if i == seto:
                    labels_str.append("Iris-setosa")
                elif i == vers:
                    labels_str.append("Iris-versicolor")
                elif i == virg:
                    labels_str.append("Iris-virginica")

            return render_template('result.html', acc=a, labels_str=labels_str, true_labels=class_labels)

        elif request.form['submit_button'] == 'Visualize dataset':
            # columns = list(df_full.columns)
            # label = list(df_full[columns[-1]])
            label = df_full.columns[-1]
            img = io.BytesIO()
            sns_plot = sns.pairplot(df_full, hue=label)
            sns_plot.savefig(img, format='png')
            img.seek(0)
            url = base64.b64encode(img.getvalue()).decode()

            return render_template('visualize.html', plot_url=url)


if __name__ == "__main__":
    app.run(debug=True)
