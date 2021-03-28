import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def normalize(mat):
    for feature in range(mat.shape[1]):
        mean = np.mean(mat[:, feature])
        sigma = np.std(mat[:, feature])
        mat[:, feature] =(mat[:, feature]-mean)/sigma
    return mat


def scratch_pca(data, dim=5): 
    value, vector = np.linalg.eig(np.cov(data.T))
    new_mat = vector[:, :dim].T.dot(data.T)
    variance_ratio = np.sum(value[:dim])/np.sum(value)
    variance = vector[:, :dim]
    return new_mat.T, variance_ratio, variance


def main():
    data_T = pd.read_csv('dataset_T.csv')
    data_X = pd.read_csv('dataset_X.csv')
    data_X, data_T = data_X.to_numpy(), data_T.to_numpy()
    data_X, data_T = data_X[:, 1:], data_T[:, 1].reshape(-1, 1)
    data_X = data_X.astype(np.float64)
    data_T = data_T.astype(np.float64)
    features_names = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','RAINFALL',
                      'RH','SO2','THC','WD_HR','W_D','W_S','WS_HR']
    x_norm = normalize(data_X)
    x_trans, ratio, pc = scratch_pca(x_norm)
    print(ratio)

    print(fold_test(x_norm, data_T))
    print(fold_test(x_trans, data_T))
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(pc[:, 0].reshape(1, -1)), cmap='hot', aspect='auto', alpha=0.7)
    for i in range(17):
        c = round(float(pc[i, 0]), 2)
        plt.text(i, 0, str(c), va='center', ha='center')
    fig.colorbar(im)
    plt.xticks(np.linspace(0, 16, 17), features_names)
    plt.show()
    plt.close()

    # plt.scatter(x_trans[:, 0], x_trans[:, 1], c=data_T, cmap='hot', vmin=np.min(data_T), vmax=np.max(data_T), alpha=0.7)
    # plt.colorbar()
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()
    # plt.close()
    # pca = PCA(n_components=2)
    # x_trans = pca.fit_transform(x_norm)
    # print(pca.explained_variance_ratio_)
    # plt.scatter(x_trans[:, 0], x_trans[:, 1])
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()
    # plt.close()


def fold_test(x, y, num_folds=5):
    rnd = np.arange(x.shape[0])
    np.random.shuffle(rnd)
    x = x[rnd, :]
    y = y[rnd, :]
    x = np.array_split(x, num_folds)
    y = np.array_split(y, num_folds)
    train_loss = 0.0
    val_loss = 0.0
    model = LinearRegression()
    for i in range(5):
        x_train, x_val = np.concatenate(x[:i] + x[i+1:]), x[i]
        y_train, y_val = np.concatenate(y[:i] + y[i+1:]), y[i]
        model.fit(x_train, y_train)
        train_loss += err_func(model.predict(x_train), y_train) 
        val_loss += err_func(model.predict(x_val), y_val)
    return train_loss/5, val_loss/5


def err_func(x, y):
    loss = np.sum((x - y)**2)
    return round(loss/(2*x.shape[0]), 2)


if __name__ == "__main__":
    main()