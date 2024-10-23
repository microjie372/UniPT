import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def visualize_features_MDF(waymo_feature, nusc_feature, index):
    """
    可视化 waymo_feature 和 nusc_feature 的 t-SNE 结果，并保存图像。
    
    参数:
    waymo_feature (numpy.ndarray): 原始特征，形状为 (1, C, D)。
    nusc_feature (numpy.ndarray): DSA 特征，形状为 (1, C, D)。
    video_name (str): 视频名称，用于图像标题。
    save_path (str): 保存图像的路径。
    """
    video_name = f"waymo-nusc-{index}"    #change your own file name
    num_classes = 4
    save_dir = "./visualfig-WN-unipt"     #change your own file path
    os.makedirs(save_dir, exist_ok=True)
    
    if waymo_feature.shape[0] != 1 or nusc_feature.shape[0] != 1:
        raise ValueError("Features should have shape (1, T, D).")

    
    waymo_feature_2d = waymo_feature.squeeze(0)
    nusc_feature_2d = nusc_feature.squeeze(0)

    scaler = StandardScaler()
    waymo_feature_2d = scaler.fit_transform(waymo_feature_2d)
    nusc_feature_2d = scaler.fit_transform(nusc_feature_2d)

    pca = PCA(n_components=2)
    reduced_waymo_data = pca.fit_transform(waymo_feature_2d)
    reduced_nusc_data = pca.fit_transform(nusc_feature_2d)

    # X_combined = np.vstack((waymo_feature_2d, nusc_feature_2d))
    X_combined = np.vstack((reduced_waymo_data, reduced_nusc_data))
      
    kmeans = KMeans(n_clusters=num_classes, random_state=0, init='k-means++')
    labels_clusters = kmeans.fit_predict(X_combined[:,::200])
    
    tsne = TSNE(n_components=2, random_state=0, method='barnes_hut', perplexity=50)  #perplexity 150, 40
    MDF_feature_tsne = tsne.fit_transform(X_combined)

    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    #ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    colors = ['red', 'blue', 'green', 'magenta']

    for i in range(num_classes):
        indices = np.where(labels_clusters == i)
        plt.scatter(MDF_feature_tsne[indices, 0], MDF_feature_tsne[indices, 1], c=colors[i], alpha=0.7, s=80)    # f'Cluster {i}'
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    save_path = os.path.join(save_dir, f"{video_name}.png")
    plt.savefig(save_path)
    #plt.show()
    plt.close()
    print(f"Feature visualization saved to {save_path}")

def main():
    for i in range(50):
        data_waymo = np.load(f'./fig-waymo-unipt/waymo_tsne_{i}.npy')        #change your own file path
        data_nusc = np.load(f'./fig-nusc-unipt/nusc_tsne_{i}.npy')           #change your own file path
       
        x = torch.from_numpy(data_waymo)
        y = torch.from_numpy(data_nusc)

        print('shape x:', x.size())
        print('shape y:', y.size())

        visualize_features_MDF(x, y, i)

if __name__ == '__main__':
    main()

