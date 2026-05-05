import seaborn as sns
import matplotlib.pyplot as plt
import data_loader

if __name__ == "__main__": 
    X_numeric = data_loader.load_all_data().select_dtypes(include="number")
    corr_matrix = X_numeric.corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr_matrix,
        cmap="Oranges",
        center=0,
        vmin=-1, vmax=1,
        xticklabels=True,
        yticklabels=True
    )
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig("../correlation_matrix.png", dpi=150)
    plt.show()
