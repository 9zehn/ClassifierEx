import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers
for idx, num_samples in enumerate([750,1500,2250,3000]):
    # TODO: Fill in a valid feature range
    feature_range = list(range(1,101))
    ratios = []

    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        X,_ = make_classification(n_samples=num_samples,
                                n_features=num_features,
                                n_informative=num_features,
                                random_state=42,
                                n_classes=2,
                                n_redundant=0,
                                n_clusters_per_class=1)

        # TODO: Choose random query point from X
        query_index = np.random.randint(0,len(X))
        query_point = X[query_index]

        remove_point = np.argwhere(np.all(X==query_point,axis=1))
        X_temp = np.delete(X,remove_point,axis=0)

        # TODO: remove query pt from X so it isn't used in distance calculations

        # TODO: Calculate distances
        delta = X_temp - query_point
        temp_result = np.sum(delta**2, axis = 1)

        distances = np.sqrt(temp_result)
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('hw5_plot.png')
plt.show()
