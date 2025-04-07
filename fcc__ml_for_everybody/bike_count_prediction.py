import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FILENAME = "datafile.csv"
DATASET_COLS = [
    "bike_count",
    "hour",
    "temp",
    "humidity",
    "wind",
    "visibility",
    "dew_pt_temp",
    "solar_rad",
    "rainfall",
    "snowfall",
    "seasons",
    "holiday",
    "functioning_day",
]
Y_LABEL = "bike_count"
df = pd.read_csv(FILENAME)

# If we want to drop columns from the CSV file
# DROP_COLS = ["Date", "Holiday", "Seasons"]
# df = pd.read_csv("datafile.csv").drop(DROP_COLUMNS, axis=1)

df.dataset_columns = DATASET_COLS

# To transform a column to something numeric.
# In this case, we are transforming values for "functional" to 1 (for 'Yes' values)
# or  0 (for 'No' values)
df["functional"] = (df["functional"] == "Yes").astype(int)

# To peek at our dataset
df.head()

# This limits our dataset to only data during a specific hour. This is what was done
# in the video but for our actual model, we _may_ want to use the hour as an additional
#  feature when training our model.
df = df[df["hour"] == 12]
df = df.drop(["hour"], axis=1)

# Explore our dataset by plotting them
for label in df.columns[
    1:
]:  # All the features / all columns except the first one (which is the target/"y-label"/ bike count)
    # To explore how a specific label/feature affect the bike count
    plt.scatter(x_axis=df[label], y_axis=df[Y_LABEL])  # TODO CONFIRM KEYWORD ARGUMENTS
    plt.title(label)
    plt.y_label("Bike Count at Noon")
    plt.x_label(label)
    plt.show()

# Get rid of the features/labels that do not seem to matter.
# For the actual model, play around with _including_ these features.
df = df.drop(["wind", "visibility", "functional"], axis=1)

# Split our data to 3 data sets: training (60%) / validation (20%) / test (20%)
# We use df.sample to ensure that the data included in the data sets are randomized
train, val, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


def get_xy(dataframe, y_label, x_labels=None):
    """Get x (features) data and y (target) from the dataframe."""
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None:
        selected_features = [c for c in dataframe.columns if c != y_label]
        X = dataframe[selected_features].values
    elif len(x_labels) == 1:
        # TODO it is possible we can simplify this? to just say
        # selected_features = x_labels. Check whether it will result to the same
        # shape as the code we currently have.
        X = dataframe[x_labels[0]].values.reshape(
            -1, 1
        )  # The reshape is to make it to a 2d array
    else:
        selected_features = x_labels
        X = dataframe[selected_features].values

    y = dataframe[y_label].values.reshape(-1, 1)  # The reshape is to make it a 2d array
    data = np.hstack((X, y))
    return data, X, y


# Get the x (features) and y (target) values from the 3 data sets: train, validation, test
_, X_train_temp, y_train_temp = get_xy(train, Y_LABEL, x_labels=["temp"])
_, X_val_temp, y_val_temp = get_xy(val, Y_LABEL, x_labels=["temp"])
_, X_test_temp, y_test_temp = get_xy(test, Y_LABEL, x_labels=["temp"])

# Train a LinearRegression model
temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)

# Output the coefficient and intercept. These are the constants that resulted from
# the training.
print(temp_reg.coef_, temp_reg.intercept_)

# Output the r^2 score; The higher the number, the higher the correlation between
# the feature and the target.
# TODO In the actual model, try doing this for every feature that's available.
temp_reg.score(X_test_temp, y_test_temp)

# Plot the results of the model
plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
# The temp(erature) data goes from -20 to 40; Get 100 values from that range.
# TODO you can verify that temperature data is within the range [-20, 40] by getting
# min and max of the temp values.
x = tf.linspace(-20, 40, 100)
# Turn the data into an array and reshape it to a 2d array.
x = np.array(x).reshape(-1, 1)
plt.plot(x, temp_reg.predict(x), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()

# Stopped at 2:52
