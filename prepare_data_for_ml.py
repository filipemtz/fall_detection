
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_data(path):
    print("Loading data...")
    with open(path, "r") as f:
        data = f.readlines()
    return np.array([l.strip().split(";") for l in data[2:]])


def select_relevant_data(data):
    print("\tSelecting relevant data.")
    # ##################################################
    # discard brain and luminosity sensors
    # ##################################################
    acelerometers_indices = [1, 2, 3]  # ankle acelerometer
    acelerometers_indices += [8, 9, 10]  # right pocket acelerometer
    acelerometers_indices += [15, 16, 17]  # belt acelerometer
    acelerometers_indices += [22, 23, 24]  # neck acelerometer
    acelerometers_indices += [29, 30, 31]  # wrist acelerometer

    angular_velocities_indices = [4, 5, 6]  # ankle angular velocity
    angular_velocities_indices += [11, 12, 13]  # right pocket angular velocity
    angular_velocities_indices += [18, 19, 20]  # belt angular velocity
    angular_velocities_indices += [25, 26, 27]  # neck angular velocity
    angular_velocities_indices += [32, 33, 34]  # wrist angular velocity

    meta_info_indices = [43, 44, 45]  # subject activity trial

    acelerometers = data[:, acelerometers_indices]
    ang_velocities = data[:, angular_velocities_indices]
    meta_info = data[:, meta_info_indices]

    return acelerometers, ang_velocities, meta_info


def fill_missing_values_and_remove_double_points(data, fill_value):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == '':
                data[i, j] = fill_value

            if data[i, j].count(".") > 1:
                idx_first = data[i, j].find(".")
                data[i, j] = data[i, j][:idx_first] + data[i, j][idx_first+1:]


def normalize(data):
    mins = np.min(data, axis=0)
    data -= mins
    maxs = np.max(data, axis=0)
    data /= maxs
    means = np.mean(data, axis=0)
    data -= means
    return data


def prepare_data(data):
    print("Preparing data...")

    acelerometers, ang_velocities, meta_info = select_relevant_data(data)

    print("\tFilling missing values and removing double points.")
    fill_missing_values_and_remove_double_points(acelerometers, '0')
    fill_missing_values_and_remove_double_points(ang_velocities, '0')

    print("\tConverting data to proper types.")
    acelerometers = acelerometers.astype(np.float64)
    ang_velocities = ang_velocities.astype(np.float64)
    meta_info = meta_info.astype(np.int32)

    # print(f"acelerometers.shape: {acelerometers.shape}")
    # print(f"ang_velocities.shape: {ang_velocities.shape}")
    # print(f"meta_info.shape: {meta_info.shape}")

    print("\tNormalizing data.")
    acelerometers = normalize(acelerometers)
    ang_velocities = normalize(ang_velocities)

    return acelerometers, ang_velocities, meta_info


def plot_data(acelerometers, ang_velocities, meta_info):
    print("Plotting data...")
    plt.figure()
    for column in range(acelerometers.shape[1]):
        plt.plot(acelerometers[:, column])

    plt.figure()
    for column in range(ang_velocities.shape[1]):
        plt.plot(ang_velocities[:, column])

    plt.show()


def assemble_experiment_data(features, meta_info):
    print("Assembling Experiment Data.")
    experiment_data = {}
    for feature, meta in zip(features, meta_info):
        # subject activity trial
        subject_idx = meta[0]
        activity_idx = meta[1]
        trial_idx = meta[2]

        if subject_idx not in experiment_data:
            experiment_data[subject_idx] = {}

        if activity_idx not in experiment_data[subject_idx]:
            experiment_data[subject_idx][activity_idx] = {}

        if trial_idx not in experiment_data[subject_idx][activity_idx]:
            experiment_data[subject_idx][activity_idx][trial_idx] = []

        experiment_data[subject_idx][activity_idx][trial_idx].append(feature)
    return experiment_data


def save_experiment_data(experiment_data, path):
    print("Saving experiment data.")
    with open(path, "wb") as f:
        pickle.dump(experiment_data, f)


def main():
    data = load_data("CompleteDataSet.csv")
    acelerometers, ang_velocities, meta_info = prepare_data(data)
    #plot_data(acelerometers, ang_velocities, meta_info)
    features = np.concatenate([acelerometers, ang_velocities], axis=1)
    print(f"features.shape: {features.shape}")
    experiment_data = assemble_experiment_data(features, meta_info)
    save_experiment_data(experiment_data, "experiment_data.pkl")
    print("Done.")


if __name__ == "__main__":
    main()
