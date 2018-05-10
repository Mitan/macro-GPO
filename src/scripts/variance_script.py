from src.DatasetUtils import *


my_save_folder_root = '../../releaseTests/updated_release/road/b5-18-log/'



start = 0
end = 34
method = 'anytime_h4'
batch_size = 5
time_steps = 20 / batch_size
all_measurements = np.zeros((end - start, time_steps+ 1))

for seed in range(start, end):
    seed_folder = my_save_folder_root + 'seed' + str(seed) + '/'
    measurements = GetAllMeasurements(root_folder=seed_folder, method_name=method, batch_size=batch_size)
    # print(measurements)
    acc_rewards  = GetAccumulatedRewards(measurements=measurements, batch_size=batch_size)
    regrets  = GetMaxValues(measurements=measurements, batch_size=batch_size)
    all_measurements[seed, : ] = regrets
    # print(acc_rewards)
print all_measurements
# print(all_measurements)
mean = np.mean(all_measurements, axis=0)
vars = np.std(all_measurements, axis=0) / np.sqrt(end - start)
print vars
all_measurements = all_measurements - mean
# print(all_measurements)
# print(np.mean(all_measurements, axis=0))
print()
square = np.square(all_measurements)
# print(square)


