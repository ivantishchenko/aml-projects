import csv
import numpy as np


def __get_predictions():
    predictions_list = []
    with open('test.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            feature_list = []
            for i in range(1, 11):
                feature_list.append(float(row['x' + str(i)]))
            feature_vec = np.array(feature_list)
            predictions_list.append((row['Id'], feature_vec.mean()))
        return predictions_list

def __write_predictions(predictions_list):
    with open('out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
        writer.writerow(['Id', 'y'])
        for prediction in predictions_list:
            writer.writerow([prediction[0], prediction[1]])

out = __get_predictions()
__write_predictions(out)
