import matplotlib.pyplot as plt
import numpy as np
import os

def plot_normalized_data():
    blur_gt_dir = 'groundtruth/'
    tracker_dir = 'data/'
    gts = {}
    blur_seqs_data = {}

    for seq_name in os.listdir(blur_gt_dir):
        path = os.path.join(blur_gt_dir, seq_name, seq_name + '.txt')
        with open(path, 'r') as f:
            lines = np.array(list(map(lambda x: np.array(x[:-1].split(',')).astype('int'), f.readlines())))
            gts[seq_name] = lines

    errors = {}
    for tracker_name in os.listdir(tracker_dir):
        tracker_data = {}
        error_data = {}
        for seq_name in os.listdir(blur_gt_dir):
            path = os.path.join(tracker_dir, tracker_name, seq_name + '.txt')
            dists = []

            with open(path, 'r') as f:
                lines = np.array(list(map(lambda x: np.array(x[:-1].split(',')).astype('int'), f.readlines())))

                for i, line in enumerate(lines):
                    tx, ty, tw, th = line[:]
                    gx, gy, gw, gh = gts[seq_name][i][:]

                    ctx, cty = tx + tw / 2, ty + th / 2
                    cgx, cgy = gx + gw / 2, gy + gh / 2
                    dist = np.sqrt((ctx - cgx) ** 2 + (cty - cgy) ** 2) / np.max((gw, gh))

                    dists.append(dist)

            error_data[seq_name] = np.round(np.average(dists), 2)
            tracker_data[seq_name] = dists

        errors[tracker_name] = error_data
        blur_seqs_data[tracker_name] = tracker_data

    for seq_name in os.listdir(blur_gt_dir):
        plt.xlabel('#' + seq_name)
        plt.ylabel('Error')
        plt.plot(blur_seqs_data['MKCFup'][seq_name], label='ours', color='indigo', linestyle='-')
        plt.plot(blur_seqs_data['BACF'][seq_name], label='bacf', color='black', linestyle='--')
        plt.plot(blur_seqs_data['CSRT'][seq_name], label='csrt', color='chocolate', linestyle='--')
        plt.plot(blur_seqs_data['DAT'][seq_name], label='dat', color='darkorange', linestyle='--')
        plt.plot(blur_seqs_data['KCF'][seq_name], label='kcf', color='dodgerblue', linestyle='-')
        plt.plot(blur_seqs_data['MCCTH-Staple'][seq_name], label='mccth', color='yellow', linestyle='-.')
        plt.plot(blur_seqs_data['Staple'][seq_name], label='staple', color='cyan', linestyle='-.')
        plt.plot(blur_seqs_data['STRCF'][seq_name], label='strcf', color='gold', linestyle='-')

        plt.legend()
        plt.show()

    for key in errors:
        print(key, errors[key], np.average(list(errors[key].values())))

if __name__ == '__main__':
    plot_normalized_data()
