import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

DELTA_R = (0.001907 * 32 * 10**-9 * 299792458) / 2 #距離間隔
DELTA_T = 40 * 10**-3 #時間間隔
INIT_DIS = 1.5 #測定初期距離

def csv2npdata(filename):
    df = pd.read_csv(filename, header = None, skiprows = 3)
    df = df.loc[:, 16: ]
    data = df.values.tolist()
    data = np.array(data)
    print("data_shape (row, col) : {}".format(data.shape) )
    data_num, data_samplenum = data.shape

    return data, data_num, data_samplenum

def clutter_data_extraction():
    data, data_num, data_samplenum = csv2npdata("./20171211/clutter/test20171211002.csv")

    x_range= []
    for i in range(data_samplenum):
        x_range.append(i * DELTA_R)

    clutter_data = np.average(data, axis=0)

    return x_range, clutter_data

def rader_spectrogram(filepath, dis_range, clutter_data):
    filename, ext = os.path.splitext( os.path.basename(filepath) )
    data, data_num, data_samplenum = csv2npdata(filepath)

    """元データスペクトログラム表示"""
    abs_data = np.absolute(data)
    plt.figure()
    plt.imshow(abs_data.T, extent=[0, data_num * DELTA_T, 0, data_samplenum * DELTA_R], aspect="auto")
    plt.title("{}".format(filename + ext))
    plt.xlabel("times[s]")
    plt.ylabel("distance[m]")
    plt.colorbar()
    plt.savefig("./result/spec_{}.png".format(filename) )
    plt.show()

    """clutter削除"""
    sub_data = data - clutter_data
    sub_abs_data = np.absolute(sub_data)
    plt.figure()
    plt.imshow(sub_abs_data.T, extent=[0, data_num * DELTA_T, 0, data_samplenum * DELTA_R], aspect="auto")
    plt.title("del_clutter{}".format(filename + ext))
    plt.xlabel("times[s]")
    plt.ylabel("distance[m]")
    plt.colorbar()
    plt.savefig("./result/del_clutter_spec_{}.png".format(filename) )
    plt.show()

def main():
    print("delta_r[m] : {}\ndelta_t[s] : {}\ninitial_distance[m] : {}".format(DELTA_R, DELTA_T, INIT_DIS) )

    dis_range, clutter_data = clutter_data_extraction()

    file_list = glob.glob("./20171211/*.csv")
    for filepath in file_list:
        rader_spectrogram(filepath, dis_range, clutter_data)

if __name__=="__main__":
    main()
