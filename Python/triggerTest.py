import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import binned_statistic


def findLevel(levels):
    lvs = levels.copy()
    
    tot_lv = np.argmax(lvs)
    back_lv = np.argmin(lvs)
    
    print(lvs, tot_lv, back_lv)

def main(fn):
    dat = sio.loadmat(f"datafiles\{fn}")['data0']

    atol = 1e-3
    normDat = dat[3650:4600].flatten()/max(dat)

    level = (np.isclose(np.diff(normDat), 0, atol=atol))

    levelDat = level * normDat[:-1]
    levelDat = levelDat[levelDat != 0]
    hist = np.histogram(levelDat, bins=int(1/atol * 0.25))
    
    test = binned_statistic(levelDat, levelDat, bins=int(1/atol), statistic='sum')
    
    levels = np.nonzero(hist[0])[0]
    testLevels = np.nonzero(test[0])[0]
    tempLevel = 0
    i = 0
    
    while i < len(testLevels):        
        curLevel = testLevels[i]     
        prevLevel = testLevels[i+1]
        
        i += 1
        
        
        
    # levels = hist[0][hist[levels] > 25]
    print(levels)
    
    dataLevels = []
    for i in range(len(levels)):
        points = hist[0][i]
        if points < len(normDat) * 0.01:
            continue
        
        lower, upper = hist[1][i], hist[1][i+1]

        mask =  (lower <= levelDat) & (levelDat <= upper)
        interesting = levelDat[mask]
        dataLevels.append(np.mean(interesting))
    
    findLevel(levels)
    # for i, le in enumerate(dataLevels):
        # print(len(le), np.mean(le), np.std(le))
    # TODO: identify which level is which in levels

    # plt.plot(normDat, label="Normalized")
    # plt.plot(levelDat, label="Leveled data")
    # plt.plot(level + 0.1, alpha=0.3, label="Level detection")

    # plt.plot(le, label=f"interesting {i}")
    
    # plt.plot(np.mean(le))

    # plt.figure()
    # plt.hist(levelDat, bins=300)



if __name__ == "__main__":
    files = glob.glob(r"C:\Users\Ueli\Desktop\Semesterarbeit\python\datafiles\*.mat")
    for f in files:
        name = f.split("\\")[-1]
        print(name)
        main(name)
        break
    # plt.legend()
    plt.show()
    