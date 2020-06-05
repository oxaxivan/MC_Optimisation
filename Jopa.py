from scipy.optimize import minimize
import numpy as np
from scipy.optimize import Bounds
import os


class Step:
    step_number = 0


def exe_runner():
    os.chdir(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo')
    os.system(r'"LightTissueIntercation.exe"')


def parameters_writer(param):
    abskwrd = 'ABSKWRD'
    scatkwrd = 'SCATKWRD'
    aniskwrd = 'ANISKWRD'

    f0 = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\settings0.txt', 'r')
    f0.seek(0, 0)
    text = f0.read()
    f0.close()

    text = text.replace(abskwrd, str(param[0]))
    text = text.replace(scatkwrd, str(param[1]))
    text = text.replace(aniskwrd, str(param[2]))

    f = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo\settings.txt', 'w')
    f.write(text)
    f.close()


def simul_result():
    s = np.zeros([2, 200])
    file = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo\output.txt')
    kwrd1 = 'For distance '
    len1 = len(kwrd1)
    kwrd2 = 'Collimated transmittance    : '
    len2 = len(kwrd2)
    kwrd3 = 'Diffusive  transmittance    : '
    len3 = len(kwrd3)
    while 1:
        # print('Seeking' + line)
        tmp = file.readline()  # troubles expected
        if 'Absorbed' in tmp:
            break
    i = 0
    # print('Start encoding \n')
    for line in file:
        if kwrd1 in line:
            s[0, i] = float(line[len1:])
            # print('Found Distance')
        if kwrd2 in line:
            s[1, i] = float(line[len2:-2])
            # print('Found CT')
        if kwrd3 in line:
            s[1, i] += float(line[len3:-2])
            # print('Found DT')
            i += 1
    file.close()
    return s


def interpolation(fd, ed):
    xp = fd[0, :]
    fp = fd[1, :]
    inter = [np.interp(ed[0], xp, fp)]
    return np.concatenate((ed, inter), axis=0)


def eval_nevyazka(fd, ed):
    arr = interpolation(fd, ed)
    return np.sum((arr[1]-arr[2]) ** 2)


def exp_data():
    file = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo\ExpData.txt')
    fulltext = file.read()
    npoints = len(fulltext.split('\n'))
    arr = np.zeros((2, npoints))
    file.seek(0, 0)
    i = 0
    for line in file:
        tmp = line.split()
        arr[0, i] = float(tmp[0])
        arr[1, i] = float(tmp[1])
        i += 1
    return arr


# exp_data = exp_data();

def nevyazka(param):
    parameters_writer(param)
    os.chdir(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo')
    os.system(r'"LightTissueIntercation.exe"')
    # exe_runner()
    fd = simul_result()
    ed = exp_data()
    tmp = eval_nevyazka(fd, ed)
    print('Step:', step.step_number, '   Nevyazka=', tmp, '\n')
    step.step_number += 1
    return tmp


def find_solution():
    start_point = np.array([4., 3600., 0.78])
    bounds = Bounds([0, 0, 0], [np.inf, np.inf, 1])
    return minimize(nevyazka, start_point, method='trust-constr', bounds=bounds, options={'disp': True})


step = Step()
print(find_solution())
