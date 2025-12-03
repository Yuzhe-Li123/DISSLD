import json
import numpy as np
import sys
import os
path = sys.path[0]


def nest_tonumpy(obj):
    assert isinstance(obj, (list, tuple))
    re = []
    for item in obj:
        if isinstance(item, (list, tuple)):
            re.append(np.array(item))
        else:
            re.append(item)
    return re


def parse_json(path):
    with open(path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    params_head = json_data[0]
    metrics_list = json_data[1:]
    for i, m in enumerate(metrics_list):
        # m: dict
        for k, v in m.items():
            if isinstance(v, (list, tuple)):
                m.update({
                    k: nest_tonumpy(v)
                })
    return params_head, metrics_list


def fectch_json_list(path):
    files = os.listdir(path)
    target_file = []
    for f in files:
        if os.path.splitext(f)[1] in ['.json']:
            target_file.append(path + '/' + f)
    return target_file


if __name__ == '__main__':
    # json_dic_path = './results/Caltech/Caltech_100_10_110_aux'
    # json_dic_path = './results/Handwritten/Handwritten_0views_ae_1.0lc_10nb_50epochs_aux'
    # json_dic_path = './results/BDGP/BDGP_10_10_20_aux'
    # json_dic_path = './results/Reuters/Reuters_3views_ae_0.1lc_10nb_20epochs_aux'
    # json_dic_path = './results/BDGP/BDGP_2v_1.0lc_30nb_0.01gm_20eps_aux'
    # json_dic_path = './results/Handwritten/Handwritten_0v_1.0lc_10nb_1000gm_50eps_aux'
    # json_dic_path = './results/Caltech101_20/Caltech101_20_0v_1.0lc_10nb_0.01gm_110eps_aux'
    json_dic_path = './results/Handwritten/Handwritten_0v_1.0lc_10nb_0.01gm_50eps_aux'
    # json_dic_path = './results/Handwritten/Handwritten_6views_ae_100_10_55_aux'
    # json_dic_path = './results/Scene15/Scene15_2v_0.7lc_10nb_0.01gm_70eps_aux'
    # json_dic_path = './results/ALOI100/ALOI100_100_10_55_aux'
    # json_dic_path = './results/YouTube_X/YouTube_X_10_10_50_aux'
    files = fectch_json_list(json_dic_path)

    for f in files:
        params_head, metrics_list = parse_json(f)

        metrics = []
        metrics_aligned = []
        metrics_unaligned = []
        exe_time = 0
        cars = []
        for rt in metrics_list:
            m = rt['View-leval'][-1]
            metrics.append(m)

            m_aligned = rt['Aligned'][-1]
            metrics_aligned.append(m_aligned)

            m_unaligned = rt['Unaligned'][-1]
            metrics_unaligned.append(m_unaligned)

            exe_time += rt['execution_time']

            car = rt['CAR'][-1]
            cars.append(car)

        metrics = np.array(metrics)
        metrics_aligned = np.array(metrics_aligned)
        metrics_unaligned = np.array(metrics_unaligned)
        cars = np.array(cars)

        var_metrics = np.mean(metrics, axis=1)
        avg_var_metrics = np.mean(var_metrics, axis=0)
        std_var_metrics = np.std(var_metrics, axis=0)

        var_metrics_aligned = np.mean(metrics_aligned, axis=1)
        avg_var_metrics_aligned = np.mean(var_metrics_aligned, axis=0)
        std_var_metrics_aligned = np.std(var_metrics_aligned, axis=0)

        var_metrics_unaligned = np.mean(metrics_unaligned, axis=1)
        avg_var_metrics_unaligned = np.mean(var_metrics_unaligned, axis=0)
        std_var_metrics_unaligned = np.std(var_metrics_unaligned, axis=0)

        # cars[]
        avg_car = np.mean(cars)
        std_car = np.std(cars)

        exe_time /= float(len(metrics))
        print('=============')
        print('=============')
        print(f'{f} performance for each view:\n {np.mean(metrics, axis=0)};\n'
              f'avg: {avg_var_metrics};\n'
              f'std: {std_var_metrics};\n'
              f'execution time:\n {exe_time};')
        print(f'CAR:\n {avg_car}; std: {std_car};')
        print(f'performance aligned:\n {var_metrics_aligned}; avg: {avg_var_metrics_aligned};')
        print(f'performance unaligned:\n {var_metrics_unaligned}; avg: {avg_var_metrics_unaligned};')









