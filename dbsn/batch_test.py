import shutil
import os
import subprocess
import numpy as np

type = "attack" #"cal_ent_svhn" #"test",

if type == "test":
    test_dirs = ["random", "ds", "ds_dp0.2", "ds_dpth0.3", "ps", "darts", "adags_lr3_con1_1gpu", "adags_lr3_con1", "adags_lr3_decayto0.5_1gpu", "adags_lr3_decayto0.5"] #cifar10
    #["random_cifar100", "ds_cifar100", "ds_dp0.2_cifar100", "ds_dpth0.3_cifar100", "ps_cifar100", "darts_cifar100",  "adags_lr3_con1_cifar100_1gpu", "adags_lr3_con1_cifar100", "adags_lr3_d205_cifar100_1gpu", "adags_lr3_d205_cifar100",] #cifar100
    dataset = 'cifar10' #'cifar100'
    gpu_map = [0,1,2]

    for dir in test_dirs:
        processes = []
        for i in range(0,3):
            #print(os.getcwd())
            dst_file = '../work/run' + dir + '_' + str(i) + '/scripts/test.py'
            shutil.copyfile('test.py', dst_file)
            os.chdir('../work/run' + dir + '_' + str(i) + '/scripts')
            #print(os.getcwd())
            if "random" in dir:
                method = "random"
            elif "ds" in dir:
                method = "densenet"
            else:
                method = "dbsn"
            restore_dir = "../epoch100.pth"
            drop_rate = 0.2 if "0.2" in dir else 0.
            droppath_rate = 0.3 if "0.3" in dir else 0.
            ifps = "--ps" if "ps" in dir or "darts" in dir else ""
            command = 'CUDA_VISIBLE_DEVICES={} python -u test.py --dataset {} --restore {} --method {} {} --drop_rate {} --droppath_rate {}'.format(gpu_map[i], dataset, restore_dir, method, ifps, drop_rate, droppath_rate)
            print(command)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            processes.append(process)
            os.chdir('../../../dbsn')
        output = [p.wait() for p in processes]
        output = [p.stdout.read() for p in processes]
        print(output)
elif type == "cal_ent_svhn":
    dataset = "cifar10"
    test_dirs = ["random", "ds", "ds_dp0.2", "ds_dpth0.3", "ps", "darts", "adags_lr3_con1", "adags_lr3_decayto0.5" if dataset == "cifar10" else "adags_lr3_d205", ]
    gpu_map = [0,1,2,3,4,5,6,7]
    suffix = "" if dataset == "cifar10" else "_" + dataset

    processes = []
    for i, dir in enumerate(test_dirs):
        dst_file = '../work/run' + dir + suffix + '_0' + '/scripts/test.py'
        shutil.copyfile('test.py', dst_file)
        os.chdir('../work/run' + dir + suffix + '_0' + '/scripts')
        #print(os.getcwd())
        if "random" in dir:
            method = "random"
        elif "ds" in dir:
            method = "densenet"
        else:
            method = "dbsn"
        restore_dir = "../epoch100.pth"
        drop_rate = 0.2 if "0.2" in dir else 0.
        droppath_rate = 0.3 if "0.3" in dir else 0.
        ifps = "--ps" if "ps" in dir or "darts" in dir else ""
        command = 'CUDA_VISIBLE_DEVICES={} python -u test.py --dataset {} --restore {} --method {} {} --drop_rate {} --droppath_rate {} --ent'.format(gpu_map[i], dataset, restore_dir, method, ifps, drop_rate, droppath_rate)
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append(process)
        os.chdir('../../../dbsn')
    output = [p.wait() for p in processes]
    output = [p.stdout.read() for p in processes]
    print(output)
elif type == "attack":
    dataset = "cifar10"
    test_dirs = ["ds","ps", "darts","adags_lr3_decayto0.5" if dataset == "cifar10" else "adags_lr3_d205","random", "ds_dp0.2", "ds_dpth0.3", "adags_lr3_con1",]#
    gpu_map = [0,1,2,3,4,5,6,7]
    suffix = "" if dataset == "cifar10" else "_" + dataset

    processes = []
    for i, dir in enumerate(test_dirs):
        dst_file = '../work/run' + dir + suffix + '_0' + '/scripts/attack.py'
        shutil.copyfile('attack.py', dst_file)
        os.chdir('../work/run' + dir + suffix + '_0' + '/scripts')
        #print(os.getcwd())
        if "random" in dir:
            method = "random"
        elif "ds" in dir:
            method = "densenet"
        else:
            method = "dbsn"
        adv_method = "fgsm"
        restore_dir = "../epoch100.pth"
        drop_rate = 0.2 if "0.2" in dir else 0.
        droppath_rate = 0.3 if "0.3" in dir else 0.
        batchSz = 90 if method == "dbsn" else 100
        ifps = "--ps" if "ps" in dir or "darts" in dir else ""
        command = 'CUDA_VISIBLE_DEVICES={} python -u attack.py --dataset {} --restore {} --method {} {} --drop_rate {} --droppath_rate {} --batchSz {} --adv_method {}'.format(gpu_map[i], dataset, restore_dir, method, ifps, drop_rate, droppath_rate, batchSz, adv_method)
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append(process)
        os.chdir('../../../dbsn')
    output = [p.wait() for p in processes]
    output = [p.stdout.read() for p in processes]
    print(output)
