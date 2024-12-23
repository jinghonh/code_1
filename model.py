import os
import pandas as pd
import numpy as np
from CORAL import CORAL
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import copy
from scipy import signal

def check(func):
    def inner(src_data, src_label, tar_data, tar_lable, test_tar_data):
        if len(src_label.shape) == 2:
            src_label = src_label.ravel()
        if len(tar_lable.shape) == 2:
            tar_lable = tar_lable.ravel()
        return func(src_data, src_label, tar_data, tar_lable, test_tar_data)
    return inner

def time_lags(data,lags):
    len_data = len(data)   #67
    years = int(len_data/12)
    remain = len_data - years*12
    data_1 = data[:len_data-remain].reshape(-1,12)   #  (60,)  (5,12)
    data_2 = data[-remain:]     #(7,)
    data_11 = data_1[:,:lags]
    data_12 = data_1[:,lags:]
    data_concat = np.concatenate((data_11,data_12),axis=1)   #(5,12)
    if remain !=0:
        data  = np.concatenate((data_concat.reshape(-1),data_2),axis=0)
    else:
        data = data_concat.reshape(-1)
    return data



def cross_correlation(target_data,move_data,lags):
    if not isinstance(target_data,np.ndarray):
        target_data = np.array(target_data)
    if not isinstance(move_data,np.ndarray):
        move_data = np.array(move_data)
    index = target_data != None

    move_data_c = move_data[index]
    target_data_c =  target_data[index]

    target_data_mean = np.mean(target_data_c)
    move_data_mean = np.mean(move_data_c)
    RXY = np.sqrt(np.sum((target_data_c-target_data_mean)**2))*np.sqrt(np.sum((move_data_c-move_data_mean)**2))

    cross_correlation_ = lambda x,y:np.sum((x-target_data_mean)*(y-move_data_mean))/RXY   #这里
    if lags ==0:
        return cross_correlation_(target_data_c,move_data_c)
    # target_data_c = target_data_c[lags:]
    # move_data_c = move_data_c[:-lags]
    move_data_c = time_lags(move_data_c,lags)

    index = target_data_c != None
    return cross_correlation_(target_data_c[index],move_data_c[index])

def data_align(x,y,maxlag=6):#x是污染物数据,y是大气数据
    lag = 0
    value = 0
    N = x.shape[0]   #68
    # y_reshaped = y.reshape(-1, 12).T   #(7,12) (12,7)
    for lags in range(maxlag+1):
        cc  = cross_correlation(target_data=x,move_data=y[-N:],lags=lags)
        if abs(cc)>value:
            value = abs(cc)   
            lag = lags
    if lag==0:
        return x,y

    y = time_lags(y,lag)
    # for i in range(y_reshaped.shape[1]):
    #     temp = y_reshaped[:, i][-lag:]
    #     y_reshaped[:, i][lag:] = y_reshaped[:, i][:-lag]
    #     y_reshaped[:, i][:lag] = temp
    # y = y_reshaped.flatten()
    return x,y

def Butterworth_filter(data,low,high,fs=0.5):
    low_,high_ = low/fs,high/fs
    mean_ = np.mean(data)
    b, a = signal.butter(N=2, Wn=[low_, high_], btype='band')
    filtered_signal = mean_ + signal.filtfilt(b, a, data)
    return filtered_signal,data-filtered_signal

@check
def TwostageTrAdaboostR2_model(src_data, src_label, tar_data, tar_lable, test_tar_data):
    ##########################TwostageTrAdaboostR2#############################
    coral = CORAL()
    src_data = coral.fit(src_data, np.concatenate((tar_data, test_tar_data)))
    TwostageTrAdaboost = TwoStageTrAdaBoostR2(base_estimator=RandomForestRegressor(n_estimators=20),
                                              sample_size=[src_data.shape[0], tar_data.shape[0]],
                                              n_estimators=10,
                                              steps=5,
                                              fold=8,
                                              learning_rate=0.7,
                                              loss="exponential", )
    cat_data = np.concatenate((src_data, tar_data), axis=0)
    cat_label = np.concatenate((src_label, tar_lable), axis=0)
    TwostageTrAdaboost_model = TwostageTrAdaboost.fit(cat_data, cat_label)
    pred_cat_data = TwostageTrAdaboost_model.predict(cat_data)
    test_tar_y = TwostageTrAdaboost_model.predict(test_tar_data)
    return pred_cat_data,test_tar_y

def run(target_input_data,target_output_data,source_input_data,source_output_data,save_path,fs=0.5,low=1/16,high=1/8):
    """
    :param target_input_data: 目标域输入数据 (样本数m*特征数量)
    :param target_output_data: 目标域输出数据 (样本数n,) n<m m-n的部分为测试集
    :param source_input_data: 源域输入数据 (样本数k*特征数量)
    :param source_output_data: 源域输出数据 (样本数k,)
    :param save_path:
    :param fs: 采样频率
    :param low: 低频阈值
    :param high: 高频阈值
    """
    f = lambda x,y:(x-np.min(x))/(np.max(x)-np.min(x))*(np.max(y)-np.min(y))+np.min(y)

    source_output_data = f(source_output_data,target_output_data)
    target_input_data = f(target_input_data,target_output_data)
    source_input_data = f(source_input_data,target_output_data)

    N = len(target_output_data)   #68

    all_data = [target_input_data,target_output_data,source_input_data,source_output_data]

    for idx,data in enumerate(all_data):
        if len(data.shape) > 1:
            input_p,input_ap = [],[]
            for d in data.T:
                input_p_d,input_ap_d = Butterworth_filter(data=d, low=low, high=high, fs=fs)
                input_p.append(input_p_d)
                input_ap.append(input_ap_d)
        else:
            output_p, output_ap = Butterworth_filter(data=data, low=low, high=high, fs=fs)

        if idx == 1:
            input_p = np.array(input_p).T
            aligned_input_p=[]
            for feature in input_p.T:
                y , input_p_a = data_align(output_p, feature)
                aligned_input_p.append(input_p_a)
            aligned_input_p = np.array(aligned_input_p)
            input_p=aligned_input_p.T
            # output_p,input_p = data_align(output_p,input_p,maxlag=6)
            gridsearch = GridSearchCV(RandomForestRegressor(),param_grid={"n_estimators":[10,20,50,100]})
            # gridsearch.fit(input_p[:N],output_p)  
            gridsearch.fit(input_p[-N:],output_p) 
            best_model = gridsearch.best_estimator_
            # input_p_predict =best_model.predict(input_p[N:])
            input_p_predict =best_model.predict(input_p[:-N])
            del input_p,output_p
            target_input_ap = np.array(copy.deepcopy(input_ap)).T
            target_output_ap = np.array(copy.deepcopy(output_ap))
            del input_ap,output_ap

        if idx == 3:
            source_input_ap = np.array(input_ap).T
            source_output_ap = np.array(output_ap)
            _,input_ap_predict = TwostageTrAdaboostR2_model(source_input_ap, source_output_ap, target_input_ap[-N:], target_output_ap, target_input_ap[:-N])
            del target_input_ap,target_output_ap

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataframe = pd.DataFrame({"周期预测值":input_p_predict,"非周期预测值":input_ap_predict,"预测值":input_p_predict+input_ap_predict})
    dataframe.to_csv(os.path.join(save_path,"predict.csv"),index=False)

    return input_p_predict,input_ap_predict

if __name__ == "__main__":
    target_input_data,target_output_data,source_input_data,source_output_data = ...,...,...,...
    save_path = ...

    run(target_input_data,target_output_data,source_input_data,source_output_data,save_path,fs=0.5,low=1/16,high=1/8)

    # 随便写一点东西，用于测试
