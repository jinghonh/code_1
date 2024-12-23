import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
# import matlab.engine
# import 处理空气质量 as air_quality
# import 处理大气因素 as atmosphere
from MMD import guassian_kernel
from CORAL import CORAL
from model import *
from rich import print
from rich.console import Console
from rich.table import Table
from tqdm import *
from wavelet import *

console = Console()
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['mathtext.default'] = 'regular'

cols = ['南北风速','东西风速','温度','紫外线','相对湿度','太阳辐射','降雨']

def get_target_domain_data(city='北京',root_path='./CityAirQualityMonthData.csv',col="NO2"):
    """
    读取城市的空气质量数据
        :param city: 城市名
        :param root_path: 文件路径
        :param col: 目标变量列名
        :return: 目标变量数据
    """
    # 判断文件路径是否存在
    if not os.path.exists(root_path):
        print("文件路径不存在")
        return
    
    start_date = '2015-01-01'
    end_date = '2021-12-31'

    air_quality_data = pd.read_csv(root_path)
    air_quality_data['日期'] = pd.to_datetime(air_quality_data['月份'], format='%Y-%m')

    air_quality_data_filtered = air_quality_data[(air_quality_data['日期'] >= start_date) & (air_quality_data['日期'] <= end_date)]
    cityData_airQuality = air_quality_data_filtered[air_quality_data_filtered['城市'] == city]
    target_domain_data = cityData_airQuality.loc[:, [col]].dropna().reset_index(drop=True)
    target_domain_datay = np.array(target_domain_data).T
    return target_domain_data

def get_origin_data(city='北京',root_path='./CityAtmosphereFactorMonthData.csv',start_date = '2015-01-01',end_date = '2021-12-31',cols = cols):
    """
    读取城市的大气因素数据
        :param city: 城市名
        :param root_path: 文件路径
        :return: 特征变量数据
    """
    # 判断文件路径是否存在
    if not os.path.exists(root_path):
        print("文件路径不存在")
        return
    atmosphere_data = pd.read_csv(root_path)
    atmosphere_data['日期'] = pd.to_datetime(atmosphere_data['年份'].astype(str) + '-' + atmosphere_data['月份'].astype(str), format='%Y-%m')

    atmosphere_data_filtered = atmosphere_data[(atmosphere_data['日期'] >= start_date) & (atmosphere_data['日期'] <= end_date)]
    cityData_atmosphere = atmosphere_data_filtered[atmosphere_data_filtered['城市'] == city]
    x = cityData_atmosphere.loc[:, cols].reset_index(drop=True)
    x = np.array(x).T
    return x

def output2screen(head,selection_data_name, data):
    """
    输出数据到屏幕
        :param selection_data_name: 数据名
        :param data: 数据
    """

    table = Table(title='[yellow]'+head,show_header=True,header_style="bold magenta")
    table.add_column("序号")
    table.add_column("数据名")
    table.add_column("数据")
    for i in range(len(selection_data_name)):
        # table.add_row('[blue]'+selection_data_name[i],'[red]'+str(data[i]))
        table.add_row(str(i+1),'[blue]'+selection_data_name[i],'[red]'+str(data[i]))
    console.print(table)

def maeoutput2screen(head,missing_rate, total_mae, mae_p, mae_ap):
    """
    输出数据到屏幕
        :param selection_data_name: 数据名
        :param data: 数据
    """

    table = Table(title='[yellow]'+head,show_header=True,header_style="bold magenta")
    table.add_column("序号")
    table.add_column("缺失率")
    table.add_column("MAE")
    table.add_column("MAE_P")
    table.add_column("MAE_AP")
    for i in range(len(missing_rate)):
        # table.add_row('[blue]'+selection_data_name[i],'[red]'+str(data[i]))
        table.add_row(str(i+1),str(missing_rate[i]),str(total_mae[i]),str(mae_p[i]),str(mae_ap[i]))
    console.print(table)


def data_add_index(data, start_date='2015-1'):
    """
    为数据添加索引
        :param data: 数据
        :param start_time: 起始时间
        :return: 添加索引后的数据
    """
    data = np.array(data)
    index = np.arange(data.shape[0])
    return np.c_[index, data]
# 计算MAE
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

if __name__ == "__main__":
    console.clear()
    y = pd.read_csv("./fig78.csv",encoding='gbk')['81'].values
    target_label = np.array(y).T.reshape(-1) #(n,)
    target_label_name = "NO2"
    selection_data = get_origin_data() #(m,n)
    selection_data_name = cols #(m,)
    selection_num = 3

    wtc_lunch = "python"
    
    f = lambda x, y: (x - min(x)) / (max(x) - min(x)) * (max(y) - min(y)) + min(y)
    #  计算MMD
    mmds = []
    for d in selection_data:
        d = f(d,target_label)
        mmd = guassian_kernel(d[:,np.newaxis],target_label[:,np.newaxis],fix_sigma=1)
        mmds.append(mmd)
    mmd_result = sorted(range(len(mmds)),key=lambda x:mmds[x])
    output2screen("目标域标签与相关数据的MMD距离:",selection_data_name,mmds)
    source_label = selection_data[mmd_result[0]]
    source_label_name = selection_data_name[mmd_result[0]]
    print("[green]选择的源域标签:",'[blue]'+source_label_name)
    if wtc_lunch == "matlab" or wtc_lunch == "all":
        # matlab.engine.shareEngine
        eng = matlab.engine.connect_matlab()
        eng.figure()
        Y = matlab.double(data_add_index(target_label).tolist())
        i = 1
        target_wtcs = []
        for x in selection_data:
            x = f(x,target_label)
            X = matlab.double(data_add_index(x).tolist())
            eng.subplot(matlab.double([2]),matlab.double([4]),matlab.double([i]))
            Rsq,period,scale,coi,wtcsig,t = eng.wtc(X,Y, nargout=6)
            eng.title("WTC: "+target_label_name+"-"+selection_data_name[i-1])
            target_wtcs.append(eng.mean(eng.mean(Rsq)))
            i += 1
        
        output2screen("目标域标签与相关数据的WTC距离:",selection_data_name,target_wtcs)
        # 去除最大的MMD
        selection_data = selection_data[mmd_result[1:]]
        selection_data_name = [selection_data_name[i] for i in mmd_result[1:]]
        target_wtcs = [target_wtcs[i] for i in mmd_result[1:]]
        Y = matlab.double(data_add_index(source_label).tolist())
        eng.figure()
        i = 1
        source_wtcs = []
        for x in selection_data:
            x = f(x,target_label)
            X = matlab.double(data_add_index(x).tolist())
            eng.subplot(matlab.double([2]),matlab.double([4]),matlab.double([i]))
            Rsq,period,scale,coi,wtcsig,t = eng.wtc(X,Y, nargout=6)
            eng.title("WTC: "+source_label_name+"-"+selection_data_name[i-1])
            source_wtcs.append(eng.mean(eng.mean(Rsq)))
            i += 1
        output2screen("源域标签与相关数据的WTC距离:",selection_data_name,source_wtcs)
    if wtc_lunch == "python" or wtc_lunch == "all":
        target_wtcs = []
        w = Wavelet(dt=1)
        for x in selection_data:
            x = f(x,target_label)
            a,b = w.wtc(x=x,y=target_label,mc = 20,wavelet= "cmor2-0.97", ifshow=False)
            target_wtcs.append(np.mean(a))
        output2screen("目标域标签与相关数据的WTC距离:",selection_data_name,target_wtcs)
        # 去除最大的MMD
        selection_data = selection_data[mmd_result[1:]]
        selection_data_name = [selection_data_name[i] for i in mmd_result[1:]]
        source_wtcs = []
        for x in selection_data:
            x = f(x,target_label)
            a,b = w.wtc(x=x,y=source_label,mc = 20,wavelet= "cmor2-0.97", ifshow=False)
            source_wtcs.append(np.mean(a))
        output2screen("源域标签与相关数据的WTC距离:",selection_data_name,source_wtcs)

#--------------------------------------------------------------------------------#
    console.rule("[bold red]数据分解", align='center')
    bf = Bf(W=[8,16])
    wdd = Wdd(level=2)

    # target_label = (target_label-min(target_label))/(max(target_label)-min(target_label))
    # del target_wtcs[mmd_result[0]]
    target_wtcs_sorted = sorted(range(len(target_wtcs)),key=lambda x:target_wtcs[x],reverse=True)
    source_wtcs_sorted = sorted(range(len(source_wtcs)),key=lambda x:source_wtcs[x],reverse=True)
    target_data_name = []
    source_data_name = []
    # exists = set()
    # # selection_num = 2
    # del target_wtcs_sorted[mmd_result[0]]
    for i in target_wtcs_sorted:
        if len(target_data_name) == selection_num:
            break
        if i == mmd_result[0]:
            continue
        current_feature = cols[i]
        
        if current_feature in ["太阳辐射", "紫外线"]:
            if "太阳辐射" in target_data_name and current_feature == "紫外线":
                continue
            elif "紫外线" in target_data_name and current_feature == "太阳辐射":
                continue
            else :
                target_data_name.append(current_feature)
        else:
            target_data_name.append(current_feature)
            
    for i in source_wtcs_sorted:
        if len(source_data_name) == selection_num:
            break
        # if i == mmd_result[0]:
        #     continue
        current_feature = selection_data_name[i]
        if current_feature in target_data_name:
            continue
        if current_feature in ["太阳辐射", "紫外线"]:
            if "太阳辐射" in source_data_name and current_feature == "紫外线":
                continue
            elif "紫外线" in source_data_name and current_feature == "太阳辐射":
                continue
            else :
                source_data_name.append(current_feature)
        else:
            source_data_name.append(current_feature)

    target_data = get_origin_data(cols=target_data_name,start_date='2015-1-1')
    source_data = get_origin_data(cols=source_data_name,start_date='2015-1-1')
    source_label = get_origin_data(cols=[source_label_name],start_date='2015-1-1')
    source_label = f(np.array(source_label).T.reshape(-1),target_label)   #从大气数据里提取出对应特征的相关数据。

    # decompose_target_label = bf.run(target_label)
    # decompose_source_label = bf.run(source_label)

    # target_label_non_periodic = np.array(decompose_target_label.iloc[:,1])
    # source_label_non_periodic = np.array(decompose_source_label.iloc[:,1])

    # target_data_non_periodic = np.zeros((target_data.shape[1],selection_num))
    # source_data_non_periodic = np.zeros((source_data.shape[1],selection_num))

    # for i in range(selection_num):
    #     decompose_target_data = bf.run(f(target_data[i],target_label))
    #     decompose_source_data = bf.run(f(source_data[i],target_label))
    #     target_data_non_periodic[:,i] = np.array(decompose_target_data.iloc[:,1])
    #     source_data_non_periodic[:,i] = np.array(decompose_source_data.iloc[:,1])

    # output2screen("目标域数据选择与分解结果:", target_data_name, target_data_non_periodic)
    # output2screen("源域数据选择与分解结果:", source_data_name, source_data_non_periodic)
    output2screen("目标域数据选择与分解结果:", target_data_name, target_data.T)
    output2screen("源域数据选择与分解结果:", source_data_name, source_data.T)

#--------------------------------------------------------------------------------#
    console.rule("[bold red]Transfer Learning", align='center')
    # coral = CORAL()
    # source_data_non_periodic = coral.fit(source_data_non_periodic, target_data_non_periodic)

    save_path = "./result"

    # run(target_data_non_periodic,target_label_non_periodic,source_data_non_periodic,source_label_non_periodic,save_path,fs=0.5,low=1/16,high=1/8)
    missing_rate_list = []
    mae_total_list = []
    mae_p_list = []
    mae_ap_list = []
    
    mea_p_normal_list = []
    mea_ap_normal_list = []
    for missing_rate in trange(2,9):
        missing_rate/=10
        missing_rate_list.append(missing_rate)
        N = np.ceil(len(target_label)*(1-missing_rate)).astype(int)
        # input_p_predict,input_ap_predict = run(target_data.T,target_label[:N],source_data.T,source_label,save_path,fs=0.5,low=1/16,high=1/8)
        input_p_predict,input_ap_predict = run(target_data.T,target_label[-N:],source_data.T,source_label,save_path,fs=0.5,low=1/16,high=1/8)
        # 计算MAE
        mae_total_list.append(mae(target_label[:-N],input_p_predict+input_ap_predict))
        output_p, output_ap = Butterworth_filter(data=target_label[:-N], fs=0.5,low=1/16,high=1/8)

        mae_p_list.append(mae(output_p,input_p_predict))
        mae_ap_list.append(mae(output_ap,input_ap_predict))


    
    maeoutput2screen("MAE结果:",missing_rate_list, mae_total_list, mae_p_list, mae_ap_list)

    console.rule("[bold red]done!", align='center')


