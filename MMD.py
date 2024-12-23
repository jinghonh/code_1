import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    ns = len(source)
    nt = len(target)
    M = np.ones((ns+nt,ns+nt))*(-1/(ns*nt))
    M[:ns,:ns] = 1/(ns*ns)
    M[nt:,nt:] = 1/(nt*nt)
    '''
    多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
     # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = np.concatenate((source, target), axis=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = np.expand_dims(total,axis=0)
    total0= np.broadcast_to(total0,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
    total1 = np.expand_dims(total,axis=1)
    total1=np.broadcast_to(total1,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = np.sum(np.square(total0-total1),axis=2).astype(np.float64) ###ERROR：上版本为np.cumsum，会导致计算出现几何倍数的L2
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # print(bandwidth_list)
    #高斯核函数的数学表达式
    kernel_val = [np.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return np.trace(sum(kernel_val)@M)#多核合并

def read_data(city='北京'):
    # 1. 读取 CSV 文件
    file_path_atmosphere = 'CityAtmosphereFactorMonthData.csv'
    file_path_air_quality = 'CityAirQualityMonthData.csv'

    atmosphere_data = pd.read_csv(file_path_atmosphere)
    air_quality_data = pd.read_csv(file_path_air_quality)

    # 2. 处理时间，转换月份为可解析的日期格式
    # 转换“年份”和“月份”列为日期对象，便于后续的时间过滤
    atmosphere_data['日期'] = pd.to_datetime(atmosphere_data['年份'].astype(str) + '-' + atmosphere_data['月份'].astype(str), format='%Y-%m')
    air_quality_data['日期'] = pd.to_datetime(air_quality_data['月份'], format='%Y-%m')

    # 3. 限制时间范围
    start_date = '2015-01-01'
    end_date = '2021-12-31'

    # 对数据进行时间过滤
    # atmosphere_data_filtered = atmosphere_data
    atmosphere_data_filtered = atmosphere_data[(atmosphere_data['日期'] >= start_date) & (atmosphere_data['日期'] <= end_date)]
    air_quality_data_filtered = air_quality_data[(air_quality_data['日期'] >= start_date) & (air_quality_data['日期'] <= end_date)]
    # 删除 'PM2.5' 列
    atmosphere_data_filtered.drop('日期', axis=1, inplace=True)
    # 4. 选择特定城市的数据，假设选择城市为“北京”
    cityData_airQuality = air_quality_data_filtered[air_quality_data_filtered['城市'] == city]
    cityData_atmosphere = atmosphere_data_filtered[atmosphere_data_filtered['城市'] == city]

    # 5. 打印列名（查看列名，如果需要）
    print(atmosphere_data.columns)
    print(air_quality_data.columns)
    cityData_atmosphere
    # 6. 提取特征和目标
    # 目标变量：空气质量中“NO2”
    y = cityData_airQuality.loc[:, ['NO2']].dropna().reset_index(drop=True)

    # 特征变量：选择从第 3 列开始的所有大气因素特征
    # x = cityData_atmosphere.iloc[:, 3:].reset_index(drop=True)
    x = cityData_atmosphere.iloc[:, 5:].reset_index(drop=True)
    plt.plot(y,'r',label='NO2',linewidth=1,marker='o',markersize=2)
    plt.show()
    # 转换为 NumPy 数组
    x = np.array(x).T
    y = np.array(y).T
    print(x.shape, y.shape)
    return x, y

if __name__ == "__main__":
    x, y = read_data(city='北京')
    # columns_to_remove = ['100m_u_component_of_wind', '100m_v_component_of_wind']
    # y = pd.read_csv("./fig78.csv",encoding='gbk')['81'].values   #(84,) (这个计算得到的mmd值和论文中结果更相近)
    # plt.plot(y,'r',label='NO2',linewidth=1,marker='o',markersize=2)   #这两个Y留一个即可，第一个是从空气数据中得到的，第二个是从折线图中提出的数据。
    
    y = np.array(y).T
    target_data = y.reshape(-1) #(n,)
    plt.plot(target_data,'r',label='NO2',linewidth=1,marker='o',markersize=2) 
    plt.show()
    selection_data = x #(m,n)   (7,84)
    selection_data_name = ['南北风速','东西风速','温度','紫外线','相对湿度','太阳辐射','降雨'] #(m,)
    # cols = ['南北风速','东西风速','温度','紫外线','相对湿度','太阳辐射','降雨']
    f = lambda x, y: (x - min(x)) / (max(x) - min(x)) * (max(y) - min(y)) + min(y)
    mmds = []
    for d in selection_data:
        d = f(d,target_data)  #d(84,)
        mmd = guassian_kernel(d[:,np.newaxis],target_data[:,np.newaxis],fix_sigma=1)
        mmds.append(mmd)
    result = sorted(range(len(mmds)),key=lambda x:mmds[x])
    source_label = selection_data[result[0]]
    source_label_name = selection_data_name[result[0]]
    print("MMD距离从小到大:",[result])
    print("选择的源域标签:",[source_label_name])
    print(mmds)