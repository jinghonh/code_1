import pandas as pd
import functools
import pywt
import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.ar_model import AutoReg
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import signal

class Wavelet:
    '''
    参考文献：
    1.A Practical Guide to Wavelet Analysis
    2.Application of the cross wavelet transform and wavelet coherence to geophysical time series
    参考代码matlab：
    http://grinsted.github.io/wavelet-coherence
    '''
    def __init__(self,dt=1):

        self.dt = dt
        self.dj = 1 / 12
        self.s0 = 2 * self.dt
        self.wt_results = None
        self.wt_periods = None
        self.empirically_derived_factors = {"cmor2-0.97":{"dof":2,"gamma":2.32,"dj0":0.6,"C":0.776,"fourier_factor":1.03}}

    def set_params(self,data):
        self.N = data.shape[0]
        self.var = np.var(data)
        self.J = np.log2(self.N * 0.17) / self.dj
        self.sj = self.s0 * np.power(2, self.dj * np.arange(int(self.J) + 1))
        self.coi = self.fourier_factor / np.sqrt(2) * self.dt * np.array(
            [1e-5] + [i for i in range(1, int(np.ceil(self.N / 2)))] + [i for i in range(self.N // 2 - 1, 0, -1)] + [1e-5])

    def wt(self,data,wavelet='cmor2-0.97',smooth=1,ifshow=False):

        self.dof = self.empirically_derived_factors[wavelet]["dof"]
        self.gamma = self.empirically_derived_factors[wavelet]["gamma"]
        self.dj0 = self.empirically_derived_factors[wavelet]["dj0"]
        self.C = self.empirically_derived_factors[wavelet]["C"]
        self.fourier_factor = self.empirically_derived_factors[wavelet]["fourier_factor"]
        self.set_params(data)
        self.wavelet = wavelet

        wavelet = pywt.ContinuousWavelet(wavelet)
        coefs, freqs = pywt.cwt(data=data, scales=self.sj, wavelet=wavelet, sampling_period=self.dt)
        self.periods = 1/freqs

        if not ifshow:
            return  coefs, self.periods

        if smooth == 2:
            wt_signif_ = self.wt_signif(data,smooth)*np.ones(coefs.shape)
        else:
            wt_signif_ = self.wt_signif(data,smooth)[:,np.newaxis]

        self.sig95 = np.abs(coefs)**2/(self.var*wt_signif_)

        x = np.arange(self.N)
        y = self.periods
        x,y = np.meshgrid(x,y)

        z = np.abs(coefs)**2/self.var

        fig,ax = plt.subplots(1,1,figsize=(8, 5.5))

        couf = ax.contourf(x,y,np.log2(z),cmap=plt.cm.viridis)
        colorbar = plt.colorbar(couf,ax=ax)
        colorbar_ticks = colorbar.get_ticks()
        colorbar.set_ticks(colorbar_ticks)
        colorbar.set_ticklabels(["$1/2^{%d}$"%(int(-i)) if i<0 else "$%d$"%(2**i) for i in colorbar_ticks])
        colorbar.ax.yaxis.set_tick_params(labelsize=16)
        colorbar.ax.yaxis.label.set_fontname('Times New Roman')

        ax.contour(x,y,self.sig95,[1],colors="k")

        t = np.arange(self.N)
        ax.fill_between(t,self.coi,[np.max(y)]*self.N,color="w",alpha=0.5)

        ax.set_yscale("log",base=2)
        ax.set_ylim(np.min(y),np.max(y))
        plt.xticks(fontsize=18,fontproperties="Times New Roman")
        plt.xlabel("时间",fontsize=18)
        plt.yticks(fontsize=18,fontproperties="Times New Roman")
        plt.ylabel("周期",fontsize=18)
        ax.invert_yaxis()

        return coefs, self.periods, fig

    def wt_signif(self,data,smooth=0):
        '''
        smooth对应论文A Practical Guide to Wavelet Analysis三种自由度
        0   不进行平滑
        1   时域平滑
        2    时域-频域平滑
        '''
        model = AutoReg(data,1)
        model_fit = model.fit()

        arfps = self.ar_fouries_power_spectrum(model_fit.params[1])
        signif = None

        if smooth == 0:
            dof = self.dof
            chisquare = chi2.isf(0.05,dof)
            signif = chisquare*arfps/dof
        elif smooth == 1:
            dof = self.dof*np.sqrt(1+(self.N*self.dt/self.gamma/self.sj)**2)
            chisquare = chi2.isf(0.05, dof)  #自由度可以为小数
            signif = chisquare * arfps / dof
        elif smooth==2:
            savg = 1/np.sum(1/self.sj)
            P = savg*np.sum(arfps/self.sj)
            smid = self.s0*2**(0.5*len(self.sj)*self.dj)
            dof = self.dof*self.N*savg/smid*np.sqrt(1+(self.N*self.dt/self.dj0)**2)
            chisquare = chi2.isf(0.05, dof)
            signif = P*chisquare*self.dt*self.dj/dof/self.C/savg
        else:
            pass
        return  signif

    def ar_fouries_power_spectrum(self,lag):

        return (1-lag**2)/(1+lag**2-2*lag*np.cos(2*np.pi*self.dt/self.periods))

    def wtc(self,x,y,mc=300,wavelet='cmor2-0.97',ifshow=False):

        self.wavelet = wavelet
        self.fourier_factor = self.empirically_derived_factors[wavelet]["fourier_factor"]
        self.set_params(x)

        coefs_x, freqs_x = pywt.cwt(data=x, scales=self.sj, wavelet=wavelet, sampling_period=self.dt)
        coefs_y, freqs_y = pywt.cwt(data=y, scales=self.sj, wavelet=wavelet, sampling_period=self.dt)
        self.periods = 1 / freqs_x

        swx = self.wtc_smooth_wavelet(1/self.sj[:,np.newaxis]*np.abs(coefs_x)**2)
        swy = self.wtc_smooth_wavelet(1/self.sj[:,np.newaxis]*np.abs(coefs_y)**2)
        swxy = self.wtc_smooth_wavelet(1/self.sj[:,np.newaxis]*coefs_x*np.conj(coefs_y))

        wtc_ = np.abs(swxy)/np.sqrt(swx*swy)

        if not ifshow:
            return wtc_,self.periods

        wtc_signif_ = self.wtc_signif(x,y,mc)

        self.sig95 = wtc_/wtc_signif_

        self.wtc_phase = np.arctan(np.imag(swxy)/np.real(swxy))*(180/np.pi)

        x = np.arange(self.N)
        y = self.periods
        x, y = np.meshgrid(x, y)

        z = wtc_

        fig,ax = plt.subplots(1,1,figsize=(8, 5.5))

        couf = ax.contourf(x,y,np.log2(z),cmap=plt.cm.viridis)
        colorbar = plt.colorbar(couf,ax=ax)
        colorbar_ticks = colorbar.get_ticks()
        colorbar.set_ticks(colorbar_ticks)
        colorbar.set_ticklabels(["$1/2^{%d}$"%(int(-i)) if i<0 else "$%d$"%(2**i) for i in colorbar_ticks])
        colorbar.ax.yaxis.set_tick_params(labelsize=18)
        colorbar.ax.yaxis.label.set_fontname('Times New Roman')

        ax.contour(x,y,self.sig95,[1],colors="k")

        t = np.arange(self.N)
        ax.fill_between(t,self.coi,[np.max(y)]*self.N,color="w",alpha=0.5)

        wtc_phase = self.wtc_phase
        mask_c = np.zeros(wtc_phase.shape,dtype=np.bool8)
        mask_c[::2,::3] = True
        mask_phase = (wtc_>0.6) & mask_c
        ax.quiver(x[mask_phase] ,y[mask_phase],np.real(swxy)[mask_phase],np.imag(swxy)[mask_phase],color="k")

        ax = plt.gca()
        ax.set_yscale("log",base=2)
        ax.set_ylim(np.min(y),np.max(y))
        plt.xticks(fontsize=18,fontproperties="Times New Roman")
        ax.set_xlabel("时间",fontsize=18)
        plt.yticks(fontsize=18,fontproperties="Times New Roman")
        ax.set_ylabel("周期",fontsize=18)
        ax.invert_yaxis()

        return wtc_, self.periods, fig

    def wtc_signif(self,x,y,m):

        model_x = AutoReg(x,1)
        model_x_fit = model_x.fit()
        ar1_x = model_x_fit.params[1]

        model_y = AutoReg(y,1)
        model_y_fit = model_y.fit()
        ar1_y = model_y_fit.params[1]

        wlc = np.zeros((len(self.sj),1000))
        sig95 = np.zeros((len(self.sj),1))
        outsidecoi = np.zeros((len(self.sj),x.shape[0]),dtype=np.bool8)

        for i in range(len(self.sj)):
            outsidecoi[i,:] = (self.periods[i] <= self.coi)

        while m > 0:
            rednoise_x = self.gen_rednoise(ar1_x)
            rednoise_y = self.gen_rednoise(ar1_y)

            coefs_rx, freqs_rx = pywt.cwt(data=rednoise_x, scales=self.sj, wavelet=self.wavelet, sampling_period=self.dt)
            coefs_ry, freqs_ry = pywt.cwt(data=rednoise_y, scales=self.sj, wavelet=self.wavelet,sampling_period=self.dt)

            swrx = self.wtc_smooth_wavelet(1 / self.sj[:, np.newaxis] * np.abs(coefs_rx) ** 2)
            swry = self.wtc_smooth_wavelet(1 / self.sj[:, np.newaxis] * np.abs(coefs_ry) ** 2)
            swrxy = self.wtc_smooth_wavelet(1 / self.sj[:, np.newaxis] * coefs_rx * np.conj(coefs_ry))
            wcr = np.abs(swrxy) / np.sqrt(swrx * swry)

            for i in range(len(self.sj)):
                cd = wcr[i,outsidecoi[i]]
                cd = np.clip(cd,0,1)
                cd = np.floor(cd * 999) + 1
                cd = cd.astype(np.uint16)
                for j in range(len(cd)):
                    wlc[i,cd[j]] += 1

            m -= 1

        for i in range(len(self.sj)):
            rsqy = (np.arange(1,1001)-0.5)/1000
            ptile = wlc[i]
            idx = (ptile != 0)
            ptile = ptile[idx]
            rsqy = rsqy[idx]
            ptile = np.cumsum(ptile)
            ptile = (ptile - .5) / ptile[-1]
            sig95[i][0] = np.interp(0.95,ptile, rsqy)

        return sig95

    def gen_rednoise(self,ar1):

        if ar1 == 0:
            return np.random.randn(self.N)

        tau = int(np.ceil(-2/np.log(np.abs(ar1))))

        return scipy.signal.lfilter([1,0],[1,-ar1],np.random.randn(tau+self.N))[tau:]

    def wtc_smooth_wavelet(self,wave):

        npad =  int(2**(np.ceil(np.log2(self.N))))
        k = np.arange(1,npad//2+1)
        k = k*(2*np.pi/npad)
        k = np.concatenate(([0],k,-k[-2::-1]))**2
        snorm = self.sj*self.dt
        dtp = np.float32 if not np.iscomplex(wave).any() else np.complex64
        smooth_wave = np.zeros(wave.shape,dtype=dtp )

        for i in range(len(self.sj)):
            F = np.exp(-0.5*(snorm[i]**2)*k)
            smooth = np.fft.ifft(F*np.fft.fft(wave[i],npad))
            smooth_wave[i] = np.real(smooth[:self.N]) if not np.iscomplex(wave).any() else smooth[:self.N]

        dj0 = 0.6
        dj0steps = dj0 / (self.dj * 2)

        kernel = np.array([np.mod(dj0steps, 1), *([1]*(2 * round(dj0steps)-1)), np.mod(dj0steps, 1)]) / \
                 (2 * np.round(dj0steps) - 1 + 2 * np.mod(dj0steps, 1))

        return scipy.signal.convolve2d(smooth_wave,kernel[:,np.newaxis],mode="same")

class Wdd:
    def __init__(self, wavelet_discrete="db1", level=1):
        """
        初始化参数
        wavelet_discrete: 小波类型，例如 'db1'
        level: 分解的层数
        """
        self.wavelet_discrete = wavelet_discrete
        self.level = level
        self.reconstruct_data = None

    def run(self, data):
        """
        对输入的 NumPy 数组数据进行小波包变换和重构
        data: 输入的时间序列数据 (1D NumPy 数组)
        """
        mode = pywt.Modes.periodic  # 使用周期边界处理模式
        node = {}
        reconstruct_data = {}

        # 使用小波包变换对数据进行分解
        wp = pywt.WaveletPacket(data=data, wavelet=self.wavelet_discrete, mode=mode)
        
        # 获取指定分解层次的所有节点
        for n in wp.get_level(self.level, "freq"):
            node[n.path] = n.data

        # 重构分解的数据
        for key, value in node.items():
            reconstruct_wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet_discrete, mode=mode)
            reconstruct_wp[key] = value
            reconstruct_data[key] = reconstruct_wp.reconstruct()

        self.reconstruct_data = reconstruct_data

        # 可视化原始数据和分解数据
        time = np.arange(len(data))
        Grid = GridSpec(2**self.level + 1, 10)
        fig = plt.figure(figsize=(10, 6))

        # 绘制原始数据
        ax = plt.subplot(Grid[0, :8])
        ax.plot(time, data, c="red")
        ax.set_title("原始数据")
        ax.grid(True, linestyle='--')

        # 绘制每个分解后的子信号
        for j, key in enumerate(reconstruct_data.keys()):
            ax1 = plt.subplot(Grid[j + 1, :8])
            ax1.plot(time, reconstruct_data[key], c="blue")
            ax1.set_title(f"重构子信号 {key}")
            ax1.grid(True, linestyle='--')

        plt.tight_layout()
        # plt.show()

        return reconstruct_data

    def data_save(self, filepath):
        """
        保存重构的数据和绘图
        filepath: 文件保存路径 (不包含扩展名)
        """
        if self.reconstruct_data is None:
            raise ValueError("未运行小波分解")

        # 保存数据到 CSV 文件
        reconstruct_df = pd.DataFrame(self.reconstruct_data)
        reconstruct_df.to_csv(filepath + ".csv", index=False)
        
        # 保存绘图为 JPG 文件
        plt.savefig(filepath + '.jpg')

class Bf:
    def __init__(self, mode="带通滤波器", order=2,W=["1", "2"]):
        """
        初始化滤波参数
        mode: 滤波器类型 ("低通滤波器", "带通滤波器", "高通滤波器", "带阻滤波器")
        order: 滤波器的阶数 (默认值为 2)
        """
        self.mode = mode
        self.order = order
        self.W = W
        self.reconstruct_data = None

    def run(self, data):
        """
        对输入的 NumPy 数组数据进行滤波
        data: 输入的时间序列数据 (1D NumPy 数组)
        """
        fs = 0.5  # 采样频率的一个示例
        try:
            if self.mode == "低通滤波器" and self.W[0]:
                low_ = 1 / (max(int(self.W[0]), 2.1) * fs)
                b, a = signal.butter(N=self.order, Wn=low_, btype='lowpass')
            elif self.mode == "带通滤波器" and self.W[0] and self.W[1]:
                low_, high_ = 1 / (max(int(self.W[1]), 2.1) * fs), 1 / (max(int(self.W[0]), 2.1) * fs)
                b, a = signal.butter(N=self.order, Wn=[low_, high_], btype='bandpass')
            elif self.mode == "高通滤波器" and self.W[1]:
                high_ = 1 / (max(int(self.W[1]), 2.1) * fs)
                b, a = signal.butter(N=self.order, Wn=high_, btype='highpass')
            elif self.mode == "带阻滤波器" and self.W[0] and self.W[1]:
                low_, high_ = 1 / (max(int(self.W[1]), 2.1) * fs), 1 / (max(int(self.W[0]), 2.1) * fs)
                b, a = signal.butter(N=self.order, Wn=[low_, high_], btype='bandstop')
            else:
                raise ValueError("滤波器参数设置错误")
            
            # 对数据进行滤波处理
            filtered_signal = signal.filtfilt(b, a, data)
        except Exception as e:
            raise RuntimeError(f"滤波过程出错: {e}")

        # 计算滤波后的残差信号
        mean_ = np.mean(data)
        filtered_signal = filtered_signal + mean_
        res = data - filtered_signal

        # 存储重构数据
        self.reconstruct_data = pd.DataFrame({"滤波成分": filtered_signal, "剩余成分": res})

        # 可视化原始数据和滤波结果
        time = np.arange(len(data))
        fig = plt.figure(figsize=(10, 6))
        Grid = GridSpec(3, 10)

        # 绘制原始数据
        ax = plt.subplot(Grid[0, :8])
        ax.plot(time, data, c="red")
        ax.set_title("原始数据")
        ax.grid(True, linestyle='--')

        # 绘制滤波成分
        ax1 = plt.subplot(Grid[1, :8])
        ax1.plot(time, filtered_signal, c="blue")
        ax1.set_title("滤波成分")
        ax1.grid(True, linestyle='--')

        # 绘制剩余成分
        ax2 = plt.subplot(Grid[2, :8])
        ax2.plot(time, res, c="green")
        ax2.set_title("剩余成分")
        ax2.grid(True, linestyle='--')

        plt.tight_layout()
        # plt.show()
        return self.reconstruct_data

    def data_save(self, filepath):
        """
        保存重构的数据和绘图
        filepath: 文件保存路径 (不包含扩展名)
        """
        if self.reconstruct_data is None:
            raise ValueError("未运行滤波处理")

        # 保存数据到 CSV 文件
        self.reconstruct_data.to_csv(filepath + ".csv", index=False)
        
        # 保存绘图为 JPG 文件
        plt.savefig(filepath + '.jpg')
