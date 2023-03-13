# End user video experience modelling
# 网络侧估计终端用户视频体验建模


## Question A: Network side estimation end-user video experience modeling



## 一. 问题描述（problem description）

随着无线宽带网络的升级，以及智能终端的普及，越来越多的用户选择在移动智能终端上用应用客户端 APP 观看网络视频，这是一种基于TCP的视频传输及播放。看网络视频影响用户体验的两个关键指标是“初始缓冲等待时间”和在视频播放过程中的“卡顿缓冲时间”，我们可以用“初始缓冲时延”和“卡顿时长占比”（卡顿时长占比 = 卡顿时长/视频播放时长）来定量评价用户体验。

已有相关研究表明影响“初始缓冲时延”和“卡顿时长占比”的主要因素有“初始缓冲峰值速率”、“播放阶段平均下载速率”、“端到端环回时间（E2E RTT）”，以及视频参数。然而这些因素和“初始缓冲时延”和“卡顿时长占比”之间的内在准确关系并不明确。

请根据附件提供的实验数据，建立用户体验评价变量（初始缓冲时延，卡顿时长占比）与网络侧变量（初始缓冲峰值速率，播放阶段平均下载速率，E2E RTT)之间的函数关系。



With the upgrade of wireless broadband network and the popularity of intelligent terminal, more and more users choose to watch network video on mobile intelligent terminal with application client APP, which is a kind of video transmission and playback based on TCP. Two key indicators that affect user experience when watching network videos are "initial buffer waiting time" and "stuck buffer time" during video playback. We can use "initial buffer delay" and "stuck duration ratio" (stuck duration ratio = stuck duration/video playback duration) to evaluate user experience quantitatively.
Relevant studies have shown that the main factors affecting "initial buffer delay" and "instant duration ratio" include "initial peak buffer rate", "average download rate during playback phase", "end-to-end loopback time (E2E RTT)", and video parameters. However, the exact internal relationship between these factors and "initial buffer delay" and "timeout duration ratio" is not clear.
According to the experimental data provided in the attachment, please establish the functional relationship between the user experience evaluation variables (initial buffering delay, instant duration ratio) and the network variables (initial buffering peak rate, average download rate during playback phase, E2E RTT).



![image-20230309034836914](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/image-20230309034836914.png)





### 数据集变量解析

1. **初始缓冲时延**： 等于初始缓冲准备阶段的时延 + 初始缓冲下载阶段的时延（包含准备时延）。

   在实时视频播放业务中，由于网络拥塞以及无线信道的信道容量变化，可能导致视频播放过程中出现中断，视频播放中断严重影响用户的观看体验，频繁的视频播放中断会导致用户最终中止播放过程。为了降低播放中断概率，通常先在播放前缓冲一段时间，叫做**初始缓冲时延**。在 初始缓冲时延期间，一般采取播放片头、或广告短片等方式，降低客户的等待感。典型的平均时延值为 4s 。

2. **卡顿时长占比**，即播放过程重缓冲总时延/有效播放总时长，卡顿时长占比 = 卡顿时长 / 视频播放时长。

3. **初始缓冲峰值速率**(kbps) ，是视频初始缓冲阶段达到的最大瞬时速率，和缓存数据量、E2E RTT(ms)，以及当时当地无线线路能力、无限负载强相关，是初始缓冲时延的最为关键的决定因素之一。

4. **播放阶段平均下载速率**，即播放期间感知速率，播放阶段下行总流量/(停止播放的时间戳-首次开始播放的时间戳)。

   



![image-20230313162817407](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131629419.png)



5. **E2E RTT**(端到端环回时间), 是无线通信运业务网络架构确定的一个通信参数。

<img src="https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131629996.png" alt="image-20230313162829676" style="zoom: 150%;" />

                                 移动视频业务E2E 组网示意图 （简单了解一下，无须深入）





6. **vMOS** 

   主观测试评价标准，MOS（Mean Opinion Score，或者Mean conversation-Opinion Score）叫做 平均会话评价分，是视频和语音质量主观测试的一种评价标准。它是一种五分制判断标尺，可以用数字或者文字表达。

   Excellent（优）＝5
   Good（良）      ＝4
   Fair（中）         ＝3
   Poor（差）       ＝2
   Bad（劣）         ＝1

   

   vMOS 是视频节目的主观测试评价标准，用来评估用户观看视频的质量，即用户体验评价变量。综合考虑视频源质量、播放过程中的初始时延。卡顿占比、视频播放时长，来对整个视频体验进行MOS打分。vMOS由sQuality（片源质量）、sLoading（初始缓冲时长）、和sStalling（卡顿）三项分值决定。

   

   影响sQuality得分的可能因素有：用户选择(资费)，片源清晰度，终端屏幕分辨率&处理器视频能力(如编解码算法支持，最高画质支持)，及可获得带宽(如可获得带宽不足，则有可能导致实际播放的最高画质受限)。

   

   影响sLoading得分的主要网络指标是视频初始缓冲峰值速率和E2E RTT（反映OTT视频的架构性时延）。统计研究表明，vMOS与E2E RTT负相关，vMOS随E2E RTT的减少而增大；vMOS与初始缓冲峰值速率正相关，vMOS随初始缓冲峰值速率增大而增大。

   

   影响sStalling得分的主要网络指标是视频全程感知速率。



## 二. 问题分析（problem analysis）

本问题是关于建立用户体验评价变量(初始缓冲时延，卡顿时长占比)与网络侧变量(初始缓冲峰值速率，播放阶段平均下载速率，E2E RTT)之间的函数关系的求解。
我们首先分析影响初始缓冲时延的因素，建立起这些相互影响因素的网络结构图。
其次我们利用题目所给数据，建立初始化缓冲时延与初始缓冲峰值速率，播放阶段平均下载速率，E2E RTT之间的函数关系。我们采取统计回归来拟合方法，分析这些变量对用户 体验评价变量的影响，分别建立相关函数模型。最后利用建立好的模型来估测初始缓冲时延与卡顿占比之间的关系，进行残差分析，验证模型的正确性。

This problem is about solving the functional relationship between user experience evaluation variables (initial buffer delay, instant duration ratio) and network variables (initial buffer peak rate, average download rate during playback phase, E2E RTT).

Firstly, we analyze the factors that affect the initial buffer delay and establish the network structure diagram of these factors.

Secondly, we use the data given in the title to establish the functional relationship between the initial buffer delay and the initial buffer peak rate, the average download rate in the playback phase, E2E RTT. We adopted statistical regression to fit the method, analyzed the influence of these variables on the user experience evaluation variables, and established correlation function models respectively. Finally, the established model is used to estimate the relationship between the initial buffer delay and the deadweight ratio, and the residual analysis is carried out to verify the correctness of the model.







### 1、初始化缓存时延(Y)与各单变量之间的相关分析



  我们首先需要获得各个影响变量与初始缓冲时延和卡顿时长占比的函数关系式，进而分析各因素之间关系，最后建立函数模型。


<img src="https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131735908.png" alt="image-20230313154730001" style="zoom:50%;" />


![image-20230313153614359](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131602233.png)





1）分析**初始缓冲时延** Y 与 **初始缓冲峰值速率** x1 之间的关系，首先作出Y -x1 的散点图。



![image-20230313152519099](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131602307.png)



由此图，我们可以分析得出从图1 可以发现，随着x1 的增加，Y 值有明显的指数递减趋势，拟合得出Y 与x1 是反比例函数关系（其中ε是随机误差）。函数关系式如下：

  ![image-20230313152939252](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131602128.png)

2）分析**初始缓冲时延** Y 与 **E2E RTT**(端到端环回时间) x2 的关系，作出Y-x2 的散点图



![image-20230313152746457](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131602421.png)



  由实验数据拟合可知，y 与x2 为正态分布函数, 函数关系式如下：

   ![image-20230313153011803](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601964.png)



3）分析**初始缓冲时延** Y 与 **播放阶段平均下载速率**  x3 的关系，作出Y-x3 的散点图

![image-20230313153030146](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601203.png)



  由实验数据拟合得出Y 与x3 为指数函数关系，函数关系式如下：                                               

![image-20230313153103771](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601108.png)



### 2、多变量的影响分析：



1）在模型（1）、（2）、（3）中，回归变量 x1、x2、x3 对因变量Y的影响都是互相独立的，即初始缓冲时延Y的均值与初始缓冲峰值速率x1 的非线性关系由回归系数a1,b2 确定，而不依赖于E2E RTTx2，播放阶段平均下载速率x3。



![image-20230313153858217](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601501.png)



2）为了寻找因变量之间的关系，采用excel 对它们进行相关系数的计算。为表明变量之间的关系，我们简单的用x1,x2,x3 的乘积代表它们之间的交互作用，于是将模型（4）增加三项。得到最终模型：



 ![image-20230313154001656](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601092.png)



用 matlab的统计工具 对最终函数模型进行验证性分析， 得到的结果如 下：



![image-20230313154047732](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601598.png)



表 2显示 ，R2 = 0.924指因变量 y（初始缓冲 时延 ）的 92.4%可由模型确定， F值远超过 F的检验临界值， p远小于 α，因而模型（ 因而模型（ 4）从整体上看来是可用的。





### 3、卡顿占比（Z）与初始化缓存时延（Y）的相关分析





![image-20230313154759921](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131601925.png)


通过 多次拟合，发现 卡段占比 z和 初始化缓存时延 y的相关性系数 R2=0.0736，说明卡顿占比 与初始缓冲时延具有交互性，其线性关系明显， 得到关系：

 ![image-20230313155050268](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303131600410.png)



  ![image-20230313155121263](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/image-20230313155121263.png)

  

结果分析：表3显示，R2 = 0.951 指因变量y（初始缓冲时延）的95.1%可由模型确定，F值远远超过F的检验临界值，p远小于α，因而模型（6）从整体上看来是可用的。



### 三. 总结

 通过以上研究，我们得到以下三个结论：

1.  在视频播放之前进行缓冲，能够降低播放中断概率。

2. 视频播放中断的原因是视频分组不能及时有效的到达接收端进行解码，导致接收端播放队列为空。

3.  初始缓冲时延能够减少播放中断，但是过大的初始缓冲时延也会影响用户的观看体验。

   因此, 初始缓冲时延是衡量视频传输服务质量的重要指标，初始缓冲时延能够减少播放卡顿中断，但是过大的初始缓冲时延也会影响用户的观看体验。此外，初始缓冲峰值速率的大小，直接影响到初始缓冲时延。

Through the above research, we can draw the following three conclusions:

1. Buffer the video before playing to reduce the probability of playing interruption.

2. Video playback is interrupted because video packets cannot reach the receiving end for decoding in a timely and effective manner, resulting in an empty playback queue on the receiving end.

3. Initial buffer delay can reduce playback interruption, but too large initial buffer delay will affect users' viewing experience.

Therefore, initial buffer delay is an important indicator to measure video transmission service quality. Initial buffer delay can reduce playback delay, but too large initial buffer delay will also affect users' viewing experience. In addition, the initial buffering peak rate directly affects the initial buffering delay.

本文提出了一个基于运营数据统计建模分析的以客户端体验为中心的网络视频质量评估方法，可以解决主观评估苛刻的条件、人为因素影响、实施步骤复杂、代价昂贵、实时性不好等缺点，同时满足准确地反映视频的体验质量的要求，而且评估质量更加精确可靠和客观，可以应用于实时视频通信中的质量评估。本文模型对评估体系进行分析优化，提高客户观看网络视频体验提供了完美的数据支持，可以成功解决网络视频业务的系统优化和客户支持等问题。

This paper proposes a network video quality evaluation method based on statistical modelling analysis of operational data and cantering on client experience, which can solve the shortcomings of harsh conditions of subjective evaluation, human factors, complex implementation steps, expensive, and poor real-time performance. At the same time, it meets the requirements of accurately reflecting the video experience quality, and the evaluation quality is more accurate, reliable and objective. It can be applied to quality evaluation of real-time video communication. The model in this paper analyses and optimizes the evaluation system, provides perfect data support to improve customers' experience of watching network video, and can successfully solve the problems of system optimization and customer support .



#### 附件：主要代码 main source code (matlab)

```matlab
Y-X1:

function [fitresult, gof] = createFit(X1, y)
[xData, yData] = prepareCurveData( X1, y );
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -1 -Inf];
opts.StartPoint = [6214.27813636366 -0.147293002006473 59.0073454326789];
opts.Upper = [Inf -1 Inf];
[fitresult, gof] = fit( xData, yData, ft, opts );
figure( 'Name', 'Y-X1' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. X1', 'Y-X1', 'Location', 'NorthEast' );
xlabel X1
ylabel y
grid on

Y-X2:
function [fitresult, gof] = createFit(X2, y)
[xData, yData] = prepareCurveData( X2, y );
ft = fittype( 'gauss1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0];
opts.StartPoint = [30552 106 15.4163423583732];
[fitresult, gof] = fit( xData, yData, ft, opts );
figure( 'Name', 'Y-X1' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. X2', 'Y-X1', 'Location', 'NorthEast' );
xlabel X2
ylabel y
grid on


Y-X3:
function [fitresult, gof] = createFit(X3, y)
[xData, yData] = prepareCurveData( X3, y );
ft = fittype( 'exp1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [3414.89573662046 -4.98287982878925e-05];
[fitresult, gof] = fit( xData, yData, ft, opts );
figure( 'Name', 'Y-X1' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. X3', 'Y-X1', 'Location', 'NorthEast' );
xlabel X3
ylabel y
grid on



```
