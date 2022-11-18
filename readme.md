# Computer Vision Learning

此项目包含了本人学习使用的一些基本图像处理算法，一部分由自己编写，一部分来自于 Github，所有的算法代码全部都包含
大量注释，方便自己学习使用。

+ blur (模糊特征) 
  + [blur_kernel](https://github.com/xuweilin2014/learning/tree/master/blur/blur_kernel)：生成模糊核
  + [angle](https://github.com/xuweilin2014/learning/tree/master/blur/angle)：判断图像模糊方向，参考论文：
    [Parametric Blur Estimation for Blind Restoration of Natural Images: Linear Motion and Out-of-Focus](https://ieeexplore.ieee.org/document/6637096)
  + [auto_correlation](https://github.com/xuweilin2014/learning/tree/master/blur/feature/auto_correlation)：自相关特征，参考论文：[Image Partial Blur Detection and Classification](https://jiaya.me/all_final_papers/blur_detect_cvpr08.pdf)
  + [local_kurtosis](https://github.com/xuweilin2014/learning/tree/master/blur/feature/local_kurtosis) 和 [gradient_magnitude](https://github.com/xuweilin2014/learning/tree/master/blur/feature/gradient_magnitude)：峰度特征和梯度幅度特征，用来区分模糊图像和清晰图像，参考论文：[Discriminative Blur Detection Features](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/papers/blurdetect_cvpr14.pdf)
+ correlation filter (相关滤波)
    + [kcf](https://github.com/xuweilin2014/learning/tree/master/cf/kcf)：Kernelized Correlation Filters 跟踪算法实现 
    + [dsst](https://github.com/xuweilin2014/learning/tree/master/cf/dsst)：DSST 跟踪算法实现
    + [mosse](https://github.com/xuweilin2014/learning/tree/master/cf/mosse)：MOSSE 跟踪算法实现
+ corner detection (角点检测)
    + [harris 角点检测](https://github.com/xuweilin2014/learning/blob/master/corner_detection/harris_corner_detection.py)
+ [dark channel (暗通道先验去雾)](https://github.com/xuweilin2014/learning/blob/master/dark_channel/dehaze.py)
+ [deep sort 多目标跟踪](https://github.com/xuweilin2014/learning/blob/master/deep_sort/deep_sort_app.py) 
+ 混合高斯背景建模
    + [bgfg_gaussmix2](https://github.com/xuweilin2014/learning/blob/master/gauss_mix/bgfg_gaussmix2_%E5%B8%A6%E6%B3%A8%E9%87%8A%E7%89%88%E6%9C%AC.cpp)：OpenCV 中混合高斯背景建模的实现，添加了注释和说明
    + [gauss_mix](https://github.com/xuweilin2014/learning/blob/master/gauss_mix/gauss_mix.py)：混合高斯背景建模的算法代码，用 python 实现，参考 OpenCV 代码和论文：[Improved adaptive Gaussian mixture model for background subtraction](https://ieeexplore.ieee.org/document/1333992)
+ 混合高斯模型
    + [gaussian_mixture_model](https://github.com/xuweilin2014/learning/blob/master/gaussian/gaussian_mixture_model.py)：手动实现高斯混合模型，并且使用 E-M 算法迭代求解模型的参数
+ 导向滤波
    + [guided_filter](https://github.com/xuweilin2014/learning/blob/master/guided_filter/main.py)：导向滤波算法实现
+ 卡尔曼滤波
    + [kalman_filter](https://github.com/xuweilin2014/learning/blob/master/kalman_filter/kf/kalman_filter.py)：卡尔曼滤波算法实现
    + [extended_kalman_filter](https://github.com/xuweilin2014/learning/blob/master/kalman_filter/ekf/extended_kalman_filter.py)：扩展卡尔曼滤波算法的实现
+ 局部二进制特征
    + [local_binary_pattern](https://github.com/xuweilin2014/learning/blob/master/local_binary_pattern/local_binary_pattern.py)：LBP 特征的实现
+ [Mean Shift 跟踪算法](https://github.com/xuweilin2014/learning/blob/master/mean_shift_tracking/mean_shift_tracker.py)
+ MKCF
    + [mkcf.cpp](https://github.com/xuweilin2014/learning/blob/master/mkcf/mkcf.cpp)：使用多 KCF 跟踪器来实现多目标跟踪，来自论文：[Multiple Object Tracking with Kernelized Correlation Filters in Urban Mixed
Traffic](https://arxiv.org/abs/1611.02364)
    + [mkcf.py](https://github.com/xuweilin2014/learning/blob/master/mkcf/mkcf.py)：多 KCF 跟踪器的实现与运用，实现了其 python 版本
+ SVM