# SCGRU-PyTorch

PyTorch implementation of "One-shot Pruning of Gated Recurrent Unit Neural Network by Sensitivity for Time-series Prediction" by Hong Tang, Xiangzheng Ling, Liangzhi Li (*Member, IEEE*) et al.



![In PowerLoad Task](https://github.com/imLingo/Pictures/blob/master/powerLoad_prune_performance_compared_new.tif)

![In LAN Task](https://github.com/imLingo/Pictures/raw/master/traffic_prune_performance_compared.tiff)

## Requirements

Following packages are required for this project

- Python 3.6+

- PyTorch-GPU 0.4.1

- tqdm, csv, time
- numpy, pandas
- matplotlib
- argparse



## Usage

1.  train a simple local areal network  traffic

```
python main.py --k_level 0.01 --sensitivity 3.754
```

2. training comparison model GVGRUs

```
python GVGRU.py --model [Model_Name]
```



## Results

### In Local Area Network traffic Task

1.  The test results of the standard GRU (**Baseline**).

![LAN Baseline](https://github.com/imLingo/Pictures/blob/master/LAN_baseline.JPG)

2. The test results of SCGRU (**Our**).

![LAN SCGRU]()



### In Power Load Task

1.  The test results of the standard GRU (**Baseline**).

![Power load Baseline](https://github.com/imLingo/Pictures/blob/master/power_baseline.JPG)

2. The test results of SCGRU (**Our**).

![Power load SCGRU]()



