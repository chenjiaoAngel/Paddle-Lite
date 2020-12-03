# Blur detection model.

## Input 
Please refer to preprocess.py to see the detailed preprocess steps. 
## Output
The model is two-class blur classification model, the output is softmax type, dimension 0 denote the clear score and dimension 1 denotes the blur score.
## Evaluation result on test_v2 test dataset.

Our model is MobileNet_v2_0.25，compared with online model.

|       Model       | Precision | Recall |  F1  |      speed      | size  |
| :---------------: | :-------: | :----: | :--: | :-------------: | :---: |
|   online model    |   0.99    |  0.57  | 0.72 |   windows 2ms   | 176k  |
| MobileNet_v2_0.25 |   0.84    |  0.87  | 0.86 | 6.8ms(1 thread) | 1300k |

The speed is test on RK3288, more details:

~~~shell
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
-- downsample_MobileNetV2_x0_25    avg = 6.8444 ms

Threads=2 Warmup=10 Repeats=30
-- downsample_MobileNetV2_x0_25    avg = 5.0259 ms

Threads=4 Warmup=10 Repeats=30
-- downsample_MobileNetV2_x0_25    avg = 4.3612 ms
~~~

