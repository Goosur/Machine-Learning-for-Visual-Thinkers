## Dataset
Your dataset -- What does it measure? How is it organized? How did you preprocess and/or clean it?

My dataset measured the follow variables of cars in 2015.

- Make	Manufacturer (e.g. Chevrolet, Toyota, etc.)
- Model:        Car model (e.g. Impala, Prius, ...)
- Type:         Vehicle category (Small, Hatchback, Sedan, Sporty, Wagon, SUV, 7Pass)
- LowPrice:	    Lowest MSRP (in $1,000)
- HighPrice:	Highest MSRP (in $1,000)
- Drive:    	Type of drive (FWD, RWD, AWD)
- CityMPG:  	City miles per gallon (EPA)
- HwyMPG:   	Highway miles per gallon (EPA)
- FuelCap:  	Fuel capacity (in gallons)
- Length:   	Length (in inches)
- Width:    	Width (in inches)
- Height:   	Height (in inches)
- Wheelbase:	Wheelbase (in inches)
- UTurn:    	Diameter (in feet) needed for a U-turn
- Weight:   	Curb weight (in pounds)
- Acc030:   	Time (in seconds) to go from 0 to 30 mph
- Acc060:   	Time (in seconds) to go from 0 to 60 mph
- QtrMile:  	Time (in seconds) to go ¼ mile
- PageNum:  	Page number in the Consumer Reports New Car Buying Guide
- Size:     	Small, Midsized, or Large

I preprocessed it by filtering out non numeric data in order for stats to be calculated from the data.

## Algorithm
Your algorithm -- How did you design your transformation matrices to normalize by range and by Z-score?

I designed my transformation matrices as we learned in class. For range normalization I translated by -min and scaled by the inverse of the range. For z score normalization I translated by -mean and scaled by the inverse of the standard deviation. I tried to design my functions in a way that I could normalize any sized dataset.

## Visualization
Your visualization technique(s) -- What did you choose to do and why?

I chose to just do the pairplots this time to show that normalizing doesn't affect the way the data looks when plotted. The relationships between variables stay the same and their plots are also the same but on different scales. Normalizing by range brings all data values between 0 and 1. Normalizing by z score brings all data values roughly between -3 and 3

Again the pairplots are rediculously large.

Cars normalized by range:\
![cars_norm_range](https://i.imgur.com/KXygCPQ.png)

Cars normalized by z score:\
![cars_norm_z](https://i.imgur.com/N0B0uGS.png)

Cars original:\
![cars](https://i.imgur.com/uAIs41o.png)

## Results
Your results -- What are the stats (mean, standard deviation, covariance) of the original and normalized datasets? Did normalization help you uncover new patterns in the data, remove outliers, or alleviate another source of bias in your statistics? What did this project help you learn about your dataset and/or machine learning?

Stats for original/normalized datasets below. Through looking at covariance, normalization helped me see just how much variables were related without the different units making things unclear. For example the Acc60 and CityMPG relation is clearer when compared to other variables because of normalization. The z score normalization made it clear that there were some significant outliers in the data. Looking at the mins and maxs of the z score normalization you can see z scores of 3+ which is well outside most of the values of the dataset. Like I mentioned earlier there was lots of unit bias in this dataset and normalization helped eliminate that bias and allow a clearer look at the data. This project helped me learn that normalization is very powerful in machine learning. Eliminating bias and making data easier to work with is very important for analyzing data. This may seem obvious but for me there was a big difference between hearing that it is usefull and normalizing data and seeing the effects myself.

```
Cars numeric stats:

Minimum Value:       
LowPrice       12.270
HighPrice      15.395
CityMPG        12.000
HwyMPG         18.000
FuelCap         9.000
Length        145.000
Width          63.000
Wheelbase      92.000
Height         49.000
UTurn          32.000
Weight       2085.000
Acc030          1.600
Acc060          4.100
QtrMile        12.400
PageNum        98.000
dtype: float64

Maximum Value:
LowPrice       84.3
HighPrice     194.6
CityMPG        37.0
HwyMPG         44.0
FuelCap        33.5
Length        224.0
Width          81.0
Wheelbase     131.0
Height         79.0
UTurn          45.0
Weight       6265.0
Acc030          4.4
Acc060         12.8
QtrMile        19.4
PageNum       223.0
dtype: float64

Median:
LowPrice       29.7675
HighPrice      42.0100
CityMPG        20.0000
HwyMPG         28.0000
FuelCap        18.5000
Length        190.0000
Width          73.0000
Wheelbase     110.0000
Height         58.0000
UTurn          39.0000
Weight       3772.5000
Acc030          3.0500
Acc060          7.9000
QtrMile        16.2000
PageNum       148.5000
dtype: float64

Mean:
LowPrice       32.808082
HighPrice      49.124309
CityMPG        20.781818
HwyMPG         29.363636
FuelCap        18.004545
Length        187.281818
Width          73.281818
Wheelbase     110.154545
Height         61.427273
UTurn          39.063636
Weight       3846.000000
Acc030          3.069091
Acc060          7.937273
QtrMile        16.102727
PageNum       154.036364
dtype: float64

Standard Devation:
LowPrice      15.926386
HighPrice     28.196937
CityMPG        4.546158
HwyMPG         5.536745
FuelCap        4.374224
Length        14.468017
Width          3.629977
Wheelbase      7.782816
Height         6.602000
UTurn          2.335515
Weight       867.496080
Acc030         0.553843
Acc060         1.645169
QtrMile        1.310190
PageNum       36.865719
dtype: float64

Variance:
LowPrice        253.649761
HighPrice       795.067238
CityMPG          20.667556
HwyMPG           30.655546
FuelCap          19.133832
Length          209.323520
Width            13.176731
Wheelbase        60.572227
Height           43.586405
UTurn             5.454629
Weight       752549.449541
Acc030            0.306742
Acc060            2.706580
QtrMile           1.716598
PageNum        1359.081234
dtype: float64
```
![original_covariance](https://i.imgur.com/FDtkSaV.png)

```
Cars numeric normalized by range stats:

Minimum Value:
LowPrice     0.0
HighPrice    0.0
CityMPG      0.0
HwyMPG       0.0
FuelCap      0.0
Length       0.0
Width        0.0
Wheelbase    0.0
Height       0.0
UTurn        0.0
Weight       0.0
Acc030       0.0
Acc060       0.0
QtrMile      0.0
PageNum      0.0
dtype: float64

Maximum Value:
LowPrice     1.0
HighPrice    1.0
CityMPG      1.0
HwyMPG       1.0
FuelCap      1.0
Length       1.0
Width        1.0
Wheelbase    1.0
Height       1.0
UTurn        1.0
Weight       1.0
Acc030       1.0
Acc060       1.0
QtrMile      1.0
PageNum      1.0
dtype: float64

Median:
LowPrice     0.242920
HighPrice    0.148517
CityMPG      0.320000
HwyMPG       0.384615
FuelCap      0.387755
Length       0.569620
Width        0.555556
Wheelbase    0.461538
Height       0.300000
UTurn        0.538462
Weight       0.403708
Acc030       0.517857
Acc060       0.436782
QtrMile      0.542857
PageNum      0.404000
dtype: float64

Mean:
LowPrice     0.285132
HighPrice    0.188216
CityMPG      0.351273
HwyMPG       0.437063
FuelCap      0.367532
Length       0.535213
Width        0.571212
Wheelbase    0.465501
Height       0.414242
UTurn        0.543357
Weight       0.421292
Acc030       0.524675
Acc060       0.441066
QtrMile      0.528961
PageNum      0.448291
dtype: float64

Standard Devation:
LowPrice     0.221108
HighPrice    0.157345
CityMPG      0.181846
HwyMPG       0.212952
FuelCap      0.178540
Length       0.183139
Width        0.201665
Wheelbase    0.199559
Height       0.220067
UTurn        0.179655
Weight       0.207535
Acc030       0.197801
Acc060       0.189100
QtrMile      0.187170
PageNum      0.294926
dtype: float64

Variance:
LowPrice     0.048889
HighPrice    0.024757
CityMPG      0.033068
HwyMPG       0.045348
FuelCap      0.031876
Length       0.033540
Width        0.040669
Wheelbase    0.039824
Height       0.048429
UTurn        0.032276
Weight       0.043071
Acc030       0.039125
Acc060       0.035759
QtrMile      0.035033
PageNum      0.086981
dtype: float64
```
![range_covariance](https://i.imgur.com/TSxj4yY.png)

```
Cars numeric normalized by standard deviation stats:

Minimum Value:
LowPrice    -1.295465
HighPrice   -1.201679
CityMPG     -1.940542
HwyMPG      -2.061797
FuelCap     -2.067968
Length      -2.935809
Width       -2.845438
Wheelbase   -2.343321
Height      -1.890965
UTurn       -3.038287
Weight      -2.039271
Acc030      -2.664679
Acc060      -2.343124
QtrMile     -2.839034
PageNum     -1.526969
dtype: float64

Maximum Value:
LowPrice     3.247917
HighPrice    5.182885
CityMPG      3.583775
HwyMPG       2.655594
FuelCap      3.558659
Length       2.549501
Width        2.135966
Wheelbase    2.690653
Height       2.673910
UTurn        2.553413
Weight       2.801247
Acc030       2.414041
Acc060       2.969290
QtrMile      2.528155
PageNum      1.879233
dtype: float64

Median:
LowPrice    -0.191788
HighPrice   -0.253463
CityMPG     -0.172760
HwyMPG      -0.247416
FuelCap      0.113785
Length       0.188735
Width       -0.077992
Wheelbase   -0.019948
Height      -0.521502
UTurn       -0.027372
Weight      -0.085114
Acc030      -0.034628
Acc060      -0.022760
QtrMile      0.074583
PageNum     -0.150864
dtype: float64

Mean:
LowPrice     4.642751e-17
HighPrice   -1.352454e-16
CityMPG     -1.695613e-16
HwyMPG       3.229740e-17
FuelCap     -5.652044e-16
Length      -4.844610e-16
Width       -3.552714e-16
Wheelbase   -1.033517e-15
Height      -6.136505e-16
UTurn        8.074349e-16
Weight      -6.459479e-16
Acc030       5.571301e-16
Acc060      -8.881784e-16
QtrMile      1.566424e-15
PageNum      6.863197e-17
dtype: float64

Standard Devation:
LowPrice     1.004577
HighPrice    1.004577
CityMPG      1.004577
HwyMPG       1.004577
FuelCap      1.004577
Length       1.004577
Width        1.004577
Wheelbase    1.004577
Height       1.004577
UTurn        1.004577
Weight       1.004577
Acc030       1.004577
Acc060       1.004577
QtrMile      1.004577
PageNum      1.004577
dtype: float64

Variance:
LowPrice     1.009174
HighPrice    1.009174
CityMPG      1.009174
HwyMPG       1.009174
FuelCap      1.009174
Length       1.009174
Width        1.009174
Wheelbase    1.009174
Height       1.009174
UTurn        1.009174
Weight       1.009174
Acc030       1.009174
Acc060       1.009174
QtrMile      1.009174
PageNum      1.009174
dtype: float64
```
![z_score_covariance](https://i.imgur.com/GmF40q4.png)