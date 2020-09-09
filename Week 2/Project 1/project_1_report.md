# Report
## Question
- A question I wanted to answer by analyzing my chosen dataset was whether type of car and 0 to 60 Acceleration were related to City MPG.
    - I chose this dataset because I worked with it in the Dealing with Data 1 course recently.
    - Through this analysis I hope to learn which variables in cars are related such as Acceleration from 0 to 60 and City MPG.
## Visualizations
- What I learned about the data by visualizing it this way is that there appears to be a relationship between Acceleration from 0 to 60 and City MPG. There also seems to be a best type of car for getting high City MPG. Which aligns with the logical assumption that faster cars guzzle more gas. 
- I chose to visualize my dataset these ways because each of them shows a clear relationship between variables. The Joint Plot helps me answer my question because it shows a trend of lower City MPG as cars have faster 0 to 60 Acceleration.

**Bar/Box Plots**: These plots show the relationship between type of car and City MPG
![BarBoxPlots](https://i.imgur.com/mPoS4e4.png)

**Joint Plot**: This plot shows the relationship between Acceleration from 0 to 60 and City MPG by density.
![Jointplot](https://i.imgur.com/1oa2prc.png)

**Pair Plot**: This plot is rediculously large but shows trends between every variable in the dataset as a whole.
![Pairplot](https://i.imgur.com/2NOor4x.png)

## Numeric Results
- From these values I learned that the relationship between City MPG and Acceleration from 0 to 60 have a positive relationship. Meaning As Acceleration from 0 to 60 increases City MPG increases so faster cars have worse city miles per gallon.
- I computed these values because they give a general overview of the data. More specifically to my question I computed covariance to determine the relationship between variables. They helped me answer my question by confirming there is a possible relationship between City MPG and 0 to 60 Acceleration.

**Units**
| LowPrice  | $1000   |
|-----------|---------|
| HighPrice | $1000   |
| CityMPG   | MPG     |
| HwyMPG    | MPG     |
| FuelCap   | Gallons |
| Length    | Inches  |
| Width     | Inches  |
| Wheelbase | Inches  |
| Height    | Inches  |
| UTurn     | Feet    |
| Weight    | Pounds  |
| Acc030    | Seconds |
| Acc060    | Seconds |
| QtrMile   | Seconds |


**Minimum Value**
| HighPrice | 15.395   |
|-----------|----------|
| CityMPG   | 12.000   |
| HwyMPG    | 18.000   |
| FuelCap   | 9.000    |
| Length    | 145.000  |
| Width     | 63.000   |
| Wheelbase | 92.000   |
| Height    | 49.000   |
| UTurn     | 32.000   |
| Weight    | 2085.000 |
| Acc030    | 1.600    |
| Acc060    | 4.100    |
| QtrMile   | 12.400   |

**Maximum Value**
| LowPrice  | 84.3   |
|-----------|--------|
| HighPrice | 194.6  |
| CityMPG   | 37.0   |
| HwyMPG    | 44.0   |
| FuelCap   | 33.5   |
| Length    | 224.0  |
| Width     | 81.0   |
| Wheelbase | 131.0  |
| Height    | 79.0   |
| UTurn     | 45.0   |
| Weight    | 6265.0 |
| Acc030    | 4.4    |
| Acc060    | 12.8   |
| QtrMile   | 19.4   |

**Median**
| LowPrice  | 29.7675   |
|-----------|-----------|
| HighPrice | 42.0100   |
| CityMPG   | 20.0000   |
| HwyMPG    | 28.0000   |
| FuelCap   | 18.5000   |
| Length    | 190.0000  |
| Width     | 73.0000   |
| Wheelbase | 110.0000  |
| Height    | 58.0000   |
| UTurn     | 39.0000   |
| Weight    | 3772.5000 |
| Acc030    | 3.0500    |
| Acc060    | 7.9000    |
| QtrMile   | 16.2000   |

**Mean**
| LowPrice  | 32.808082   |
|-----------|-------------|
| HighPrice | 49.124309   |
| CityMPG   | 20.781818   |
| HwyMPG    | 29.363636   |
| FuelCap   | 18.004545   |
| Length    | 187.281818  |
| Width     | 73.281818   |
| Wheelbase | 110.154545  |
| Height    | 61.427273   |
| UTurn     | 39.063636   |
| Weight    | 3846.000000 |
| Acc030    | 3.069091    |
| Acc060    | 7.937273    |
| QtrMile   | 16.102727   |

**Standard Deviation**
| LowPrice  | 15.926386  |
|-----------|------------|
| HighPrice | 28.196937  |
| CityMPG   | 4.546158   |
| HwyMPG    | 5.536745   |
| FuelCap   | 4.374224   |
| Length    | 14.468017  |
| Width     | 3.629977   |
| Wheelbase | 7.782816   |
| Height    | 6.602000   |
| UTurn     | 2.335515   |
| Weight    | 867.496080 |
| Acc030    | 0.553843   |
| Acc060    | 1.645169   |
| QtrMile   | 1.310190   |

**Variance**
| LowPrice  | 253.649761    |
|-----------|---------------|
| HighPrice | 795.067238    |
| CityMPG   | 20.667556     |
| HwyMPG    | 30.655546     |
| FuelCap   | 19.133832     |
| Length    | 209.323520    |
| Width     | 13.176731     |
| Wheelbase | 60.572227     |
| Height    | 43.586405     |
| UTurn     | 5.454629      |
| Weight    | 752549.449541 |
| Acc030    | 0.306742      |
| Acc060    | 2.706580      |
| QtrMile   | 1.716598      |

**Covariance**
|           | LowPrice    | HighPrice    | CityMPG      | HwyMPG       | FuelCap     | Length       | Width       | Wheelbase   | Height      | UTurn       | Weight       | Acc030      | Acc060      | QtrMile     |
|-----------|-------------|--------------|--------------|--------------|-------------|--------------|-------------|-------------|-------------|-------------|--------------|-------------|-------------|-------------|
| LowPrice  | 253.649761  | 406.936682   | -46.809019   | -52.270663   | 39.946901   | 108.191215   | 27.569747   | 56.967171   | 2.533396    | 14.726527   | 7601.549183  | -6.672430   | -19.469346  | -15.945119  |
| HighPrice | 253.649761  | 406.936682   | -46.809019   | -52.270663   | 39.946901   | 108.191215   | 27.569747   | 56.967171   | 2.533396    | 14.726527   | 7601.549183  | -6.672430   | -19.469346  | -15.945119  |
| CityMPG   | -46.809019  | -71.268675   | 20.667556    | 23.511259    | -15.340284  | -47.433361   | -12.937948  | -24.461384  | -11.566472  | -7.729108   | -3261.201835 | 1.621635    | 5.071510    | 3.850142    |
| HwyMPG    | -52.270663  | -75.959022   | 23.511259    | 30.655546    | -18.275980  | -50.993328   | -15.011676  | -27.634696  | -19.789825  | -8.784821   | -4026.009174 | 1.572811    | 4.713845    | 3.542118    |
| FuelCap   | 39.946901   | 57.622420    | -15.340284   | -18.275980   | 19.133832   | 51.769349    | 13.537239   | 26.741493   | 16.632902   | 7.713470    | 3453.876147  | -1.130133   | -3.506226   | -2.553315   |
| Length    | 108.191215  | 158.000408   | -47.433361   | -50.993328   | 51.769349   | 209.323520   | 42.461134   | 103.488157  | 44.236280   | 28.303003   | 10282.467890 | -3.021485   | -11.096839  | -7.883344   |
| Width     | 27.569747   | 38.246628    | -12.937948   | -15.011676   | 13.537239   | 42.461134    | 13.176731   | 21.359716   | 14.768390   | 6.514012    | 2854.440367  | -0.833411   | -2.773903   | -1.960409   |
| Wheelbase | 56.967171   | 85.699429    | -24.461384   | -27.634696   | 26.741493   | 103.488157   | 21.359716   | 60.572227   | 25.355379   | 14.696497   | 5446.449541  | -1.346555   | -4.903061   | -3.525196   |
| Height    | 2.533396    | -17.708445   | -11.566472   | -19.789825   | 16.632902   | 44.236280    | 14.768390   | 25.355379   | 43.586405   | 8.458799    | 4056.036697  | 0.768374    | 2.238974    | 2.186897    |
| UTurn     | 14.726527   | 20.392384    | -7.729108    | -8.784821    | 7.713470    | 28.303003    | 6.514012    | 14.696497   | 8.458799    | 5.454629    | 1613.009174  | -0.461318   | -1.592302   | -1.135038   |
| Weight    | 7601.549183 | 10410.322578 | -3261.201835 | -4026.009174 | 3453.876147 | 10282.467890 | 2854.440367 | 5446.449541 | 4056.036697 | 1613.009174 | 75259.449541 | -199.166055 | -620.702752 | -439.800917 |
| Acc030    | -6.672430   | -11.481786   | 1.621635     | 1.572811     | -1.130133   | -3.021485    | -0.833411   | -1.346555   | 0.768374    | -0.461318   | -199.166055  | 0.306742    | 0.865933    | 0.692746    |
| Acc060    | -19.469346  | -33.309973   | 5.071510     | 4.713845     | -3.506226   | -11.096839   | -2.773903   | -4.903061   | 2.238974    | -1.592302   | -620.702752  | 0.865933    | 2.706580    | 2.137512    |
| QtrMile   | -15.945119  | -27.953229   | 3.850142     | 3.542118     | -2.553315   | -7.883344    | -1.960409   | -3.525196   | 2.186897    | -1.135038   | -439.800917  | 0.692746    | 2.137512    | 1.716598    |

## Conclusions
- The audience should take away from the Bar and Box Plots that there is a best type of car for City Miles per Gallon, from the Joint Plot that there is a positive relationship between City MPG and Acceleration from 0 to 60, and from the Pair Plot that there are many possible relations between many variables when it comes to cars. When it comes to numeric data the audience should take away from covariance that there is a relationship between City MPG and Acceleration from 0 to 60.
- What I learned about about the world by analyzing this dataset is that there are relationships between a lot of things and you may or may not see it until you visualize some data.
- What I learned about data by engaging with this project is that it is hard to pick which stats to back up your conclusions with. Dealing with all of these numbers can get overwhelming if you don't know specifically what you want to do with them.