# Chainplot
Easy plotting via method chaining in python (yet another matplotlib wrapper)

Api ~~stolen~~ inspired by ggplot2 (and some of pandas)

## Example:
```python
import pandas as pd
from chainplot import Plot

data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv')

(Plot(data)
 .map(x='Sepal.Length', y='Petal.Length', by='Species')
 .layer_points())
```
 
