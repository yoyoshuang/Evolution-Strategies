# Evolution Strategies on mnist task

This project uses [Evolution Strategies](https://blog.openai.com/evolution-strategies/) on mnist task. 

The structure of the network is followed by [here](https://blog.openai.com/nonlinear-computation-in-linear-networks/).

## Results (ES vs BP)


| accuracy  | Layer number  | iteration times| npop| optimize method|GPU/CPU
|:------------- |:---------------:| -------------:| -------:|------:|-----:|
| 92.05% | 1 |       1000| none   | BP |        CPU |
| 99.11% | 3   | 20000 | none     | BP|         CPU|
| 90.94% |1       | 20000| 50     | ES |        CPU |
| 91.31% | 1      |  20000 | 50      | ES |        GPU |
| 83.32%| 3    | 10000| 10    | ES|         CPU |
| 85.8% | 3      |10000 | 20     | ES |         CPU |
| 85.53% | 3      |20000 | 10     | ES|         CPU |
| 82.5% | 3   |  10000 | 50    | ES|         CPU |
| 84.19% | 3    | 10000 | 10  | ES |        GPU |

## Question

The [writer](https://blog.openai.com/nonlinear-computation-in-linear-networks/) says ES method can get better results than BP (backpropagation) method.

But I results showed that it is NO, I don't know where is the problem.

