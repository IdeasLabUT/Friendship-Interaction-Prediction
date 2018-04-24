# Leveraging Friendship Networks for Dynamic Link Prediction in Social Interaction Networks

This repository contains the MATLAB code to reproduce the experiments in our [ICWSM 2018 paper "Leveraging Friendship Networks for Dynamic Link Prediction in Social Interaction Networks"](https://arxiv.org/abs/1804.08584). We demonstrate that incorporating Facebook friendship networks into dynamic link prediction of future interaction networks (constructed from wall posts) significantly improves accuracy of interaction link predictions. However, incorporating predicted future friendship networks decreases interaction link prediction accuracy compared to incorporating actual friendships.

## Requirements

The MATLAB code requires first downloading and adding the following two MATLAB packages to your MATLAB path:

- [Dynamic Link Prediction Evaluation MATLAB Toolbox](https://github.com/IdeasLabUT/Dynamic-Link-Prediction-Evaluation)
- [Dynamic Stochastic Block Models MATLAB Toolbox](https://github.com/IdeasLabUT/Dynamic-Stochastic-Block-Model)

The raw Facebook data collected by Viswanath et al. (2009) can be obtained from http://socialnetworks.mpi-sws.org/data-wosn2009.html.

## Contents

Run scripts in the following order:

1. `ImportBothAdjMatFacebook.m`: Script to import raw CSV files into friendship and interaction network adjacency matrices and save as a MAT file.
2. `FilterAdjMatFacebook.m`: Script to filter out nodes with low degree to reduce size of network.
3. `FBFriendshipInteractionPrediction.m`: Script to run a variety of link predictors and evaluate accuracy. Some of the link predictors may take a significant amount of time to run. Computation time can be reduced (especially for the dynamic stochastic block model) if you have the MATLAB Parallel Computing Toolbox installed.

Additional files:

- `LICENSE.txt`: License for this software.
- `unixtime2serial.m`: MATLAB function used to convert Unix timestamp into MATLAB serial date format

## References

Junuthula, R. R., Xu, K. S., & Devabhaktuni, V. K. (2018). Leveraging friendship networks for dynamic link prediction in social interaction networks. In Proceedings of the 12th International AAAI Conference on Web and Social Media (to appear). [arXiv:1804.08584](https://arxiv.org/abs/1804.08584)

Viswanath, B., Mislove, A., Cha, M., & Gummadi, K. P. (2009). On the evolution of user interaction in Facebook. In Proceedings of the 2nd ACM Workshop on Online Social Networks (pp. 37–42). New York, New York, USA: ACM Press. https://doi.org/10.1145/1592665.1592675

## License

Distributed with a BSD license; see `LICENSE.txt`