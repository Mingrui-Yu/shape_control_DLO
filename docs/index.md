# Shape Control of Deformable Linear Objects with Offline and Online Learning of Local Linear Deformation Models

This paper has been accepted by the 2022 International Conference on Robotics and Automation (ICRA 2022).

The journal version of this work has been accepted by **IEEE Transactions on Robotics (T-RO)**, in which the method is further improved. [[website](https://mingrui-yu.github.io/shape_control_DLO_2/)]

<p align="center">
<iframe width="640" height="360" src="https://www.youtube.com/embed/au4TDZFrFHc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

(There is a small mistake in the video that the diameters of the DLOs used for collecting offline data should be 6mm~30mm, not 3mm~15mm.)

[[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9812244)] [[arXiv](https://arxiv.org/abs/2109.11091)]  [[Code](https://github.com/Mingrui-Yu/shape_control_DLO)]

## Abstract
The shape control of deformable linear objects (DLOs) is challenging, since it is difficult to obtain the deformation models. Previous studies often approximate the models in purely offline or online ways. In this paper, we propose a scheme for the shape control of DLOs, where the unknown model is estimated with both offline and online learning. The model is formulated in a local linear format, and approximated by a neural network (NN). First, the NN is trained offline to provide a good initial estimation of the model, which can directly migrate to the online phase. Then, an adaptive controller is proposed to achieve the shape control tasks, in which the NN is further updated online to compensate for any errors in the offline model caused by insufficient training or changes of DLO properties. The simulation and real-world experiments show that the proposed method can precisely and efficiently accomplish the DLO shape control tasks, and adapt well to new and untrained DLOs.

Please refer to our paper for more details and results.


## Citation
Please cite our paper if you find it helpful :)

```
@INPROCEEDINGS{yu2022shape,
  author={Yu, Mingrui and Zhong, Hanzhong and Li, Xiang},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Shape Control of Deformable Linear Objects with Offline and Online Learning of Local Linear Deformation Models}, 
  year={2022},
  volume={},
  number={},
  pages={1337-1343},
  doi={10.1109/ICRA46639.2022.9812244}}
```
