# *x*R-EgoPose

This repository contains the unoffical pyTorch implementation of xR-EgoPose in the paper ["*x*R-EgoPose: Egocentric 3D Human Pose from an HMD Camera"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tome_xR-EgoPose_Egocentric_3D_Human_Pose_From_an_HMD_Camera_ICCV_2019_paper.pdf) (ICCV 2019, oral). (**Failed to implement similar performance reported on the paper.**)

I tried to implement (the paper didn't mention explicitly.)
* Neural Network Architecture
* Data Augmentation Strategy
* Training Details

## License and Citation

```
@inproceedings{tome2019xr,
  title={xR-EgoPose: Egocentric 3D Human Pose from an HMD Camera},
  author={Tome, Denis and Peluse, Patrick and Agapito, Lourdes and Badino, Hernan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={7728--7738},
  year={2019}
}
```

The license agreement for the data usage implies citation of the paper. Please notice that citing the dataset URL instead of the publication would not be compliant with this license agreement.

# Data Preparation

Follow the instruction of [Official xR-EgoPose Dataset Repository](https://github.com/facebookresearch/xR-EgoPose) to download.  
After downloading, open **data/config.yml**, and write your own dataset path.

# Results of implemented xR-EgoPose

</ul>
<table>
<thead>
<tr>
<th align="center">Dataset</th>
<th align="center">Ours</th>
<th align="center">CVPRW19</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Set5</td>
<td align="center">-- / --</td>
<td align="center">30.13 / 0.853</td>
</tr>
<tr>
<td align="center">Set14</td>
<td align="center">-- / --</td>
<td align="center">26.98 / 0.746</td>
</tr>
<tr>
<td align="center">Urban100</td>
<td align="center">-- / --</td>
<td align="center">24.31 / 0.716</td>
</tr>  
</tbody></table>

# Training

hi
