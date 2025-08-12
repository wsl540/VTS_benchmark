# Variable-length Time Series Classification: Benchmarking, Analysis and Effective Spectral Pooling Strategy
We present the first comprehensive benchmark for variable-length time series classification tasks, evaluating the effectiveness of 22 previously widely-used length normalization methods across 14 publicly available VTS datasets and 8 backbones, and propose a novel spectral pooling layer to process variable-length time series.



# Baseline Model
- **MLP** - Time series classification from scratch with deep neural networks: A strong baseline [\[IJCNN 2017\]](https://arxiv.org/pdf/1611.06455.pdf)[\[Code\]](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline)
- **LSTM** - A comparison of pooling methods on LSTM models for rare acoustic event classification [\[ICASSP 2020\]](https://arxiv.org/pdf/2002.06279.pdf)
- **FCN** - Time series classification from scratch with deep neural networks: A strong baseline [\[IJCNN 2017\]](https://arxiv.org/pdf/1611.06455.pdf)[\[Code\]](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline)
- **Resnet** - Time series classification from scratch with deep neural networks: A strong baseline [\[IJCNN 2017\]](https://arxiv.org/pdf/1611.06455.pdf)[\[Code\]](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline)
- **Inception** - Inceptiontime: Finding alexnet for time series classification [\[Data Mining and Knowledge Discovery 2020\]](https://arxiv.org/pdf/1909.04939.pdf)[\[Code\]](https://github.com/hfawaz/InceptionTime)
- **Transformer** - Attention is all you need [\[NeurlPS 2017\]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [\[Code\]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)
- **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting  [\[AAAI 2021\]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [\[Code\]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
- **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [\[ICLR 2023\]](https://openreview.net/pdf?id=ju_Uqw384Oq) [\[Code\]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)

# length normalization strategies
##  Pre-processing
### Padding
- Zero Pad (Pre/Post/Outer/Mid)
- Noise Pad (Pre/Post/Outer)
- Edge Pad (Pre/Post/Outer)
- STRP Pad
- Random Pad
- Zoom Pad

### Truncation
- Truncate (Pre/Post/Outer)

### Resampling
- Linear Interpolate
- Frequency Selection

### Warping
- Nearest Guided Warping- α
- Nearest Guided Warping- αβ

## Pooling
- Adaptive Max Pooling
- Adaptive Average Pooling
- Spectral Pooling

# Usage
1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```
2. Prepare Data. You can obtain datasets from [the website](https://www.timeseriesclassification.com/dataset.php), then place the downloaded data in the folder ./data/ and make sure the file format is .tsv. For example

```
./data/ShakeGestureWiimoteZ/ShakeGestureWiimoteZ_TRAIN.tsv
```


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder ./scripts/. For example
```
bash FCN.sh
```

4. Develop your own model.
- Add the model file to the folder `./models`. 
- Include the newly added model in the `Exp_Main.model_dict` of `./exp/exp_main.py`.
- Create the corresponding scripts under the folder `./scripts`.

# Acknowledgement
Thanks to the open source of the following projects:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [vary_length_time_series](https://github.com/uchidalab/vary_length_time_series)
