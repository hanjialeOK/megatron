# DATA

## 简介

该分支相较于官方 main 分支的区别是：

- 增加 data/ 文件夹，存放训练数据集。
- 增加 preprocess_data.sh，用于生成 .bin 和 .idx 数据。
- 修改 examples/pretrain_bert_distributed.sh，适配 ARNOLD 平台。
- 修改 examples/pretrain_bert_distributed_with_mp.sh，适配 ARNOLD 平台。
- 修改 如果 .npy 文件不存在，则在每个 worker 里都生成。参考 [Code fixes for local-storage-only environment #356](https://github.com/NVIDIA/Megatron-LM/issues/356)
  - megatron/data/dataset_utils.py: torch.distributed.get_rank() % torch.cuda.device_count()
  - megatron/data/gpt_dataset.py: torch.distributed.get_rank() % torch.cuda.device_count()
  - 在训练脚本中预编译 megatron/data/helpers.cpython-39-x86_64-linux-gnu.so，防止其他 worker 不编译该文件导致出错。参考 [Cannot import C++ compiled "helpers" #143](https://github.com/NVIDIA/Megatron-LM/issues/143)
- 修改 megatron/training.py: os.system("nvidia-smi")，分别在加载模型和加载数据后打印显存占用情况，在日志中搜索 `GPU Memory Used` 可以找到打印结果。环境变量 `SLEEP` 可以指定在两次打印处睡眠的时间。

## 镜像

目前使用的基础镜像是 [maybe_magatron_base:latest](https://cloud.bytedance.net/image/544076)

## 数据准备

数据来自知乎的一篇文章 [[细读经典]Megatron论文和代码详细分析(2)](https://zhuanlan.zhihu.com/p/388830967)

在项目根目录下首先需要执行脚本生成 _ss.json 文件。

```c
bash preprocess_data.sh
```

然后再一次执行上面的指令生成 .bin 和 .idx 文件。

## 训练

普通分布式

```c
# bert
bash examples/pretrain_bert_distributed.sh
# gpt
bash examples/pretrain_gpt_distributed.sh
```

模型并行

```c
# bert
bash examples/pretrain_bert_distributed_with_mp.sh
# gpt
bash examples/pretrain_gpt_distributed_with_mp.sh
```
