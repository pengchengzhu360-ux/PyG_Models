# PyG_Models 项目说明

本项目包含了基于 PyTorch Geometric (PyG) 实现的多种图神经网络模型，包括 DimeNet, GCN 和 SchNet。

---

## 📂 模型说明

### 1. DimeNet / DimeNet++
位于 `dimenet/` 文件夹。
* **运行方式**：包含使用 Jupyter Notebook 编写的训练代码和推理代码。
* **模型切换**：
    * **DimeNet**：请使用 `config.yaml` 配置文件。
    * **DimeNet++**：请使用 `config_pp.yaml` 配置文件。
* **注意**：修改代码中的路径指向对应的配置文件即可完成模型切换。

### 2. GCN (Graph Convolutional Networks)
位于 `gcn-master/` 文件夹。
* **运行方式**：请参考该文件夹下的 `README.md` 获取详细的终端运行指令。
* **功能**：运行后代码会自动进行模型训练，并在新的数据集上进行推理，最终输出预测精度。

### 3. SchNet
位于 `schnet/` 文件夹。
* **运行方式**：包含模型训练代码，直接运行即可。
* **补充说明**：目前的推理代码可参考(https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_pretrained_schnet.py)，后续我会将集成好的推理代码更新至本项目。
