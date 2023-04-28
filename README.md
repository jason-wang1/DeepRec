# DeepRec
DeepRec 是一个易用、易扩展、模块化的深度学习推荐算法项目，采用 `TensorFlow 2`实现
- 通过子类化`tf.keras.Model`来构建每一个算法模型
- 每个模型充分利用`tf.keras.Layer`的各种子类之间相互组合来实现对应的功能
- 使用`tf.data`API，轻松读取各种格式数据及预处理，从容应对海量样本
- 每次训练的模型、样本、训练参数均定义在`config/`文件夹下的一个配置文件中，运行`main/run.py`并传入配置文件路径即可开始训练
- 每个算法模型均使用公开数据集进行了验证

## 环境要求
* TensorFlow 2.6
* Python 3.8

## 数据集
- [淘宝展示广告点击率预估数据集](https://tianchi.aliyun.com/dataset/56)
- [Avazu数据集](https://www.kaggle.com/c/avazu-ctr-prediction/data?spm=a2c4g.11186623.0.0.46319507CFcyDY)

## 实现的模型
| 模型   |  描述   |
| ---- | ---- |
|   [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)   |   在线性模型基础上进行了二阶特征交叉，且适用于高维稀疏特征样本   |
|   [AFM](https://arxiv.org/pdf/1708.04617.pdf)   |   在FM交叉特征中引入注意力机制，学习不同特征交叉的重要程度，降低无效特征交叉的权重   |
|   [DeepFM](https://www.ijcai.org/proceedings/2017/0239.pdf)   |   在wide侧使用FM进行二阶特征交叉，Deep侧使用多层神经网络   |
|   TwoTowerDeepFM   |   深度向量双塔模型，并把FM双塔化，实现了user塔与item塔的显示特征交叉   |
|   [ESMM](https://arxiv.org/abs/1804.07931)   |   通过预估CTR、CTCVR来间接预估CVR，缓解传统CVR预估的样本选择偏差、样本稀疏问题   |
|   [DIN](https://arxiv.org/pdf/1706.06978.pdf)   |   根据候选item对用户行为序列中的item加权   |

## 特征输入编码规则
特征编码为dict结构：
* key为特征名
* value为dict结构，包含两个kv元素：
  * 第一个kv：
    * key为"index"
    * value为tensor，dtype支持int32、float32、string
  * 第二个kv：
    * key为"value"
    * value为tensor，dtype支持float32

注意：
* 所有tensor必须为二维张量，形状为[batch_size, tag_size]
  * batch_size：批量训练时批次大小
  * tag_size：对于单一特征为1；序列特征为约定的一个大于1的固定值，每个序列特征的tag_size可不同
* 对于某一特定的特征，其index和value两个tensor的形状相同

示例：
```
{
    '101': {
    'index': <tf.Tensor: shape=(3, 1), dtype=int32>, 
    'value': <tf.Tensor: shape=(3, 1), dtype=float32
    }, 
    '901': {
    'index': <tf.Tensor: shape=(3, 1), dtype=float32>, 
    'value': <tf.Tensor: shape=(3, 1), dtype=float32>
    }, 
    '109_14': {
    'index': <tf.Tensor: shape=(3, 5), dtype=string>, 
    'value': <tf.Tensor: shape=(3, 5), dtype=float32>
    }
}
```


## 配置文件
总共两个配置文件：
1. 第一个是样本配置文件，约定了从磁盘样本文件加载/转换到TensorFlow模型输入所要求格式
2. 第二个是模型训练过程中的配置文件

通过配置，能够方便地进行以下操作：
* 增加、删除特征
* 指定特征预处理方法
* 指定模型结构及其具体的参数，如DNN形状
* 指定优化器参数、损失函数、观测指标、正则方式等
### 样本配置文件
json格式
* train_input_path：样本文件所在磁盘路径
* data_config：样本参数
  * input_type：磁盘中的样本文件格式，现仅支持CSV格式，即"CSVInput"
  * input_attributes：有序字典，顺序为csv文件中的字段顺序
    * key为特征或标签名称
    * value为特征或标签的属性
      * field_type：对于单一特征为SingleFeature；对于不带权重的序列特征为TagFeature；对于带权重的序列特征为WeightTagFeature
      * index_type：支持int、float、string三种类型
      * pad_num：序列特征截断或填充后的数量，序列特征必须填写
  * label_fields：训练模型时所用到的label
  * feature_fields：训练模型时所用到的feature
  * batch_size：训练批次大小
  * prefetch_size：提前读取一批样本到内存中，减少模型训练时读取数据延迟

示例：
```json
{
  "train_input_path": "..\\data\\sample_test.csv",
  "data_config": {
    "label_fields": ["click", "conversion"],
    "feature_fields": ["101", "901", "109_14", "210", "902", "903", "904"],
    "input_attributes": {
      "sample_id": {"field_type": "SingleFeature", "index_type": "int"},
      "click": {"field_type": "ClassLabel"},
      "conversion": {"field_type": "ClassLabel"},
      "101": {"field_type": "SingleFeature", "index_type": "int"},
      "901": {"field_type": "SingleFeature", "index_type": "float"},
      "109_14": {"field_type": "WeightTagFeature", "index_type": "int", "pad_num": 5},
      "210": {"field_type": "TagFeature", "index_type": "int", "pad_num": 5},
      "902": {"field_type": "SingleFeature", "index_type": "string"},
      "903": {"field_type": "WeightTagFeature", "index_type": "string", "pad_num": 5},
      "904": {"field_type": "TagFeature", "index_type": "string", "pad_num": 5}
    },
    "batch_size":3,
    "prefetch_size": 0,
    "input_type": "CSVInput"
  }
}
```

### 训练模型配置文件
json格式：
* model_config：模型参数
  * model_type：模型类型，现支持ESMM、DIN、DeepFM、AFM、FM
  * feature_groups：每个塔有一个特征簇
    * key为特征簇名称
    * value为特征名及其预处理方式（转换为从0开始的连续的自然数，作为后续embedding层的输入）
      * input_names：特征名称，离散特征为单个特征值，连续特征为一个列表
      * hash_bucket_size：类别特征hash。当index_type为int或string时可以使用此方法
      * boundaries：连续特征分桶。当index_type为float时可以使用此方法
      * int_vocab_list：int类别特征vocab
      * str_vocab_list：string类别特征vocab
  * embedding_dim：embedding维度
  * deep_hidden_units：DNN形状
  * l2_reg：l2正则系数
* train_config：训练参数
  * epochs：训练轮次
  * steps_per_epoch：每轮训练批次
  * loss：损失函数
  * metrics：评价指标
  * optimizer：优化器
    * name：优化器类型
    * learning_rate：学习率

示例：
```json
{
  "model_config": {
    "model_type": "ESMM",
    "feature_groups": {
      "user": [
        {"input_names": "101", "hash_bucket_size": 100000},
        {"input_names": "901", "boundaries": [0.0, 0.2, 0.6, 1.0]},
        {"input_names": "109_14", "hash_bucket_size": 10000},
        {"input_names": "903", "hash_bucket_size": 10000}
      ],
      "item": [
        {"input_names": "210", "hash_bucket_size": 100000},
        {"input_names": "902", "hash_bucket_size": 100000},
        {"input_names": "904", "hash_bucket_size": 1000}
      ],
      "cross": [
        {"input_names": ["101", "902"], "hash_bucket_size": 100000}
      ]
    },
    "embedding_dim": 16,
    "deep_hidden_units": [128, 64, 32, 8, 1],
    "l2_reg": 0.0001
  },
  "train_config": {
    "epochs": 2,
    "steps_per_epoch":100,
    "loss": ["binary_crossentropy", "binary_crossentropy"],
    "metrics": ["AUC"],
    "optimizer": {
      "name": "adam",
      "learning_rate": {
          "schedule": "exponential_decay",
          "initial_learning_rate": 0.0001,
          "decay_steps": 10000,
          "decay_rate": 0.5
      }
    }
  }
}
```