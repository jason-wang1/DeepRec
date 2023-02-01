# DeepRec
DeepRec 是一个易用、易扩展、模块化的深度学习推荐算法项目，采用 `TensorFlow 2`实现
- 通过子类化`tf.keras.Model`来构建每一个算法模型
- 每个模型充分利用各种`tf.keras.Layer`子类之间的相互组合来实现对应的功能
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
|   [DeepFM](https://www.ijcai.org/proceedings/2017/0239.pdf)   |   在wide侧使用FM进行二阶特征交叉，Deep侧使用多层神经网络   |
|   TwoTowerDeepFM   |   深度向量双塔模型，并把FM双塔化，实现了user塔与item塔的显示特征交叉   |

