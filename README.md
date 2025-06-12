# 印刷字体识别系统
## 简介
使用 Python 编写，带UI。  
可实现字体倾斜校正、行和字符分割、字符识别。  

倾斜校正使用传统的投影法。  
行分割采用 自适应扩展+连通域辅助+智能padding，字符分割采用 垂直投影+连通域+极小值法细分粘连字符。  
字体识别使用HOG算法，Chars74K数据集训练。  
该数据集可在[这里](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz)找到。  

详细信息已发布于：[阿里云社区](https://developer.aliyun.com/article/1666792)

还有一点，这个系统虽然对手写体具有一定的区分能力，但大部分情况下还是差强人意。  
可以试一下使用`torchvision `的 datasets 手写数据集来训练。

```python
from torchvision import datasets
train = datasets.EMNIST(root='data', split='letters', train=True, download=True)
```

## 使用方法
本项目集成性很高，使用简单。
1. 下载测试数据集和本仓库
2. 将数据集放到本仓库目录下改为`EnglishFnt.tgz`。
3. 运行
```bash
pip install -r requirements.txt   # 下载软件包
python main.py                    # 执行
```

## 后记
该系统还有诸多问题亟需解决。Such as the damn UI.  
不过作为课设应该是合格了。

Jessarin
2025/6/12
