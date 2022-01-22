from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# 源代码链接：
# https://zhuanlan.zhihu.com/p/429964726?utm_source=wechat_session&utm_medium=social&utm_oi=985859367580811264
# 定义GCN空域图卷积神经网络
class GCNConv(MessagePassing, ABC):
    # 网络初始化
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 节点属性向量的维度
        :param out_channels: 经过图卷积之后，节点的特征表示维度
        """
        # 定义伽马函数为求和函数,aggr='add'
        super(GCNConv, self).__init__(aggr='add')
        # 定义最里面那个线性变换
        # 具体到实现中就是一个线性层
        self.linear_change = torch.nn.Linear(in_channels, out_channels)

    # 定义信息汇聚函数
    def message(self, x_j, norm):
        # 正则化
        # norm.view(-1,1)将norm变为一个列向量
        # x_j是节点的特征表示矩阵
        return norm.view(-1, 1) * x_j

    # 前向传递，进行图卷积
    def forward(self, x, edge_index):
        """
        :param x:图中的节点，维度为[节点数,节点属性相邻维度数]
        :param edge_index: 图中边的连接信息,维度为[2,边数]
        :return:
        """
        # 添加节点到自身的环
        # 因为节点最后面汇聚相邻节点信息时包含自身
        # add_self_loops会在edge_index边的连接信息表中，
        # 添加形如[i,i]这样的信息
        # 表示一个节点到自身的环
        # 函数返回[边的连接信息，边上的属性信息]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 进行线性变换
        x = self.linear_change(x)
        # 计算外面的正则化
        row, col = edge_index
        # 获取节点的度
        deg = degree(col, x.size(0), dtype=x.dtype)
        # 带入外面的正则化公式
        deg_inv_sqrt = deg.pow(-0.5)
        # 将未知的值设为0，避免下面计算出错
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # 正则化部分
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 进行信息传递和融合
        # propagate会自动调用self.message函数，并将参数传递给它
        return self.propagate(edge_index, x=x, norm=norm)


# 测试我们刚才定义的图卷积神经网络
if __name__ == '__main__':
    # 实例化一个图卷积神经网络
    # 并假设图节点属性向量的维度为16，图卷积出来的节点特征表示向量维度为32
    conv = GCNConv(16, 32)
    # 随机生成一个节点属性向量
    # 5个节点，属性向量为16维
    x = torch.randn(4, 16)
    # 随机生成边的连接信息
    # 假设有3条边
    edge_index = [
        [0, 1, 1, 2, 1, 3],
        [1, 0, 2, 1, 3, 1]
    ]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # 进行图卷积
    output = conv(x, edge_index)
    # 输出卷积之后的特征表示矩阵
    print(output.data)