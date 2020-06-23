# NER-0
使用的网络模型是bilstm+crf模型
模型结构为：![model](https://github.com/jinlianchao185874/NER-0/blob/master/model.png)
数据集总共684条数据，训练使用600条，评估使用84条，实验结果如下：![jieguo](https://github.com/jinlianchao185874/NER-0/blob/master/train%20and%20%20eval.jpg)
使用一条数据进行预测结果如下：测试数据为：'李明今天想去华为公司面试'结果为：['person: 李', 'location:', 'organzation:天华公司']
该项目最大的一个问题是数据集过少，总共就684条数据，按照词频建立词库表所有的都算上才有1866个，相比于有几万个的词库来说太小了。

