from bert4vec import Bert4Vec

model = Bert4Vec(mode='paraphrase-multilingual-minilm') 
sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
         '给我推荐一款红色的车', '我喜欢北京']
sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
         '给我推荐一款黑色的车', '我不喜欢北京']

similarity = model.similarity(sent1, sent2, return_matrix=False)
# similarity函数支持的默认输入参数有：batch_size=64, return_matrix=False
print(similarity)
