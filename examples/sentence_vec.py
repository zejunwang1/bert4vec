from bert4vec import Bert4Vec

# 支持四种模式：simbert-base/roformer-sim-base/roformer-sim-small/paraphrase-multilingual-minilm
model = Bert4Vec(mode='roformer-sim-small')   
sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
             '给我推荐一款红色的车', '我喜欢北京']

vecs = model.encode(sentences, convert_to_numpy=True, normalize_to_unit=False)
# encode函数支持的默认输入参数有：batch_size=64, convert_to_numpy=False, normalize_to_unit=False

print(vecs.shape)
print(vecs)
