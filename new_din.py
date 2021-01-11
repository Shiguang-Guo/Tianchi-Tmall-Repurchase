import numpy as np

from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


def get_xy_fd():
    # 对基础特征进行 embedding
    feature_columns = [SparseFeat('user', vocabulary_size=3, embedding_dim=10),
                       SparseFeat('gender', vocabulary_size=2, embedding_dim=4),
                       SparseFeat('item_id', vocabulary_size=3, embedding_dim=8),
                       SparseFeat('cate_id', vocabulary_size=2, embedding_dim=4),
                       DenseFeat('pay_score', 1)]

    # 指定历史行为序列对应的特征
    behavior_feature_list = ["item_id", "cate_id"]

    # 构造 ['item_id', 'cate_id'] 这两个属性历史序列数据的数据结构: hist_item_id, hist_cate_id
    # 由于历史行为是不定长数据序列，需要用 VarLenSparseFeat 封装起来，并指定序列的最大长度为 4 
    # 注意,对于长度不足4的部分会用0来填充,因此 vocabulary_size 应该在原来的基础上 + 1
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=2 + 1, embedding_name='cate_id'), maxlen=4)]

    # 基础特征数据
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])
    cate_id = np.array([1, 2, 2])
    pay_score = np.array([0.1, 0.2, 0.3])

    # 构造历史行为序列数据
    # 构造长度为 4 的 item_id 序列,不足的部分用0填充
    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    # 构造长度为 4 的 cate_id 序列,不足的部分用0填充
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])

    # 构造实际的输入数据
    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id, 'pay_score': pay_score}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    # 构造 DIN 模型
    model = DIN(dnn_feature_columns=feature_columns, history_feature_list=behavior_feature_list)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10)