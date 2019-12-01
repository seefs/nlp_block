

import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from keras import backend as K
import numpy as np
import copy

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[1]:

# path
curPath  = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
dataPath = os.path.join(rootPath, "data")
log_dir   = os.path.join(dataPath,  "log")
modelPath = os.path.join(dataPath, "model")

# file
sqlite3_file = os.path.join(dataPath, "sqlite3", "data.db3")
h5_file      = os.path.join(modelPath, "atec_nlp_layer_{}.h5")

# log
if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(modelPath) or not os.path.isdir(modelPath):
    os.mkdir(modelPath)
	
# cfg
sys.path.append(rootPath)

# Custom Class
from _token import TokenizerChg
from _block import b2mts                        # 分块
from _layer import mark_slice                   # 分块标记
from _tool  import pad
from _model import create_model



def run_block_model(text):
    token_chg = TokenizerChg(db_path=sqlite3_file, debug_log=False)
    tokenizer = Tokenizer(num_words=None)
    
    max_seq_len = 10
    t1, t2, m1, m2, cnt1, cnt2 = [], [], [], [], [], []
    for t in text:
        tokens1 = [['_C', 0]] + token_chg.tokens_mode2(text=t[0])
        tokens2 = [['_C', 0]] + token_chg.tokens_mode2(text=t[1])
        modes1  = [int(mode) for item in tokens1 for mode in item[1:]]
        modes2  = [int(mode) for item in tokens2 for mode in item[1:]]
        tokens1 = [item[0] if pos==0 else '_S' for item in tokens1 for pos in range(len(item[1:]))]
        tokens2 = [item[0] if pos==0 else '_S' for item in tokens2 for pos in range(len(item[1:]))]
        max_seq_len = max(max_seq_len, len(modes1), len(modes2))
        
#        print("text1:", t[0])
#        print("tokens1:", tokens1)
#        print("modes1:", modes1)
#        print ("----------------------------------")
#        print("text2:", t[1])
#        print("tokens2:", tokens2)
#        print("modes2:", modes2)
#        print ("----------------------------------")
        t1.append(tokens1)
        t2.append(tokens2)
        m1.append(modes1)
        m2.append(modes2)
        cnt1.append(len(modes1))
        cnt2.append(len(modes2))
        
    data_cnt = 1
    if True:
        for i in range(len(t1)):
            t1[i][0] = '_' + str(data_cnt)
            data_cnt+=1
        for i in range(len(t2)):
            t2[i][0] = '_' + str(data_cnt)
            data_cnt+=1
           
    t1 = np.array(t1)
    t2 = np.array(t2)
    m1 = np.array(m1)
    m2 = np.array(m2)
    cnt1 = np.array(cnt1)
    cnt2 = np.array(cnt2)
    # pad
    max_seq_len = max_seq_len + 2
    t1 = pad(t1, max_seq_len) #pad_sequences不支持中文
    t2 = pad(t2, max_seq_len) #pad_sequences不支持中文
    m1 = pad_sequences(m1, maxlen=max_seq_len, dtype='int32', padding='post', truncating='post', value=0.)
    m2 = pad_sequences(m2, maxlen=max_seq_len, dtype='int32', padding='post', truncating='post', value=0.)
    print("       t1", t1.shape)         # 中文字符 (用不上)
    print("       t2", t2.shape)
    print("       m1", m1.shape)         # 字类型, 原类型:0~16,17,26 训练时类型:0~16
    print("       m2", m2.shape)
    print("     cnt1", cnt1.shape)       # 字长
    print("     cnt2", cnt2.shape)
    print("max_seq_len:", max_seq_len)
    
    # seq--fit text
    tokenizer.fit_on_texts(t1.flatten())
    tokenizer.fit_on_texts(t2.flatten())
    max_vocab_len = len(tokenizer.word_counts) + 1
    # seq--token id
    _ids1 = [tokenizer.texts_to_sequences(_row) for _row in t1]
    _ids1 = [[i for t in item for i in t] for item in _ids1]
    _ids2 = [tokenizer.texts_to_sequences(_row) for _row in t2]
    _ids2 = [[i for t in item for i in t] for item in _ids2]
    x1 = np.array(_ids1)
    x2 = np.array(_ids2)
    
    print("       x1", x1.shape)         # 字编号
    print("       x2", x2.shape)
#    print("cnt1:", cnt1)
#    print("cnt2:", cnt2)
    print("t1:", t1)
#    print("t2:", t2)
    print("x1:", x1)
#    print("x2:", x2)
    
    # 词性--分块
    mark1, mark2, block1, block2, scale1, scale2 = b2mts([m1, m2, cnt1, cnt2], max_seq_len)
    mark1 = mark_slice([mark1, cnt1])
    mark2 = mark_slice([mark2, cnt2])
    cnt1 = tf.expand_dims(cnt1, -1)
    cnt2 = tf.expand_dims(cnt2, -1)
    print("cnt1:", cnt1)
        
    # model
    new_model = create_model(max_vocab_len, max_seq_len, h5_file=h5_file, debug=True)
    if os.path.isfile(h5_file.format(max_vocab_len)):
        new_model.load_weights(h5_file.format(max_vocab_len))
                    
    pred = new_model([x1, x2, m1, m2, mark1, mark2, block1, block2, scale1, scale2, cnt1, cnt2])
    print ("pred: %s" % (pred))
    new_model.save_weights(h5_file.format(max_vocab_len), overwrite=True)
    

def tokens_parsing_main():
# 比较 True
    text = [
#        ["蚂蚁花呗不用会扣钱吗", "花呗不用会收利息吗"],
        ["我的收钱码为什么不能用蚂蚁花呗", "收钱码开通花呗怎么不能用"],
#        ["花呗在哪里找到", "花呗在哪，找不到"], 
#        ["花呗负会不会有影响", "花呗用超了有什么影响吗"], 
#        ["蚂蚁借呗只能分六个月还款", "蚂蚁借呗为什么只有一个期数"], 
        ["电费花呗怎么没发交", "为什么交电费不支持花呗付了"],
#        ["花呗支付以后，为什么没有获得奖励金", "超市使用花呗付款后没有奖励金"],
#        ["蚂蚁花呗咋个开通不了", "条件都满足为什么还开通不了花呗"],
#        ["蚂蚁借呗只能分六个月还款", "蚂蚁借呗为什么只有一个期数"],
#        ["为什不能花呗收钱", "蚂蚁花呗不能收款"],
#        ["怎样取出花呗里商家退款", "退款去花呗了要怎么样拿出来"],
#        ["可以提前还花呗分期付款的钱吗", "花呗分期以后想提前还款"],
# 比较 False
#        ["花呗分期用不了", "线下签约花呗分期"], 
        ["我今天用了花呗，是不是明天一定要还的", "这个单子用的不是花呗"], 
#        ["系统什么时候评估借呗开通条件", "有哪些条件开通借呗"], 
#        ["可以用花呗冲流量吗", "花呗可以绑定etc吗"], 
#        ["如何能使花呗额度提高", "花呗额度每个月都会提额吗"], 
#        ["花呗服务费这么贵", "花呗收款服务费 什么意思"],
        ["借呗怎么就不能用了", "支付宝跟换手机号码，借呗就不能用了"],
#        ["花呗出个系统忙是什么意思", "花呗，最低还款是什么意思"],
#        ["一次消费花呗有金额限制吗", "花呗付款次数有限制吗"],
#        ["微贷产品已还款，怎么还不能使用花呗", "花呗还清还不能用"],
# 比较测试----未调整句子
#        ["花呗逾期，现在为什么不能使用", "花呗逾期，现在不能用"]
       ]

# 比较测试
    run_block_model(text)




if __name__ == "__main__":
    tokens_parsing_main()




