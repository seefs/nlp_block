

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
h5_file      = os.path.join(modelPath, "atec_nlp_block_{}.h5")

# log
if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(modelPath) or not os.path.isdir(modelPath):
    os.mkdir(modelPath)
	
# cfg
sys.path.append(rootPath)

# Custom Class
from _loader_bl import LoadData, save_pred_result
from _token  import TokenizerChg
from _losses import f1, expand_dims_f1, cross_entropy_loss, accuracy, create_learning_rate_scheduler
from _model_bl import create_model




# In[2]:



data = LoadData(sample_size=None, train_enable=True, test_enable=True)

if data.train_enable:
    print("-----------train_t1", data.train_t1.shape)         # 中文字符 (用不上)
    print("           train_t2", data.train_t2.shape)
    print("           train_x1", data.train_x1.shape)         # 字编号
    print("           train_x2", data.train_x2.shape)
    print("           train_m1", data.train_m1.shape)         # 字类型, 原类型:0~16,17,26 训练时类型:0~16
    print("           train_m2", data.train_m2.shape)
    print("         train_cnt1", data.train_cnt1.shape)       # 字长
    print("         train_cnt2", data.train_cnt2.shape)
    print("            train_y", data.train_y.shape)          # 标签
    
    print("        train_mark1", data.train_mark1.shape)      # 分块后的mark
    print("        train_mark2", data.train_mark2.shape)
    print("       train_block1", data.train_block1.shape)     # 分块后的块ID
    print("       train_block2", data.train_block2.shape)
    print("       train_scale1", data.train_scale1.shape)     # 分块后的块权重
    print("       train_scale2", data.train_scale2.shape)

if data.test_enable:
    print("------------test_t1", data.test_t1.shape)          # 中文字符 (用不上)
    print("            test_t2", data.test_t2.shape)
    print("            test_x1", data.test_x1.shape)
    print("            test_x2", data.test_x2.shape)
    print("            test_m1", data.test_m1.shape)
    print("            test_m2", data.test_m2.shape)
    print("          test_cnt1", data.test_cnt1.shape)
    print("          test_cnt2", data.test_cnt2.shape)
    print("             test_y", data.test_y.shape)
    
    print("          test_mark1", data.test_mark1.shape)      # 分块后的mark
    print("          test_mark2", data.test_mark2.shape)
    print("         test_block1", data.test_block1.shape)     # 分块后的块ID
    print("         test_block2", data.test_block2.shape)
    print("         test_scale1", data.test_scale1.shape)     # 分块后的块权重
    print("         test_scale2", data.test_scale2.shape)


# 实际字长:
print("        max_vocab", data.max_vocab_len)
print("      max_seq_len", data.max_seq_len)



# In[3]:


block_model = create_model(data.max_vocab_len, data.max_seq_len, h5_file=h5_file, debug=False, mean='mean', save_data=False)

#pred = block_model([data.train_x1,     data.train_x2,       #编号
#                    data.train_m1,     data.train_m2,       #类型
#                    data.train_mark1,  data.train_mark2,    #块标记
#                    data.train_block1, data.train_block2,   #块ID
#                    data.train_scale1, data.train_scale2])
#print ("pred: %s" % (tf.shape(pred)))

# Training Parameters
total_epoch_count = 2 #200
batch_size        = 50
#display_step      = 1

callbacks = [ 
    create_learning_rate_scheduler(max_learn_rate=1e-1,
        end_learn_rate=1e-12,
        warmup_epoch_count=10,
        total_epoch_count=total_epoch_count),
    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True), 
    keras.callbacks.TensorBoard(log_dir=log_dir)
]

block_model.fit(x=data.get_train_data(),
          y=data.train_y, 
          #validation_data=[data.get_test_data(), data.test_y], 
          validation_split=0.1,
          batch_size=batch_size,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=callbacks)

block_model.save_weights(h5_file.format(data.max_vocab_len), overwrite=True)



# In[5]:


# 验证(训练ok后再打开)
#if data.train_enable:
#    _, train_acc, train_f1 = block_model.evaluate(data.get_train_data(), data.train_y)
#    print(" eval--train acc", train_acc)
#    print(" eval--train f1", train_f1)
#    
#if data.test_enable:
#    _, test_acc, test_f1  = block_model.evaluate(data.get_test_data(), data.test_y)
#    print(" eval--test acc", test_acc)
#    print(" eval--test f1", test_f1)



# In[7]:

# 加载weights再次验证(不用)
#new_model = create_model(data.max_vocab_len, data.max_seq_len, h5_file=h5_file, debug=False, mean='mean', save_data=False)
#new_model.load_weights(h5_file.format(data.max_vocab_len))
#if data.train_enable:
#    _, train_acc, train_f1 = block_model.evaluate(data.get_train_data(), data.train_y)
#    print("train acc", train_acc)
#    print("train f1", train_f1)
#if data.test_enable:
#    _, test_acc, test_f1  = block_model.evaluate(data.get_test_data(), data.test_y)
#    print(" test acc", test_acc)
#    print(" test f1", test_f1)



# In[8]:


def run_pred_data():
    new_model = create_model(data.max_vocab_len, data.max_seq_len, h5_file=h5_file, debug=False, mean='mean', save_data=False)
    new_model.debug = True
    if os.path.isfile(h5_file.format(data.max_vocab_len)):
        new_model.load_weights(h5_file.format(data.max_vocab_len))
    
    if data.train_enable:
        pred = new_model([data.get_train_data()])
        loss = cross_entropy_loss(pred, data.train_y)
        print ("  pred--loss: %s" % (loss))
        acc = accuracy(data.train_y, pred)
        print ("  pred--acc:  %s" % (acc))
        f1_score = expand_dims_f1(data.train_y, pred)  # 用f1有问题
        print ("  pred--f1:   %s" % (f1_score))
        res_string = 'loss=%s,   acc=%s,   f1_score=%s'%(np.array(loss), np.array(acc), np.array(f1_score))
        
        train_y = tf.expand_dims(data.train_y, -1)
        train_y = tf.cast(train_y, tf.float32)
        pred_y = tf.cast(tf.argmax(pred, -1), tf.float32)
        pred_y = tf.expand_dims(pred_y, -1)
        pred_c   = tf.concat([pred, train_y, pred_y], -1)
        save_pred_result(data.train_t1, data.train_t2, pred.numpy(), data.train_y, res_string, name='train_bl')
        
    if data.test_enable:
        pred = new_model([data.get_test_data()])
        loss = cross_entropy_loss(pred, data.test_y)
        print ("  pred--loss: %s" % (loss))
        acc = accuracy(data.test_y, pred)
        print ("  pred--acc:  %s" % (acc))
        f1_score = expand_dims_f1(data.test_y, pred)  # 用f1有问题
        print ("  pred--f1:   %s" % (f1_score))
        res_string = 'loss=%s,   acc=%s,   f1_score=%s'%(np.array(loss), np.array(acc), np.array(f1_score))
        
        test_y = tf.expand_dims(data.test_y, -1)
        test_y = tf.cast(test_y, tf.float32)
        pred_y = tf.cast(tf.argmax(pred, -1), tf.float32)
        pred_y = tf.expand_dims(pred_y, -1)
        pred_c   = tf.concat([pred, test_y, pred_y], -1)
        save_pred_result(data.test_t1, data.test_t2, pred.numpy(), data.test_y, res_string, name='test_bl')
    
run_pred_data()




