# 超级词性

### 词性是如何来的？语言是如何来的？<br>

本文用非常通用的模型，非常简单的代码，从中解释一些难以发现的新理论。<br>
什么是词性？什么是词性与词性之间的关系？常识不可以尽信。<br>
其实，通过此文，已经隐约说明了词性的来源，语言的来源。<br>
为了不让常识埋没真相，希望此文能有更多的生存空间。<br>


## 1.文件(公共模块)
* _loader.py  ---预处理数据
* _block.py   ---预处理数据
* _token.py   ---分词
* _losses.py
* _layer.py
* _model.py
* _tool.py    ---其他, 编码与log初始化


## 2.编码命令
* 目录


### 2.1 训练(不可用, 非常卡)
// 在8G内存电脑上测试, 数据多时超级卡
python_w run_block.py - --------------------- del _w


### 2.2 训练(可用)
* 1)不训练，仅显示参数
python_w runtest_all\runtest_model_calc.py --mean=mean

* 2)训练10步，一般电脑都能运行
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=mean --batch_size=40 --training_steps=10

* 3)训练10步，更换指标为max_mean，f1值最大
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=max_mean --batch_size=40 --training_steps=10 --learning_rate=5e-5

* 4)训练60步，
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=mean --batch_size=120 --training_steps=30 --start_check_step=10 --learning_rate=2e-5
	

### 2.3 预测(可用)
* 1)混合指标(max_mean)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean

* 2)单一指标(mean)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=mean

* 3)单一指标(max)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max

* 输出中间数据(速度有些慢, 用于调试最佳序数，见3.1)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean --save_np_data=true

* 查看详细结果：
data\debug\result_train_calc.txt
* 查看中间输出：(用于调试最佳序数，见3.1)
data\debug\brd_sum.txt



### 3.1 参数分析（双指标效果更好）

* 1)显示基本指标
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path=

* 2)检查p=0.49~0.501之前的数据，看下是否预测准确，这一段比较难控制
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=mid_range

* 3)更新以上2组系数与基准线后, 再计算最佳的基准线
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=max_base

* 4)固定max组系数后, 重新确认最佳系数（有点慢）
    bit=0~16, 分17次分别确定系数
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=max_coefficient --bit=2

* 5)固定mean组系数后, 重新确认最佳系数（有点慢）
    bit=0~16, 分17次分别确定系数
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=mean_coefficient --bit=9

* 6)显示最终混合指标
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=merge_show

* 7)混合指标，重新确认最佳范围参数：
    bit=0~9, 并要手动注释掉范围数组(partB)的对应行(与bit对应的范围重复)
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=merge_coefficient
	

### 3.2 参数混合（暂时用不上）
  由于同类型的参数差不多，可以用已训练的数据初始化，加快训练速度。暂时不考虑。
runtest_all\runtest_merge.py
python_w runtest_all\runtest_merge.py



### 4.1.预处理--截取数据长度、分词

* 1)检查长度是否相等(预处理错误时，输出每条数据长度)
python_w runtest\runtest_load.py  --test_type=data_length_check

* 2)截取数据长度
 正例100, 反例100, 一共200
python_w runtest\runtest_load.py  --test_type=data_take_out --tcnt=100 --fcnt=100
 正例500, 反例500, 一共1000
python_w runtest\runtest_load.py  --test_type=data_take_out --tcnt=500 --fcnt=500
 正例100, 反例100, 从第200+1开始
python_w runtest\runtest_load.py  --test_type=data_take_out --tcnt=500 --fcnt=500 --first_cnt=200
 共取200, 正反例个数不确定
python_w runtest\runtest_load.py  --test_type=data_take_out --tcnt=999 --fcnt=999 --allcnt=200

* 3)保存分词过程
python_w runtest\runtest_load.py  --test_type=save_tokens


* 查看截取数据：
data\csv\redata_1_to_200\train_xxx.csv

* 查看详细分词过程：
data\debug\text_split_train.txt



### 4.2.预处理--词性编码

* 1)编码-->填充隐藏词性
python_w runtest\runtest_seq.py  --test_type=m2n

* 2)词-->编码-->填充隐藏词性
python_w runtest\runtest_seq.py  --test_type=tokens_parsing
	

### 4.3.预处理--分词

* 1)对一个或几个的句子分词
  要在代码中增加句子内容
python_w runtest\runtest_split.py






























句子匹配
=============================================
// 1.预测与结果:
python runtest_all\runtest_model.py
---------------------------------------------
 result:
	           train_x1 (1000, 55)
	           train_x2 (1000, 55)
	           train_m1 (1000, 55)
	           train_m2 (1000, 55)
	          train_mi1 (1000, 55)
	          train_mi2 (1000, 55)
	           train_n1 (1000, 24)
	           train_n2 (1000, 24)
	         train_cnt1 (1000, 1)
	         train_cnt2 (1000, 1)
	            train_y (1000,)
	        sents_len 1000
	        max_vocab 1123
	      max_seq_len 55
	    max_modes_len 24
	  pred--loss: tf.Tensor(0.6202628, shape=(), dtype=float32)
	  pred--acc:  tf.Tensor(0.737, shape=(), dtype=float32)
	  pred--f1:   tf.Tensor(0.76496863, shape=(), dtype=float32)
	save train_calc result cnt=1000
---------------------------------------------


// 2.正常训练文件:
python runtest_all\runtest_model_calc.py


// 3.异常训练文件(卡死机, 不能用):
python run_block


// 4.单元测试
// 加载, 设置数量
python runtest\runtest_load.py

// 序列化m2n
python runtest\runtest_seq.py

// 单句分词
python runtest\runtest_split.py


