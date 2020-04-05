# 超级词性

### 词性是如何来的？语言是如何来的？

本文用非常通用的模型，非常简单的代码，从中解释一些难以发现的新理论。<br>
什么是词性？什么是词性与词性之间的关系？常识不可以尽信。<br>
其实，通过此文，已经隐约说明了词性的来源，语言的来源。<br>
为了让真相不被常识埋没，只能让专家来判断，并给出有说服力的结论。<br>

### 代码功能

代码的功能，是比较句子相似度。为了简单，只选取1000个例子，包括500个正例，500个负例。<br>
过去的模型在设计时主要考虑词与词的关系，并没有直接考虑词性与词性的关系。<br>
但是这个是这样做的，其中为每个词重新给定词性，是一项非常耗时的工作。<br>


## 1.文件
* 公共模块
```
_loader.py -------------预处理数据
_block.py --------------预处理数据
_token.py --------------分词
_losses.py
_layer.py
_model.py
_tool.py ---------------其他，编码与log初始化
```
* 训练
```
run_block.py
runtest_all\runtest_model_calc.py
runtest_all\runtest_model.py
```
* 测试
```
runtest_all\runtest_param.py
runtest_all\runtest_merge.py  
runtest\runtest_load.py  
runtest\runtest_seq.py  
runtest\runtest_split.py  
```

## 2.训练与评估指令

### 2.1.训练(不建议用，非常卡)
* 在8G内存电脑上测试，1000条时非常卡；100条没问题  
```
python run_block.py
```


### 2.2.训练(可用)
```
1)仅显示参数，	不训练  
python_w runtest_all\runtest_model_calc.py --mean=mean

2) 训练10步，能在普通电脑运行  
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=mean --batch_size=40 --training_steps=10

3) 训练10步，为了保持f1值最大，更换指标为max_mean；好像不收敛  
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=max_mean --batch_size=40 --training_steps=10 --learning_rate=5e-5

4) 训练60步  
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=mean --batch_size=120 --training_steps=30 --start_check_step=10 --learning_rate=2e-5
```
* 运行结果：  　[查看大图](/images/screenshot/train/train2.jpg)<br>
![train1](/images/screenshot/train/train1.jpg)

### 2.3.预测(可用)
```
1)混合指标(max_mean)，双指标效果更好
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean

2)单一指标(mean)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=mean

3)单一指标(max)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max

4)输出中间数据(速度有些慢，用于调试最佳序数，见3.1)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean --save_np_data=true

5)删除隐藏模式(add_hide_seq)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean --add_hide_seq=false

```

* 查看详细结果：  
[data\debug\result_train_calc.txt](data/debug/result_train_calc.txt)
* 查看中间输出：(用于调试最佳序数，见3.1)  
[data\debug\brd_sum.txt](data/debug/brd_sum.txt)

* 运行结果(max_mean)：  　[查看大图](/images/screenshot/long/3.4.5_max_mean.jpg)<br>
![train1](/images/screenshot/short/3.4.5_max_mean.jpg)

* 运行结果(mean)：  　[查看大图](/images/screenshot/long/3.4.5_mean.jpg)<br>
![train1](/images/screenshot/short/3.4.5_mean.jpg)

* 运行结果(max)：  　[查看大图](/images/screenshot/long/3.4.5_max.jpg)<br>
![train1](/images/screenshot/short/3.4.5_max.jpg)

* 运行结果(add_hide_seq)：  　[查看大图](/images/screenshot/long/4_delete_seq.jpg)<br>
![train1](/images/screenshot/short/4_delete_seq.jpg)


### 3.参数分析
### 3.1 计算最佳系数
* 前提: 输出中间数据(见2.3)

```
1)显示基本参数
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path=

2)检查p=0.49~0.501之前的数据，看下是否预测准确，这一段比较难控制
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=mid_range

3)更新以上2组系数与基准线后，再计算最佳的基准线
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=max_base

4)固定max组系数后，重新确认最佳系数（有点慢）
  bit=0~16，分17次分别确定系数
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=max_coefficient --bit=2

5)固定mean组系数后，重新确认最佳系数（有点慢）
  bit=0~16，分17次分别确定系数
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=mean_coefficient --bit=9

6)显示最终混合指标
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=merge_show

7)混合指标，重新确认最佳范围参数：
  bit=0~9，并要手动注释掉范围数组(partB)的对应行(与bit对应的范围重复)
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=merge_coefficient
```

### 3.2 参数混合
* 由于同类型的参数相对接近，可以用已训练的数据初始化新参数，加快训练速度。  
*   暂时不考虑使用。  
```
python runtest_all\runtest_merge.py
```


### 4.预处理
### 4.1.截取数据长度、分词

```
1)检查长度是否相等(预处理错误时，输出每条数据长度)
python_w runtest\runtest_load.py	--test_type=data_length_check

2)截取数据长度
# 正例100，反例100，一共200  
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=100 --fcnt=100
# 正例500，反例500，一共1000  
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=500 --fcnt=500
# 正例100，反例100，从第200+1开始  
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=500 --fcnt=500 --first_cnt=200
# 共取200，正反例个数不确定  
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=999 --fcnt=999 --allcnt=200

3)保存分词过程
python_w runtest\runtest_load.py	--test_type=save_tokens
```

* 查看截取数据：  
data\csv\redata_1_to_200\train_xxx.csv

* 查看详细分词过程：
[data\debug\text_split_train.txt](data/debug/text_split_train.txt)

``` 
句子  0: 怎么更改xx手机号码
jieba  : ['怎么', '更改', 'xx', '手机号码']
mode0_list : [['怎么', '13'], ['更改', '3'], ['xx', '2'], ['手机号码', '2']]
mode1_list : [['怎么', '13'], ['更改', '3'], ['xx', '2'], ['手机号码', '2']]
mode2_list : [['怎么', '13'], ['更改', '3'], ['xx', '2'], ['手机号码', '2']]
mode3_list : [['怎么更改', '13', '3'], ['xx手机号码', '2', '2']]
```

### 4.2.预处理--词性编码

```
1)编码-->填充隐藏词性
python_w runtest\runtest_seq.py  --test_type=m2n

2)词-->编码-->填充隐藏词性
python_w runtest\runtest_seq.py  --test_type=tokens_parsing
```
* 运行结果：  　[查看大图](/images/screenshot/long/3.4.1_tokens_parsing.jpg)<br>
![train1](/images/screenshot/short/3.4.1_tokens_parsing.jpg)

### 4.3.分词
* 对一个或几个的句子分词，注意  在代码中增加句子内容  
```
python runtest\runtest_split.py
```


