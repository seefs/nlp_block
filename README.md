
// 1.预测与结果:
python runtest_all\runtest_model.py

// result:
//	           train_x1 (1000, 55)
//	           train_x2 (1000, 55)
//	           train_m1 (1000, 55)
//	           train_m2 (1000, 55)
//	          train_mi1 (1000, 55)
//	          train_mi2 (1000, 55)
//	           train_n1 (1000, 24)
//	           train_n2 (1000, 24)
//	         train_cnt1 (1000, 1)
//	         train_cnt2 (1000, 1)
//	            train_y (1000,)
//	        sents_len 1000
//	        max_vocab 1123
//	      max_seq_len 55
//	    max_modes_len 24
//	  pred--loss: tf.Tensor(0.6202628, shape=(), dtype=float32)
//	  pred--acc:  tf.Tensor(0.737, shape=(), dtype=float32)
//	  pred--f1:   tf.Tensor(0.76496863, shape=(), dtype=float32)
//	save train_calc result cnt=1000


// 2.训练(正常):
// 这个文件比较乱
python runtest_all\runtest_model_calc.py


// 3.训练(卡死机, 不能用):
python run_block


// 4.单元测试
// 加载, 设置数量
python runtest\runtest_load.py

// 序列化m2n
python runtest\runtest_seq.py

// 单句分词
python runtest\runtest_split.py


