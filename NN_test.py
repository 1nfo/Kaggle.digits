import sys


def test_result(path):
	xt = load_test()
	with open(path, "rb") as f:
		result, cost_hist, trai_errs, vali_errs, hidden_layer_num = pickle.load(f)
	pred = predict(result.x, xt)
	z = zip(range(1,len(pred)+1),pred)
	with open("test_result", "wb") as f:
		f.write("ImageId,Label\n")
		f.writelines(map(lambda x: str(x[0])+","+str(x[1])+"\n", z))


if len(sys.argv)>1:
	from NN import *
	test_result(sys.argv[1])
else:
	print "A model file path need specifying."