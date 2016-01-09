from NN import *
import pickle
import sys

if len(sys.argv) > 1:
	model = sys.argv[1]
else:
	model="model"
with open(model, "rb") as f:
    result, cost_hist, trai_errs, vali_errs, hidden_layer_num = pickle.load(f)

x, y = load_train()
print result
print (predict(result.x, x) == y).mean()
plt.plot(cost_hist)
plt.show()