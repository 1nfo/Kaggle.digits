from NN import *
import pickle

with open("model", "rb") as f:
    result, cost_hist, trai_errs, vali_errs = pickle.load(f)

x, y = load_train()
print (predict(result.x, x) == y).mean()
plt.plot(cost_hist)
plt.show()

