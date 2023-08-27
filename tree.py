from treenode import TreeNode
from input_reader import *
import numpy as np
import time 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import time
import seaborn as sns

class DecisionTree:

    def __init__(self, train_path, min_samples_split=7, max_depth=10, stop_crit=0, split_crit=0):
        self.dataset = parse_train_images_bin(train_path)[0]
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.stop_crit = stop_crit
        self.split_crit = split_crit
        self.Xtest = None

    def grow_tree(self):
        # Create the root 
        self.root = TreeNode(self.dataset, depth = 0, min_samples_split=self.min_samples_split, max_depth=self.max_depth,\
            stop_crit=self.stop_crit, split_crit=self.split_crit)
        
        # Grow the root 
        t_start = time.time()
        self.root.expand_node()
        t_end = time.time()
        print("[LOG] Sec 3.1a: Training Time is ", t_end - t_start)

    def get_test_data(self, test_path):
        self.Xtest = parse_test_images(test_path)

    def predict_val(self, val_path):
        Xval, Yval = parse_train_images_bin(val_path, False)
        preds = []
    
        for test in Xval:
            preds.append(self.root.predict(test))
        
        acc = 0
        tot = 0

        for i in range(len(Yval)):
            tot += 1
            if (preds[i]==Yval[i]):
                acc += 1

        # print("------------------------------------------------")
        # print("[LOGS: Sec 3.1a] Accuracy:", acc*100/tot)
        # print("------------------------------------------------")

        scores = precision_recall_fscore_support(Yval, preds, average='macro')

        # print("The accuracy is {:2f}".format(acc*100))
        # print("The precision is {:.2f}".format(scores[0]*100))
        # print("The recall is {:.2f}".format(scores[1]*100))

        conf_matrix = confusion_matrix(Yval, preds)
        # print(conf_matrix)

        plt.figure(figsize=(8,6), dpi=100)
        sns.set(font_scale = 1.1)

        # Plot Confusion Matrix using Seaborn heatmap()
        ax = sns.heatmap(conf_matrix, annot=True, cmap='Purples')

        ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(['Negative', 'Positive'])
        ax.set_ylabel("Actual", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(['Negative', 'Positive'])

        # set plot title
        ax.set_title("Confusion Matrix for the Scikit Implementation", fontsize=14, pad=20)

        filename = "bin_own_conf.png"
        # plt.show()
        plt.savefig(filename)

    def predict_test(self):
        preds = []
        for test in self.Xtest:
            preds.append(self.root.predict(test))
        # print("------------------------------------------------")
        # print("Predictions are: ")
        # print(preds)
        # print("-----------------------------------------------")
        return preds 
    
if __name__ == "__main__":
    tree = DecisionTree("./data/train", split_crit=0)

    print("Created tree")
    tree.grow_tree()

    print("[LOGS: Sec 3.1a] Trained Decision Tree")

    print("[LOGS: Sec 3.1a] Finding Training Accuracy:")
    tree.predict_val("./data/train")

    tree.predict_val("./data/validation")
    

    print("[LOGS: Sec 3.1a] Predicting on Test Set:")
    
    # tree.get_test_data(test_path, test_img_ids)


