from tree import DecisionTree
from input_reader import *
import sys, getopt
import os
import matplotlib.pyplot as plt

##############################################
# SCIKIT IMPORTS 
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
##############################################

def write_csv(outfile, predictions, image_ids):
    f = open(outfile, "w")
    # print(outfile)
    # print(len(predictions), len(image_ids))
    preds_mod = [(pred+1)//2 for pred in predictions]
    for i in range(len(predictions)):
        line = image_ids[i] + ", " + str(preds_mod[i])
        print(line, file=f)

    f.close()

##########################################################
## DECISION TREES FOR BINARY CLASSIFICATION
##########################################################

###############################################
## PRELIMINARIES
## TAKE COMMAND LINE INPUTS
###############################################

def main(argv):
    train_path = ""
    test_path = ""
    out_path = ""
    opts, args = getopt.getopt(argv,"hi:o:",["train_path=","test_path=","out_path="])
    for opt, arg in opts:
        if (opt == "--train_path"):
            train_path = arg
        elif (opt == "--test_path"):
            test_path = arg
        elif (opt == "--out_path"):
            out_path = arg

    if (train_path == "" or test_path == "" or out_path==""):
        print("[ERROR] Incorrect/Missing Command Line Args")
        exit(1)

    val = 0
    test_img_ids = []

    X_train, y_train = parse_train_images_bin(train_path, flag=False)
    X_test = parse_test_images("./data/test_sample", test_img_ids)

    if (val ==1):
        Xval, yval = parse_train_images_bin("./data/validation", flag=False)

    ###############################################
    ## SECTION 3.1 (b) 
    ## SCIKIT LEARN IMPLEMENTATION 
    ###############################################

    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=7)

    print("[LOG] Sec 3.1b: Created Decision Tree Classifier")

    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)

    if (val ==1):
        val_preds = clf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_31b.csv")
    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1b: Complete.")

    ###############################################
    ## SECTION 3.1 (c) 
    ## SCIKIT LEARN IMPLEMENTATION GRID
    ###############################################

    selector = SelectKBest(f_classif, k=10)
    X_train_new = selector.fit_transform(X_train, y_train)

    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=4)
    clf.fit(X_train_new,y_train)

    if (val ==1):
        Xval_new  = selector.transform(Xval)
        val_preds = clf.predict(Xval_new)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    print("[LOG] Sec 3.1c: Created Decision Tree Classifier")

    X_test_new = selector.transform(X_test)

    predictions = clf.predict(X_test_new)

    outpath = os.path.join(out_path, "test_31c.csv")
    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1c: Complete.")

    ###############################################
    ## SECTION 3.1 (d) 
    ## SCIKIT LEARN IMPLEMENTATION - CCP ALPHAS
    ###############################################

    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=7, ccp_alpha=0.002)

    print("[LOG] Sec 3.1d: Created Decision Tree Classifier")

    clf = clf.fit(X_train,y_train)

    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf, 
    #                filled=True)

    # fig.savefig("scikit_ccp_tree.png")

    predictions = clf.predict(X_test)

    if (val ==1):
        val_preds = clf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_31d.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1d: Complete.")

    ###############################################
    ## SECTION 3.1 (e) 
    ## SCIKIT RANDOM FOREST
    ###############################################

    print("[LOG] Sec 3.1e: Created Random Forest Classifier")

    rf = RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_split=5, n_estimators=100)
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    if (val ==1):
        val_preds = rf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))

    outpath = os.path.join(out_path, "test_31e.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1e: Complete.")

    ###############################################
    ## SECTION 3.1 (f) 
    ## XGBOOST
    ###############################################

    print("[LOG] Sec 3.1f: Created XGBoost Classifier")

    le = LabelEncoder()
    y_train_xg = le.fit_transform(y_train)

    xg_clf = xgb.XGBClassifier(max_depth=5, n_estimators=40, subsample=0.6)
    xg_clf.fit(X_train, y_train_xg)

    predictions = xg_clf.predict(X_test)

    predictions = le.inverse_transform(predictions)

    if (val ==1):
        val_preds = xg_clf.predict(Xval)
        val_preds = le.inverse_transform(val_preds)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_31f.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1f: Complete.")

    ###############################################
    ## SECTION 3.1 (h) 
    ## BEST IMPLEMENTATION
    ###############################################

    print("[LOG] Sec 3.1h: Created Best - RF Classifier")

    rf = RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_split=5, n_estimators=100)
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    if (val ==1):
        val_preds = rf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_31h.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1h: Complete.")

    ###############################################
    ## SECTION 3.1 (a) 
    ## OWN IMPLEMENTATION OF DECISION TREES
    ###############################################

    decision_tree = DecisionTree(train_path, split_crit=1)

    print("[LOG] Sec 3.1a: Created Decision Tree")
    print("[LOG] Sec 3.1a: Training Decision Tree...")

    decision_tree.grow_tree()

    print("[LOG] Sec 3.1a: Trained Decision Tree")

    print("[LOG] Sec 3.1a: Predicting on Test Set:")
    
    decision_tree.get_test_data(test_path)
    predictions = decision_tree.predict_test()

    # Write to output CSV file 
    outpath = os.path.join(out_path, "test_31a.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.1a: Complete.")

if __name__ == "__main__":
    main(sys.argv[1:])