from tree import DecisionTree
from input_reader import *
import sys, getopt
import os

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
    # preds_mod = [(pred+1)//2 for pred in predictions]
    for i in range(len(predictions)):
        line = image_ids[i] + ", " + str(predictions[i])
        print(line, file=f)

    f.close()

##########################################################
## DECISION TREES FOR MULTI CLASSIFICATION
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

    X_train, y_train = parse_train_images_multi(train_path, flag=False)
    X_test = parse_test_images("./data/test_sample", test_img_ids)

    if (val ==1):
        Xval, yval = parse_train_images_multi("./data/validation", flag=False)

    ###############################################
    ## SECTION 3.2 (a) 
    ## SCIKIT LEARN IMPLEMENTATION 
    ###############################################

    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=7)

    print("[LOG] Sec 3.2a: Created Decision Tree Classifier")

    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)

    if (val ==1):
        val_preds = clf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_32a.csv")
    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.2a: Complete.")

    ###############################################
    ## SECTION 3.2 (b) 
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

    print("[LOG] Sec 3.2b: Created Decision Tree Classifier")

    X_test_new = selector.transform(X_test)

    predictions = clf.predict(X_test_new)

    outpath = os.path.join(out_path, "test_32b.csv")
    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.2b: Complete.")

    ###############################################
    ## SECTION 3.2 (c) 
    ## SCIKIT LEARN IMPLEMENTATION - CCP ALPHAS
    ###############################################

    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=7, ccp_alpha=0.001)

    print("[LOG] Sec 3.2d: Created Decision Tree Classifier")

    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)

    if (val ==1):
        val_preds = clf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_32c.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.2c: Complete.")

    ###############################################
    ## SECTION 3.2 (d) 
    ## SCIKIT RANDOM FOREST
    ###############################################

    print("[LOG] Sec 3.2d: Created Random Forest Classifier")

    rf = RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_split=5, n_estimators=200)
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    if (val ==1):
        val_preds = rf.predict(Xval)
        acc = accuracy_score(yval, val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))

    outpath = os.path.join(out_path, "test_32d.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.2d: Complete.")

    ###############################################
    ## SECTION 3.2 (e) 
    ## XGBOOST
    ###############################################

    print("[LOG] Sec 3.2e: Created XGBoost Classifier")

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

    outpath = os.path.join(out_path, "test_32e.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.2e: Complete.")

    ###############################################
    ## SECTION 3.2 (h) 
    ## BEST IMPLEMENTATION
    ###############################################

    print("[LOG] Sec 3.2h: Created Best - RF Classifier")

    le = LabelEncoder()
    y_train_xg = le.fit_transform(y_train)

    xg_clf = xgb.XGBClassifier(max_depth=5, n_estimators=40, subsample=0.6)
    xg_clf.fit(X_train, y_train_xg)

    predictions = xg_clf.predict(X_test)

    predictions = le.inverse_transform(predictions)

    if (val ==1):
        val_preds = xg_clf.predict(Xval)
        val_preds = le.inverse_transform(val_preds)
        scores = precision_recall_fscore_support(yval, val_preds, average='macro')
        print("The validation accuracy is {:2f}".format(acc*100))
        print("The validation precision is {:.2f}".format(scores[0]*100))
        print("The validation recall is {:.2f}".format(scores[1]*100))

    outpath = os.path.join(out_path, "test_32h.csv")

    write_csv(outpath, predictions, test_img_ids)
    print("[LOG] Sec 3.2h: Complete.")

if __name__ == "__main__":
    main(sys.argv[1:])