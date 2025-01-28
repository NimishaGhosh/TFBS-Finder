
class MetricsHolder:
    def __init__(self) -> None:
        self.folderName = None
        self.subFolderName = None
        self.val_TP = None
        self.val_TN = None
        self.val_FP = None
        self.val_FN = None
        self.val_SN = None
        self.val_SP = None
        self.val_ACC = None
        self.val_MCC = None
        self.val_PR_AUC = None
        self.val_ROC_AUC = None

        self.test_TP = None
        self.test_TN = None
        self.test_FP = None
        self.test_FN = None
        self.test_SN = None
        self.test_SP = None
        self.test_ACC = None
        self.test_MCC = None
        self.test_PR_AUC = None
        self.test_ROC_AUC = None

    def get_folderName(self):
        return self.folderName

    def set_folderName(self, folderName):
        self.folderName = folderName

    def get_subFolderName(self):
        return self.subFolderName

    def set_subFolderName(self, subFolderName):
        self.subFolderName = subFolderName

    def get_val_counts(self):
        return self.val_TP, self.val_TN, self.val_FP, self.val_FN

    def set_val_counts(self, tp, tn, fp, fn):
        self.val_TP = tp
        self.val_TN = tn
        self.val_FP = fp
        self.val_FN = fn

    def get_val_metrics(self):
        return self.val_SN, self.val_SP, self.val_ACC, self.val_MCC

    def set_val_metrics(self, sn, sp, acc, mcc):
        self.val_SN = sn
        self.val_SP = sp
        self.val_ACC = acc
        self.val_MCC = mcc

    def get_val_auc(self):
        return self.val_PR_AUC, self.val_ROC_AUC

    def set_val_auc(self, pr_auc, roc_auc):
        self.val_PR_AUC = pr_auc
        self.val_ROC_AUC = roc_auc

    def get_test_counts(self):
        return self.test_TP, self.test_TN, self.test_FP, self.test_FN

    def set_test_counts(self, tp, tn, fp, fn):
        self.test_TP = tp
        self.test_TN = tn
        self.test_FP = fp
        self.test_FN = fn

    def get_test_metrics(self):
        return self.test_SN, self.test_SP, self.test_ACC, self.test_MCC

    def set_test_metrics(self, sn, sp, acc, mcc):
        self.test_SN = sn
        self.test_SP = sp
        self.test_ACC = acc
        self.test_MCC = mcc

    def get_test_auc(self):
        return self.test_PR_AUC, self.test_ROC_AUC

    def set_test_auc(self, pr_auc, roc_auc):
        self.test_PR_AUC = pr_auc
        self.test_ROC_AUC = roc_auc

    def reset(self):
        self.__init__()

    def reset_except_folderName(self):
        f_name = self.folderName
        self.__init__()
        self.folderName = f_name
        
    def to_dict(self) -> dict:
        """Return all attributes as a dictionary."""
        return {
            "folderName": self.folderName,
            "subFolderName": self.subFolderName,
            "val_TP": self.val_TP,
            "val_TN": self.val_TN,
            "val_FP": self.val_FP,
            "val_FN": self.val_FN,
            "val_SN": self.val_SN,
            "val_SP": self.val_SP,
            "val_ACC": self.val_ACC,
            "val_MCC": self.val_MCC,
            "val_PR_AUC": self.val_PR_AUC,
            "val_ROC_AUC": self.val_ROC_AUC,
            "test_TP": self.test_TP,
            "test_TN": self.test_TN,
            "test_FP": self.test_FP,
            "test_FN": self.test_FN,
            "test_SN": self.test_SN,
            "test_SP": self.test_SP,
            "test_ACC": self.test_ACC,
            "test_MCC": self.test_MCC,
            "test_PR_AUC": self.test_PR_AUC,
            "test_ROC_AUC": self.test_ROC_AUC,
        }