import keras.backend as K
from keras.metrics import Metric

class CustomMetrics(Metric):
    def __init__(self, num_classes, **kwargs):
        super(CustomMetrics, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.tpr = [self.add_weight(name=f'true_positive_rate_{i}', initializer='zeros') for i in range(num_classes)]
        self.tnr = [self.add_weight(name=f'true_negative_rate_{i}', initializer='zeros') for i in range(num_classes)]
        self.fpr = [self.add_weight(name=f'false_positive_rate_{i}', initializer='zeros') for i in range(num_classes)]
        self.fnr = [self.add_weight(name=f'false_negative_rate_{i}', initializer='zeros') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # First delete all previous values
        for i in range(self.num_classes):
            self.tpr[i].assign(0)
            self.tnr[i].assign(0)
            self.fpr[i].assign(0)
            self.fnr[i].assign(0)

        y_true = K.cast(y_true, 'int32')
        y_pred = K.argmax(y_pred, axis=-1)

        for class_index in range(self.num_classes):
            true_positives = K.sum(K.cast(K.equal(y_true, class_index) & K.equal(y_pred, class_index), 'float32'))
            true_negatives = K.sum(K.cast(K.equal(y_true, class_index) & K.not_equal(y_pred, class_index), 'float32'))
            false_positives = K.sum(K.cast(K.not_equal(y_true, class_index) & K.equal(y_pred, class_index), 'float32'))
            false_negatives = K.sum(K.cast(K.not_equal(y_true, class_index) & K.not_equal(y_pred, class_index), 'float32'))

            self.tpr[class_index].assign_add(true_positives / (true_positives + false_negatives + K.epsilon()))
            self.tnr[class_index].assign_add(true_negatives / (true_negatives + false_positives + K.epsilon()))
            self.fpr[class_index].assign_add(false_positives / (true_negatives + false_positives + K.epsilon()))
            self.fnr[class_index].assign_add(false_negatives / (true_positives + false_negatives + K.epsilon()))


    def result(self):
        return self.tpr, self.tnr, self.fpr, self.fnr


print(model.metrics[2].result().numpy())


def conf_els(conf: np.ndarray, labels: list[str]) -> list[tuple[str, int, int, int, int]]:
    results = list()
    for label, row in zip(labels, conf):
        tp_i = row[labels.index(label)]
        fp_i = sum(row) - tp_i
        fn_i = sum(conf[labels.index(label)]) - tp_i
        tn_i = sum(map(sum, conf)) - tp_i - fp_i - fn_i
        results.append((label, tp_i, fp_i, fn_i, tn_i))

    return np.array(results, dtype=object)

def conf_data(metrics: list[tuple[str, int, int, int, int]]) -> dict[str, int]:
    tp = [int(metric[1]) for metric in metrics]
    fp = [int(metric[2]) for metric in metrics]
    fn = [int(metric[3]) for metric in metrics]
    tn = [int(metric[4]) for metric in metrics]
    print(tp, fp, fn, tn)

    tpr = [tp_i / (tp_i + fn_i) for tp_i, fn_i in zip(tp, fn)]
    ppv = [tp_i / (tp_i + fp_i) for tp_i, fp_i in zip(tp, fp)]
    tnr = [tn_i / (tn_i + fp_i) for tn_i, fp_i in zip(tn, fp)]
    fpr = [fp_i / (fp_i + tn_i) for fp_i, tn_i in zip(fp, tn)]

    rv = {'tpr': tpr, 'ppv': ppv, 'tnr': tnr, 'fpr': fpr}
    return rv
    # tp, fp, fn, tn = [int(metrics[i]) for i in range(1, len(metrics))]
    # rv = {'tpr':[tp/(tp+fn)], 'ppv':[tp/(tp+fp)], 'tnr':[tn/(tn+fp)], 'fpr':[fp/(fp+tn)] }
    # return rv

conf_elements = conf_els(conf_mtx, labels)

def calc(conf_elements):
    tpr = [tp / (tp + fn) for _, tp, _, fn, _ in conf_elements]
    tnr = [tn / (tn + fp) for _, _, fp, _, tn in conf_elements]
    fpr = [fp / (fp + tn) for _, _, fp, _, tn in conf_elements]
    fnr = [fn / (fn + tp) for _, tp, _, fn, _ in conf_elements]
    
    return {'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr}

rates = calc(conf_elements)
tpr, fpr = zip(*sorted(zip(rates['tpr'], rates['fpr'])))

from sklearn import metrics
roc_auc = metrics.auc(tpr, fpr)
disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()
plt.show()
