from sklearn.base import TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pylab as plt
import seaborn as sn
import io
import base64

class OutlierExtractor(TransformerMixin):

    def __init__(self, **kwargs):
        self.threshold = kwargs.pop('neg_conf_val', -10.0)
        self.kwargs = kwargs
    
    def transform(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (X[lcf.negative_outlier_factor_ > self.threshold, :],
                y[lcf.negative_outlier_factor_ > self.threshold])
    
    def fit(self, *args, **kwargs):
        return self

def CreateOutliersBoxplot(features, df):

    #tempFeatures = features.columns.toList()

    plt.figure(figsize=(16,10))

    for i, col in enumerate(features):
        try:
            if len(features) > 30:
                plt.subplot(int(len(features)/10)+1,10,i + 1)
            else:
                plt.subplot(int(len(features)/5)+1,5,i + 1)
            #plt.subplot(3,6,i + 1)
            sn.boxplot(y=col, data=df, palette='Greens', fliersize=3)
        except:
            pass

    plt.tight_layout()

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    plt.clf()

    return my_base64_jpgData
