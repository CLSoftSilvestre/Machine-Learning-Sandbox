
import matplotlib.pylab as plt
import base64
import io

def CreateImage(test, pred, yLabel):
    # Add train image into model
    plot_y_test = test.reset_index()
    del plot_y_test['index']

    plt.plot(plot_y_test[0:100], color='#2c3e50', label='Real')
    plt.plot(pred[0:100], color='#18bc9c', label='Predicted')
    plt.xlabel('Predictions')
    plt.ylabel(yLabel)
    plt.legend(loc='lower right')

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    plt.clf()

    return my_base64_jpgData