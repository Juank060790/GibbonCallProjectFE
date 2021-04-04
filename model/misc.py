'''
Helper functions for error analysis.
'''

from sklearn.metrics import confusion_matrix 
import itertools 

def plotConfusionMatrix(matrix: np.ndarray, title: str = "Confusion Matrix", 
                        classes: list = [0, 1], cmap: plt.cm = plt.cm.Reds):
    '''
    * Plot confusion matrix.
    '''

    plt.imshow(matrix, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    #Basically a nested for loop: for i in range(2), for j in range(2)
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
