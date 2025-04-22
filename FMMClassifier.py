import numpy as np
import pandas as pd
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_K_chess_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_Poker_data

class FuzzyMMC:

    def __init__(self, sensitivity=1, exp_bound=1):
        '''
        Constructor for FuzzyMMC class
        '''
        self.sensitivity = sensitivity
        self.hyperboxes = None
        self.classes = np.array([])
        self.exp_bound = exp_bound




    def membership(self, pattern):
        '''
        Calculates membership values a pattern

        Returns an ndarray of membership values of all hyperboxes
        '''
        min_pts = self.hyperboxes[:, 0, :]
        max_pts = self.hyperboxes[:, 1, :]

        a = np.maximum(0, (1 - np.maximum(0, (self.sensitivity * np.minimum(1, pattern - max_pts)))))
        b = np.maximum(0, (1 - np.maximum(0, (self.sensitivity * np.minimum(1, min_pts - pattern)))))

        return np.sum(a + b, axis=1) / (2 * len(pattern))


    def overlap_contract(self, index):
        '''
        Check if any classwise dissimilar hyperboxes overlap
        '''
        contracted = False
        for test_box in range(len(self.hyperboxes)):

            if self.classes[test_box] == self.classes[index]:
                # Ignore same class hyperbox overlap
                continue

            expanded_box = self.hyperboxes[index]
            box = self.hyperboxes[test_box]

            ## TODO: Refactor for vectorization
            vj, wj = expanded_box
            vk, wk = box

            delta_new = delta_old = 1
            min_overlap_index = -1
            for i in range(len(vj)):
                if vj[i] < vk[i] < wj[i] < wk[i]:
                    delta_new = min(delta_old, wj[i] - vk[i])

                elif vk[i] < vj[i] < wk[i] < wj[i]:
                    delta_new = min(delta_old, wk[i] - vj[i])

                elif vj[i] < vk[i] < wk[i] < wj[i]:
                    delta_new = min(delta_old, min(wj[i] - vk[i], wk[i] - vj[i]))

                elif vk[i] < vj[i] < wj[i] < wk[i]:
                    delta_new = min(delta_old, min(wj[i] - vk[i], wk[i] - vj[i]))

                if delta_old - delta_new > 0:
                    min_overlap_index = i
                    delta_old = delta_new

            if min_overlap_index >= 0:
                i = min_overlap_index
                # We need to contract the expanded box
                if vj[i] < vk[i] < wj[i] < wk[i]:
                    vk[i] = wj[i] = (vk[i] + wj[i])/2

                elif vk[i] < vj[i] < wk[i] < wj[i]:
                    vj[i] = wk[i] = (vj[i] + wk[i])/2

                elif vj[i] < vk[i] < wk[i] < wj[i]:
                    if (wj[i] - vk[i]) > (wk[i] - vj[i]):
                        vj[i] = wk[i]

                    else:
                        wj[i] = vk[i]

                elif vk[i] < vj[i] < wj[i] < wk[i]:
                    if (wk[i] - vj[i]) > (wj[i] - vk[i]):
                        vk[i] = wj[i]

                    else:
                        wk[i] = vj[i]

                self.hyperboxes[test_box] = np.array([vk, wk])
                self.hyperboxes[index] = np.array([vj, wj])
                contracted = True

        return contracted



    def train_pattern(self, X, Y):
        '''
        Main function that trains a fuzzy min max classifier
        Note:
        Y is a one-hot encoded target variable
        '''
        target = Y

        if target not in self.classes:

            # Create a new hyberbox
            if self.hyperboxes is not None:
                self.hyperboxes = np.vstack((self.hyperboxes, np.array([[X, X]])))
                self.classes = np.hstack((self.classes, np.array([target])))

            else:
                self.hyperboxes = np.array([[X, X]])
                self.classes = np.array([target])

        else:

            memberships = self.membership(X)
            memberships[np.where(self.classes != target)] = 0
            memberships = sorted(list(enumerate(memberships)), key=lambda x: x[1], reverse=True)

            # Expand the most suitable hyperbox
            count = 0
            while True:
                index = memberships[count][0]
                min_new = np.minimum(self.hyperboxes[index, 0, :], X)
                max_new = np.maximum(self.hyperboxes[index, 1, :], X)

                if self.exp_bound * len(np.unique(self.classes)) >= np.sum(max_new - min_new):
                    self.hyperboxes[index, 0] = min_new
                    self.hyperboxes[index, 1] = max_new
                    break
                else:
                    count += 1

                if count == len(memberships):
                    self.hyperboxes = np.vstack((self.hyperboxes, np.array([[X, X]])))
                    self.classes = np.hstack((self.classes, np.array([target])))
                    index = len(self.hyperboxes) - 1
                    break



            contracted = self.overlap_contract(index)




    # def fit(self, X, Y):
    #     '''
    #     Wrapper for train_pattern
    #     '''
    #     for x, y in zip(X, Y):
    #         self.train_pattern(x, y)
            
            
    def fit(self, X, Y, epochs):
        """
        Erweitert: Wiederhole das Training 'epochs'-Mal
        """
        for epoch in range(epochs):
            # Ein "Durchgang" über alle Trainingsdaten
            for x, y in zip(X, Y):
                self.train_pattern(x, y)

            # print(f"Epoch {epoch+1}/{epochs}, #Hyperboxes={len(self.hyperboxes)}")


    def predict(self, X):
        '''
        Predict the class of the pattern X
        '''
        classes = np.unique(self.classes)
        results = []
        memberships = self.membership(X)
        max_prediction = 0
        pred_class = 0
        for _class in classes:
            mask = np.zeros((len(self.hyperboxes),))
            mask[np.where(self.classes == _class)] = 1
            p = memberships * mask
            prediction, class_index = np.max(p), np.argmax(p)
            if prediction > max_prediction:
                max_prediction = prediction
                pred_class = class_index

        return max_prediction, self.classes[pred_class]


    def score(self, X, Y):
        '''
        Scores the classifier
        '''
        count = 0
        for x, y in zip(X, Y):
            _, pred = self.predict(x)
            if y == pred:
                count += 1

        return count / len(Y)
    
    
    
if __name__ == "__main__":
    
    #X_train, y_train, X_test, y_test = load_iris_data()
    X_train, y_train, X_test, y_test = load_K_chess_data_splitted()

    _max = 1
    _min = 0.1
    
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()
    
    X_train = (X_train - _min) / (_max - _min)
    
    

    f = FuzzyMMC()
    f.fit(X_train, y_train, 100)
    f.score(X_test, y_test)
    print("Trained")
    print(f.score(X_test, y_test))