import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers

# y = w1*x1 + w2*x2 + b

def gen_sequential_model():
        model = Sequential([
                Input(2, name='input_layer'),
                #실험할때 weight값을 고정하고 하는것이 좋음(kernel_initializer)
                Dense(16, activation='sigmoid', name='hidden_layer1',
                      kernel_initializer= initializers.RandomNormal(mean = 0.0, stddev=0.05, seed= 42)),
                Dense(1, activation='relu', name='output_layer',
                      kernel_initializer= initializers.RandomNormal(mean = 0.0, stddev=0.05, seed= 42))
                ])

        model.summary()
        # print(model.layers[0].get_weights())
        # print(model.layers[1].get_weights())
        #stochastic gradient descent, SGD LossFunction => min squared multi 일때는 cross..
        model.compile(optimizer='sgd', loss='mse')
        return model

def gen_linear_regression_dataset(numofsamples=1000, w1=3, w2=5, b=10):
        np.random.seed(0)
        X = np.random.rand(numofsamples, 2) #random 2차원 array를 만듦
        # print(X)
        # print(X.shape)

        coef = np.array([w1,w2])
        bias = b
        #
        # print(coef)
        # print(coef.shape)

        y = np.matmul(X, coef.transpose()) + bias
        # print(y)
        # print(y.shape)

        return X, y

#loss/epoch 그래프
def plot_loss_curve(history):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15,10))

        plt.plot(history.history['loss'][1:])
        plt.plot(history.history['val_loss'][1:])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc = 'upper right')
        plt.show()

#예측값 비교
def predict_new_sample(model, x, w1=3, w2=5, b = 10):
        x = x.reshape(1,2)
        y_pred = model.predict(x)[0][0]
        y_actual = w1*x[0][0] + w2*x[0][1] + b

        print("y actual value = ", y_actual)
        print("y predicted value = ", y_pred)

model = gen_sequential_model()
X, y = gen_linear_regression_dataset(numofsamples=2000)
history = model.fit(X, y, epochs=100, verbose=2, validation_split=0.3)
plot_loss_curve(history)
print('train loss=', history.history['loss'][-1])
print('test loss=', history.history['val_loss'][-1])

predict_new_sample(model, np.array([0.3,0.8]))
