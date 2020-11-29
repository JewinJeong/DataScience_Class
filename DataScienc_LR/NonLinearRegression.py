import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers


def gen_sequential_model():
    model = Sequential([
        Input(3, name='input_layer'),
        # 실험할때 weight값을 고정하고 하는것이 좋음(kernel_initializer)
        Dense(16, activation='sigmoid', name='hidden_layer1',
              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
        Dense(16, activation='sigmoid', name='hidden_layer2',
              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
        Dense(1, activation='relu', name='output_layer',
              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
    ])

    model.summary()
    # print(model.layers[0].get_weights())
    # print(model.layers[1].get_weights())
    # stochastic gradient descent, SGD LossFunction => min squared multi 일때는 cross..
    model.compile(optimizer='sgd', loss='mse')
    return model

def gen_nonlinear_regression_dataset(numofsamples=3000, w1=3, w2=5, w3=10, a=1, e=20):
    np.random.seed(0)
    X = np.random.rand(numofsamples, 3)
    print(X)
    print(X.shape)
    y = list()
    for i in range(numofsamples):
        y.append(a + w1*X[i][0] + w2*X[i][1]**2 + w3*X[i][2]**3 + e)

    y = np.array(y)
    print(y)
    print(y.shape)

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
def predict_new_sample(model, x, w1=3, w2=5, w3=10, a=1, e=20):
        x = x.reshape(1,3)
        y_pred = model.predict(x)[0][0]
        y_actual = a + w1*x[0][0] + w2*x[0][1]**2 +w3*x[0][2]**3 + e

        print("y actual value = ", y_actual)
        print("y predicted value = ", y_pred)



model = gen_sequential_model()
X, y = gen_nonlinear_regression_dataset()
y = np.array(y)
history = model.fit(X, y, epochs=1500, verbose=2, validation_split=0.3)
plot_loss_curve(history)
print('train loss=', history.history['loss'][-1])
print('test loss=', history.history['val_loss'][-1])

predict_new_sample(model, np.array([0.3,0.8,0.4]))


