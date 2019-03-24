import tensorflow as tf
import sys
import csv
import numpy

def load_csv(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        dataset = list(reader)
    
    i = 0
    for row in dataset:
        if i > 0:
            y_train.append(int(row[0]))
            val = row[1:]
            x_train.append(val)
        i = i + 1;
        
def load_csvTest(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        dataset = list(reader)
    
    i = 0
    for row in dataset:
        if i > 0:
            x_test.append(row)
        i = i + 1;
    
y_train = []
x_train = []
load_csv(sys.argv[1])

x_test = []
load_csvTest(sys.argv[2])

testLabel = open(sys.argv[3], 'w')

x_train = numpy.array(x_train, dtype=numpy.float32)
x_test = numpy.array(x_test, dtype=numpy.float32)

x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
x_train, x_test = x_train/255.0, x_test/255.0
y_train = numpy.array(y_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=70)
predictions = model.predict(x_test)
testLabel.write("ImageId,Label\n")
i = 1
for row in predictions:
    testLabel.write(str(i) + "," + str(numpy.argmax(row)) + "\n")
    i = i + 1