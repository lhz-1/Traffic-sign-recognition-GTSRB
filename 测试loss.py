from Display_image import X_train,X_test,y_train,y_test
from Traffic_Model import Model
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



model_name = 'traffic net'  # 设置模型名称，后续对结果保存时候会用到

EPOCHS = 3 # 设置周期 (Epochs)
BATCH_SIZE = 128  # 设置批大小（批尺寸)


x = tf.placeholder(tf.float32, (None, 32, 32, 3))  #创建一个tf.float32类型 形状为（NULL，32，32，3）的tensor类# 形参，在执行的时候再赋给具体的值
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# 模型训练
rate = 0.001

logits, layers = Model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def loss(X_data, y_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})
    return loss



def evaluate(X_data, y_data):

    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    train_losss = []
    train_accuracys = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        n_train_right = 0

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            n_train_right += sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y}) * len(y_train[offset:end])


        train_loss = loss(X_test, y_test)
        train_losss.append(train_loss)
        train_accuracy = n_train_right / num_examples
        train_accuracys.append(train_accuracy)
        validation_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i + 1))
        print("Train          loss = {:.3f}".format(train_loss))
        print("Train      Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    x = range(0,EPOCHS)
    y1 = train_losss
    y2 = train_accuracys
    plt.plot(x, y1, marker='o',label= 'loss')
    plt.plot(x, y2, marker='*',label='accuracy')
    plt.title('loss and accuracy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('loss_and_accuracy.png')
    plt.show()
