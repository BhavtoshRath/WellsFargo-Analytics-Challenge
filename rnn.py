from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from script_1 import user_features


truncated_backprop_length = 5
state_size = 4
num_classes = 21  # Number of unique purchase categories (smaller the number, better is the prediction)
echo_step = 3  # Shift of purchase sequence within shiftSeq.
batch_size = 3

customer = raw_input("Please enter customer's masked id: ")
customer = int(customer)

'''input: Customer's masked id.
return: Actual purchase sequence, Sifted purchase sequence  '''


def shiftedSeq(user):
    seq = user_features[user]
    seq = seq[0:len(seq) - (len(seq)%batch_size)]
    new_seq = np.roll(seq, echo_step)
    new_seq[0:echo_step] = 0

    seq = seq.reshape((batch_size, -1))
    new_seq = new_seq.reshape((batch_size, -1))

    return seq, new_seq


'''R-NN internals.'''
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
# Initializing the weights and biases
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    states_series.append(next_state)
    current_state = next_state

# Computing prediction probabilities
logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

# Computing the loss for each batch data
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

# Optimization step
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


'''TensorFlow graphs is executed below within a session. '''
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    batch_count = []
    pred_dict = dict()
    seq, new_seq = shiftedSeq(customer)

    _current_state = np.zeros((batch_size, state_size))

    _predicted_state = np.zeros((truncated_backprop_length, batch_size))

    num_batches = len(user_features[customer])//batch_size//truncated_backprop_length

    # Generate batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * truncated_backprop_length
        end_idx = start_idx + truncated_backprop_length

        batch_seq = seq[:, start_idx:end_idx]
        batch_new_seq = new_seq[:, start_idx:end_idx]

        # Run RNN to generate predictions
        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
            [total_loss, train_step, current_state, predictions_series],
            feed_dict={
                batchX_placeholder: batch_seq,
                batchY_placeholder: batch_new_seq,
                init_state: _current_state
            })

        loss_list.append(_total_loss)
        batch_count.append(batch_idx + 1)

        for i in range(len(_predictions_series)):
            for j in range(len(_predictions_series[i])):
                t = (_predictions_series[i][j])
                label = np.where(t == np.amax(t))[0][0]
                _predicted_state[i, j] = label
        pred_dict[customer] = _predicted_state

        print('For masked_id: ', customer, 'Batch number: ', batch_idx)
        print('Current batch data: ', batch_seq)
        print('Shifted batch data: ', batch_new_seq)
        print('Predicted batch data: ', _predicted_state)


'''Plotting the loss calculated for each batch'''
plt.plot(batch_count, loss_list)
plt.ylabel('Loss (error) calculated for entered masked_id')
plt.xlabel('Batch number')
plt.show()


