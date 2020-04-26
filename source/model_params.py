#general prameters
train_size = 0.8

#Predictor Parameter
num_epochs = 100
batch_size = 50 
sequence_size = 5
input_size = sequence_size * 1
learning_rate = 1e-5


#Policy Network Parameters
bench_mark_output_size = 20
input_channels = 1
epochs = 1
window_size = 5
policy_learning_rate = 1e-3
policy_batch_size = 1
transaction_commission = 0.0025	