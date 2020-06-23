import bilsm_crf_model
import process_data
import numpy as np

EPOCHS = 5
model, (train_x, train_y) = bilsm_crf_model.create_model()
model.fit(train_x[:600], train_y[:600],batch_size=64,epochs=EPOCHS)
model.save('model/crf.h5')

result=model.evaluate(train_x[600:], train_y[600:],batch_size=64,verbose=2)
print(result)







