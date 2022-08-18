#!/usr/bin/env python3
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.models import Model

######## TEST - direct output - crash pattern - openCL  #########################
x0 = Input(shape=(4, 4, 8))

x = Conv2D( 8, (1, 1))(x0)
x1 = Conv2D( 8, (1, 1))(x)
x = Conv2D( 8, (1, 1))(x1)

model = Model([x0], [x, x1], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('direct_output.h5')
model.save('direct_output')

######## TEST - indirect output - crash pattern - openCL  #########################
x0 = Input(shape=(4, 4, 8))

x = Conv2D( 8, (1, 1))(x0)
x1 = Conv2D( 8, (1, 1))(x)
x2 = Activation('relu')(x1)
x = Conv2D( 8, (1, 1))(x1)

model = Model([x0], [x, x2], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('indirect_output.h5')
model.save('indirect_output')