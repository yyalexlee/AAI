
#===============================================================================
# set callbacks
#===============================================================================
def set_callbacks(filepath):
    callbacks_list = [

      callbacks.EarlyStopping(
          monitor='val_loss',
          mode='min',
          patience=30,
      ),
 
      callbacks.ModelCheckpoint(
          filepath=filepath,
          monitor='val_loss',
          save_best_only=True,
      ),

      callbacks.ReduceLROnPlateau(monitor='val_loss', 
          factor=0.2,
          patience=5, 
          min_lr=0.001
      )

    ]
   
    return(callbacks_list)

#===============================================================================
# set model architecture
#===============================================================================
def basicCNN_terc(INPUT_SHAPE):
    af = 'selu'
    otm = 'adam'   
    lf = 'sparse_categorical_crossentropy'
    #lf = 'categorical_crossentropy'
    kl = 'l1_l2'
    model = Sequential(
        [
            Conv2D(8, [3,3], input_shape=INPUT_SHAPE, activation=af, padding='valid', strides=1,
                             kernel_regularizer=kl),
            BatchNormalization(),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(16, [3,3], activation=af, padding='valid', strides=1),
            BatchNormalization(),
            #Dropout(0.2),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(32, [3,3], activation=af, padding='valid', strides=1),
            BatchNormalization(),
            #Dropout(0.2),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(64, [3,3], activation=af, padding='valid', strides=1),
            BatchNormalization(),
            #Dropout(0.2),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(32, [3,3], activation=af, padding='valid', strides=1,name='target_layer'),
            Flatten(),
            #Dense(50, activation=af),
            #Dense(256, activation=af,kernel_regularizer=regularizers.l2(0.2)),
            #Dense(16, activation=af,kernel_regularizer=regularizers.l1l2(0.4)),
            Dense(16, activation=af,kernel_regularizer=kl),
            Dense(3, activation='softmax'),
       ]
    )

    model.compile(optimizer=otm, loss=lf, metrics=['accuracy'])

    print(model.summary())
    return model




def basicCNN_terc_3d(INPUT_SHAPE):
    af = 'selu'
    otm = 'adam'
    lf = 'sparse_categorical_crossentropy'
    ki = 'he_uniform'
    kl = 'l1_l2'
    model = Sequential(
        [
            Conv3D(8, [3,3,3], input_shape=INPUT_SHAPE, activation=af, padding='valid', strides=1, 
                               kernel_initializer=ki),
            MaxPooling3D((4,4,2), strides=(4,4,1), padding='valid'),
            BatchNormalization(center=True,scale=True),
            #Dropout(0.2),
            Conv3D(16, [3,3,3], activation=af, padding='same', strides=1, kernel_initializer=ki),
            MaxPooling3D((4,4,2), strides=(4,4,1), padding='same'),
            BatchNormalization(center=True,scale=True),
            #Dropout(0.2),
            Conv3D(32, [3,3,3], activation=af, padding='valid', strides=1, kernel_initializer=ki),
            Flatten(),
            #Dropout(0.2),
            Dense(16, activation=af,kernel_initializer=ki,kernel_regularizer=kl),
            Dense(3, activation='softmax'),
        ]
    )

    model.compile(optimizer=otm, loss=lf, metrics=['accuracy'])
    print(model.summary())

    return model




def basicCNN(INPUT_SHAPE):
    af = 'tanh'
    otm = 'adam'   
    lf = 'mean_squared_error'
    model = Sequential(
        [
            Conv2D(8, [3,3], input_shape=INPUT_SHAPE, activation=af, padding='valid', strides=1),
            BatchNormalization(),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(16, [3,3], activation=af, padding='valid', strides=1),
            BatchNormalization(),
            Dropout(0.2),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(32, [3,3], activation=af, padding='valid', strides=1),
            BatchNormalization(),
            Dropout(0.2),
            MaxPooling2D((2,2), strides=2, padding='valid'),
            Conv2D(64, [3,3], activation=af, padding='valid', strides=1),
            Flatten(),
            #Dense(50, activation=af),
            #Dense(256, activation=af,kernel_regularizer=regularizers.l2(0.2)),
            Dense(16, activation=af,kernel_regularizer=regularizers.l2(0.4)),
            Dense(1, activation=None),
        ]
    )

    model.compile(optimizer=otm, loss=lf, metrics=['cosine_similarity'])

    print(model.summary())
    return model



'''
def basicCNN(INPUT_SHAPE):
    af = 'tanh'
    otm = 'adam'
    lf = 'mean_squared_error'
    model = Sequential(
        [
            Conv2D(32, [3,3], input_shape=INPUT_SHAPE, activation=af, padding='valid', strides=1),
            MaxPooling2D((2,2), strides=2, padding='same'),
            Conv2D(16, [3,3], activation=af, padding='same', strides=1, name='target_layer'),
            Dropout(0.2),
            MaxPooling2D((2,2), strides=2, padding='same'),
            Conv2D(8, [3,3], activation=af, padding='same', strides=1),
            Flatten(),
            #Dropout(0.2),
            Dense(16, activation=af,kernel_regularizer=regularizers.l2(0.2)),
            Dense(1, activation=None),
        ]
    )

    model.compile(optimizer=otm, loss=lf, metrics=['cosine_similarity'])
    print(model.summary())

    return model
'''
