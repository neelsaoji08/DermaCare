from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Dropout,MaxPooling2D,Input
from keras.callbacks import ReduceLROnPlateau

    

def model_training_personal(X_train,X_valid,y_train,y_valid,size,
                            optimizer="Adam",loss="categorical_crossentropy",
                            metrics=['accuracy'],epochs=30,batch_size=128):

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)
    
    model = Sequential()
    # Create Model Structure
    model.add(Input(shape=[size[0], size[1], 3]))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dropout(rate=0.2))

    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=7, activation='softmax'))
    model.compile(optimizer=optimizer, loss= loss, metrics=metrics)

    model.summary()


    history=model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_valid, y_valid),
    callbacks=[learning_rate_reduction]
    )

    return model,history


def transfer_learning_model(base_model,X_train,X_test,y_train,y_test,
                            optimizer="Adam",loss="categorical_crossentropy",
                            metrics=['accuracy'],epochs=30,batch_size=128):

    x = Flatten()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(7, activation='softmax')(x)


    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    history=model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test)
    )

    return model,history

    


    

