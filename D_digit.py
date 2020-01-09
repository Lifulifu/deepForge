from model import build_discriminator_digit
from util import onehot, load_mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

xtr, ytr, xte, yte = load_mnist()
ytr = onehot(ytr, 10)
yte = onehot(yte, 10)

model = build_discriminator_digit()
model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0002),
            metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
checkpoint = ModelCheckpoint('outputs/D_digit.hdf5', monitor='val_accuracy')
model.fit(xtr, ytr, 
    epochs=2000,
    batch_size=128,
    validation_split=0.2,
    callbacks=[earlystopping, checkpoint]
)
