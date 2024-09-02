from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


train_datagen = ImageDataGenerator(
    rescale=1./255,               
    rotation_range=20,            
    width_shift_range=0.2,        
    height_shift_range=0.2,       
    shear_range=0.2,              
    zoom_range=0.2,              
    horizontal_flip=True,         
    fill_mode='nearest'           
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'images',             
    target_size=(224, 224),       
    batch_size=32,                
    class_mode='categorical'      
)

# Load the validation data
validation_generator = validation_datagen.flow_from_directory(
    'images',         
    target_size=(224, 224),       
    batch_size=32,               
    class_mode='categorical'     
)



base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Build the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
model.save('person_classifier_model.h5')