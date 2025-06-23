# 🤖📊 Chapter 6: LSTM Model

Welcome back, gesture recognition experts! So far, we have prepared our data by extracting keypoints from video frames, organizing them into sequences, encoding labels, and splitting into training and test sets. Now it’s time to build the core of our system: a model that understands motion over time to recognize hand signs. ✋➡️🤟

## 🧠⏳ The Challenge: Understanding Motion

Recognizing a gesture isn’t just about a single static pose—it’s about how the pose changes across frames. Traditional neural networks process inputs independently and lack memory of past frames. We need a network that remembers sequence context.

LSTMs (Long Short-Term Memory networks) are designed for sequential data. They have internal gates that control what to remember or forget over time, enabling them to capture temporal dependencies—ideal for gestures where the order of movements matters. 🔄

## Introducing the LSTM Model

This is where **LSTMs (Long Short-Term Memory)** come in. LSTMs are a special kind of **recurrent neural network (RNN)** designed specifically to work with sequential data. Unlike basic neural networks, LSTM layers have internal mechanisms (often called "gates") that allow them to selectively remember or forget information from past time steps in the sequence.

Think of an LSTM as having a kind of "memory cell" that runs through the sequence. At each frame (time step), the LSTM processes the new keypoint data for that frame along with the information it has stored in its memory cell from the previous frames. It then updates its memory and produces an output for the current frame, which is passed along to the next time step or the next layer.

This memory capability makes LSTMs perfectly suited for tasks where the order and temporal dependencies in the data are important, such as:

*   Understanding sentences (the meaning of a word depends on previous words).
*   Predicting stock prices (current price depends on past prices and trends).
*   And, in our case, recognizing hand signs from sequences of body poses!

By using an LSTM, our model can learn the *patterns of movement* captured in our 30-frame keypoint sequences.

## 🏗️ Building the LSTM Model with Keras 🏗️

Using Keras, we stack layers to build our model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # Number of classes (e.g., 3)

```

Let's break down the layers we added to our `Sequential` model:

| Layer Type | Units | Output Shape | Description                      |
| ---------- | ----- | ------------ | -------------------------------- |
| LSTM       | 64    | (30, 64)     | Processes input sequence         |
| LSTM       | 128   | (30, 128)    | Learns complex temporal patterns |
| LSTM       | 64    | (64)         | Summarizes sequence to vector    |
| Dense      | 64    | (64)         | Fully connected                  |
| Dense      | 32    | (32)         | Further processing               |
| Dense      | 3     | (3)          | Outputs class probabilities      |

```
Note: return_sequences=True keeps output as a sequence for stacking LSTMs; the last LSTM outputs a single vector (return_sequences=False) for classification layers.
```

## ⚙️ Compiling the Model

Before we can train the model, we need to configure how it will learn. This is done with the `compile` step:

```python
# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

* **Adam optimizer:** Efficient and widely used for training neural networks. ⚡

* **Categorical crossentropy loss:** Measures how well predicted probabilities match true labels. 🎯

* **Categorical accuracy:** Tracks accuracy during training. ✅

## 🚂 Training the Model

Now that the model is defined and compiled, we can train it using our prepared training data (`X_train`, `y_train`):

```python
# Train the model
# epochs=500: How many times to iterate over the entire training dataset
model.fit(X_train, y_train, epochs=500)
```

*   `model.fit(X_train, y_train, ...)`: This is where the learning happens! The model takes the training sequences (`X_train`) and their true labels (`y_train`) and adjusts its parameters (`weights` and `biases`) to reduce the `loss`.

*   `epochs=500`: An epoch means the model has seen every sequence in the training set once. Training for many epochs allows the model to refine its understanding of the data patterns. The output you see during training shows the loss and accuracy improving (hopefully!) over the epochs.

**Important Note:** Training can take some time, especially with larger datasets or more complex models. The notebook shows output for each epoch. You might see the accuracy go up and down a bit, but the general trend should be improvement. Training is stopped manually in the example notebook after ~140 epochs, but you could let it run for the full 500 or stop it when the accuracy on the training data plateaus.

**Tip -> Use early stopping to prevent overfitting:**

```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=500, callbacks=[early_stopping])
```

## Evaluating the Model 📈
Check how your model performs on unseen test data:

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

## 📝 Model Summary

After defining the model, you can print its summary to see its structure and the number of parameters:

```python
# Print model summary
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 30, 64)            442112
_________________________________________________________________
 lstm_1 (LSTM)               (None, 30, 128)           98816
_________________________________________________________________
 lstm_2 (LSTM)               (None, 64)                49408
_________________________________________________________________
 dense (Dense)               (None, 64)                4160
_________________________________________________________________
 dense_1 (Dense)             (None, 32)                2080
_________________________________________________________________
 dense_2 (Dense)             (None, 3)                 99
=================================================================
Total params: 596,675
Trainable params: 596,675
Non-trainable params: 0
_________________________________________________________________
```
*(Note: The exact layer names like `lstm`, `lstm_1` might vary slightly depending on how many times you've built models in the same notebook session).*

## How the Data Flows Through the Model (Simplified)

Let's visualize the journey of one sequence of keypoints through our defined model layers:

```mermaid
sequenceDiagram
    participant Input Sequence (30x1662);
    participant LSTM Layer 1 (64 units);
    participant LSTM Layer 2 (128 units);
    participant LSTM Layer 3 (64 units);
    participant Dense Layers (64, 32 units);
    participant Output Layer (3 units, Softmax);
    participant Prediction (Probabilities);

    Input Sequence->>LSTM Layer 1: Processes frame 0, then 1, ..., 29
    Note over LSTM Layer 1: Remembers/forgets info across frames,<br/>Outputs a sequence (30x64).
    LSTM Layer 1-->>LSTM Layer 2: Pass sequence output
    LSTM Layer 2->>LSTM Layer 3: Processes sequence further
    Note over LSTM Layer 3: Processes final sequence,<br/>Outputs a single vector (64 values).
    LSTM Layer 3-->>Dense Layers: Pass summarized vector
    Dense Layers->>Output Layer (3 units, Softmax): Process vector, calculate scores for each class
    Output Layer (3 units, Softmax)-->>Prediction (Probabilities): Convert scores to probabilities (e.g., [0.1, 0.8, 0.1])

    Note over Input Sequence (30x1662): One sequence of 30 frames, each 1662 keypoints.
    Note over Prediction (Probabilities): Sums to 1.
```

This diagram shows how the sequence data is processed sequentially by the LSTM layers, summarized, and then classified by the Dense layers.

## 💾 Saving and Loading the Model 

Save your trained model for future use:

```python
model.save('action.h5')
```

Load it later without retraining:

```python
from keras.models import load_model
model = load_model('action.h5')
```

Optionally, save weights and architecture separately:

```python
model.save_weights('action_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
```

## 📝 Note: Using the Saved Model Without Training 

If you don’t want to train the model from scratch, you can directly use the pre-trained action.h5 model provided with this project. This means you can skip the entire training process and jump straight into real-time gesture recognition! 🚀

```python
from keras.models import load_model

# Load the pre-trained model
model = load_model('action.h5')
```
```
✅ No need for training if you're using the provided model — just load it and start recognizing hand signs in real-time!
```




## Final Thoughts 🎉

You’ve now built a powerful model that learns temporal patterns in hand gestures. Next up, we’ll integrate this model into a real-time system that predicts gestures live from webcam input. 🎥🖐️

Ready to see your model in action? Let’s move on!

All you need to do now is run the `RealtimeTest.ipynb` file to see your model in action and making predictions.


## 🎉 Thank You!

Congratulations on reaching the end of this tutorial series! 🙌 You've gone from raw video data to building a full deep learning model capable of recognizing hand gestures using LSTMs. Whether you're applying this to sign language interpretation, human-computer interaction, or your own creative project—we hope this journey gave you both the skills and confidence to explore further. Thank you for learning with us, and keep building amazing things! 🚀🤖💡