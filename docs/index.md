## 📘 Tutorial : Real Time Hand Sign Interpretation using LSTM

This repository includes a step-by-step guide for building and understanding the system:

## Visual Overwiew

![Hand Detection Example](/docs/image.png)


```mermaid
flowchart TD
    A0["MediaPipe Library
"]
    A1["Keypoint Extraction
"]
    A2["Sequence Handling
"]
    A3["LSTM Model
"]
    A4["Real-Time Inference Loop
"]
    A5["Label Mapping and Encoding
"]
    A6["Data Splitting
"]
    A0 -- "Provides Landmarks" --> A1
    A1 -- "Feeds Frame Data" --> A2
    A2 -- "Provides Sequence Data" --> A6
    A5 -- "Provides Labels" --> A6
    A6 -- "Supplies Training Data" --> A3
    A3 -- "Provides Predictions" --> A4
    A4 -- "Uses for Processing" --> A0
    A4 -- "Maps Predictions" --> A5
```

## 💡 Features

- **Real-time detection** using webcam feed
- **MediaPipe** for precise hand landmark tracking
- **LSTM-based** classification on sequential landmark data
- Supports typical sign language gestures like “hello,” “thank you,” etc.
- Easy to extend: train new gestures by adding labeled sequences

## 🧪 Results

- **hello** → ✅
- **thanks** → ✅
- **iloveyou** → ✅
- Additional gestures: _you, yes, no, please, etc

Accuracy on test set: **~98%**  

- [Chapter 1: MediaPipe Library](docs/chapter1.md)
- [Chapter 2: Keypoint Extraction](docs/chapter2.md)
- [Chapter 3: Sequence Handling](docs/chapter3.md)
- [Chapter 4: Label Mapping , Encoding and Splitting](docs/chapter4.md)
- [Chapter 5: LSTM Model](docs/chapter5.md)
- [Chapter 6: Real-Time Inference](docs/chapter6.md)




