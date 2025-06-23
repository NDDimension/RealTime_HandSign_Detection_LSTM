# 🔁 Chapter 3: Sequence Handling

In Chapter 2, we extracted keypoints from each frame. But recognizing a sign like "hello" needs more than one frame — it requires tracking motion over time. That’s what sequence handling solves.

## 📉 The Problem: Single Frames Don't Show Motion

Single frames give only a snapshot — no context of what happened before or after. To understand motion, we need consecutive keypoint frames, just like flipping through pages in an animation.

## 📦 What is a "Sequence" in Our Project?

A sequence is a group of 30 consecutive frames of keypoint data. Each one represents a short clip of movement — essential for detecting dynamic gestures.

**sequence_length = 30**
**Each sequence = 30 frames × 1662 values** (from Chapter 2)

## 📏 Why a Fixed Length?

Models like LSTM require consistent input shapes.

- Fixed size = predictable input

- Our LSTM is built for 30-frame sequences

So, all sequences must be exactly 30 frames long.

## 🧪 How We Handle Sequences in Practice

From the dataset:

```
MP_Data/
├── hello/
│   ├── 0/          <-- This is one sequence (video example)
│   │   ├── 0.npy   <-- Keypoints for frame 0
│   │   ├── 1.npy   <-- Keypoints for frame 1
│   │   ...
│   │   └── 29.npy  <-- Keypoints for frame 29
│   ├── 1/          <-- Another sequence
│   │   ├── 0.npy
│   │   ...
│   ... (e.g., 30 sequences for 'hello')
├── thanks/
│   ... (e.g., 30 sequences for 'thanks')
└── iloveyou/
    ... (e.g., 30 sequences for 'iloveyou')
```
Each .npy holds a 1662-length keypoint vector. We load them like this:

```python
import os
import numpy as np
# Assuming DATA_PATH, actions, and sequence_length are defined as in the notebook
# e.g., DATA_PATH = "D:\\LSTM\\MP_Data"
#       actions = np.array(['hello', 'thanks', 'iloveyou'])
#       sequence_length = 30

sequences, labels = [], [] # Lists to store our data

# Loop through each action (e.g., 'hello', 'thanks', 'iloveyou')
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    # Find all sequence folders (the numbered ones) for this action
    sequence_folders = sorted([d for d in os.listdir(action_path) if d.isdigit()], key=int)

    # Loop through each sequence folder (e.g., '0', '1', '2', ...)
    for sequence in sequence_folders:
        window = [] # This list will temporarily hold keypoints for one sequence

        # Loop through each frame file within the sequence folder (e.g., '0.npy', '1.npy', ...)
        for frame_num in range(sequence_length):
            file_path = os.path.join(action_path, sequence, f"{frame_num}.npy")

            # Check if the file exists before loading (important!)
            if os.path.exists(file_path):
                res = np.load(file_path) # Load the keypoint data for this frame
                window.append(res)       # Add the keypoint data (a 1662-value array) to our window list
            else:
                print(f"❌ Missing file: {file_path}")
                break # Skip this sequence if any frame is missing

        # Only add the sequence if it's complete (has all 'sequence_length' frames)
        if len(window) == sequence_length:
            sequences.append(window) # Add the complete sequence (list of 30 arrays) to our main sequences list
            labels.append(label_map[action]) # Add the corresponding label for this sequence

        else:
            print(f"⚠️ Skipped incomplete sequence: {action}/{sequence}")

```

This code:

1. Loops through each action and sequence

2. Loads 30 .npy files per sequence

3. Builds a 30-frame "window"

4. Adds it to sequences only if complete


## 🧱  The Final Data Structure

After running the loading code, the `sequences` list contains all the raw data, structured by sequence. We then convert this list of lists into a single NumPy array, which is the standard format for machine learning inputs.

Let's look at the shape of this NumPy array:

```python
# After the loading code...
X = np.array(sequences)
print(X.shape)
```

```
(90, 30, 1662)
```

This output `(90, 30, 1662)` tells us:

*   `90`: We have a total of 90 sequences (e.g., 3 actions \* 30 sequences per action).
*   `30`: Each sequence has a length of 30 frames.
*   `1662`: Each frame's data (each element in the sequence) is a vector of 1662 keypoint values.

This 3D NumPy array `X` is the properly formatted input data ready for our model.

Here's a simple diagram illustrating how the data is structured at this point:

```mermaid
graph LR
    A["MP_Data Folder"] --> B["Action Folders (hello, thanks, iloveyou)"];
    B --> C["Sequence Folders (0, 1, 2, ...)"];
    C --> D["Frame Files (0.npy, 1.npy, ...)"];
    D --> E["Keypoint Vector (e.g., 1662 values)"];
    E --> F["Collect 30 vectors into a window"];
    F --> G["One Sequence (30x1662 array)"];
    G --> H["Combine all sequences"];
    H --> I["Final Training Data (90x30x1662 array)"];

    subgraph Processing
        D -- np.load() --> E;
        C -- Loop through frames --> F;
        B -- Loop through sequences --> G;
        A -- Loop through actions --> H;
    end

    Note over I: This is the X (input) data for the model.
```

This process of loading the individual frame data and grouping it into fixed-length sequences is essential for training a model that understands the *movement* of a hand sign, not just static poses.

## Conclusion

We now group frames into consistent 30-frame sequences so our model can learn motion. These sequences become the foundation of our input data.

Next, we’ll explore how we pair each sequence with the correct label like "hello" or "thanks".

[Label Mapping , Encoding and Splitting](chapter4.md)
