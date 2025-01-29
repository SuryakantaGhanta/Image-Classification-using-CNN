# Image-Classification-using-CNN
## Data Description: 
The dataset is provided as a zip file containing a collection of images organized into 15 distinct folders. Each folder corresponds to a unique class of animals, where the folder name serves as the class label. The images within the dataset are standardized to dimensions of 224 x 224 x 3, making them compatible for use in convolutional neural networks (CNNs) for image classification tasks.

The dataset represents the following animal classes:  
- Bear  
- Bird  
- Cat  
- Cow  
- Deer  
- Dog  
- Dolphin  
- Elephant  
- Giraffe  
- Horse  
- Kangaroo  
- Lion  
- Panda  
- Tiger  
- Zebra  

This dataset is well-suited for machine learning and deep learning applications, particularly in training and evaluating models to classify images into their respective animal categories. It provides a structured and diverse set of examples, enabling effective model training and validation.
## Data Preparation for Analysis:
Here’s a rephrased and clear explanation of the process, presented in an organized manner:

---

The 15 image folders from the zip file are extracted, and the paths of the images are collected using the `glob` library. These paths are stored in two separate lists: `image_data` (for the image data) and `labels` (for the corresponding class labels).

### Data Preprocessing Workflow:

1. **Iterating Through Files and Labels**  
   The images are processed through a loop, where `file_paths` represents the paths of the images for a specific class, and `class_name` represents the corresponding class label.

   ```python
   for file_paths, class_name in all_file_paths:
       for file_path in file_paths:
           # Load the image
           image = cv2.imread(file_path)
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

           # Resize the image to a uniform size (128 x 128)
           resized_image = cv2.resize(image, (128, 128))

           # Normalize the pixel values to the range [0, 1]
           normalized_image = resized_image / 255.0

           # Append the processed image data and its label
           image_data.append(normalized_image)
           labels.append(class_label_mapping[class_name])
   ```

2. **Conversion to NumPy Arrays**  
   After preprocessing, the image data and labels are converted to NumPy arrays to facilitate compatibility with machine learning models.  
   ```python
   X = np.array(image_data)  # Contains normalized image data
   y = np.array(labels)      # Contains corresponding labels
   ```

The final variable `y` contains the class labels for all images, encoded numerically as per the mapping defined in `class_label_mapping`.

This structured approach ensures the dataset is ready for training a convolutional neural network (CNN) while maintaining uniform image dimensions and normalized pixel values.
## Train Test Split of the data and the Model building:
### Splitting the Data for Training and Testing:

To ensure a robust evaluation of the model's performance, the dataset is split into training and testing subsets using the following line of code:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This step divides the processed dataset as follows:

1. **Purpose of `train_test_split`**:  
   The function is used to allocate a portion of the data for training the model and the rest for testing its accuracy and generalization ability.

2. **Key Parameters**:  
   - `X`: The feature dataset, containing normalized image data.  
   - `y`: The target labels associated with the images.  
   - `test_size=0.2`: Specifies that 20% of the data will be reserved for testing, while the remaining 80% will be used for training.  
   - `random_state=42`: Ensures reproducibility of the data split by setting a fixed seed for randomization.

3. **Outputs**:  
   - `X_train`: The training subset of image data.  
   - `X_test`: The testing subset of image data.  
   - `y_train`: The training labels corresponding to `X_train`.  
   - `y_test`: The testing labels corresponding to `X_test`.

### Importance in Data Preparation Workflow:

- The **training set (`X_train`, `y_train`)** is used to train the convolutional neural network (CNN), allowing the model to learn patterns and features from the data.  
- The **testing set (`X_test`, `y_test`)** is used to assess the model's performance on unseen data, ensuring it can generalize effectively to new inputs.

This structured division is a crucial step in validating the reliability and accuracy of the machine learning model across various metrics such as accuracy, precision, recall, and F1-score.
