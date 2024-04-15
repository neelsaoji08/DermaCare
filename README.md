# Skin Cancer Classification Project

This project aims to classify skin cancer images using machine learning techniques. The dataset used in this project is the HAM10000 dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

## Steps to Run the Project

1. **Download the Dataset**:
   - Download the dataset from the provided Kaggle link.

2. **Install Necessary Libraries**:
   - Ensure you have all the necessary libraries installed. You can install them using pip:
     ```
     pip install -r requirements.txt
     ```

3. **Organize the Dataset**:
   - Add all the images to a folder named `HAM10000_images_part_1`.
   - Put this image folder along with all other dataset files in a folder called `dataset`.

4. **Run the Training Pipeline**:
   - Open and run `main_pipeline.ipynb` to train and save the model.
   - You can also adjust the size of the images used for training within this notebook.

5. **Run the Front-end**:
   - To run the front-end interface, use the following command:
     ```
     streamlit run main.py
     ```

6. **Make Predictions**:
   - With the front-end running, you will be able to make predictions on new images.

## Notes

- Ensure you have Python installed on your system.
- This project uses Streamlit for the front-end interface.
