Drinking Status Prediction App 🍷

A complete End-to-End Machine Learning and Deep Learning web application built to predict an individual's drinking status based on health examination indicators, utilizing a custom Artificial Neural Network (ANN) built with TensorFlow/Keras and deployed via Streamlit.

🚀 Project Overview

Lifestyle factors and routine health parameters often hold hidden correlations. This project uses tabular health checkup data (such as blood pressure, cholesterol levels, hemoglobin, and anthropometric measurements) to train a classification neural network that predicts whether an individual has a drinking habit (DRK_YN).

The pipeline is split into two primary phases:

Model Training & Artifact Generation: Preprocessing data, handling missing values, scaling features, training a regularized multi-layer ANN with Early Stopping, and saving the trained model and scaler.

Interactive Web Application: A fully functional Streamlit frontend allowing users to input their health indicators and instantly receive real-time inference, probability breakdown charts, and input summaries.

🛠️ Tech Stack & Libraries

Language: Python 3.x

Deep Learning / ML: TensorFlow, Keras, Scikit-Learn

Data Manipulation: Pandas, NumPy

Deployment & UI: Streamlit

Persistence: Joblib

📂 Project Structure

├── ANN_FOR_SOMKING_&_DRINKING_PREDICTION.keras   # Trained Keras Sequential Model
├── scaler.pkl                                   # Fitted StandardScaler for input normalization
├── app.py / script.py                           # Main application/training script containing Streamlit UI
└── README.md                                    # Project documentation


⚙️ Installation & Setup

Follow these steps to set up and run the project locally on your machine.

1. Clone the Repository / Copy Files

Ensure you have all the necessary code files and dataset in your local working directory.

2. Install Dependencies

Install the required Python packages using pip:

pip install tensorflow pandas numpy scikit-learn streamlit joblib matplotlib


3. Run the Training Pipeline (Optional)

If you wish to retrain the model from scratch using your dataset, update the dataset file path in the script:

df = pd.DataFrame(pd.read_csv("path/to/your/smoking_driking_dataset_Ver01.csv"))


Then execute the training script to generate the .keras model and .pkl scaler.

4. Launch the Streamlit App

To run the interactive web interface, navigate to the folder containing your script in your terminal and execute:

streamlit run app.py


(Replace app.py with the actual filename of your script if named differently).

🖥️ App Features

Interactive Health Form: Input vital statistics, blood panels (cholesterol, glucose, triglycerides), liver function markers (SGOT AST/ALT, Gamma GTP), and lifestyle status (smoking).

Automated Validation: Real-time warnings (e.g., checking if Systolic BP is logically higher than Diastolic BP).

Probability Visualization: Interactive bar charts and progress bars depicting the exact likelihood of the individual being classified as a drinker vs. non-drinker.

Tabbed Layout: Clean separation between analytical prediction outcomes and an audit trail of the raw inputs processed by the model.

📈 Model Architecture

The deep learning model is built using a sequential API structure featuring:

Input Layer: Matches the dimensionality of the scaled feature set.

Hidden Layers: Multiple dense layers utilizing ReLU activation functions combined with L2 Regularization to prevent overfitting.

Dropout Layer: Fraction-based dropouts ($25\%$) introduced to enhance generalization.

Output Layer: Single neuron with a sigmoid activation function for binary classification.

Optimization: Compiled using the Adam optimizer (learning rate = $0.005$) and binary_crossentropy loss, supported by EarlyStopping monitoring validation loss.

📝 License

This project is open-source and available for educational and personal use.
