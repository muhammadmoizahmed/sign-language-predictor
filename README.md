# 🤟 Sign Language Predictor

This is a **Sign Language Recognition** project built using Python, OpenCV, and Machine Learning. It uses a pre-trained model to predict hand signs from live camera input.

## 📂 Project Structure

```
signlanguageproject/
│
├── src/                   # Python source code files (main app, helper scripts)
├── models/                # Trained ML model (.pkl file)
├── dataset/               # CSV dataset used for training
├── samples/               # Sample images for testing
├── requirements.txt       # List of required Python libraries
```

## 🧠 Features

- Real-time hand gesture recognition using webcam.
- Machine Learning model trained on custom dataset.
- User-friendly GUI interface for predictions.
- Predicts alphabets based on hand signs.

## 🔧 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/muhammadmoizahmed/sign-language-predictor.git
cd sign-language-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python src/app.py
```

> 💡 Make sure your webcam is working before running the app.

## 📝 Dataset

- A CSV file located in the `dataset/` folder was used for model training.
- You can modify or extend the dataset as per your project needs.

## 📦 Model

- The model is stored in the `models/` folder as a `.pkl` file.
- It is loaded at runtime to make predictions from the webcam feed.

## 📸 Sample Test Data

Some sample hand gesture images are available in the `samples/` folder for testing and evaluation purposes.

## 🙋‍♂️ Author

- **Muhammad Moiz Ahmed**
- GitHub: [@muhammadmoizahmed](https://github.com/muhammadmoizahmed)

## 📃 License

This project is licensed under the MIT License.
