# Music Recommendation System Using Facial Expression  

This project recommends music based on the user's detected facial expressions. It combines deep learning for emotion recognition with Spotify's API to deliver personalized playlists.  

## 🛠️ Setup Instructions  

### Prerequisites  
- Python 3.x  
- Required libraries (install via `pip`):  
  ```bash
  pip install tensorflow opencv-python numpy pandas spotipy flask
  ```

### 🚀 Installation  
1. **Download Resources**  
   - Facial detection model: [Haar Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)  
   - Dataset: [FER2013 from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) (or use files in this repository).  

2. **Train the Model**  
   - Run `main.py` to train the emotion recognition model:  
     ```bash
     python main.py
     ```  
   - The trained model will be saved as `emotion_model.h5`.  

3. **Test the Model**  
   - Validate performance using `test.py`:  
     ```bash
     python test.py
     ```  

4. **Web Interface**  
   - A simple HTML/CSS frontend is provided to capture real-time facial expressions.  

5. **Spotify Integration**  
   - Register your app in the [Spotify Developer Dashboard](https://developer.spotify.com/) to obtain API credentials.  
   - Update `app.py` with your `CLIENT_ID` and `CLIENT_SECRET`.  
   - Run the Flask server:  
     ```bash
     python app.py
     ```  
   - The system will fetch songs based on detected emotions.  

## 📂 Project Structure  
```
├── main.py            # Model training script  
├── test.py            # Model testing script  
├── app.py             # Flask server + Spotify integration  
├── emotion_model.h5   # Trained model (generated after training)  
├── static/            # CSS/JS files for the web interface  
├── templates/         # HTML files  
├── haarcascades/      # Haar Cascades XML files  
└── fer2013/           # Dataset (optional if using custom data)  
```

## 🌟 Features  
- Real-time facial emotion detection (Happy, Sad, Angry, etc.).  
- Dynamic Spotify playlist recommendations based on mood.  
- Lightweight web interface for user interaction.  
