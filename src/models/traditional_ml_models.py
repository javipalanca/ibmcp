"""
Traditional machine learning models with advanced feature engineering
"""
import numpy as np
import pandas as pd
import librosa
import cv2
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import stats
import optuna
from pathlib import Path
import joblib

class AdvancedFeatureExtractor:
    """Advanced feature extraction for plant ultrasonic audio classification"""
    
    def __init__(self):
        self.audio_features_names = []
        self.image_features_names = []
        
    def extract_audio_features(self, audio_path: str, sr: int = 22050) -> Dict[str, float]:
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, duration=30.0)
            
            features = {}
            
            # 1. Basic statistical features
            features.update(self._extract_statistical_features(y, prefix='time'))
            
            # 2. Spectral features
            features.update(self._extract_spectral_features(y, sr))
            
            # 3. MFCCs
            features.update(self._extract_mfcc_features(y, sr))
            
            # 4. Chroma features
            features.update(self._extract_chroma_features(y, sr))
            
            # 5. Tonnetz features
            features.update(self._extract_tonnetz_features(y, sr))
            
            # 6. Tempo and rhythm
            features.update(self._extract_rhythm_features(y, sr))
            
            # 7. Zero crossing rate
            features.update(self._extract_zcr_features(y, sr))
            
            # 8. Spectral contrast
            features.update(self._extract_spectral_contrast_features(y, sr))
            
            # 9. Mel-scale features
            features.update(self._extract_mel_features(y, sr))
            
            # 10. Harmonic and percussive components
            features.update(self._extract_harmonic_percussive_features(y, sr))
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features from {audio_path}: {e}")
            return {}
    
    def _extract_statistical_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract statistical features from signal"""
        features = {}
        features[f'{prefix}_mean'] = np.mean(signal)
        features[f'{prefix}_std'] = np.std(signal)
        features[f'{prefix}_max'] = np.max(signal)
        features[f'{prefix}_min'] = np.min(signal)
        features[f'{prefix}_median'] = np.median(signal)
        features[f'{prefix}_skewness'] = stats.skew(signal)
        features[f'{prefix}_kurtosis'] = stats.kurtosis(signal)
        features[f'{prefix}_rms'] = np.sqrt(np.mean(signal**2))
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.update(self._extract_statistical_features(spectral_centroid, 'spectral_centroid'))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.update(self._extract_statistical_features(spectral_bandwidth, 'spectral_bandwidth'))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.update(self._extract_statistical_features(spectral_rolloff, 'spectral_rolloff'))
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features.update(self._extract_statistical_features(spectral_flatness, 'spectral_flatness'))
        
        return features
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, float]:
        """Extract MFCC features"""
        features = {}
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        for i in range(n_mfcc):
            features.update(self._extract_statistical_features(mfccs[i], f'mfcc_{i}'))
        
        # Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        for i in range(n_mfcc):
            features.update(self._extract_statistical_features(mfcc_delta[i], f'mfcc_delta_{i}'))
            features.update(self._extract_statistical_features(mfcc_delta2[i], f'mfcc_delta2_{i}'))
        
        return features
    
    def _extract_chroma_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract chroma features"""
        features = {}
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        for i in range(12):
            features.update(self._extract_statistical_features(chroma[i], f'chroma_{i}'))
        
        return features
    
    def _extract_tonnetz_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract tonnetz features"""
        features = {}
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        
        for i in range(6):
            features.update(self._extract_statistical_features(tonnetz[i], f'tonnetz_{i}'))
        
        return features
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract rhythm and tempo features"""
        features = {}
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Beat features
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            features.update(self._extract_statistical_features(beat_intervals, 'beat_interval'))
        
        return features
    
    def _extract_zcr_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract zero crossing rate features"""
        features = {}
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.update(self._extract_statistical_features(zcr, 'zcr'))
        return features
    
    def _extract_spectral_contrast_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral contrast features"""
        features = {}
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        for i in range(spectral_contrast.shape[0]):
            features.update(self._extract_statistical_features(spectral_contrast[i], f'spectral_contrast_{i}'))
        
        return features
    
    def _extract_mel_features(self, y: np.ndarray, sr: int, n_mels: int = 128) -> Dict[str, float]:
        """Extract mel-scale features"""
        features = {}
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Statistical features of mel spectrogram
        features.update(self._extract_statistical_features(mel_db.flatten(), 'mel_spectrogram'))
        
        return features
    
    def _extract_harmonic_percussive_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract harmonic and percussive component features"""
        features = {}
        
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Features from harmonic component
        features.update(self._extract_statistical_features(y_harmonic, 'harmonic'))
        
        # Features from percussive component
        features.update(self._extract_statistical_features(y_percussive, 'percussive'))
        
        # Harmonic-percussive ratio
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        
        if percussive_energy > 0:
            features['harmonic_percussive_ratio'] = harmonic_energy / percussive_energy
        else:
            features['harmonic_percussive_ratio'] = np.inf
        
        return features
    
    def extract_image_features(self, image_path: str) -> Dict[str, float]:
        """Extract features from spectrogram images"""
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            features = {}
            
            # 1. Basic statistical features
            features.update(self._extract_statistical_features(image.flatten(), 'pixel'))
            
            # 2. Texture features using GLCM
            features.update(self._extract_glcm_features(image))
            
            # 3. Local Binary Pattern features
            features.update(self._extract_lbp_features(image))
            
            # 4. Histogram features
            features.update(self._extract_histogram_features(image))
            
            # 5. Gradient features
            features.update(self._extract_gradient_features(image))
            
            return features
            
        except Exception as e:
            print(f"Error extracting image features from {image_path}: {e}")
            return {}
    
    def _extract_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix features"""
        features = {}
        
        # Compute GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for distance in distances:
            glcm = graycomatrix(image, distances=[distance], angles=angles, 
                             levels=256, symmetric=True, normed=True)
            
            # Extract properties
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            for prop in props:
                values = graycoprops(glcm, prop).flatten()
                features[f'glcm_{prop}_d{distance}_mean'] = np.mean(values)
                features[f'glcm_{prop}_d{distance}_std'] = np.std(values)
        
        return features
    
    def _extract_lbp_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract Local Binary Pattern features"""
        features = {}
        
        # Parameters for LBP
        radius_values = [1, 2, 3]
        n_points_values = [8, 16, 24]
        
        for radius in radius_values:
            for n_points in n_points_values:
                lbp = local_binary_pattern(image, n_points, radius, method='uniform')
                
                # Histogram of LBP
                hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, density=True)
                
                # Statistical features of histogram
                features.update(self._extract_statistical_features(
                    hist, f'lbp_r{radius}_p{n_points}'
                ))
        
        return features
    
    def _extract_histogram_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract histogram-based features"""
        features = {}
        
        # Intensity histogram
        hist, _ = np.histogram(image.ravel(), bins=256, density=True)
        features.update(self._extract_statistical_features(hist, 'hist'))
        
        # Entropy
        hist_normalized = hist / np.sum(hist)
        hist_normalized = hist_normalized[hist_normalized > 0]  # Remove zeros
        features['entropy'] = -np.sum(hist_normalized * np.log2(hist_normalized))
        
        return features
    
    def _extract_gradient_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract gradient-based features"""
        features = {}
        
        # Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        features.update(self._extract_statistical_features(grad_magnitude.flatten(), 'grad_magnitude'))
        features.update(self._extract_statistical_features(grad_direction.flatten(), 'grad_direction'))
        
        return features


class OptimizedMLModels:
    """Traditional ML models with hyperparameter optimization"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
    def create_xgboost_model(self, trial: Optional[optuna.Trial] = None) -> xgb.XGBClassifier:
        """Create XGBoost model with optional hyperparameter optimization"""
        if trial:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        return xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            **params
        )
    
    def create_lightgbm_model(self, trial: Optional[optuna.Trial] = None) -> lgb.LGBMClassifier:
        """Create LightGBM model with optional hyperparameter optimization"""
        if trial:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            }
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        return lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            **params
        )
    
    def create_svm_model(self, trial: Optional[optuna.Trial] = None) -> SVC:
        """Create SVM model with optional hyperparameter optimization"""
        if trial:
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) or trial.suggest_float('gamma_value', 1e-6, 1e-1, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            }
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
        else:
            params = {
                'C': 10,
                'gamma': 'scale',
                'kernel': 'rbf',
            }
        
        return SVC(
            random_state=self.random_state,
            probability=True,
            **params
        )
    
    def create_random_forest_model(self, trial: Optional[optuna.Trial] = None) -> RandomForestClassifier:
        """Create Random Forest model with optional hyperparameter optimization"""
        if trial:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30) if trial.suggest_categorical('max_depth_none', [True, False]) else None,
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
            }
        
        return RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            **params
        )
    
    def preprocess_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                          feature_selection_k: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features with scaling and selection"""
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(feature_selection_k, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = selector.transform(X_val_scaled)
        
        # Store for later use
        self.scalers['robust'] = scaler
        self.feature_selectors['selectk'] = selector
        
        return X_train_selected, X_val_selected
    
    def create_ensemble_model(self, models: Dict[str, any]) -> VotingClassifier:
        """Create ensemble model using voting"""
        estimators = [(name, model) for name, model in models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble


def process_dataset_features(data_loader, feature_extractor: AdvancedFeatureExtractor, 
                           use_audio: bool = True, use_images: bool = True) -> pd.DataFrame:
    """Process entire dataset to extract features"""
    
    # Load classification data
    classification_df = data_loader.load_classification_data()
    
    all_features = []
    
    for idx, row in classification_df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing sample {idx}/{len(classification_df)}")
        
        features = {'filename': row['filename'], 'label': row['label']}
        
        # Extract audio features if requested and file exists
        if use_audio:
            # You'll need to implement the logic to find corresponding audio files
            # This is a placeholder
            audio_path = None  # Find actual audio path
            if audio_path and Path(audio_path).exists():
                audio_features = feature_extractor.extract_audio_features(audio_path)
                features.update(audio_features)
        
        # Extract image features if requested and file exists
        if use_images:
            # You'll need to implement the logic to find corresponding image files
            # This is a placeholder
            image_path = None  # Find actual image path
            if image_path and Path(image_path).exists():
                image_features = feature_extractor.extract_image_features(image_path)
                features.update(image_features)
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Example usage
    feature_extractor = AdvancedFeatureExtractor()
    ml_models = OptimizedMLModels()
    
    # Create sample models
    xgb_model = ml_models.create_xgboost_model()
    lgb_model = ml_models.create_lightgbm_model()
    rf_model = ml_models.create_random_forest_model()
    
    print("Traditional ML models created successfully!")
    print(f"XGBoost parameters: {xgb_model.get_params()}")
    print(f"LightGBM parameters: {lgb_model.get_params()}")
    print(f"Random Forest parameters: {rf_model.get_params()}")
