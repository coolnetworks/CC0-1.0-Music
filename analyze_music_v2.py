#!/usr/bin/env python3
"""
Advanced music analysis with audio feature extraction.
- Extracts librosa features (tempo, spectral, MFCC)
- Detects vocals using spectral analysis
- Uses ML to categorize uncategorized files based on labeled training data
"""

import os
import sys
import json
import warnings
import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

warnings.filterwarnings('ignore')

# These imports are heavy, only import when needed
def get_librosa():
    import librosa
    return librosa

def get_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    return RandomForestClassifier, StandardScaler, cross_val_score


# Category definitions (same as v1 for consistency)
CATEGORIES = {
    'classical': ['bach', 'mozart', 'beethoven', 'chopin', 'joplin', 'waltz', 'sonata', 'symphony', 'classical', 'orchestra', 'concerto', 'minuet', 'fugue', 'prelude', 'nocturne', 'etude'],
    'piano': ['piano'],
    'jazz': ['jazz', 'swing', 'blues', 'bebop', 'saxophone', 'trumpet', 'bossa', 'ragtime'],
    'electronic': ['electronic', 'synth', 'techno', 'edm', 'loop', 'digital', 'dubstep', 'trance', 'house'],
    'rock': ['rock', 'guitar', 'metal', 'punk', 'grunge'],
    'action': ['action', 'battle', 'fight', 'combat', 'war', 'intense', 'powerful', 'energy'],
    'epic': ['epic', 'heroic', 'triumph', 'victory', 'grand'],
    'horror': ['horror', 'scary', 'creepy', 'dark', 'evil', 'haunted', 'spooky', 'nightmare', 'terror'],
    'ambient': ['ambient', 'atmospheric', 'calm', 'peaceful', 'relaxing', 'meditation', 'serene', 'tranquil', 'chill'],
    'nature': ['nature', 'forest', 'ocean', 'rain', 'wind', 'water', 'birds', 'thunder'],
    'cinematic': ['cinematic', 'movie', 'film', 'trailer', 'dramatic', 'score', 'soundtrack'],
    'game': ['game', 'arcade', 'retro', '8-bit', 'chiptune', 'pixel', 'level', 'boss'],
    'holiday': ['christmas', 'holiday', 'festive', 'winter', 'jingle', 'carol'],
    'world': ['celtic', 'asian', 'african', 'latin', 'folk', 'ethnic', 'tribal', 'irish', 'spanish', 'indian', 'arabic', 'chinese', 'japanese'],
    'comedy': ['funny', 'comedy', 'silly', 'cartoon', 'quirky', 'wacky', 'goofy'],
    'romantic': ['love', 'romantic', 'romance', 'wedding', 'heart'],
    'sad': ['sad', 'melancholy', 'sorrow', 'grief', 'lonely', 'tears'],
    'happy': ['happy', 'joy', 'cheerful', 'upbeat', 'fun', 'playful', 'uplifting'],
    'mystery': ['mystery', 'suspense', 'detective', 'spy', 'intrigue', 'secret'],
    'scifi': ['sci-fi', 'scifi', 'space', 'alien', 'future', 'robot', 'cyber'],
    'western': ['western', 'cowboy', 'saloon', 'desert', 'wild west'],
}

VOCAL_KEYWORDS = ['vocal', 'voice', 'sing', 'choir', 'lyric', 'song', 'singer', 'acapella', 'spoken']


def categorize_by_name(name):
    """Categorize by filename keywords."""
    name_lower = name.lower()
    has_vocal_hint = any(kw in name_lower for kw in VOCAL_KEYWORDS)

    for category, keywords in CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return category, has_vocal_hint

    return 'uncategorized', has_vocal_hint


def extract_features(filepath, sr=22050, duration=30):
    """
    Extract audio features from a file.
    Only analyzes first `duration` seconds for speed.
    """
    librosa = get_librosa()

    try:
        # Load audio (first 30 seconds for speed)
        y, sr = librosa.load(filepath, sr=sr, duration=duration, mono=True)

        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            return None

        features = {}

        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        # Zero crossing rate (percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # Spectral flatness (noise vs tonal)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(flatness))

        # MFCC (timbre fingerprint) - first 13 coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(contrast))

        return features

    except Exception as e:
        return None


def detect_vocals(filepath, sr=22050, duration=30):
    """
    Detect vocals using spectral analysis heuristics.
    Returns (has_vocals: bool, confidence: float)

    Vocals typically have:
    - Strong energy in 300Hz-3400Hz range (voice fundamental + harmonics)
    - Specific harmonic patterns
    - Characteristic spectral envelope
    """
    librosa = get_librosa()

    try:
        y, sr = librosa.load(filepath, sr=sr, duration=duration, mono=True)

        if len(y) < sr * 0.5:
            return False, 0.0

        # Get spectrogram
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        # Voice frequency range (300Hz - 3400Hz)
        voice_low = 300
        voice_high = 3400
        voice_mask = (freqs >= voice_low) & (freqs <= voice_high)

        # Extended range for harmonics (up to 8kHz)
        harmonic_high = 8000
        harmonic_mask = (freqs >= voice_low) & (freqs <= harmonic_high)

        # Calculate energy ratios
        total_energy = np.sum(S ** 2)
        voice_energy = np.sum(S[voice_mask] ** 2)
        harmonic_energy = np.sum(S[harmonic_mask] ** 2)

        if total_energy == 0:
            return False, 0.0

        voice_ratio = voice_energy / total_energy
        harmonic_ratio = harmonic_energy / total_energy

        # Spectral flatness in voice range (vocals are less flat/more tonal)
        voice_spec = S[voice_mask]
        if voice_spec.size > 0:
            mean_voice = np.mean(voice_spec, axis=0)
            mean_voice = mean_voice[mean_voice > 0]
            if len(mean_voice) > 0:
                geo_mean = np.exp(np.mean(np.log(mean_voice + 1e-10)))
                arith_mean = np.mean(mean_voice)
                voice_flatness = geo_mean / (arith_mean + 1e-10)
            else:
                voice_flatness = 1.0
        else:
            voice_flatness = 1.0

        # Harmonic-to-noise ratio approximation
        # Vocals have distinct harmonic structure
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_energy_hpss = np.sum(harmonic ** 2)
        total_energy_hpss = np.sum(y ** 2)

        if total_energy_hpss > 0:
            harmonic_ratio_hpss = harmonic_energy_hpss / total_energy_hpss
        else:
            harmonic_ratio_hpss = 0

        # Scoring
        score = 0.0

        # Voice range energy (0-0.4)
        if voice_ratio > 0.3:
            score += 0.3
        elif voice_ratio > 0.2:
            score += 0.2
        elif voice_ratio > 0.1:
            score += 0.1

        # Harmonic content (0-0.3)
        if harmonic_ratio_hpss > 0.7:
            score += 0.3
        elif harmonic_ratio_hpss > 0.5:
            score += 0.2
        elif harmonic_ratio_hpss > 0.3:
            score += 0.1

        # Low flatness in voice range (tonal, not noise) (0-0.3)
        if voice_flatness < 0.3:
            score += 0.3
        elif voice_flatness < 0.5:
            score += 0.2
        elif voice_flatness < 0.7:
            score += 0.1

        # Threshold - use 0.7 for higher precision
        # 0.5 has too many false positives from harmonic instruments
        has_vocals = score >= 0.7
        confidence = min(score, 1.0)

        return has_vocals, round(confidence, 3)

    except Exception as e:
        return False, 0.0


def process_file(args):
    """Process a single file (for parallel processing)."""
    filepath, music_dir = args

    try:
        from mutagen.mp3 import MP3

        # Basic info
        try:
            audio = MP3(str(filepath))
            duration = round(audio.info.length, 2)
        except:
            duration = 0

        # Category from filename
        category, vocal_hint = categorize_by_name(filepath.stem)

        # Source folder
        rel_path = filepath.relative_to(music_dir)
        source = rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'

        # Extract audio features
        features = extract_features(str(filepath))

        # Detect vocals
        has_vocals, vocal_confidence = detect_vocals(str(filepath))

        result = {
            'file': filepath.name,
            'path': str(filepath),
            'source': source,
            'duration': duration,
            'filename_category': category,
            'vocal_hint': vocal_hint,
            'features': features,
            'vocal_detected': has_vocals,
            'vocal_confidence': vocal_confidence,
        }

        return result

    except Exception as e:
        return {
            'file': filepath.name,
            'path': str(filepath),
            'error': str(e)
        }


def train_classifier(results):
    """Train a classifier on labeled files to predict categories."""
    RandomForestClassifier, StandardScaler, cross_val_score = get_sklearn()

    # Get files with known categories and valid features
    labeled = []
    for r in results:
        if r.get('filename_category') != 'uncategorized' and r.get('features'):
            labeled.append(r)

    if len(labeled) < 50:
        print(f"  Only {len(labeled)} labeled files with features, need at least 50")
        return None, None, None

    # Prepare feature matrix
    feature_names = sorted(labeled[0]['features'].keys())
    X = []
    y = []

    for r in labeled:
        feat_vec = [r['features'].get(fn, 0) for fn in feature_names]
        X.append(feat_vec)
        y.append(r['filename_category'])

    X = np.array(X)
    y = np.array(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    # Cross-validation
    scores = cross_val_score(clf, X_scaled, y, cv=5)
    print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    # Train on full data
    clf.fit(X_scaled, y)

    return clf, scaler, feature_names


def predict_categories(results, clf, scaler, feature_names, min_confidence=0.3):
    """Predict categories for uncategorized files."""
    predictions = []

    for r in results:
        if r.get('filename_category') == 'uncategorized' and r.get('features'):
            feat_vec = [r['features'].get(fn, 0) for fn in feature_names]
            X = np.array([feat_vec])
            X_scaled = scaler.transform(X)

            # Get prediction and confidence
            pred = clf.predict(X_scaled)[0]
            proba = clf.predict_proba(X_scaled)[0]
            confidence = float(max(proba))

            if confidence >= min_confidence:
                predictions.append({
                    'file': r['file'],
                    'path': r['path'],
                    'predicted_category': pred,
                    'prediction_confidence': round(confidence, 3)
                })

    return predictions


def save_progress(results, output_file):
    """Save intermediate results."""
    with open(output_file, 'w') as f:
        json.dump({'partial': True, 'count': len(results), 'results': results}, f)


def main():
    music_dir = Path('/home/cn/Projects/CC0-1.0-Music')
    output_file = music_dir / 'analysis_results_v2.json'
    progress_file = music_dir / 'analysis_progress.json'
    model_file = music_dir / 'category_model.pkl'

    # Find all MP3 files
    mp3_files = list(music_dir.rglob('*.mp3'))
    print(f"Found {len(mp3_files)} MP3 files")

    # Check for existing progress
    processed = {}
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                data = json.load(f)
                if data.get('partial'):
                    processed = {r['path']: r for r in data.get('results', [])}
                    print(f"Resuming from {len(processed)} previously processed files")
        except:
            pass

    # Process files
    results = list(processed.values())
    to_process = [f for f in mp3_files if str(f) not in processed]

    print(f"Processing {len(to_process)} files...")

    # Use multiprocessing for speed
    batch_size = 100
    args_list = [(f, music_dir) for f in to_process]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_file, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)

            # Progress update and save
            if (i + 1) % batch_size == 0:
                print(f"  [{len(results)}/{len(mp3_files)}] Processed, saving progress...")
                save_progress(results, progress_file)

    print(f"Feature extraction complete: {len(results)} files")

    # Train classifier
    print("\nTraining category classifier...")
    clf, scaler, feature_names = train_classifier(results)

    predictions = []
    if clf is not None:
        print("Predicting categories for uncategorized files...")
        predictions = predict_categories(results, clf, scaler, feature_names)
        print(f"  Made {len(predictions)} predictions with confidence >= 0.3")

        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump({'clf': clf, 'scaler': scaler, 'feature_names': feature_names}, f)

    # Build summary
    print("\nBuilding summary...")

    # Apply predictions to results
    pred_map = {p['path']: p for p in predictions}

    category_stats = defaultdict(int)
    predicted_category_stats = defaultdict(int)
    source_stats = defaultdict(int)
    vocal_stats = {'detected': 0, 'not_detected': 0, 'filename_hint': 0}
    total_duration = 0
    errors = 0

    final_results = []
    for r in results:
        if 'error' in r:
            errors += 1
            continue

        # Combine prediction with result
        pred = pred_map.get(r['path'])
        if pred:
            r['predicted_category'] = pred['predicted_category']
            r['prediction_confidence'] = pred['prediction_confidence']
            predicted_category_stats[pred['predicted_category']] += 1

        final_results.append(r)

        category_stats[r['filename_category']] += 1
        source_stats[r['source']] += 1
        total_duration += r.get('duration', 0)

        if r.get('vocal_detected'):
            vocal_stats['detected'] += 1
        else:
            vocal_stats['not_detected'] += 1

        if r.get('vocal_hint'):
            vocal_stats['filename_hint'] += 1

    # Find files with vocals
    vocal_files = [r for r in final_results if r.get('vocal_detected')]
    vocal_files_sorted = sorted(vocal_files, key=lambda x: -x.get('vocal_confidence', 0))

    summary = {
        'total_files': len(final_results),
        'total_duration_seconds': round(total_duration, 2),
        'total_duration_hours': round(total_duration / 3600, 2),
        'processing_errors': errors,
        'filename_categories': dict(sorted(category_stats.items(), key=lambda x: -x[1])),
        'predicted_categories': dict(sorted(predicted_category_stats.items(), key=lambda x: -x[1])),
        'sources': dict(sorted(source_stats.items(), key=lambda x: -x[1])),
        'vocal_detection': {
            'detected_vocals': vocal_stats['detected'],
            'no_vocals_detected': vocal_stats['not_detected'],
            'filename_hints': vocal_stats['filename_hint'],
        },
        'top_vocal_files': [
            {'file': r['file'], 'confidence': r['vocal_confidence']}
            for r in vocal_files_sorted[:50]
        ],
        'results': final_results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total files analyzed: {len(final_results)}")
    print(f"Total duration: {round(total_duration / 3600, 2)} hours")
    print(f"Processing errors: {errors}")

    print("\nFilename-based categories:")
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1])[:10]:
        pct = count / len(final_results) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    if predicted_category_stats:
        print(f"\nML Predictions (for {sum(predicted_category_stats.values())} uncategorized files):")
        for cat, count in sorted(predicted_category_stats.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {count}")

    print(f"\nVocal Detection:")
    print(f"  Detected vocals: {vocal_stats['detected']}")
    print(f"  No vocals: {vocal_stats['not_detected']}")
    print(f"  Filename hints: {vocal_stats['filename_hint']}")

    if vocal_files_sorted:
        print(f"\nTop files with detected vocals:")
        for r in vocal_files_sorted[:10]:
            print(f"  [{r['vocal_confidence']:.2f}] {r['file']}")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
