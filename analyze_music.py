#!/usr/bin/env python3
"""Analyze music files to categorize and get statistics."""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict
from mutagen.mp3 import MP3

warnings.filterwarnings('ignore')

def get_duration(filepath):
    """Get duration of MP3 file."""
    try:
        audio = MP3(str(filepath))
        return round(audio.info.length, 2)
    except Exception:
        return 0

def categorize_by_name(name):
    """Categorize by filename keywords."""
    name_lower = name.lower()

    categories = {
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

    # Check for vocal indicators in filename
    vocal_keywords = ['vocal', 'voice', 'sing', 'choir', 'lyric', 'song', 'singer', 'acapella', 'spoken']
    has_vocal_hint = any(kw in name_lower for kw in vocal_keywords)

    for category, keywords in categories.items():
        if any(kw in name_lower for kw in keywords):
            return category, has_vocal_hint

    return 'uncategorized', has_vocal_hint

def main():
    music_dir = Path('/home/cn/Projects/CC0-1.0-Music')
    output_file = music_dir / 'analysis_results.json'

    # Find all MP3 files
    mp3_files = list(music_dir.rglob('*.mp3'))
    print(f"Found {len(mp3_files)} MP3 files")

    results = []
    stats = defaultdict(int)
    category_stats = defaultdict(int)
    source_stats = defaultdict(int)
    total_duration = 0
    vocal_hints = []

    for i, filepath in enumerate(mp3_files):
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(mp3_files)}] Processing...")

        duration = get_duration(filepath)
        category, has_vocal_hint = categorize_by_name(filepath.stem)

        # Get source folder
        rel_path = filepath.relative_to(music_dir)
        source = rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'

        result = {
            'file': filepath.name,
            'path': str(filepath),
            'source': source,
            'duration': duration,
            'category': category,
            'vocal_hint': has_vocal_hint
        }
        results.append(result)

        total_duration += duration
        category_stats[category] += 1
        source_stats[source] += 1
        stats['total'] += 1

        if has_vocal_hint:
            vocal_hints.append(filepath.name)

    # Final save
    summary = {
        'total_files': stats['total'],
        'total_duration_seconds': round(total_duration, 2),
        'total_duration_hours': round(total_duration / 3600, 2),
        'files_with_vocal_hints': len(vocal_hints),
        'categories': dict(sorted(category_stats.items(), key=lambda x: -x[1])),
        'sources': dict(sorted(source_stats.items(), key=lambda x: -x[1])),
        'vocal_hint_files': vocal_hints,
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total files: {stats['total']}")
    print(f"Total duration: {round(total_duration / 3600, 2)} hours")
    print(f"Files with vocal hints in name: {len(vocal_hints)}")

    print("\nBy source:")
    for source, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    print("\nBy category:")
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    if vocal_hints:
        print(f"\nFiles likely with vocals:")
        for f in vocal_hints[:20]:
            print(f"  - {f}")
        if len(vocal_hints) > 20:
            print(f"  ... and {len(vocal_hints) - 20} more")

    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
