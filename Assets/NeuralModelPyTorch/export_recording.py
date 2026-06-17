"""
Утилита для конвертации FaceRigRecording из Unity YAML в JSON для обучения.

Unity сериализует ScriptableObject в YAML. Этот скрипт парсит его
и выдаёт JSON в формате, который ожидает dataset.py.

Использование:
  python export_recording.py --input path/to/FaceRigRecording.asset --output recording.json

ВАЖНО: tag_order определяет фиксированный порядок мышц в векторе.
Этот же порядок ДОЛЖЕН использоваться при инференсе в Unity (ManifoldProjectionStep).
"""

import argparse
import json
import yaml
from pathlib import Path


# Маппинг числовых значений enum FaceMuscleAnchorTag → строковые имена.
# ОБНОВИ этот словарь если добавишь новые теги в Unity!
ANCHOR_TAG_NAMES = {
    0: "LAngleLipUpTarget",
    1: "RAngleLipUpTarget",
    2: "LAngleLipDownTarget",
    3: "RAngleLipDownTarget",
    4: "LMiddleLipUp",
    5: "RMiddleLipUp",
    6: "LMiddleLipDown",
    7: "RMiddleLipDown",
    8: "LBrowInside",
    9: "RBrowInside",
    10: "LBrowMiddleOutside",
    11: "RBrowMiddleOutside",
    12: "LEyeInner",
    13: "REyeInner",
    14: "BridgeOfTheNose",
    15: "LBrowOutside",
    16: "RBrowOutside",
    17: "LEyeOuter",
    18: "REyeOuter",
    19: "MouthCenter",
}


def parse_unity_yaml(yaml_path: str) -> dict:
    """
    Парсит Unity YAML asset.
    Unity YAML начинается с %YAML 1.1 и содержит теги !u! которые нужно обработать.
    """
    with open(yaml_path, "r") as f:
        lines = f.readlines()

    # Убираем Unity-специфичные строки
    clean_lines = []
    for line in lines:
        if line.startswith("%") or line.startswith("---"):
            continue
        # Убираем Unity теги типа !u!114
        if "!u!" in line:
            line = line.split("!u!")[0] + line.split("}")[-1] if "}" in line else line
        clean_lines.append(line)

    clean_yaml = "\n".join(clean_lines)
    data = yaml.safe_load(clean_yaml)
    return data


def extract_recording(data: dict) -> dict:
    """
    Извлекает клипы и фреймы из распарсенного YAML.
    """
    mono = data.get("MonoBehaviour", {})
    clips_raw = mono.get("clips", [])

    # Собираем tag_order из первого фрейма первого клипа
    tag_order = None

    clips = []
    for clip_raw in clips_raw:
        clip_name = clip_raw.get("name", "unnamed")
        frames_raw = clip_raw.get("frames", [])
        frames = []

        for frame_raw in frames_raw:
            timestamp = frame_raw.get("timestamp", 0.0)
            activations_raw = frame_raw.get("activations", [])

            # Определяем tag_order из первого фрейма
            if tag_order is None:
                tag_order = []
                for act in activations_raw:
                    tag_id = act.get("tag", 0)
                    tag_name = ANCHOR_TAG_NAMES.get(tag_id, f"Unknown_{tag_id}")
                    tag_order.append(tag_name)

            # Извлекаем значения активаций в порядке tag_order
            activation_values = [act.get("value", 0.0) for act in activations_raw]

            frames.append({
                "timestamp": timestamp,
                "activations": activation_values,
            })

        clips.append({
            "name": clip_name,
            "frames": frames,
        })

    return {
        "tag_order": tag_order or [],
        "clips": clips,
    }


def main():
    parser = argparse.ArgumentParser(description="Export FaceRigRecording to JSON")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to FaceRigRecording.asset (Unity YAML)")
    parser.add_argument("--output", type=str, default="recording.json",
                        help="Output JSON path (default: recording.json)")

    args = parser.parse_args()

    print(f"Parsing: {args.input}")
    data = parse_unity_yaml(args.input)
    recording = extract_recording(data)

    total_frames = sum(len(c["frames"]) for c in recording["clips"])
    print(f"Found {len(recording['clips'])} clips, {total_frames} total frames")
    print(f"Tag order ({len(recording['tag_order'])} muscles): {recording['tag_order']}")

    with open(args.output, "w") as f:
        json.dump(recording, f, indent=2)

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
