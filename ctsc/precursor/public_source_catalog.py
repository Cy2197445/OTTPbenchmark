'Official public source catalog and source-group mapping for DSRL assets.'

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


UNIFIED38_DOWNLOAD_MANIFEST_ID = "Unified38_download_manifest_v1"
UNIFIED38_SOURCE_HOMEPAGE = "https://huggingface.co/datasets/YYY-45/DSRL"
UNIFIED38_LICENSE_NOTE = "See the upstream dataset card and original benchmark licenses."
SOURCE_GROUP_TO_PLATFORM = {
    "bullet_safety_gym_like": "bullet_safety_gym_suite",
    "safety_gymnasium_like": "safety_gymnasium_suite",
    "velocity_like": "safety_gymnasium_velocity_suite",
    "metadrive_like": "safe_metadrive_suite",
}


DSRL_TASK_CATALOG: dict[str, dict[str, str]] = {
    # Bullet-Safety-Gym source group
    "AntCircle": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntCircle-v0-200-5728.hdf5",
    },
    "AntRun": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntRun-v0-150-1816.hdf5",
    },
    "BallCircle": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyBallCircle-v0-80-886.hdf5",
    },
    "BallRun": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyBallRun-v0-80-940.hdf5",
    },
    "CarCircle": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle-v0-100-1450.hdf5",
    },
    "CarRun": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarRun-v0-40-651.hdf5",
    },
    "DroneCircle": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyDroneCircle-v0-100-1923.hdf5",
    },
    "DroneRun": {
        "source_group": "bullet_safety_gym_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyDroneRun-v0-140-1990.hdf5",
    },
    # Safety-Gymnasium source group
    "CarButton1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarButton1Gymnasium-v0-250-2656.hdf5",
    },
    "CarButton2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarButton2Gymnasium-v0-300-3755.hdf5",
    },
    "CarCircle1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle1Gymnasium-v0-250-1271.hdf5",
    },
    "CarCircle2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle2Gymnasium-v0-400-940.hdf5",
    },
    "CarGoal1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarGoal1Gymnasium-v0-120-1671.hdf5",
    },
    "CarGoal2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarGoal2Gymnasium-v0-200-4105.hdf5",
    },
    "CarPush1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarPush1Gymnasium-v0-200-2871.hdf5",
    },
    "CarPush2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarPush2Gymnasium-v0-250-4407.hdf5",
    },
    "PointButton1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointButton1Gymnasium-v0-200-2268.hdf5",
    },
    "PointButton2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointButton2Gymnasium-v0-250-3288.hdf5",
    },
    "PointCircle1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointCircle1Gymnasium-v0-200-1098.hdf5",
    },
    "PointCircle2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointCircle2Gymnasium-v0-300-895.hdf5",
    },
    "PointGoal1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5",
    },
    "PointGoal2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal2Gymnasium-v0-200-3442.hdf5",
    },
    "PointPush1": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointPush1Gymnasium-v0-150-2379.hdf5",
    },
    "PointPush2": {
        "source_group": "safety_gymnasium_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointPush2Gymnasium-v0-200-3242.hdf5",
    },
    # Velocity source group
    "AntVelocity": {
        "source_group": "velocity_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntVelocityGymnasium-v1-250-2249.hdf5",
    },
    "HalfCheetahVelocity": {
        "source_group": "velocity_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyHalfCheetahVelocityGymnasium-v1-250-2495.hdf5",
    },
    "HopperVelocity": {
        "source_group": "velocity_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyHopperVelocityGymnasium-v1-250-2240.hdf5",
    },
    "SwimmerVelocity": {
        "source_group": "velocity_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetySwimmerVelocityGymnasium-v1-200-1686.hdf5",
    },
    "Walker2dVelocity": {
        "source_group": "velocity_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyWalker2dVelocityGymnasium-v1-300-2729.hdf5",
    },
    # MetaDrive source group
    "easysparse": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easysparse-v0-85-1000.hdf5",
    },
    "easymean": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easymean-v0-85-1000.hdf5",
    },
    "easydense": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easydense-v0-85-1000.hdf5",
    },
    "mediumsparse": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediumsparse-v0-50-1000.hdf5",
    },
    "mediummean": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediummean-v0-50-1000.hdf5",
    },
    "mediumdense": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediumdense-v0-50-1000.hdf5",
    },
    "hardsparse": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-hardsparse-v0-85-1000.hdf5",
    },
    "hardmean": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-hardmean-v0-85-1000.hdf5",
    },
    "harddense": {
        "source_group": "metadrive_like",
        "source": "dsrl_official_hf",
        "url": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-harddense-v0-85-1000.hdf5",
    },
}


ARCHIVE_REFERENCE_EXPANSION_TASKS: tuple[str, ...] = (
    # Bullet-Safety-Gym additions
    "AntRun",
    "CarCircle",
    "DroneCircle",
    "DroneRun",
    # Velocity additions
    "HopperVelocity",
    "SwimmerVelocity",
    "HalfCheetahVelocity",
    "Walker2dVelocity",
    # MetaDrive additions
    "easysparse",
    "easymean",
    "easydense",
)


def clean_public_task_name(name_or_path: str | Path) -> str:
    stem = Path(str(name_or_path)).stem
    if stem.startswith("Safety"):
        stem = stem[len("Safety") :]
    if stem.startswith("SafeMetaDrive-"):
        stem = stem[len("SafeMetaDrive-") :]
    stem = re.sub(r"Gymnasium-v\d+-\d+-\d+$", "", stem)
    stem = re.sub(r"-v\d+-\d+-\d+$", "", stem)
    stem = re.sub(r"-\d+-\d+$", "", stem)
    return stem or Path(str(name_or_path)).stem


def infer_public_task_source_group(task_name: str, dataset_path: str | None = None) -> str:
    cleaned = clean_public_task_name(task_name)
    if cleaned in DSRL_TASK_CATALOG:
        return str(DSRL_TASK_CATALOG[cleaned]["source_group"])
    joined = f"{cleaned} {dataset_path or ''}".lower()
    if "metadrive" in joined:
        return "metadrive_like"
    if "velocity" in cleaned.lower():
        return "velocity_like"
    bullet_tokens = ("antcircle", "antrun", "ballcircle", "ballrun", "carcircle", "carrun", "dronecircle", "dronerun")
    if cleaned.lower() in bullet_tokens:
        return "bullet_safety_gym_like"
    if any(token in cleaned for token in ("Point", "Goal", "Button", "Push", "Circle1", "Circle2")) or "gymnasium" in joined:
        return "safety_gymnasium_like"
    return "unknown_source_group"


def dsrl_catalog_records(task_names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    selected = list(task_names) if task_names is not None else sorted(DSRL_TASK_CATALOG.keys())
    records: list[dict[str, Any]] = []
    for task_name in selected:
        if task_name not in DSRL_TASK_CATALOG:
            raise KeyError(f"Unknown DSRL task name: {task_name}")
        payload = dict(DSRL_TASK_CATALOG[task_name])
        payload["task_name"] = task_name
        payload["source_homepage"] = UNIFIED38_SOURCE_HOMEPAGE
        payload["license"] = UNIFIED38_LICENSE_NOTE
        payload["download_manifest_id"] = UNIFIED38_DOWNLOAD_MANIFEST_ID
        payload["source_platform"] = SOURCE_GROUP_TO_PLATFORM.get(
            str(payload["source_group"]),
            "unknown_source_platform",
        )
        records.append(payload)
    return records
