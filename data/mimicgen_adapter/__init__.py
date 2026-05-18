"""MimicGen-style data augmentation adapter for SO101 PickPlace.

Pipeline:
    real_demos (LeRobot) → segmenter → segments
    segments + new_scenes → replayer (in sim) → augmented LeRobot dataset
"""
