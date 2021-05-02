from unit_tests import generator_test, keypoint_test, dense_motion_test
# to run tests:
# pytest unit_test.py
"""
print("Tests for KeypointDetector")
kp = keypoint_test.KeypointTest()

def test_keypoints():
    kp.test()
"""

print("Tests for DenseMotionNetwork")
dm = dense_motion_test.DenseMotionTest()

def test_create_heatmap_representations():
    dm.test_create_heatmap_representations()

def test_create_sparse_motions():
    dm.test_create_sparse_motions()

test_create_sparse_motions()

"""
print("Tests for OcclusionAwareGenerator")
generator = generator_test.GeneratorTest()

def test_generator_training():
    generator.test_training()

def test_output_range():
    generator.test_output_range()
"""