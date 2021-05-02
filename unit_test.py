from unit_tests import generator_test, keypoint_test

print("Tests for KeypointDetector")
kp = keypoint_test.KeypointTest()

def test_keypoint():
    kp.test()

print("Tests for OcclusionAwareGenerator")
generator = generator_test.GeneratorTest()

def test_generator_training():
    generator.test_training()

def test_output_range():
    generator.test_output_range()

# to run tests:
# pytest unit_test.py