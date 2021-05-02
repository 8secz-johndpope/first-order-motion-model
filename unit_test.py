from unit_tests import generator_test

print("Tests for OcclusionAwareGenerator")
generator = generator_test.GeneratorTest()

def test_generator_training():
    generator.test_training()

def test_output_range():
    generator.test_output_range()
