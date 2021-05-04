from modules.keypoint_detector import KPDetector
import torch

class KeypointTest():
    def __init__(self):
        self.BLOCK_EXPANSION = 8
        self.NUM_KP = 10
        self.NUM_CHANNELS = 3
        self.MAX_FEATURES = 8
        self.NUM_BLOCKS = 1
        self.TEMP = 0.1
        self.kp = KPDetector(self.BLOCK_EXPANSION, self.NUM_KP, self.NUM_CHANNELS,\
                             self.MAX_FEATURES, self.NUM_BLOCKS, self.TEMP, True)

    def test_output_shape(self):
        # output test 1
        _in = torch.zeros(self.NUM_KP, self.NUM_CHANNELS, self.MAX_FEATURES, self.MAX_FEATURES) # input 
        _out = self.kp.forward(_in) # run

        shape = (self.NUM_KP, self.NUM_KP, 2, 2) # expected output shape
        assert _out['jacobian'].size() == shape, "output shape is incorrect"

        # output test 2
        _in = torch.zeros(20, self.NUM_CHANNELS, self.MAX_FEATURES, self.MAX_FEATURES) # input 
        _out = self.kp.forward(_in) # run

        shape = (20, self.NUM_KP, 2, 2) # expected output shape
        assert _out['jacobian'].size() == shape, "output shape is incorrect"

