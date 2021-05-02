from modules.dense_motion import DenseMotionNetwork
from modules.keypoint_detector import KPDetector
import torch

class DenseMotionTest():
    def __init__(self):
        self.BLOCK_EXPANSION = 8
        self.NUM_BLOCKS = 1
        self.MAX_FEATURES = 8
        self.NUM_KP = 10
        self.NUM_CHANNELS = 3
        self.dm = DenseMotionNetwork(self.BLOCK_EXPANSION, self.NUM_BLOCKS, self.MAX_FEATURES, \
                                     self.NUM_KP, self.NUM_CHANNELS)

        self.TEMP = 0.1
        self.kp = KPDetector(self.BLOCK_EXPANSION, self.NUM_KP, self.NUM_CHANNELS,\
                             self.MAX_FEATURES, self.NUM_BLOCKS, self.TEMP, True)

    def test_create_heatmap_representations(self):
        _in = torch.zeros(self.NUM_KP, self.NUM_CHANNELS, self.MAX_FEATURES, self.MAX_FEATURES) # input 
        _out = self.kp.forward(_in) # get keypoints

        # expected output shape
        shape = (self.NUM_KP, 11, 1, self.MAX_FEATURES, self.MAX_FEATURES)
        heatmap_rep = self.dm.create_heatmap_representations(_in, _out, _out)
        assert heatmap_rep.detach().size() == shape, "wrong output size"
        
        _in = torch.zeros(30, self.NUM_CHANNELS, 20, 20) # input 
        _out = self.kp.forward(_in) # get keypoints

        # expected output shape
        shape = (30, 11, 1, 20, 20)
        heatmap_rep = self.dm.create_heatmap_representations(_in, _out, _out)
        assert heatmap_rep.detach().size() == shape, "wrong output size"

    def test_create_sparse_motions(self):
        _in = torch.zeros(self.NUM_KP, self.NUM_CHANNELS, self.MAX_FEATURES, self.MAX_FEATURES) # input 
        _out = self.kp.forward(_in) # get keypoints

        # expected output shape
        shape = (self.NUM_KP, 11, self.MAX_FEATURES, self.MAX_FEATURES, 2)
        sparse_motions = self.dm.create_sparse_motions(_in, _out, _out)
        assert sparse_motions.detach().size() == shape, "wrong output size"
        
        _in = torch.zeros(30, self.NUM_CHANNELS, 20, 20) # input 
        _out = self.kp.forward(_in) # get keypoints

        # expected output shape
        shape = (30, 11, 20, 20, 2)
        sparse_motions = self.dm.create_sparse_motions(_in, _out, _out)
        assert sparse_motions.detach().size() == shape, "wrong output size"