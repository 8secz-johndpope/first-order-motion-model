from modules.generator import OcclusionAwareGenerator
import torch

class GeneratorTest():
    def __init__(self):
        self.NUM_CHANNELS = 3
        self.NUM_KP = 2
        self.BLOCK_EXPANSION = 8
        self.MAX_FEATURES = 16
        self.NUM_DOWN_BLOCKS = 1
        self.NUM_BOTTLENECK_BLOCKS = 1
        self.generator = OcclusionAwareGenerator(self.NUM_CHANNELS, self.NUM_KP, self.BLOCK_EXPANSION, \
            self.MAX_FEATURES, self.NUM_DOWN_BLOCKS, self.NUM_BOTTLENECK_BLOCKS)
        self.output = None
    
    def test_training(self):
        # example input - weights are initialised to 0
        _in = torch.zeros((256,3,256,256))
        # run the generator training step once
        self.output = self.generator.forward(_in, _in, _in)['prediction'].detach()

        # ensure that weights have been initialised after training for 1 iteration and something has changed
        assert not torch.equal(self.output, _in), "weights did not change after training"
    
    def test_output_range(self): 
        # output of each weight should be in the range(0-1) after applying sigmoid
        for i,w in enumerate(self.output):
            for j,x in enumerate(w):
                for k,y in enumerate(x):
                    for z in y:
                        z = float(z.item())
                        assert 0 <= z <= 1, str(z) + " not in range(0-1)"

        
        