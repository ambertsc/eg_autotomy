import unittest

# import tests here

from eg_tests.test_check_connected import TestCheckConnected
from eg_tests.test_envs import TestBackAndForthEnv, TestAdaptiveWalkEnv
from eg_tests.test_mlps import TestMLPPolicy, TestHebbianMLP, TestABCHebbianMLP

from eg_tests.test_rnns import TestGatedRNNPolicy

if __name__ == "__main__":

    unittest.main(verbosity=2)
