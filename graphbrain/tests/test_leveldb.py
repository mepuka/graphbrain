import unittest

# Check if LevelDB backend is available
try:
    import graphbrain.memory.leveldb
    LEVELDB_AVAILABLE = True
except ImportError:
    LEVELDB_AVAILABLE = False

from graphbrain.tests.hypergraph import Hypergraph


@unittest.skipUnless(LEVELDB_AVAILABLE, "LevelDB backend not available (plyvel import failed)")
class TestLevelDB(Hypergraph, unittest.TestCase):
    def setUp(self):
        self.hg_str = 'test.hg'
        super().setUp()
