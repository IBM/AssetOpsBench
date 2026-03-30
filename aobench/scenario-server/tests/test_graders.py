import unittest

from scenario_server.grading.graders import cosine_similarity 

class TestGraders(unittest.TestCase):

    def test_cosine_similarity(self):

        x = "apple"
        y = "pear"

        similarity_score = cosine_similarity(x,y)

        self.assertIsInstance(similarity_score,float)
        self.assertTrue(similarity_score < 1. )
        self.assertTrue(similarity_score > 0. )

        print(f"{similarity_score=}")
    
    def test_cosine_similarity_serial(self):
        """Multiple single calls"""

        x = "apple"
        y = "pear"
        z = "zebra"

        ss_xy = cosine_similarity(x,y)
        ss_yz = cosine_similarity(y,z)
        ss_xz = cosine_similarity(x,z)

        self.assertTrue(ss_xy > 0 )
        self.assertTrue(ss_yz > 0 )
        self.assertTrue(ss_xz > 0 )

        print(f"{ss_xy=}")
        print(f"{ss_yz=}")
        print(f"{ss_xz=}")
    
    def test_cosine_similarity_vectorize(self):
        """One vectorized call"""

        x = ['apple', 'pear', 'apple']
        y = ['pear', 'zebra', 'zebra']

        simscores = cosine_similarity(x,y)

        self.assertEqual( len(simscores), 3)

        [print(f"{ss=}") for ss in simscores]


if __name__ == '__main__':
    unittest.main()



