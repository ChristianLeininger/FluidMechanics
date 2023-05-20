import unittest
import numpy as np
from utils import create_density_field, compute_density, set_density, streaming_operator

class Testing(unittest.TestCase):
    def test_create_density_field(self):
        a = create_density_field()
        self.assertEqual(a.shape, (10, 15, 9))

    def test_streaming_operator(self):
        c = np.array([[0, 0, -1,  0, 1, -1, -1,  1, 1],
                      [0, 1,  0, -1, 0, 1,  -1,  1, -1]]).T
        x_dim = 10 
        y_dim = 15
        lattic_dim = 9
        stay_d = 0
        right_d = 1
        up_d = 2
        left_d = 3 
        down_d = 4
        rightUp_d = 5
        leftUp_d = 6 
        rightDown_d = 7
        leftDown_d = 8
        density_field_whc = create_density_field(x_dim=x_dim, y_dim=y_dim, lattic_dim=lattic_dim)
        density_field_whc_right_d = set_density(density_field_whc, x_pos=0, y_pos=0, direction=right_d, amout=10)
        self.assertEqual(density_field_whc_right_d[0,0, right_d], 10)
        density_field_whc_right_d = streaming_operator(density_field_whc_right_d, c)
        compute_density_right_d = compute_density(density_field_whc_right_d)
        self.assertEqual(compute_density_right_d[0,1], 10)
        
        density_field_whc_up_d = set_density(density_field_whc, x_pos=0, y_pos=0, direction=up_d, amout=10)
        self.assertEqual(density_field_whc_up_d[0,0, up_d], 10)
        density_field_whc_up_d = streaming_operator(density_field_whc_up_d, c)
        compute_density_up_d = compute_density(density_field_whc_up_d)
        self.assertEqual(compute_density_up_d[9,0], 10)
        
        density_field_whc_left_d = set_density(density_field_whc, x_pos=0, y_pos=0, direction=left_d, amout=10)
        self.assertEqual(density_field_whc_up_d[0,0, left_d], 10)
        density_field_whc_left_d = streaming_operator(density_field_whc_left_d, c)
        compute_density_left_d = compute_density(density_field_whc_left_d)
        self.assertEqual(compute_density_left_d[0,14], 10)
        
        
        density_field_whc_down_d = set_density(density_field_whc, x_pos=0, y_pos=0, direction=down_d, amout=10)
        self.assertEqual(density_field_whc_up_d[0,0, down_d], 10)
        density_field_whc_down_d = streaming_operator(density_field_whc_down_d, c)
        compute_density_down_d = compute_density(density_field_whc_down_d)
        self.assertEqual(compute_density_down_d[1,0], 10)
        
        
        
        
        
        
        
if __name__ == '__main__':
    unittest.main()