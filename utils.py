import numpy as np
import matplotlib.pyplot as plt



def set_density(field, x_pos=0, y_pos=0, direction=0, amout=10):
    """_summary_

    Args:
        field (np.ndarray): density field
        x_pos (int, optional): x_pos  . Defaults to 0.
        y_pos (int, optional): _description_. Defaults to 0.
        direction (int, optional): _description_. Defaults to 0.
        amout (int, optional): _description_. Defaults to 10.

    Returns:
        np.ndarray: augmented density field 
    """
    assert isinstance(field, np.ndarray)
    assert 0 <= x_pos and 0 <= y_pos
    assert field.shape[0] >= x_pos
    assert field.shape[1] >= y_pos
    field[x_pos, y_pos, direction] = amout
    return field



def create_density_field(x_dim=10, y_dim=15, lattic_dim=9):
    """_summary_

    Args:
        x_dim (int, optional): _description_. Defaults to 10.
        y_dim (int, optional): _description_. Defaults to 15.
        lattic_dim (int, optional): _description_. Defaults to 9.

    Returns:
        np.ndarray: array of zeros with shape (x_dim, y_dim, lattic_dim)
    """
    return np.zeros((x_dim, y_dim, lattic_dim))



def compute_density(density_field_whc):
    """_summary_

    Args:
        density_field_whc (_type_): _description_

    Returns:
        density field: same width and height as density_field_whc but only one channel
    """
    return np.sum(density_field_whc, axis=-1)


def streaming_operator(density_field_whc, c):
    """ moves the particels in the 
        density field in the direction of c

    Args:
        density_field_whc (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_dim = density_field_whc.shape[0]
    y_dim = density_field_whc.shape[1]
    lattic_dim = density_field_whc.shape[2]
    
    for i in range(lattic_dim):
        density_field_whc[:,:,i] = np.roll(density_field_whc[:,:,i], shift=c[i], axis=(0, 1))
    return density_field_whc