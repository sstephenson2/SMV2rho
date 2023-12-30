from dataclasses import dataclass

@dataclass
class TParameters:
    """Additional parameters."""
    m: float = None  # This parameter depends on the choice of Vp or Vs
    alpha0: float = -2.3e4
    alpha1: float = -1e-5
    K: float = 90e9

@dataclass
class VpConstants:
    """Constants for Vp."""
    v0: float = -9.3521e-01
    b: float = 1.69478e-03
    d0: float = 2.55911
    dp: float = -4.76050e-04
    c: float = 1.674065
    k: float = 1.953466e-02
    m: float = -4e-4  # Default m value for Vp

@dataclass
class VsConstants:
    """Constants for Vs."""
    v0: float = -6.0777e-01
    b: float = 1.0345e-03
    d0: float = 1.4808
    dp: float = -2.9773e-04
    c: float = 7.3740e-01
    k: float = 2.0041e-02
    m: float = -2.3e-4  # Default m value for Vs

@dataclass
class Constants:
    """
    Master class to control instantiation of VpConstants and VsConstants.

    Usage:
    Initialize Constants class to create instances for VpConstants 
    and VsConstants.

    Example:
    ```
    constants = Constants()
    constants.get_constants('Vp', v0=-0.9, b=0.001, d0=2.5, 
                            dp=-0.0005, c=1.7, k=0.019, m=1.5)
    constants.get_constants('Vs', v0=-0.9, b=0.001, d0=2.5, 
                            dp=-0.0005, c=1.7, k=0.019, m=2.5)
    vp_instance = constants.vp_constants
    vs_instance = constants.vs_constants
    ```

    """

    vp_constants: VpConstants = None
    vs_constants: VsConstants = None

    def get_constants(self, data_type: str, **kwargs):
        """
        Get constants instance based on data_type ('Vp' or 'Vs') and 
        assign them to class attributes.
        """
        m = kwargs.get('m')
        if data_type == 'Vp':
            if m is None:
                kwargs['m'] = -4e-4  # Default m value for Vp if not provided
            self.vp_constants = VpConstants(**kwargs)
        elif data_type == 'Vs':
            if m is None:
                kwargs['m'] = -2.3e-4  # Default m value for Vs if not provided
            self.vs_constants = VsConstants(**kwargs)
        else:
            raise ValueError("Invalid data_type. Must be 'Vp' or 'Vs'.")