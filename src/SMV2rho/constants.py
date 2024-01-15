from dataclasses import dataclass

@dataclass
class MaterialConstants:
    """
    This class represents material properties for geophysical calculations.

    Attributes:
    ----------
    alpha0 (float): Thermal expansivity coefficient at reference temperature.
                    Default is 1e-5.
    alpha1 (float): Temperature derivative of thermal expansion coefficient.
                    Default is 2.9e-8.
    K (float): Bulk modulus, invariant with pressure and temperature.
               Default is 90e9.
    alpha0_unc (float): Uncertainty in alpha0. Default is 0.5e-5.
    alpha1_unc (float): Uncertainty in alpha1. Default is 0.5e-8.
    K_unc (float): Uncertainty in K. Default is 20e9.
    """

    # parameter values
    alpha0: float = 1e-5
    alpha1: float = 2.9e-8
    K: float = 90e9

    # parameter uncertainties
    alpha0_unc: float = 0.5e-5
    alpha1_unc: float = 0.5e-8
    K_unc: float = 20e9

@dataclass
class VpConstants:
    """
    This class represents constants for compressional wave velocity (Vp).

    Attributes:
    ----------
    v0 (float): Initial velocity at reference conditions. Default 
                is -9.3521e-01.
    b (float): Velocity gradient as function of pressure at constant 
                temperature. Default is 1.69478e-03.
    d0 (float): Velocity gradient with respect to density at standard 
                temperature and pressure. Default is 2.55911.
    dp (float): Pressure dependence of velocity gradient with respect to 
                density.  Default is -4.76050e-04.
    c (float): Exponential drop-off magnitude. Default is 1.674065.
    k (float): Exponential drop-off of Vp at low pressure. Default 
                is 1.953466e-02.
    m (float): Velocity gradient as function of temperature at constant 
                pressure. Default is -4e-4.
    v0_unc (float): Uncertainty in v0. Default is None.
    b_unc (float): Uncertainty in b. Default is None.
    d0_unc (float): Uncertainty in d0. Default is None.
    dp_unc (float): Uncertainty in dp. Default is None.
    c_unc (float): Uncertainty in c. Default is None.
    k_unc (float): Uncertainty in k. Default is None.
    m_unc (float): Uncertainty in m. Default is 1e-4.
    """

    # parameter values
    v0: float = -9.3521e-01
    b: float = 1.69478e-03
    d0: float = 2.55911
    dp: float = -4.76050e-04
    c: float = 1.674065
    k: float = 1.953466e-02
    m: float = -4e-4

    # parameter uncertainties
    v0_unc: float = None
    b_unc: float = None
    d0_unc: float = None
    dp_unc: float = None
    c_unc: float = None
    k_unc: float =  None
    m_unc: float = 1e-4

@dataclass
class VsConstants:
    """
    This class represents constants for shear wave velocity (Vs).

    Attributes:
    ----------
    v0 (float): Initial velocity at reference conditions. Default 
                is -6.0777e-01.
    b (float): Velocity gradient as function of pressure at constant 
                temperature.  Default is 1.0345e-03.
    d0 (float): Velocity gradient with respect to density at standard 
                temperature and pressure. Default is 1.4808.
    dp (float): Pressure dependence of velocity gradient with respect to 
                density.   Default is -2.9773e-04.
    c (float): Exponential drop-off magnitude. Default is 7.3740e-01.
    k (float): Exponential drop-off of Vs at low pressure. Default 
                is 2.0041e-02.
    m (float): Velocity gradient as function of temperature at constant 
                pressure.  Default is -2.3e-4.
    v0_unc (float): Uncertainty in v0. Default is None.
    b_unc (float): Uncertainty in b. Default is None.
    d0_unc (float): Uncertainty in d0. Default is None.
    dp_unc (float): Uncertainty in dp. Default is None.
    c_unc (float): Uncertainty in c. Default is None.
    k_unc (float): Uncertainty in k. Default is None.
    m_unc (float): Uncertainty in m. Default is 1e-4.
    """

    # parameter values
    v0: float = -6.0777e-01
    b: float = 1.0345e-03
    d0: float = 1.4808
    dp: float = -2.9773e-04
    c: float = 7.3740e-01
    k: float = 2.0041e-02
    m: float = -2.3e-4

    # parameter uncertainties
    v0_unc: float = None
    b_unc: float = None
    d0_unc: float = None
    dp_unc: float = None
    c_unc: float = None
    k_unc: float =  None
    m_unc: float = 1e-4


@dataclass
class Constants:
    """
    Master class controlling instantiation of VpConstants and VsConstants.

    Usage:
    Initialize Constants class to create instances for VpConstants 
    and VsConstants.

    Example:
    ```
    constants = Constants()
    constants.get_v_constants('Vp', v0=-0.9, b=0.001, d0=2.5, 
                              dp=-0.0005, c=1.7, k=0.019, m=1.5)
    constants.get_v_constants('Vs', v0=-0.9, b=0.001, d0=2.5, 
                              dp=-0.0005, c=1.7, k=0.019, m=2.5)
    vp_instance = constants.vp_constants
    vs_instance = constants.vs_constants
    ```
    """
    # Instance of VpConstants
    vp_constants: VpConstants = None

    # Instance of VsConstants
    vs_constants: VsConstants = None

    # Instance of MaterialConstants
    material_constants: MaterialConstants = None

    def get_v_constants(self, data_type: str, **kwargs):
        """
        Get constants instance based on data_type ('Vp' or 'Vs') and 
        assign them to class attributes.
        
        Parameters:
        - data_type (str): Indicates whether VpConstants or VsConstants 
          should be instantiated.
        - **kwargs: Variable keyword arguments representing constant values.

        Raises:
        - ValueError: If an invalid data_type is provided.
        """
        if data_type == 'Vp':
            self.vp_constants = VpConstants(**kwargs)
        elif data_type == 'Vs':
            self.vs_constants = VsConstants(**kwargs)
        else:
            raise ValueError("Invalid data_type. Must be 'Vp' or 'Vs'.")
    
    def get_material_constants(self, **kwargs):
        """
        Get material constants instance and assign them to class attributes.
        
        Parameters:
        - **kwargs: Variable keyword arguments representing constant values.
        """
        self.material_constants = MaterialConstants(**kwargs)