from dataclasses import dataclass

@dataclass
class MaterialConstants:
    """
    Constants representing material properties for geophysical calculations.

    Attributes:
    alpha0: float = 1e-5  # Thermal expansivity coefficient at reference temperature
    alpha1: float = 2.9e-8  # Temperature derivative of thermal expansion coefficient
    K: float = 90e9  # Bulk modulus (invariant with pressure and temperature)
    """
    alpha0: float = 1e-5
    alpha1: float = 2.9e-8
    K: float = 90e9

@dataclass
class VpConstants:
    """
    Constants for compressional wave velocity (Vp).

    Attributes:
    v0: float = -9.3521e-01   # Initial velocity at reference conditions
    b: float = 1.69478e-03    # Velocity gradient as function of pressure 
                                at constant temperature
    d0: float = 2.55911       # Velocity gradient with respect to density
                                at standard temperature and pressure
    dp: float = -4.76050e-04  # Pressure dependence of velocity 
                                gradient with respect to density
    c: float = 1.674065       # Exponential drop-off magnitude
    k: float = 1.953466e-02   # Exponential drop-off of Vp at low pressure
    m: float = -4e-4          # Velocity gradient as function of temperature 
                                at constant pressure
    """
    v0: float = -9.3521e-01
    b: float = 1.69478e-03
    d0: float = 2.55911
    dp: float = -4.76050e-04
    c: float = 1.674065
    k: float = 1.953466e-02
    m: float = -4e-4

@dataclass
class VsConstants:
    """
    Constants for shear wave velocity (Vs).

    Attributes:
    v0: float = -6.0777e-01  # Initial velocity at reference conditions
    b: float = 1.0345e-03    # Velocity gradient as function of pressure at 
                               constant temperature
    d0: float = 1.4808       # Velocity gradient with respect to density at 
                               standard temperature and pressure
    dp: float = -2.9773e-04  # Pressure dependence of velocity gradient with 
                               respect to density
    c: float = 7.3740e-01    # Exponential drop-off magnitude
    k: float = 2.0041e-02    # Exponential drop-off of Vs at low pressure
    m: float = -2.3e-4       # Velocity gradient as function of temperature at
                               constant pressure
    """
    v0: float = -6.0777e-01
    b: float = 1.0345e-03
    d0: float = 1.4808
    dp: float = -2.9773e-04
    c: float = 7.3740e-01
    k: float = 2.0041e-02
    m: float = -2.3e-4

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

    vp_constants: VpConstants = None  # Instance of VpConstants
    vs_constants: VsConstants = None  # Instance of VsConstants
    material_constants: MaterialConstants = None  # Instance of MaterialConstants

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