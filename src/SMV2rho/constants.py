# module for collating and organising constants used in SMV2rho

from dataclasses import dataclass, field
import sys

@dataclass
class Constants:
    """
    Master class to collect velocity to density constants for Vp and Vs.

    Usage:
    Initialize Constants class by specifying the 'data_type' 
    ('Vp' or 'Vs') and optional specific parameters
    (v0, b, d0, dp, c, k) for overriding default values.

    Arguments:
    - data_type (str): Type of data, must be either 'Vp' or 'Vs'.
    - v0 (float, optional): Velocity parameter 'v0'. Default is -9.3521e-01.
    - b (float, optional): Velocity parameter 'b'. Default is 1.69478e-03.
    - d0 (float, optional): Velocity parameter 'd0'. Default is 2.55911.
    - dp (float, optional): Velocity parameter 'dp'. Default is -4.76050e-04.
    - c (float, optional): Velocity parameter 'c'. Default is 1.674065.
    - k (float, optional): Velocity parameter 'k'. Default is 1.953466e-02.

    Example:
    To create an instance with custom values:
    ```
    custom_constants = Constants(
                            data_type='Vp', 
                            v0=-0.9, 
                            b=0.001, 
                            d0=2.5,
                            dp=-0.0005,
                            c=1.7,
                            k=0.019)
    ```

    Any parameters not explicitly set to custom values will take the default
    value for the velocity profile type given.

    """


    data_type: str  # Type of data: 'Vp' or 'Vs'
    v0: float = None
    b: float = None
    d0: float = None
    dp: float = None
    c: float = None
    k: float = None

    default_params = {
        'Vp': {
            'v0': -9.3521e-01,
            'b': 1.69478e-03,
            'd0': 2.55911,
            'dp': -4.76050e-04,
            'c': 1.674065,
            'k': 1.953466e-02
        },
        'Vs': {
            'v0': -9.3521e-01,
            'b': 1.6948e-03,
            'd0': 2.5591,
            'dp': -4.7605e-04,
            'c': 1.6741,
            'k': 1.9535e-02
        }
    }

    def __post_init__(self):
        """
        Initializes the Constants object after the initial creation.
        Overrides default parameters if specific parameters are provided.

        Raises:
        - ValueError: If the provided data_type is invalid ('Vp' or 'Vs').
        """

        # Check if the provided data_type is valid ('Vp' or 'Vs')
        if self.data_type not in self.default_params:
            raise ValueError("Invalid data_type. Must be 'Vp' or 'Vs'.")

        # Create a copy of default parameters to avoid 
        # modifying the class-level defaults
        params = self.default_params[self.data_type].copy()

        # If specific parameters are provided during 
        # initialization, override defaults
        if self.v0 is not None:
            params['v0'] = self.v0
        if self.b is not None:
            params['b'] = self.b
        if self.d0 is not None:
            params['d0'] = self.d0
        if self.dp is not None:
            params['dp'] = self.dp
        if self.c is not None:
            params['c'] = self.c
        if self.k is not None:
            params['k'] = self.k

        # Set the attributes directly based on the chosen parameters
        for attr, value in params.items():
            setattr(self, attr, value)

@dataclass
class TemperatureDependentConstants(Constants):
    """
    Class to extend Constants class with temperature-dependent constants and 
    additional parameters.

    Usage:
    Initialize TemperatureDependentConstants class by specifying the 
    'data_type' ('Vp' or 'Vs'), optional specific parameters, 
    temperature-related constants, additional parameters 'alpha0', 'alpha1', 
    'K', and parameter 'm'.

    Arguments:
    - alpha0 (float, optional): Additional parameter 'alpha0'. 
            Default is 1e-5.
    - alpha1 (float, optional): Additional parameter 'alpha1'. 
            Default is 2.9e-8.
    - K (float, optional): Additional parameter 'K'. Default is 90e9.
    - m (int, optional): Parameter 'm' based on 'data_type' (Vp or Vs). 
      Default values are:
        - For 'Vp': -4e-4
        - For 'Vs': -2.3e-4

    Example:
    To create an instance with custom values:
    ```
    custom_constants = TemperatureDependentConstants(
                            data_type='Vp', 
                            alpha0=1e-4, 
                            alpha1=3.5e-8, 
                            K=85e9,
                            m=-3e-4)
    ```

    The 'm' parameter can be overridden based on the 'data_type' specified.
    Any parameters not explicitly set to custom values will take the default
    value for the velocity profile type given.

    """

    alpha0: float = None
    alpha1: float = None
    K: float = None
    m: int = None

    default_additional_params = {
        'alpha0': 1e-5,
        'alpha1': 2.9e-8,
        'K': 90e9
    }

    def __post_init__(self):
        """
        Initializes the TemperatureDependentConstants object after the 
        initial creation.
        Overrides default additional parameters if specific parameters are 
        provided.
        Sets the parameter 'm' based on the 'data_type' (Vp or Vs).

        Raises:
        - ValueError: If the provided data_type is invalid ('Vp' or 'Vs').
        """

        super().__post_init__()

        # Create a copy of default additional parameters to avoid modifying 
        # the class-level defaults
        additional_params = self.default_additional_params.copy()

        # If specific additional parameters are provided during 
        # initialization, override defaults
        if self.alpha0 is not None:
            additional_params['alpha0'] = self.alpha0
        if self.alpha1 is not None:
            additional_params['alpha1'] = self.alpha1
        if self.K is not None:
            additional_params['K'] = self.K

        # Set the attributes directly based on the chosen additional 
        # parameters
        for attr, value in additional_params.items():
            setattr(self, attr, value)

        # Set parameter 'm' based on 'data_type' (Vp or Vs) if not explicitly 
        # provided
        if self.m is None:
            if self.data_type == 'Vp':
                self.m = -4e-4
            elif self.data_type == 'Vs':
                self.m = -2.3e-4
            else:
                raise ValueError("Invalid data_type. Must be 'Vp' or 'Vs'.")


sys.exit()



@dataclass
class stpV2rhoConstants:
    """
    Master class to collect velocity to density constants.
    """
    
    v0: float = field(default = -9.3521e-01)
    b:  float = field(default = 1.69478e-03)
    d0: float = field(default = 2.55911)
    dp: float = field(default = -4.76050e-04)
    c:  float = field(default = 1.674065)
    k:  float = field(default = 1.953466e-02)

@dataclass
class TemperatureDependentConstants(stpV2rhoConstants):
    """
    Constants for calculating temperature dependence.
    """
    mP: float = field(default = -4e-4)
    mT: float = field(default = -2.3e-4)
    alpha0: float = field(default = 1e-5)
    alpha1: float = field(default = 2.9e-8)
    K: float = field(default = 90e9)

