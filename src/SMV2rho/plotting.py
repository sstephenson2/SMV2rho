# some useful plotting routines

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.odr import Model, Data, ODR

#####################################################################

def plot_panels(data, plot_type='scatter', cmap='YlGnBu', 
                titles=None, xlabels=None, ylabels=None,
                z_values=None, figure_scale=1.0, save_path=None):
    """
    Plot multiple panels with customizable settings, each with data series in
    different colors and a legend.

    Parameters:
        data (list of lists): A list where each element represents a separate 
            plot. Each element is a list of dictionaries, where each 
            dictionary represents a data series and contains 'x', 'y', and 
            'label' values. Multiple data series for a plot should be passed 
            as separate dictionaries within the nested list.
        plot_type (str): Type of plot to create ('scatter' or 'line'). Default
            is 'scatter'.
        cmap (str): Color map for scatter plot. Default is 'YlGnBu'.
        titles (list or None): Titles for each panel. Can be a list of strings
            or None to omit titles.
        xlabels (list or None): X-axis labels for each panel. Can be a list of
            strings or None to omit x-axis labels.
        ylabels (list or None): Y-axis labels for each panel. Can be a list of
            strings or None to omit y-axis labels.
        z_values (None or list): Z-values for color mapping in scatter plot.
            Can be None or a list of values.
        figure_scale (float): Scale factor for the figure size. Default 
            is 1.0.
        save_path (str or None): File path to save the plot as an image. 
            None to display the plot without saving.

    Returns:
        None

    Example:
        data = [
            [
                {'x': [1, 2, 3], 'y': [2, 3, 1], 'label': 'Dataset 1'},
                {'x': [1, 2, 3], 'y': [1, 1, 2], 'label': 'Dataset 2'}
            ],
            [
                {'x': [4, 5, 6], 'y': [3, 4, 4], 'label': 'Dataset 3'}
            ]
        ]
        titles = ['Panel 1', 'Panel 2']
        xlabels = ['X Label 1', 'X Label 2']
        ylabels = ['Y Label 1', 'Y Label 2']
        z_values = [1, 2]  # For color mapping in scatter plot
        plot_panels(data, plot_type='scatter', cmap='viridis', titles=titles,
                    xlabels=xlabels, ylabels=ylabels, z_values=z_values,
                    figure_scale=1.2, save_path='plot.png')
    """

    n_panels = len(data)

    # Create the figure and axes for the panels
    fig, axs = plt.subplots(1, n_panels, figsize=(8*figure_scale,
                                                  6*figure_scale),
                            facecolor='w', edgecolor='k', squeeze=False)

    # Iterate over the panels
    for i, ax in enumerate(axs[0]):
        if titles:
            ax.set_title(titles[i], fontweight='bold', fontsize=12, pad=20)
        if xlabels:
            # Place x-axis label at the top
            ax.xaxis.set_label_position('top') 
            ax.set_xlabel(xlabels[i], fontweight='bold', 
                          fontsize=10)
        if ylabels:
            if i == 0 or i == n_panels-1:
                ax.set_ylabel(ylabels[i], fontweight='bold', 
                              fontsize=10)

        # Plot data on the panel
        lines = []  # To collect line objects for legend placement
        for j, dataset in enumerate(data[i]):
            if 'label' in dataset:
                label = dataset['label']
            else:
                label = None

            if plot_type == 'scatter':
                if z_values is not None:
                    sc = ax.scatter(dataset['x'], dataset['y'], c=z_values[j], 
                                    cmap=cmap, edgecolor='k', label=label)
                else:
                    sc = ax.scatter(dataset['x'], dataset['y'], color=f'C{j}', 
                                    edgecolor='k', label=label)
            elif plot_type == 'line':
                line, = ax.plot(dataset['x'], dataset['y'], color=f'C{j}', 
                                label=label)
                lines.append(line)
            else:
                raise ValueError("Invalid plot type. Choose"
                                 " 'scatter' or 'line'.")

        # Add colorbar to the plot if z values are provided
        if z_values is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.05, shrink=0.7)
            cbar.set_label('Z Values', fontweight='bold', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

        # Set border around each axis
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='both', which='both', direction='in', width=0.5)
        ax.yaxis.set_tick_params(pad=5)
        ax.xaxis.set_tick_params(pad=5)
        # Adjust x-axis label position
        ax.xaxis.set_label_coords(0.5, 1.1) 
        # Adjust y-axis label position
        ax.yaxis.set_label_coords(1.3, 0.5)

        # Adjust label positions for the right-hand panel
        if i == n_panels-1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            # Place x-axis ticks at the top
            ax.xaxis.tick_top()  
        else:
            # Remove y-axis labels and annotations for middle panels
            ax.yaxis.set_visible(False)
            ax.xaxis.tick_top()  # Place x-axis ticks at the top
            ax.annotate('', xy=(0, 0), xytext=(0, 0), 
                        xycoords='axes fraction',
                        textcoords='offset points', annotation_clip=False)

        # Show legend if label is given
        if label:
            ax.legend(lines, [line.get_label() for line in lines], 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, -0.02),
                      fancybox=True, shadow=True, ncol=1, frameon=False)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Save the plot as a high-resolution image if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_file_tree(master_path, include_files=False, 
                   suppress_pycache=False, suppress_hidden=False):
    """
    Draw a file tree diagram for the specified master directory.

    Args:
        master_path (str): The path to the master directory.
           include_files (bool, optional): Include files in the 
           file tree if True,
           otherwise only include directories. Defaults to False.
        suppress_pycache (bool, optional): 
            Suppress the '__pycache__' directory if True. Defaults to False.
        suppress_hidden (bool, optional): 
            Suppress hidden files and directories if True. Defaults to False.
    """
    def print_tree(path, prefix='', is_last=False):
        """
        Recursively print the file tree starting from the specified path.

        Args:
            path (str): The current path in the file tree.
            prefix (str, optional): The prefix string for indentation. 
                Defaults to an empty string.
            is_last (bool, optional): 
                True if the current item is the last one in its parent 
                directory, False otherwise. Defaults to False.
        """
        # Check if the current path is a directory
        if os.path.isdir(path):
            # Check if the directory is '__pycache__' and suppress it if needed
            if suppress_pycache and os.path.basename(path) == '__pycache__':
                return

            # Print the current directory name with appropriate prefix
            print(prefix, '`- ' if is_last else '|- ', 
                  os.path.basename(path), sep='')
            # Update the prefix for the subdirectories
            prefix += '   ' if is_last else '|  '

            # Get the list of files and directories in the current directory
            files = [f for f in os.listdir(path) 
                     if include_files or os.path.isdir(os.path.join(path, f))]
            if suppress_hidden:
                files = [f for f in files if not f.startswith('.')]
            files.sort()

            # Recursively print the file tree for each file/directory 
            # in the current directory
            for i, file in enumerate(files):
                is_last = (i == len(files) - 1)
                print_tree(os.path.join(path, file), prefix, is_last)

    # Print the master directory name
    print(os.path.basename(master_path))
    # Start printing the file tree from the master directory
    print_tree(master_path)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_text_file(file_path):
    """
    Print the contents of a text file exactly as it is saved.

    Args:
        file_path (str): The path to the text file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_geographic_locations(lon, lat, color, projection='Mollweide', 
                              title='Geographic Locations', 
                              third_field_label='Third Field', 
                              colorbar_range=None, **kwargs):
    """
    Plot geographic locations on a map with color-coded points.

    This function generates a map projection specified by 'projection' 
    (default is Mollweide) and plots geographic locations defined by 
    longitude ('lon') and latitude ('lat'). The color of the points is 
    determined by the 'color' parameter, and an optional third field
    can be labeled on the colorbar. You can customize the color range 
    using 'colorbar_range'.

    Parameters:
        lon (array-like): Array of longitudes for the locations.
        lat (array-like): Array of latitudes for the locations.
        color (array-like): Array of values determining the color of each 
            point.
        projection (str, optional): Map projection type. Default is 
            'Mollweide'.
        title (str, optional): Title for the plot. Default is 
            'Geographic Locations'.
        third_field_label (str, optional): Label for the third field 
            on the colorbar.
        colorbar_range (list, optional): Custom color range for the colorbar. 
            Default is determined by the data.
        **kwargs: Additional keyword arguments to pass to the scatter plot.

    Example:
        lon = [0, 90, 180, -90]
        lat = [0, 30, 60, -30]
        color = [10, 20, 30, 40]
        plot_geographic_locations(lon, lat, color, 
                                  projection='PlateCarree',
                                  title='Sample Locations', 
                                  third_field_label='Value',
                                  colorbar_range=[0, 50], s=100, 
                                  cmap='coolwarm')

    """
    # Create the figure and axes
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, projection=getattr(ccrs, projection)())

    # Set the extent of the map
    ax.set_global()

    # Add coastlines
    ax.coastlines(resolution='110m', linewidth=0.5)

    # Determine the colorbar range if not specified
    if colorbar_range is None:
        colorbar_range = [np.min(color), np.max(color)]

    # Plot the points with colors
    sc = ax.scatter(lon, lat, c=color, cmap='viridis', edgecolor='black', 
                    linewidth=0.5, s=80, transform=ccrs.PlateCarree(), 
                    vmin=colorbar_range[0], vmax=colorbar_range[1], **kwargs)

    # Add a colorbar with the specified range
    cbar = fig.colorbar(sc, shrink=0.8, pad=0.02)
    cbar.set_label(third_field_label, fontsize=12)

    # Set plot title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)

    # Set tick label sizes
    ax.tick_params(axis='both', labelsize=10)

    # Customize the gridlines
    ax.gridlines(linewidth=0.5, linestyle='dotted', alpha=0.5)

    # Adjust subplot spacing
    fig.tight_layout(pad=2)

    # Show the plot
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define a function to calculate the ODR line equation and plot it
def calculate_odr_line_equation2(x, y, ax):
    """
    Calculate the ODR (Orthogonal Distance Regression) line equation 
    and plot it.

    This function performs an ODR regression on the given data and plots the
    resulting regression line on the specified axis. It also writes the 
    equation of the regression line in the top left corner of the plot.

    Parameters:
        x (numpy.ndarray): Array of x-values.
        y (numpy.ndarray): Array of y-values.
        ax (matplotlib.axes.Axes): The axis on which to plot the 
            regression line.

    Example:
        x = np.random.rand(20)
        y = 2 * x + np.random.rand(20)
        fig, ax = plt.subplots()
        ax.scatter(x, y, label='Data')
        calculate_odr_line_equation2(x, y, ax)
        ax.legend()
        plt.show()

    Note:
        This function filters out data points where x < 1.5 before performing
        the ODR regression.

    """
    # Filter out points with x < 1.5
    mask = x >= 1.5
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Define the ODR model
    def linear_func(B, x):
        return B[0] * x + B[1]

    linear_model = Model(linear_func)

    # Create ODR data object
    data = Data(x_filtered, y_filtered)

    # Perform ODR regression
    odr = ODR(data, linear_model, beta0=[1.0, 0.0])
    odr_result = odr.run()

    # Get the regression parameters
    slope = odr_result.beta[0]
    intercept = odr_result.beta[1]

    # Plot the ODR regression line
    ax.plot(x_filtered, linear_func([slope, intercept], x_filtered), color='r', linestyle='--')

    # Write the equation in the top left corner
    equation = f"y = {slope:.2f}x + {intercept:.2f}"
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, ha='left', va='top')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to plot three scatter plots with regression and custom labels
def plot_three_scatter(data1, data2, data3, labels=None, 
                       x_labels=None, y_labels=None):
    """
    Plot three scatter plots with regression lines and custom labels.

    Parameters:
        data1 (numpy.ndarray): Data for the first scatter plot as a 
            2D array (x, y).
        data2 (numpy.ndarray): Data for the second scatter plot as a 
            2D array (x, y).
        data3 (numpy.ndarray): Data for the third scatter plot as a 
            2D array (x, y).
        labels (list, optional): List of labels for each plot. 
            Default is [None, None, None].
        x_labels (list, optional): List of x-axis labels for each plot. 
            Default is [None, None, None].
        y_labels (list, optional): List of y-axis labels for each plot. 
            Default is [None, None, None].

    Example:
        x1 = np.random.rand(20)
        y1 = 2 * x1 + np.random.rand(20)
        data1 = np.column_stack((x1, y1))

        x2 = np.random.rand(20)
        y2 = 3 * x2 + np.random.rand(20)
        data2 = np.column_stack((x2, y2))

        x3 = np.random.rand(20)
        y3 = 4 * x3 + np.random.rand(20)
        data3 = np.column_stack((x3, y3))

        labels = ['Plot 1', 'Plot 2', 'Plot 3']
        x_labels = ['X Label 1', 'X Label 2', 'X Label 3']
        y_labels = ['Y Label 1', 'Y Label 2', 'Y Label 3']

        plot_three_scatter(data1, data2, data3, labels, x_labels, y_labels)

    """
    # Create a new figure and set its size
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Define color for all scatter plots
    scatter_color = 'skyblue'

    def plot_scatter_with_regression(ax, data, label, x_label, y_label):
        x, y = data[:, 0], data[:, 1]
        ax.scatter(x, y, edgecolors='black', facecolors=scatter_color)
        ax.set_xlabel(x_label if x_label else 'X')
        ax.set_ylabel(y_label if y_label else 'Y')
        calculate_odr_line_equation2(x, y, ax)
        if label:
            ax.set_title(label)

    if labels is None:
        labels = [None, None, None]
    if x_labels is None:
        x_labels = [None, None, None]
    if y_labels is None:
        y_labels = [None, None, None]

    # Panel 1: Scatter plot for data1
    plot_scatter_with_regression(axes[0], data1, labels[0], 
                                 x_labels[0], y_labels[0])

    # Panel 2: Scatter plot for data2
    plot_scatter_with_regression(axes[1], data2, labels[1], 
                                 x_labels[1], y_labels[1])

    # Panel 3: Scatter plot for data3
    plot_scatter_with_regression(axes[2], data3, labels[2], 
                                 x_labels[2], y_labels[2])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the figure
    plt.show()