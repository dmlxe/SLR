class Dataset:
    """Defines a Dataset"""
    def __init__(self):
        """Initializes the Dataset"""
        self._data = [
            [23, 651],
            [26, 762],
            [30, 856],
            [34, 1063],
            [43, 1190],
            [48, 1298],
            [52, 1421],
            [57, 1440],
            [58, 1518]
        ]
        self._dataset_size = len(self._data)
    
    def get_dataset_size(self):
        """Returns the size of the Dataset."""
        return self._dataset_size
    
    def get_x_values(self):
        """Returns list of x values (independent variable)."""
        x_values = []
        for point in self._data:
            x_values.append(point[0])
        return x_values
    
    def get_y_values(self):
        """Returns list of y values (dependent variable)."""
        y_values = []
        for point in self._data:
            y_values.append(point[1])
        return y_values

class LinearRegressionCalculator:
    """Class for calculating linear regression coefficients."""
    
    def __init__(self, dataset: Dataset):
        """
        Initializes the calculator with a Dataset and calculates the summations.
        
        Args:
            dataset: Dataset object containing the data points
        """
        self._dataset = dataset
        self._dataset_size = dataset.get_dataset_size()
        
        # Calculate all summations during initialization
        self._sum_x = self._calculate_sum_x()
        self._sum_y = self._calculate_sum_y()
        self._sum_xy = self._calculate_sum_xy()
        self._sum_xx = self._calculate_sum_xx()
    
    def _calculate_sum_x(self):
        """
        Calculates the sum of x values.
        """
        sum_x = 0
        x_values = self._dataset.get_x_values()
        for x in x_values:
            sum_x += x
        return sum_x
    
    def _calculate_sum_y(self):
        """
        Calculates the sum of y values.
        """
        sum_y = 0
        y_values = self._dataset.get_y_values()
        for y in y_values:
            sum_y += y
        return sum_y
    
    def _calculate_sum_xy(self):
        """
        Calculates the sum of x*y products.
        """
        sum_xy = 0
        x_values = self._dataset.get_x_values()
        y_values = self._dataset.get_y_values()
        for x, y in zip(x_values, y_values):
            sum_xy += x * y
        return sum_xy
    
    def _calculate_sum_xx(self):
        """
        Calculates the sum of x squared values.
        """
        sum_xx = 0
        x_values = self._dataset.get_x_values()
        for x in x_values:
            sum_xx += x * x
        return sum_xx
    
    def calculate_b0(self):
        """
        Calculates the B0 coefficient (intercept).
        """        
        return ((self._sum_xx * self._sum_y) - (self._sum_x * self._sum_xy)) / ((self._dataset_size * self._sum_xx) - (self._sum_x * self._sum_x))
    
    def calculate_b1(self):
        """
        Calculates the B1 coefficient (slope).
        """
        return ((self._dataset_size * self._sum_xy) - (self._sum_x * self._sum_y)) / ((self._dataset_size * self._sum_xx) - (self._sum_x * self._sum_x))

class SimpleLinearRegression:
    """Main class for performing simple linear regression."""
    
    def __init__(self):
        """Initializes the linear regression model."""
        self._dataset = Dataset()
        self._calculator = LinearRegressionCalculator(self._dataset)
        self.b0 = self._calculator.calculate_b0()
        self.b1 = self._calculator.calculate_b1()
    
    def predict(self, x):
        """
        Predicts the y value for a given x value.
        
        Args:
            x: Input value (independent variable)
        """
        return self.b0 + self.b1 * x
    
    def get_equation(self):
        """
        Returns the regression equation as text.
        """
        return f"y = {self.b0} + {self.b1} * x"

def main():
    # Create and use the model
    slr = SimpleLinearRegression()
    print(f"Regression equation: {slr.get_equation()}")
    # Get input from user
    x = float(input())
    # Calculate and display prediction
    y_prediction = slr.predict(x)
    print(f"For x = {x} -> predicted y = {y_prediction}")

if __name__ == "__main__":
    main()
