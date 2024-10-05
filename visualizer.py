import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook

class DataProcessor:
    def __init__(self, db_name='data.db'):
        self.engine = create_engine(f'sqlite:///{db_name}')
    
    def load_data_to_db(self, training_file, ideal_file, test_file):
        """Load CSV data into SQLite database."""

        train_df = pd.read_csv(training_file)
        train_df.to_sql('training_data', self.engine, if_exists='replace', index=False)

        ideal_df = pd.read_csv(ideal_file)
        ideal_df.to_sql('ideal_functions', self.engine, if_exists='replace', index=False)

        test_df = pd.read_csv(test_file)
        test_df.to_sql('test_data', self.engine, if_exists='replace', index=False)

    def select_ideal_functions(self):
        """Select the four ideal functions that best fit the training data."""
        connection = self.engine.connect()
        ideal_df = pd.read_sql('ideal_functions', connection)
        train_df = pd.read_sql('training_data', connection)
        selected_functions = []

        for train_col in ['y1', 'y2', 'y3', 'y4']:
            min_deviation = float('inf')
            best_function = None
            
            for ideal_col in ideal_df.columns[1:]:
                deviation = ((train_df[train_col] - ideal_df[ideal_col]) ** 2).sum()
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_function = ideal_col
            
            selected_functions.append(best_function)
        
        connection.close()
        return selected_functions

    def map_test_data(self, selected_functions):
        """Map test data to the selected ideal functions."""
        connection = self.engine.connect()
        test_df = pd.read_sql('test_data', connection)
        ideal_df = pd.read_sql('ideal_functions', connection)
        results = []

        for index, row in test_df.iterrows():
            # x, y = index, row
            x, y = row['x'], row['y']
            best_fit = None
            min_deviation = float('inf')
            
            for func in selected_functions:
                deviation = abs(y - ideal_df.loc[ideal_df['x'] == x, func].values[0])
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_fit = func
            
            results.append((x, y, best_fit, min_deviation))
        #database saving
        results_df = pd.DataFrame(results, columns=['x', 'y', 'Ideal Function', 'Deviation'])
        results_df.to_sql('mapped_test_data', self.engine, if_exists='replace', index=False)
        connection.close()

    def visualize_data(self, selected_functions):
        """Visualize the training data, test data, and ideal functions."""
        connection = self.engine.connect()
        output_notebook()
        output_file("visualization.html")
        
        p = figure(title="Data Visualization", x_axis_label='X', y_axis_label='Y')
        #plotdata
        train_df = pd.read_sql('training_data', connection)
        for col in ['y1', 'y2', 'y3', 'y4']:
            p.line(train_df['x'], train_df[col], legend_label=f'Training Data {col}', line_width=2)
        
        #plot ideafunctions
        ideal_df = pd.read_sql('ideal_functions', connection)
        for func in selected_functions:
            p.line(ideal_df['x'], ideal_df[func], legend_label=f'Ideal Function {func}', line_dash='dashed')
        
        #plot testdata
        test_df = pd.read_sql('test_data', connection)
        p.circle(test_df['x'], test_df['y'], size=5, color='red', legend_label='Test Data')
        
        show(p)
        connection.close()

training_file = 'train.csv'
ideal_file = 'ideal.csv'
test_file = 'test.csv'

processor = DataProcessor()
processor.load_data_to_db(training_file, ideal_file, test_file)
selected_functions = processor.select_ideal_functions()
processor.map_test_data(selected_functions)
processor.visualize_data(selected_functions)
