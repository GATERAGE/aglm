import os
import joblib
import pandas as pd

class Predictor:
    """
    A class to manage the machine learning model for making predictions.
    """
    def __init__(self, model_dir):
        """
        Initializes the Predictor object by loading the model from the specified directory.

        Args:
            model_dir (str): The directory containing the trained model.
        """
        self.model = self.load_model(model_dir)

    def load_model(self, model_dir):
        """
        Load a trained machine learning model from the specified directory.

        Args:
            model_dir (str): The directory containing the trained model.

        Returns:
            The loaded machine learning model or None if an error occurs.
        """
        model_filename = os.path.join(model_dir, 'trained_model.pkl')
        try:
            return joblib.load(model_filename)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def predict(self, features):
        """
        Make predictions using the loaded machine learning model.

        Args:
            features (DataFrame): A Pandas DataFrame containing the features of new data.

        Returns:
            A Pandas Series containing the predicted labels, or None if an error occurs.
        """
        if self.model is not None:
            try:
                predictions = self.model.predict(features)
                return pd.Series(predictions, name='predicted_label')
            except Exception as e:
                print(f"Failed to make predictions: {e}")
        return None

def main(input_dir, model_dir, output_dir):
    """
    Main function to make predictions using a trained model and save the results.

    Args:
        input_dir (str): The directory containing the new data features.
        model_dir (str): The directory containing the trained machine learning model.
        output_dir (str): The directory where the prediction results will be saved.
    """
    predictor = Predictor(model_dir)

    if predictor.model is None:
        print("Model loading failed. Exiting.")
        return

    try:
        features = pd.read_csv(os.path.join(input_dir, 'new_data_features.csv'))
        predictions = predictor.predict(features)
        if predictions is not None:
            output_filename = os.path.join(output_dir, 'predictions.csv')
            predictions.to_csv(output_filename, index=False)
            print(f"Predictions saved to {output_filename}.")
        else:
            print("No predictions to save.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage when running this script directly
    input_dir = "data/new_data"
    model_dir = "models"
    output_dir = "results"
    main(input_dir, model_dir, output_dir)
