import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
import io
import sys
import os
import joblib
from Linear_regression_auto import load_and_preprocess, show_key_stats, prepare_data, train_and_save_model, \
    evaluate_model
from titanic import load_and_prepare_data, explore_data, sigmoid_demo, cost_function, train_and_evaluate


class TestLinearRegressionAuto(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

        # Prepare test data for Linear_regression_auto.py
        self.features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year']
        self.target = 'mpg'

    def test_load_and_preprocess(self):
        """
        Test case for load_and_preprocess() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            df = load_and_preprocess("auto-mpg.csv")

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if data is loaded correctly
            expected_columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                                'acceleration', 'model-year']

            if (isinstance(df, pd.DataFrame) and
                    all(col in df.columns for col in expected_columns) and
                    " Data loaded and cleaned." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", True, "functional")
                print("TestLoadAndPreprocess = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
                print("TestLoadAndPreprocess = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
            print(f"TestLoadAndPreprocess = Failed | Exception: {e}")

    def test_show_key_stats(self):
        """
        Test case for show_key_stats() function.
        """
        try:
            # Load data first
            df = load_and_preprocess("auto-mpg.csv")

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            show_key_stats(df)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if output contains expected information
            output = captured_output.getvalue()

            if (" Mean Displacement:" in output and
                    "  Minimum Horsepower:" in output):
                self.test_obj.yakshaAssert("TestShowKeyStats", True, "functional")
                print("TestShowKeyStats = Passed")
            else:
                self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
                print("TestShowKeyStats = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
            print(f"TestShowKeyStats = Failed | Exception: {e}")

    def test_prepare_data(self):
        """
        Test case for prepare_data() function.
        """
        try:
            # Load data first
            df = load_and_preprocess("auto-mpg.csv")

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if data is prepared correctly
            if (isinstance(X_train, np.ndarray) and
                    isinstance(X_test, np.ndarray) and
                    (isinstance(y_train, pd.Series) or isinstance(y_train, np.ndarray)) and
                    (isinstance(y_test, pd.Series) or isinstance(y_test, np.ndarray)) and
                    hasattr(scaler, 'transform') and
                    " Data prepared and split." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestPrepareData", True, "functional")
                print("TestPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
                print("TestPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
            print(f"TestPrepareData = Failed | Exception: {e}")

    def test_train_and_save_model(self):
        """
        Test case for train_and_save_model() function.
        """
        try:
            # Load and prepare data first
            df = load_and_preprocess("auto-mpg.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)

            # Create a temporary file for testing
            test_model_path = "test_linear_model.pkl"

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            model = train_and_save_model(X_train, y_train, test_model_path)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if model is trained and saved correctly
            file_exists = os.path.exists(test_model_path)

            if (hasattr(model, 'coef_') and
                    hasattr(model, 'intercept_') and
                    file_exists and
                    f" Model trained and saved to '{test_model_path}'" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", True, "functional")
                print("TestTrainAndSaveModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
                print("TestTrainAndSaveModel = Failed")

            # Clean up test file
            if file_exists:
                os.remove(test_model_path)

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
            print(f"TestTrainAndSaveModel = Failed | Exception: {e}")

    def test_evaluate_model(self):
        """
        Test case for evaluate_model() function.
        """
        try:
            # Load and prepare data first
            df = load_and_preprocess("auto-mpg.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            model = train_and_save_model(X_train, y_train)

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            evaluate_model(model, X_test, y_test)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if evaluation output is correct
            output = captured_output.getvalue()

            if (" Mean Squared Error:" in output and
                    " Sample Predictions:" in output):
                self.test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
            print(f"TestEvaluateModel = Failed | Exception: {e}")


class TestTitanic(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

    def test_load_and_prepare_data(self):
        """
        Test case for load_and_prepare_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            df = load_and_prepare_data("titanic.csv")

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if data is loaded and prepared correctly
            expected_columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

            if (isinstance(df, pd.DataFrame) and
                    all(col in df.columns for col in expected_columns) and
                    df['sex'].dtype == 'int64' and  # Check if sex is encoded
                    df['embarked'].dtype == 'int64' and  # Check if embarked is encoded
                    " Data loaded, cleaned, and encoded." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", True, "functional")
                print("TestLoadAndPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
                print("TestLoadAndPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
            print(f"TestLoadAndPrepareData = Failed | Exception: {e}")

    def test_explore_data(self):
        """
        Test case for explore_data() function.
        """
        try:
            # Load data first
            df = load_and_prepare_data("titanic.csv")

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            explore_data(df)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if output contains expected information
            output = captured_output.getvalue()

            if (" Fare - Max:" in output and
                    "Std Dev:" in output):
                self.test_obj.yakshaAssert("TestExploreData", True, "functional")
                print("TestExploreData = Passed")
            else:
                self.test_obj.yakshaAssert("TestExploreData", False, "functional")
                print("TestExploreData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestExploreData", False, "functional")
            print(f"TestExploreData = Failed | Exception: {e}")

    def test_sigmoid_demo(self):
        """
        Test case for sigmoid_demo() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            sigmoid_demo()

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if output contains expected information
            output = captured_output.getvalue()

            if " Sigmoid(0) = 0.5000" in output:
                self.test_obj.yakshaAssert("TestSigmoidDemo", True, "functional")
                print("TestSigmoidDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
                print("TestSigmoidDemo = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
            print(f"TestSigmoidDemo = Failed | Exception: {e}")

    def test_cost_function(self):
        """
        Test case for cost_function() function.
        """
        try:
            # Create test data
            y_true = np.array([0, 1, 0, 1, 1])
            y_pred_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

            # Call the function
            cost = cost_function(y_true, y_pred_prob)

            # Check if cost is calculated correctly
            # For binary cross-entropy, the cost should be positive
            if isinstance(cost, float) and cost > 0:
                self.test_obj.yakshaAssert("TestCostFunction", True, "functional")
                print("TestCostFunction = Passed")
            else:
                self.test_obj.yakshaAssert("TestCostFunction", False, "functional")
                print("TestCostFunction = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCostFunction", False, "functional")
            print(f"TestCostFunction = Failed | Exception: {e}")

    def test_train_and_evaluate(self):
        """
        Test case for train_and_evaluate() function.
        """
        try:
            # Load and prepare data first
            df = load_and_prepare_data("titanic.csv")

            features = ['pclass', 'sex', 'age', 'fare', 'embarked']
            X = df[features]
            y = df['survived']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a temporary file for testing
            test_model_path = "test_titanic_model.pkl"

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Call the function
            train_and_evaluate(X_train, y_train, X_test, y_test, test_model_path)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if model is trained and evaluated correctly
            file_exists = os.path.exists(test_model_path)
            output = captured_output.getvalue()

            if (file_exists and
                    " Model trained and saved to" in output and
                    " Log Loss (Custom Cost):" in output and
                    " Sample Predictions:" in output):
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", True, "functional")
                print("TestTrainAndEvaluate = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("TestTrainAndEvaluate = Failed")

            # Clean up test file
            if file_exists:
                os.remove(test_model_path)

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print(f"TestTrainAndEvaluate = Failed | Exception: {e}")


if __name__ == '__main__':
    unittest.main()
