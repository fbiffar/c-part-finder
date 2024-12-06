import pandas as pd

def extract_category_details(csv_filepath, unique_id):
    """
    Extracts the details of a category with a matching unique_id from the given CSV file.
    
    Parameters:
        csv_filepath (str): The path to the CSV file.
        unique_id (int): The unique ID to search for.
    
    Returns:
        dict: A dictionary containing the details of the matching row, or None if not found.
    """
    try:
        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(csv_filepath)

        # Find the row where unique_id matches
        matching_row = df.loc[df['unique_id'] == unique_id]

        # If a matching row exists, return it as a dictionary
        if not matching_row.empty:
            return matching_row.iloc[0].to_dict()
        
        # Return None if no matching row is found
        return None
    
    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} does not exist.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_filepath} is empty or improperly formatted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
