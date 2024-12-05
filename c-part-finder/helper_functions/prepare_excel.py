import pandas as pd

def clean_column(data, column_name):
    # Extract the middle part of the string using string slicing and regular expressions
    data[column_name] = data[column_name].str.extract(r'<p>(.*?)</p>', expand=False)
    return data

def import_data():
    # Replace 'your_file.xlsx' with your actual file path
    file_path = 'c-part-finder/eShop_Struktur_ETH.xlsx'
    column_ebene_2 = 'Column6'
    column_produkt_bezeichnung = 'Column12'
    column_produkt_bn = 'Column11'
    # Load the Excel file
    data = pd.read_excel(file_path)
    data = clean_column(data, 'Column6')
    return data[[column_ebene_2, column_produkt_bezeichnung, column_produkt_bn]]


def get_unique_names_column(data, column_name):
    # Extract unique values from the specified column
    unique_values = data[column_name].drop_duplicates().tolist()
    return unique_values


def get_unique_produkt_bezeichnung_bn(data, ebene_2):
    # Extract unique values from the produkt_bezeichnung column where ebene_2 matches the input
    column_ebene_2 = 'Column6'
    column_produkt_bezeichnung = 'Column12'
    filtered_data = data[data[column_ebene_2] == ebene_2][column_produkt_bezeichnung]
    unique_produkt_bezeichnung = filtered_data.drop_duplicates().tolist()
    return unique_produkt_bezeichnung



