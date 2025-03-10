import pandas as pd
import math
import pycountry

def get_iso_code(country_name):
    """
    Returns the ISO alpha-3 code for a given country name using pycountry.
    If no match is found, returns None.
    """
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

def main():
    # 1. Read the CSV with encoding specified
    df = pd.read_csv('data/sipri_trade_data.csv', encoding='latin-1')

    # 3. Add ISO columns for Recipient and Supplier
    df['Recipient ISO'] = df['Recipient'].apply(get_iso_code)
    df['Supplier ISO'] = df['Supplier'].apply(get_iso_code)

    # 4. Find the median year of delivery (rounding up)
    df['Delivery Year Numeric'] = df['Year(s) of delivery'].str.extract(r'(\d{4})').astype(float)
    
    median_year_deliv = df['Delivery Year Numeric'].median()
    median_year_rounded_up = math.ceil(median_year_deliv)

    # Store that median in the DataFrame
    df['Median Year of Delivery'] = median_year_rounded_up

    # Print the head
    print("\nFirst few rows of processed data:")
    print(df.head())

    # Save the modified DataFrame
    df.to_csv('data/sipri_trade_register_cleaned.csv', index=False)
    print("Saved cleaned data to sipri_trade_register_cleaned.csv")

if __name__ == '__main__':
    main()
