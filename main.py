import pandas as pd
from typing import Optional
import re

def clean_price(unit_price: Optional[str]) -> float:
    if unit_price is None or unit_price.strip() == '':
        return 0.0
    try:
        cleaned_price = re.sub(r'[$,]', '', unit_price)
        return float(cleaned_price)
    except ValueError:
        return 0.0

def clean_quantity(quantity: Optional[str]) -> int:

    if quantity is None or quantity.strip() == '':
        return 0
    


    quantity = quantity.strip().lower()

    text_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10
    }

    if quantity in text_numbers:
        return text_numbers[quantity]
    try:
        return int(quantity)
    except ValueError:
        return 0


def main():
    df = pd.read_csv('messy_sales.csv')
    print(df.columns)

    print(df.head())

    print(df["customer_name"].value_counts())

    print((df["unit_price"]).apply(clean_price))

    print(df["quantity"].apply(clean_quantity))

    print(df["customer_name"].apply(lambda x: x.strip().lower()))






if __name__ == "__main__":
    main()
