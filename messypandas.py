import pandas as pd
import numpy as np
import json
from datetime import datetime
import re

def load_json_to_dataframes(json_file):
    """
    Load JSON using pandas json_normalize for automatic flattening
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Use json_normalize to flatten nested structures
    customers_raw = pd.json_normalize(
        data['customers'],
        sep='_'
    )
    
    # Extract purchases into separate dataframe
    purchases_data = []
    for idx, customer in enumerate(data['customers']):
        customer_id = customer.get('id', f"MISSING_{idx}")
        purchases = customer.get('purchases', [])
        
        if purchases:
            for purchase in purchases:
                purchase_copy = purchase.copy()
                purchase_copy['customer_id'] = customer_id
                purchases_data.append(purchase_copy)
    
    if purchases_data:
        orders_raw = pd.json_normalize(purchases_data, sep='_')
    else:
        orders_raw = pd.DataFrame()
    
    # Extract items into separate dataframe
    items_data = []
    for idx, customer in enumerate(data['customers']):
        customer_id = customer.get('id', f"MISSING_{idx}")
        purchases = customer.get('purchases', [])
        
        if purchases:
            for purchase in purchases:
                order_id = purchase.get('id', f"{customer_id}-ORD")
                items = purchase.get('items', [])
                
                for item in items:
                    item_copy = item.copy()
                    item_copy['customer_id'] = customer_id
                    item_copy['order_id'] = order_id
                    items_data.append(item_copy)
    
    if items_data:
        items_raw = pd.json_normalize(items_data, sep='_')
    else:
        items_raw = pd.DataFrame()
    
    return customers_raw, orders_raw, items_raw


def clean_currency_column(series):
    """
    Vectorized currency cleaning using pandas string methods
    """
    return (series
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
            .replace('None', np.nan)
            .replace('', np.nan)
            .astype(float)
            .fillna(0.0))


def clean_phone_column(series):
    """
    Vectorized phone number cleaning
    """
    # Extract only digits
    digits = series.astype(str).str.replace(r'\D', '', regex=True)
    
    # Format as XXX-XXXX (take last 7 digits)
    formatted = digits.apply(lambda x: f"{x[-7:-4]}-{x[-4:]}" if len(x) >= 7 else x if x else None)
    
    return formatted.replace('', None).replace('None-None', None)


def clean_email_column(series):
    """
    Vectorized email cleaning and validation
    """
    # Convert to lowercase and strip
    cleaned = series.astype(str).str.lower().str.strip()
    
    # Validate: must contain @ and . after @
    valid_mask = cleaned.str.contains(r'@.*\.', regex=True, na=False)
    
    # Set invalid emails to None
    cleaned = cleaned.where(valid_mask, None)
    cleaned = cleaned.replace('none', None).replace('', None)
    
    return cleaned


def parse_dates_column(series):
    """
    Vectorized date parsing with multiple format support
    """
    # First attempt with coerce for all formats
    result = pd.to_datetime(series, errors='coerce', utc=True)
    
    # Convert to datetime without timezone and format as string
    result = result.dt.tz_localize(None) if result.dt.tz is not None else result
    
    # Return as string in YYYY-MM-DD format
    return result.dt.strftime('%Y-%m-%d').where(result.notna(), None)


def clean_customers_dataframe(df):
    """
    Clean customers dataframe using pandas vectorized operations
    """
    df = df.copy()
    
    # Rename columns from json_normalize format
    column_mapping = {
        'contact_email': 'email',
        'contact_phone': 'phone',
        'contact_address_street': 'street',
        'contact_address_city': 'city',
        'contact_address_state': 'state',
        'contact_address_zip': 'zip'
    }
    df = df.rename(columns=column_mapping)
    
    # Handle missing IDs
    df['id'] = df['id'].fillna('').astype(str)
    df.loc[df['id'] == '', 'id'] = [f"MISSING_{i}" for i in range((df['id'] == '').sum())]
    df = df.rename(columns={'id': 'customer_id'})
    
    # Clean email - vectorized
    df['email'] = clean_email_column(df['email'])
    
    # Clean phone - vectorized
    df['phone'] = clean_phone_column(df['phone'])
    
    # Parse registration date - vectorized
    df['registration_date'] = parse_dates_column(df['registration'])
    
    # Normalize status - vectorized
    df['status'] = df['status'].str.lower().fillna('unknown')
    
    # Clean loyalty points
    df['loyalty_points'] = pd.to_numeric(df['loyalty_points'], errors='coerce').fillna(0).astype(int)
    
    # Clean address fields - vectorized
    df['street'] = df['street'].fillna('').str.strip()
    df['city'] = df['city'].fillna('').str.strip().str.title()
    df['state'] = df['state'].fillna('').str.strip().str.upper()
    df['zip'] = df['zip'].fillna('').astype(str).str.strip()
    
    # Select and order columns
    columns_to_keep = ['customer_id', 'name', 'email', 'phone', 'street', 'city', 
                       'state', 'zip', 'registration_date', 'status', 'loyalty_points']
    
    return df[columns_to_keep]


def clean_orders_dataframe(df):
    """
    Clean orders dataframe using pandas vectorized operations
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Parse dates - vectorized
    df['order_date'] = parse_dates_column(df['date'])
    
    # Clean totals - vectorized
    df['order_total'] = clean_currency_column(df['total'])
    
    # Clean payment method - vectorized
    df['payment_method'] = (df['payment_method']
                            .fillna('unknown')
                            .str.replace('_', ' ')
                            .str.title())
    
    # Rename and select columns
    df = df.rename(columns={'id': 'order_id'})
    columns_to_keep = ['customer_id', 'order_id', 'order_date', 'order_total', 'payment_method']
    
    return df[columns_to_keep]


def clean_items_dataframe(df):
    """
    Clean items dataframe using pandas vectorized operations
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Clean prices - vectorized
    df['price'] = clean_currency_column(df['price'])
    
    # Clean quantities - vectorized
    df['quantity'] = pd.to_numeric(df['qty'], errors='coerce').fillna(1).astype(int)
    
    # Normalize categories - vectorized
    df['category'] = df['category'].fillna('Uncategorized').str.strip().str.title()
    
    # Calculate line total - vectorized
    df['line_total'] = (df['price'] * df['quantity']).round(2)
    
    # Rename name column if it exists
    if 'name' in df.columns:
        df = df.rename(columns={'name': 'item_name'})
    
    # Select columns that exist
    columns_to_keep = ['customer_id', 'order_id', 'item_name', 'price', 'quantity', 'category', 'line_total']
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    
    return df[existing_columns]


def calculate_customer_metrics(customers_df, orders_df, items_df):
    """
    Calculate aggregated metrics using pandas groupby
    """
    # Aggregate orders by customer
    if not orders_df.empty:
        order_metrics = orders_df.groupby('customer_id').agg({
            'order_id': 'count',
            'order_total': 'sum'
        }).rename(columns={
            'order_id': 'order_count',
            'order_total': 'total_spent'
        }).reset_index()
    else:
        order_metrics = pd.DataFrame(columns=['customer_id', 'order_count', 'total_spent'])
    
    # Aggregate items by customer
    if not items_df.empty:
        item_metrics = items_df.groupby('customer_id').agg({
            'quantity': 'sum'
        }).rename(columns={
            'quantity': 'total_items_purchased'
        }).reset_index()
    else:
        item_metrics = pd.DataFrame(columns=['customer_id', 'total_items_purchased'])
    
    # Merge metrics back to customers using pandas merge
    customers_df = customers_df.merge(order_metrics, on='customer_id', how='left')
    customers_df = customers_df.merge(item_metrics, on='customer_id', how='left')
    
    # Fill NaN values
    customers_df['order_count'] = customers_df['order_count'].fillna(0).astype(int)
    customers_df['total_spent'] = customers_df['total_spent'].fillna(0).round(2)
    customers_df['total_items_purchased'] = customers_df['total_items_purchased'].fillna(0).astype(int)
    
    return customers_df


def analyze_with_pandas(customers_df, orders_df, items_df):
    """
    Perform analysis using pandas operations
    """
    print("=" * 80)
    print("PANDAS-POWERED DATA ANALYSIS")
    print("=" * 80)
    
    # Data quality summary using pandas methods
    print("\nDATA QUALITY METRICS:")
    print(f"Total Customers: {len(customers_df)}")
    print(f"Missing Email: {customers_df['email'].isna().sum()} ({customers_df['email'].isna().mean()*100:.1f}%)")
    print(f"Missing Phone: {customers_df['phone'].isna().sum()} ({customers_df['phone'].isna().mean()*100:.1f}%)")
    
    incomplete_address = ((customers_df['street'] == '') | (customers_df['city'] == '')).sum()
    print(f"Incomplete Address: {incomplete_address} ({incomplete_address/len(customers_df)*100:.1f}%)")
    
    no_orders = (customers_df['order_count'] == 0).sum()
    print(f"No Orders: {no_orders} ({no_orders/len(customers_df)*100:.1f}%)")
    
    # Display dataframes
    print("\n" + "=" * 80)
    print("CUSTOMER DATA")
    print("=" * 80)
    print(customers_df.to_string(index=False))
    
    if not orders_df.empty:
        print("\n" + "=" * 80)
        print("ORDER DATA")
        print("=" * 80)
        print(orders_df.to_string(index=False))
    
    if not items_df.empty:
        print("\n" + "=" * 80)
        print("CATEGORY ANALYSIS (using groupby + agg)")
        print("=" * 80)
        
        # Pandas groupby for category analysis
        category_stats = items_df.groupby('category').agg({
            'line_total': ['sum', 'mean'],
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).round(2)
        
        category_stats.columns = ['Total_Sales', 'Avg_Sale', 'Units_Sold', 'Unique_Customers']
        category_stats = category_stats.sort_values('Total_Sales', ascending=False)
        print(category_stats)
    
    print("\n" + "=" * 80)
    print("BUSINESS QUESTIONS (Pandas-Style)")
    print("=" * 80)
    
    # Question 1: VIP Customers using pandas operations
    print("\n1. VIP CUSTOMERS (using vectorized scoring)")
    
    # Calculate VIP score - fully vectorized
    max_spent = customers_df['total_spent'].max() or 1
    max_orders = customers_df['order_count'].max() or 1
    max_points = customers_df['loyalty_points'].max() or 1
    
    customers_df['vip_score'] = (
        (customers_df['total_spent'] / max_spent) * 40 +
        (customers_df['order_count'] / max_orders) * 30 +
        (customers_df['loyalty_points'] / max_points) * 30
    )
    
    # Filter VIPs using pandas query or boolean indexing
    vips = customers_df[customers_df['vip_score'] >= 60].sort_values('vip_score', ascending=False)
    
    print(f"   Found {len(vips)} VIPs:")
    print(vips[['name', 'total_spent', 'order_count', 'loyalty_points', 'vip_score']].to_string(index=False))
    
    # Question 2: Profile Updates using pandas query
    print("\n2. PROFILE UPDATE NEEDS (using boolean indexing)")
    
    needs_update = customers_df[
        (customers_df['status'] == 'active') &
        (
            customers_df['email'].isna() |
            customers_df['phone'].isna() |
            (customers_df['city'] == '') |
            (customers_df['state'] == '')
        )
    ].copy()
    
    # Add priority column
    needs_update['priority'] = np.where(needs_update['order_count'] > 0, 'HIGH', 'MEDIUM')
    
    print(f"   Found {len(needs_update)} customers:")
    if not needs_update.empty:
        print(needs_update[['name', 'email', 'phone', 'city', 'order_count', 'priority']].to_string(index=False))
    
    # Question 3: Re-engagement using pandas filtering
    print("\n3. RE-ENGAGEMENT TARGETS (using multi-condition filters)")
    
    inactive_with_history = customers_df[
        customers_df['status'].isin(['inactive', 'pending']) &
        (customers_df['total_spent'] > 0)
    ]
    
    low_engagement = customers_df[
        (customers_df['status'] == 'active') &
        (customers_df['order_count'] <= 1) &
        (customers_df['total_spent'] > 0)
    ]
    
    total_targets = len(inactive_with_history) + len(low_engagement)
    total_ltv = inactive_with_history['total_spent'].sum() + low_engagement['total_spent'].sum()
    
    print(f"   Inactive with history: {len(inactive_with_history)}")
    print(f"   Low engagement: {len(low_engagement)}")
    print(f"   Total targets: {total_targets}")
    print(f"   Combined LTV: ${total_ltv:.2f}")
    print(f"   Recommendation: {'YES - Launch campaign' if total_targets > 0 else 'NO - Not needed'}")
    
    # Question 4: Best Category using pandas ranking
    if not items_df.empty:
        print("\n4. BEST CATEGORY (using composite scoring)")
        
        category_perf = items_df.groupby('category').agg({
            'line_total': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        })
        category_perf.columns = ['Revenue', 'Units', 'Customers']
        
        # Normalize and score - vectorized
        category_perf['score'] = (
            (category_perf['Revenue'] / category_perf['Revenue'].max()) * 50 +
            (category_perf['Units'] / category_perf['Units'].max()) * 25 +
            (category_perf['Customers'] / category_perf['Customers'].max()) * 25
        ).round(1)
        
        category_perf = category_perf.sort_values('score', ascending=False)
        print(category_perf)
        print(f"\n   Winner: {category_perf.index[0]} (Score: {category_perf['score'].iloc[0]})")
    
    # Question 5: Loyalty Analysis using pandas statistics
    print("\n5. LOYALTY PROGRAM EFFECTIVENESS (using pandas stats)")
    
    active = customers_df[customers_df['status'] == 'active']
    
    if len(active) > 0 and active['total_spent'].sum() > 0:
        points_per_dollar = active['loyalty_points'].sum() / active['total_spent'].sum()
        
        print(f"   Points per dollar: {points_per_dollar:.2f}")
        
        # Correlation analysis
        correlation = active[['total_spent', 'loyalty_points']].corr().iloc[0, 1]
        print(f"   Correlation (spending vs points): {correlation:.3f}")
        
        # Top spenders analysis using nlargest
        top_spenders = active.nlargest(3, 'total_spent')[['name', 'total_spent', 'loyalty_points']]
        print("\n   Top 3 Spenders:")
        print(top_spenders.to_string(index=False))
        
        assessment = "STRONG" if points_per_dollar > 1 and correlation > 0.8 else "MODERATE" if correlation > 0.5 else "WEAK"
        print(f"\n   Assessment: {assessment} loyalty program effectiveness")
    
    return customers_df


if __name__ == "__main__":
    print("PANDAS-NATIVE DATA WRANGLING")
    print("=" * 80)
    print("Using vectorized operations, groupby, merge, and pandas methods\n")
    
    # Load data using json_normalize
    print("Loading data with pd.json_normalize()...")
    customers_raw, orders_raw, items_raw = load_json_to_dataframes('/home/claude/messy_data.json')
    
    # Clean dataframes using pandas operations
    print("Cleaning with vectorized pandas operations...")
    customers_df = clean_customers_dataframe(customers_raw)
    orders_df = clean_orders_dataframe(orders_raw)
    items_df = clean_items_dataframe(items_raw)
    
    # Calculate metrics using groupby and merge
    print("Aggregating metrics with groupby()...")
    customers_df = calculate_customer_metrics(customers_df, orders_df, items_df)
    
    # Analyze
    customers_df = analyze_with_pandas(customers_df, orders_df, items_df)
    
    # Save using pandas to_csv
    print("\n" + "=" * 80)
    print("SAVING CLEANED DATA")
    print("=" * 80)
    
    customers_df.to_csv('/home/claude/pandas_cleaned_customers.csv', index=False)
    orders_df.to_csv('/home/claude/pandas_cleaned_orders.csv', index=False)
    items_df.to_csv('/home/claude/pandas_cleaned_items.csv', index=False)
    
    print("✓ pandas_cleaned_customers.csv")
    print("✓ pandas_cleaned_orders.csv")
    print("✓ pandas_cleaned_items.csv")
    
    print("\n" + "=" * 80)
    print("PANDAS METHODS USED")
    print("=" * 80)
    print("""
    - pd.json_normalize() - Flatten nested JSON
    - df.merge() - Join dataframes
    - df.groupby() - Aggregate data
    - df.agg() - Multiple aggregations
    - df.str methods - Vectorized string operations
    - pd.to_datetime() - Date parsing
    - df.fillna() - Handle missing values
    - df.query() / boolean indexing - Filter data
    - df.nlargest() - Top N rows
    - df.corr() - Correlation analysis
    - df.apply() - Custom functions (minimal use)
    - Vectorized operations - No loops!
    """)