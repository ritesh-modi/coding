import json
import pandas as pd
import re
from datetime import datetime
from typing import Optional, Any, Dict, List
import numpy as np

def parse_amount(amount: Any) -> float:
    """Clean and parse monetary amounts from various formats"""
    if amount is None or amount == "":
        return 0.0
    
    # Convert to string if not already
    amount_str = str(amount)
    
    # Remove currency symbols, commas, and extra whitespace
    cleaned = re.sub(r'[$,\s]', '', amount_str.strip())
    
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def parse_date(date_input: Any) -> Optional[str]:
    """Parse dates from various formats and standardize to YYYY-MM-DD"""
    if not date_input or date_input == "":
        return None
    
    date_str = str(date_input).strip()
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',              # 2023-01-15
        '%m/%d/%Y',              # 01/20/2023
        '%Y-%m-%dT%H:%M:%S',     # 2023-01-15T10:30:00
        '%Y-%m-%dT%H:%M:%S.%fZ', # 2023-01-05T08:00:00.000Z
        '%Y-%m-%dT%H:%M:%SZ',    # 2023-02-10T14:25:00Z
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None

def normalize_phone(phone_input: Any) -> Optional[str]:
    """Normalize phone numbers to a consistent format"""
    if not phone_input or phone_input == "" or phone_input is None:
        return None
    
    phone_str = str(phone_input)
    
    # Extract only digits
    digits = re.sub(r'\D', '', phone_str)
    
    # Format as XXX-XXXX
    if len(digits) >= 7:
        return f"{digits[-7:-4]}-{digits[-4:]}"
    else:
        return digits if digits else None

def normalize_email(email_input: Any) -> Optional[str]:
    """Normalize email addresses"""
    if not email_input or email_input == "" or email_input is None:
        return None
    
    email = str(email_input).strip().lower()
    
    # Basic validation - check if it looks like an email
    if '@' in email and '.' in email.split('@')[-1]:
        return email
    else:
        return None  # Invalid email

def normalize_status(status_input: Any) -> str:
    """Normalize status values"""
    if not status_input or status_input == "":
        return "unknown"
    
    status = str(status_input).strip().lower()
    
    # Map various statuses to standard values
    status_map = {
        'active': 'active',
        'inactive': 'inactive',
        'pending': 'pending',
        'unknown': 'unknown'
    }
    
    return status_map.get(status, 'unknown')

def normalize_category(category_input: Any) -> str:
    """Normalize category values"""
    if not category_input or category_input == "":
        return "Uncategorized"
    
    category = str(category_input).strip()
    
    # Standardize to Title Case
    return category.title()

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def safe_get(dictionary: Dict, *keys, default=None) -> Any:
    """Safely navigate nested dictionary"""
    result = dictionary
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result if result != "" else default

def ingest_json(json_file: str):
    """Ingest and parse the messy JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    customers = []
    orders = []
    items = []
    
    customer_list = data.get('customers', [])
    
    for idx, customer in enumerate(customer_list):
        # Extract customer data with fallbacks
        customer_id = customer.get('id')
        if customer_id is None or customer_id == "":
            customer_id = f"MISSING_{idx}"
        else:
            customer_id = str(customer_id)
        
        name = customer.get('name', '').strip()
        
        # Navigate nested contact info
        contact = customer.get('contact', {})
        email = normalize_email(safe_get(contact, 'email'))
        phone = normalize_phone(safe_get(contact, 'phone'))
        
        # Address info
        address = contact.get('address', {}) if isinstance(contact.get('address'), dict) else {}
        street = safe_get(address, 'street', default='')
        city = safe_get(address, 'city', default='').strip().title()
        state = safe_get(address, 'state', default='').strip().upper()
        zip_code = str(safe_get(address, 'zip', default=''))
        
        reg_date = parse_date(customer.get('registration'))
        status = normalize_status(customer.get('status'))
        loyalty_points = safe_int(customer.get('loyalty_points', 0))
        
        # Process purchases
        purchases = customer.get('purchases')
        if purchases is None:
            purchases = []
        
        order_count = 0
        total_spent = 0.0
        total_items = 0
        
        for purchase in purchases:
            if not isinstance(purchase, dict):
                continue
            
            order_id = purchase.get('id', f"{customer_id}-ORD-{order_count}")
            order_date = parse_date(purchase.get('date'))
            order_total = parse_amount(purchase.get('total', 0))
            payment_method = purchase.get('payment_method', 'unknown')
            
            # Process items in the order
            purchase_items = purchase.get('items', [])
            order_item_count = 0
            
            for item in purchase_items:
                if not isinstance(item, dict):
                    continue
                
                item_name = item.get('name', 'Unknown Item')
                item_price = parse_amount(item.get('price', 0))
                item_qty = safe_int(item.get('qty', 1))
                item_category = normalize_category(item.get('category', 'Uncategorized'))
                
                items.append({
                    'customer_id': customer_id,
                    'order_id': order_id,
                    'item_name': item_name,
                    'price': item_price,
                    'quantity': item_qty,
                    'category': item_category,
                    'line_total': round(item_price * item_qty, 2)
                })
                
                order_item_count += item_qty
            
            orders.append({
                'customer_id': customer_id,
                'order_id': order_id,
                'order_date': order_date,
                'order_total': order_total,
                'payment_method': payment_method.replace('_', ' ').title(),
                'item_count': order_item_count
            })
            
            order_count += 1
            total_spent += order_total
            total_items += order_item_count
        
        customers.append({
            'customer_id': customer_id,
            'name': name,
            'email': email,
            'phone': phone,
            'street': street,
            'city': city,
            'state': state,
            'zip': zip_code,
            'registration_date': reg_date,
            'status': status,
            'loyalty_points': loyalty_points,
            'order_count': order_count,
            'total_spent': round(total_spent, 2),
            'total_items_purchased': total_items
        })
    
    return pd.DataFrame(customers), pd.DataFrame(orders), pd.DataFrame(items)

def analyze_data(customers_df, orders_df, items_df):
    """Perform analysis and answer ambiguous questions"""
    
    print("=" * 80)
    print("DATA QUALITY SUMMARY")
    print("=" * 80)
    
    # Data quality metrics
    total_customers = len(customers_df)
    missing_email = customers_df['email'].isna().sum()
    missing_phone = customers_df['phone'].isna().sum()
    missing_address = ((customers_df['street'] == '') | (customers_df['city'] == '')).sum()
    no_orders = (customers_df['order_count'] == 0).sum()
    
    print(f"\nTotal Customers: {total_customers}")
    print(f"Missing Email: {missing_email} ({missing_email/total_customers*100:.1f}%)")
    print(f"Missing Phone: {missing_phone} ({missing_phone/total_customers*100:.1f}%)")
    print(f"Incomplete Address: {missing_address} ({missing_address/total_customers*100:.1f}%)")
    print(f"Customers with No Orders: {no_orders} ({no_orders/total_customers*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("CUSTOMER OVERVIEW")
    print("=" * 80)
    print(customers_df[['customer_id', 'name', 'email', 'phone', 'city', 'state', 
                        'status', 'order_count', 'total_spent', 'loyalty_points']].to_string(index=False))
    
    if not orders_df.empty:
        print("\n" + "=" * 80)
        print("ORDER SUMMARY")
        print("=" * 80)
        print(orders_df.to_string(index=False))
        
        if not items_df.empty:
            print("\n" + "=" * 80)
            print("ITEMS ANALYSIS")
            print("=" * 80)
            
            # Category breakdown
            category_summary = items_df.groupby('category').agg({
                'line_total': 'sum',
                'quantity': 'sum',
                'item_name': 'count'
            }).round(2)
            category_summary.columns = ['Total_Sales', 'Units_Sold', 'Num_Transactions']
            category_summary = category_summary.sort_values('Total_Sales', ascending=False)
            print("\nSales by Category:")
            print(category_summary.to_string())
    
    print("\n" + "=" * 80)
    print("AMBIGUOUS QUESTIONS WITH JUDGMENT CALLS")
    print("=" * 80)
    
    # Question 1: Who are our "VIP" customers?
    print("\n1. Who are our VIP customers?")
    print("   (Judgment: Consider spending, frequency, and loyalty points)")
    
    # Create a composite VIP score
    max_spent = customers_df['total_spent'].max() if customers_df['total_spent'].max() > 0 else 1
    max_orders = customers_df['order_count'].max() if customers_df['order_count'].max() > 0 else 1
    max_points = customers_df['loyalty_points'].max() if customers_df['loyalty_points'].max() > 0 else 1
    
    customers_df['vip_score'] = (
        (customers_df['total_spent'] / max_spent) * 40 +
        (customers_df['order_count'] / max_orders) * 30 +
        (customers_df['loyalty_points'] / max_points) * 30
    )
    
    vip_threshold = 60
    vips = customers_df[customers_df['vip_score'] >= vip_threshold].sort_values('vip_score', ascending=False)
    
    if not vips.empty:
        print(f"\n   Found {len(vips)} VIP customers (score >= {vip_threshold}):")
        for _, customer in vips.iterrows():
            print(f"   - {customer['name']}: ${customer['total_spent']:.2f} spent, "
                  f"{customer['order_count']} orders, {customer['loyalty_points']} points "
                  f"(VIP score: {customer['vip_score']:.1f})")
    else:
        print(f"   No customers meet the VIP threshold (>= {vip_threshold})")
    
    # Question 2: Which customers need profile updates?
    print("\n2. Which customers need PROFILE UPDATES?")
    print("   (Judgment: Prioritize active customers with missing critical info)")
    
    needs_update = customers_df[
        (customers_df['status'] == 'active') &
        ((customers_df['email'].isna()) | 
         (customers_df['phone'].isna()) | 
         (customers_df['city'] == '') |
         (customers_df['state'] == ''))
    ].copy()
    
    print(f"\n   Found {len(needs_update)} active customers needing updates:")
    for _, customer in needs_update.iterrows():
        missing = []
        if pd.isna(customer['email']):
            missing.append('email')
        if pd.isna(customer['phone']):
            missing.append('phone')
        if customer['city'] == '' or customer['state'] == '':
            missing.append('address')
        
        priority = "HIGH" if customer['order_count'] > 0 else "MEDIUM"
        print(f"   - {customer['name']}: Missing {', '.join(missing)} "
              f"({customer['order_count']} orders) - Priority: {priority}")
    
    # Question 3: Should we launch a re-engagement campaign?
    print("\n3. Should we launch a RE-ENGAGEMENT campaign?")
    print("   (Judgment: Target inactive customers with purchase history)")
    
    reengagement_targets = customers_df[
        ((customers_df['status'] == 'inactive') | (customers_df['status'] == 'pending')) &
        (customers_df['total_spent'] > 0)
    ].copy()
    
    # Also include active customers who haven't ordered recently (defined as only 1 order)
    low_engagement = customers_df[
        (customers_df['status'] == 'active') &
        (customers_df['order_count'] <= 1) &
        (customers_df['total_spent'] > 0)
    ].copy()
    
    print(f"\n   Re-engagement Targets:")
    print(f"   - Inactive with purchase history: {len(reengagement_targets)}")
    print(f"   - Low engagement (active but ≤1 order): {len(low_engagement)}")
    
    print(f"\n   Total potential: {len(reengagement_targets) + len(low_engagement)} customers")
    print(f"   Combined lifetime value: ${reengagement_targets['total_spent'].sum() + low_engagement['total_spent'].sum():.2f}")
    
    if len(reengagement_targets) + len(low_engagement) > 0:
        print(f"\n   ✓ RECOMMENDATION: Yes, launch re-engagement campaign")
        print(f"     Potential recovery of {len(reengagement_targets) + len(low_engagement)} customers")
    else:
        print(f"\n   ✗ RECOMMENDATION: No immediate need for campaign")
    
    # Question 4: What's our best-performing category?
    if not items_df.empty:
        print("\n4. What's our BEST-PERFORMING category?")
        print("   (Judgment: Consider revenue, volume, and customer reach)")
        
        category_metrics = items_df.groupby('category').agg({
            'line_total': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).round(2)
        category_metrics.columns = ['Revenue', 'Units_Sold', 'Unique_Customers']
        
        # Calculate composite performance score
        max_rev = category_metrics['Revenue'].max() if category_metrics['Revenue'].max() > 0 else 1
        max_units = category_metrics['Units_Sold'].max() if category_metrics['Units_Sold'].max() > 0 else 1
        max_customers = category_metrics['Unique_Customers'].max() if category_metrics['Unique_Customers'].max() > 0 else 1
        
        category_metrics['Performance_Score'] = (
            (category_metrics['Revenue'] / max_rev) * 50 +
            (category_metrics['Units_Sold'] / max_units) * 25 +
            (category_metrics['Unique_Customers'] / max_customers) * 25
        ).round(1)
        
        category_metrics = category_metrics.sort_values('Performance_Score', ascending=False)
        
        print("\n   Category Performance:")
        print(category_metrics.to_string())
        
        best = category_metrics.index[0]
        print(f"\n   ✓ WINNER: {best}")
        print(f"     (Balanced score considering revenue, volume, and reach)")
    
    # Question 5: Are loyalty points being used effectively?
    print("\n5. Are LOYALTY POINTS being used effectively?")
    print("   (Judgment: Analyze correlation between points and spending)")
    
    active_customers = customers_df[customers_df['status'] == 'active'].copy()
    
    if len(active_customers) > 0:
        avg_points_per_dollar = (active_customers['loyalty_points'].sum() / 
                                 active_customers['total_spent'].sum()) if active_customers['total_spent'].sum() > 0 else 0
        
        # Check if high spenders have proportional points
        high_spenders = active_customers.nlargest(3, 'total_spent')
        
        print(f"\n   Average points per dollar spent: {avg_points_per_dollar:.2f}")
        print(f"\n   Top 3 Spenders vs. Points:")
        for _, customer in high_spenders.iterrows():
            expected_points = customer['total_spent'] * avg_points_per_dollar
            points_ratio = (customer['loyalty_points'] / expected_points * 100) if expected_points > 0 else 0
            print(f"   - {customer['name']}: ${customer['total_spent']:.2f} spent, "
                  f"{customer['loyalty_points']} points ({points_ratio:.0f}% of expected)")
        
        # Overall assessment
        correlation_quality = "STRONG" if avg_points_per_dollar > 1 else "WEAK"
        print(f"\n   ✓ ASSESSMENT: {correlation_quality} correlation between spending and points")
        if correlation_quality == "WEAK":
            print(f"     Consider adjusting points allocation formula")
    
    return customers_df

if __name__ == "__main__":
    print("MESSY JSON DATA INGESTION AND WRANGLING")
    print("=" * 80)
    
    # Ingest the data
    customers_df, orders_df, items_df = ingest_json('/home/claude/messy_data.json')
    
    # Analyze and answer ambiguous questions
    customers_df = analyze_data(customers_df, orders_df, items_df)
    
    # Save cleaned data
    customers_df.to_csv('/home/claude/cleaned_customers.csv', index=False)
    orders_df.to_csv('/home/claude/cleaned_orders.csv', index=False)
    items_df.to_csv('/home/claude/cleaned_items.csv', index=False)
    
    print("\n" + "=" * 80)
    print("CLEANED DATA SAVED")
    print("=" * 80)
    print("- cleaned_customers.csv")
    print("- cleaned_orders.csv")
    print("- cleaned_items.csv")