import xml.etree.ElementTree as ET
import pandas as pd
import re
from datetime import datetime
from typing import Optional

def parse_amount(amount_str: Optional[str]) -> float:
    """Clean and parse monetary amounts from various formats"""
    if not amount_str or amount_str.strip() == "":
        return 0.0
    
    # Remove currency symbols, commas, and extra whitespace
    cleaned = re.sub(r'[$,\s]', '', amount_str.strip())
    
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def parse_date(date_str: Optional[str]) -> Optional[str]:
    """Parse dates from various formats and standardize to YYYY-MM-DD"""
    if not date_str or date_str.strip() == "":
        return None
    
    date_str = date_str.strip()
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',      # 2023-01-15
        '%m/%d/%Y',      # 01/20/2023
        '%d/%m/%Y',      # 15/01/2023
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None

def normalize_phone(phone_str: Optional[str]) -> Optional[str]:
    """Normalize phone numbers to a consistent format"""
    if not phone_str or phone_str.strip() == "":
        return None
    
    # Extract only digits
    digits = re.sub(r'\D', '', phone_str)
    
    # Format as XXX-XXXX if 7 digits, or include area code if more
    if len(digits) == 7:
        return f"555-{digits[-4:]}"
    elif len(digits) >= 10:
        return f"{digits[:3]}-{digits[3:7]}"
    else:
        return digits

def normalize_email(email_str: Optional[str]) -> Optional[str]:
    """Normalize email addresses"""
    if not email_str or email_str.strip() == "":
        return None
    
    email = email_str.strip().lower()
    
    # Basic validation - check if it looks like an email
    if '@' in email and '.' in email.split('@')[1]:
        return email
    else:
        return None  # Invalid email

def normalize_status(status_str: Optional[str]) -> str:
    """Normalize status values"""
    if not status_str or status_str.strip() == "":
        return "unknown"
    
    status = status_str.strip().lower()
    
    # Map various statuses to standard values
    status_map = {
        'active': 'active',
        'inactive': 'inactive',
        'pending': 'pending',
        'unknown': 'unknown'
    }
    
    return status_map.get(status, 'unknown')

def ingest_xml(xml_file: str):
    """Ingest and parse the messy XML file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    customers = []
    purchases = []
    
    for customer_elem in root.findall('customer'):
        # Extract customer data with fallbacks for missing values
        customer_id = customer_elem.get('id', f"MISSING_{len(customers)}")
        name = customer_elem.findtext('n', '').strip()
        email = normalize_email(customer_elem.findtext('email', ''))
        phone = normalize_phone(customer_elem.findtext('phone', ''))
        reg_date = parse_date(customer_elem.findtext('registration_date', ''))
        status = normalize_status(customer_elem.findtext('status', ''))
        
        # Extract purchase history
        purchase_history = customer_elem.find('purchase_history')
        purchase_count = 0
        total_spent = 0.0
        
        if purchase_history is not None:
            for purchase_elem in purchase_history.findall('purchase'):
                purchase_date = parse_date(purchase_elem.findtext('date', ''))
                amount = parse_amount(purchase_elem.findtext('amount', '0'))
                category = purchase_elem.findtext('category', 'Uncategorized').strip()
                
                # Normalize category capitalization
                category = category.title()
                
                purchases.append({
                    'customer_id': customer_id,
                    'purchase_date': purchase_date,
                    'amount': amount,
                    'category': category
                })
                
                purchase_count += 1
                total_spent += amount
        
        customers.append({
            'customer_id': customer_id,
            'name': name,
            'email': email,
            'phone': phone,
            'registration_date': reg_date,
            'status': status,
            'purchase_count': purchase_count,
            'total_spent': round(total_spent, 2)
        })
    
    return pd.DataFrame(customers), pd.DataFrame(purchases)

def analyze_data(customers_df, purchases_df):
    """Perform analysis and answer ambiguous questions"""
    
    print("=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    
    # Data quality metrics
    total_customers = len(customers_df)
    missing_email = customers_df['email'].isna().sum()
    missing_phone = customers_df['phone'].isna().sum()
    no_purchases = (customers_df['purchase_count'] == 0).sum()
    
    print(f"\nTotal Customers: {total_customers}")
    print(f"Missing Email: {missing_email} ({missing_email/total_customers*100:.1f}%)")
    print(f"Missing Phone: {missing_phone} ({missing_phone/total_customers*100:.1f}%)")
    print(f"Customers with No Purchases: {no_purchases} ({no_purchases/total_customers*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("CUSTOMER OVERVIEW")
    print("=" * 70)
    print(customers_df.to_string(index=False))
    
    if not purchases_df.empty:
        print("\n" + "=" * 70)
        print("PURCHASE HISTORY")
        print("=" * 70)
        print(purchases_df.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("PURCHASE ANALYSIS")
        print("=" * 70)
        
        # Category breakdown
        category_summary = purchases_df.groupby('category').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        category_summary.columns = ['Total_Sales', 'Avg_Sale', 'Num_Purchases']
        print("\nSales by Category:")
        print(category_summary.to_string())
    
    print("\n" + "=" * 70)
    print("AMBIGUOUS QUESTIONS WITH JUDGMENT CALLS")
    print("=" * 70)
    
    # Question 1: Who are our "valuable" customers?
    print("\n1. Who are our VALUABLE customers?")
    print("   (Judgment: Consider both spending and engagement)")
    
    # Create a composite score
    max_spent = customers_df['total_spent'].max()
    max_purchases = customers_df['purchase_count'].max()
    
    customers_df['value_score'] = 0
    if max_spent > 0:
        customers_df['value_score'] += (customers_df['total_spent'] / max_spent) * 50
    if max_purchases > 0:
        customers_df['value_score'] += (customers_df['purchase_count'] / max_purchases) * 50
    
    valuable = customers_df[customers_df['value_score'] > 50].sort_values('value_score', ascending=False)
    
    if not valuable.empty:
        print(f"\n   Found {len(valuable)} valuable customers (score > 50):")
        for _, customer in valuable.iterrows():
            print(f"   - {customer['name']}: ${customer['total_spent']:.2f} spent, "
                  f"{customer['purchase_count']} purchases (score: {customer['value_score']:.1f})")
    else:
        print("   No customers meet the 'valuable' threshold")
    
    # Question 2: Should we contact customers with incomplete profiles?
    print("\n2. Should we reach out to customers with INCOMPLETE profiles?")
    print("   (Judgment: Define 'incomplete' and consider engagement level)")
    
    incomplete = customers_df[
        (customers_df['email'].isna() | customers_df['phone'].isna()) &
        (customers_df['status'] == 'active')
    ]
    
    print(f"\n   Found {len(incomplete)} active customers with missing contact info:")
    for _, customer in incomplete.iterrows():
        missing = []
        if pd.isna(customer['email']):
            missing.append('email')
        if pd.isna(customer['phone']):
            missing.append('phone')
        
        recommendation = "HIGH PRIORITY" if customer['purchase_count'] > 0 else "LOW PRIORITY"
        print(f"   - {customer['name']}: Missing {', '.join(missing)} "
              f"({customer['purchase_count']} purchases) - {recommendation}")
    
    # Question 3: Which customers are at risk of churning?
    print("\n3. Which customers are AT RISK of churning?")
    print("   (Judgment: Consider inactivity and status)")
    
    # Customers who registered but never purchased, or are inactive
    at_risk = customers_df[
        ((customers_df['purchase_count'] == 0) & (customers_df['status'] == 'active')) |
        (customers_df['status'] == 'inactive')
    ]
    
    print(f"\n   Found {len(at_risk)} at-risk customers:")
    for _, customer in at_risk.iterrows():
        if customer['purchase_count'] == 0:
            reason = "Never purchased"
        else:
            reason = f"Inactive status (but spent ${customer['total_spent']:.2f})"
        
        print(f"   - {customer['name']}: {reason}")
    
    # Question 4: What's our most popular category?
    if not purchases_df.empty:
        print("\n4. What's our MOST POPULAR category?")
        print("   (Judgment: Define 'popular' - by count or revenue?)")
        
        by_count = purchases_df['category'].value_counts().iloc[0]
        by_revenue = purchases_df.groupby('category')['amount'].sum().idxmax()
        total_revenue = purchases_df.groupby('category')['amount'].sum().max()
        
        print(f"\n   By number of purchases: {purchases_df['category'].value_counts().index[0]} "
              f"({by_count} purchases)")
        print(f"   By revenue: {by_revenue} (${total_revenue:.2f})")
    
    return customers_df

if __name__ == "__main__":
    print("MESSY XML DATA INGESTION AND WRANGLING")
    print("=" * 70)
    
    # Ingest the data
    customers_df, purchases_df = ingest_xml('messy.xml')
    
    # Analyze and answer ambiguous questions
    customers_df = analyze_data(customers_df, purchases_df)
    
    # Save cleaned data
    customers_df.to_csv('cleaned_customers.csv', index=False)
    purchases_df.to_csv('cleaned_purchases.csv', index=False)
    
    print("\n" + "=" * 70)
    print("CLEANED DATA SAVED")
    print("=" * 70)
    print("- cleaned_customers.csv")
    print("- cleaned_purchases.csv")