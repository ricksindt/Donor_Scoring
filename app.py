import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Function to convert capacity ratings
def convert_capacity(capacity):
    if pd.isna(capacity) or capacity == 'Unknown' or capacity == 'Unable to rate':
        return 'Unknown'
    elif capacity == 'Principal' or '6-Figure' in capacity:
        return 'Principal'
    elif any(x in capacity for x in ['500,000,001', '50,000,001', '25,000,001', '10,000,001', '5,000,001']):
        return '5,000,001+'
    elif '1,000,001' in capacity or '2,500,001' in capacity:
        return '1,000,001-5,000,000'
    elif '250,001' in capacity or '500,001' in capacity:
        return '250,001-1,000,000'
    else:
        return '250,000 or less'

# Scoring functions
def score_total_giving(amount):
    if amount >= 1000000:
        return 100
    elif amount >= 500000:
        return 80
    elif amount >= 250000:
        return 60
    elif amount >= 200000:
        return 40
    elif amount >= 100000:
        return 10
    else:
        return 0

def score_capacity(capacity):
    if capacity == 'Principal' or capacity == '5,000,001+':
        return 100
    elif capacity == '1,000,001-5,000,000':
        return 80
    elif capacity == '250,001-1,000,000':
        return 60
    elif capacity == '250,000 or less':
        return 20
    else:  # 'Unknown'
        return 0

def score_years_given(years):
    return max(100 - (years * 5), 0)

def score_consecutive_years(years):
    return max(100 - (years * 10), 0)

def score_recency(days):
    if pd.isna(days) or days > 1460:  # More than 4 years
        return 100
    if days <= 365:  # 0-1 year
        return 0
    elif days <= 730:  # 1-2 years
        return 25
    elif days <= 1095:  # 2-3 years
        return 50
    else:  # 3-4 years
        return 75

def score_recency_inverse(days):
    if pd.isna(days) or days > 1460:  # More than 4 years
        return 100
    if days <= 730:  # 0-2 years
        return 0
    elif days <= 1095:  # 2-3 years
        return 50
    else:  # 3-4 years
        return 75

def score_prospect_status(status):
    if status == 'Stewardship':
        return 100
    elif status == 'Not a Prospect Now':
        return 50
    else:
        return 0

def score_mg_evi(evi):
    return min(evi, 100)

def score_volunteer_positions(row):
    positions = [
        row['Primary Volunteer Position 1'], row['Primary Volunteer Position 2'],
        row['Primary Volunteer Position 3'], row['Primary Volunteer Position 4'],
        row['Secondary Volunteer Position 1'], row['Secondary Volunteer Position 2'],
        row['Secondary Volunteer Position 3'], row['Secondary Volunteer Position 4']
    ]
    filled_positions = sum(1 for pos in positions if pd.notna(pos) and pos != '')
    return 100 if filled_positions == 0 else 0

def calculate_total_score(row, weights):
    return sum(row[col] * (weight / 100) for col, weight in weights.items())

def calculate_action_score(row):
    if row['Eligible for DRO? (Capacity, Giving, RM)'].lower() in ['yes', 'maybe']:
        if row['Final Score'] >= 80:
            return 'Extreme'
        elif 60 <= row['Final Score'] < 80:
            return 'High'
        elif 40 <= row['Final Score'] < 60:
            return 'Moderate'
        else:
            return 'Low'
    return 'Not Required'

# Main function to process the data
def process_data(df, custom_weights):
    # Apply the conversion to a new column
    df['Converted Account Capacity'] = df['Overall Account Capacity'].apply(convert_capacity)

    # Select relevant columns
    relevant_columns = [
        'Account ID', 'Account Name', 'Total Giving Including Pre-66', 
        'Overall Account Capacity', 'Converted Account Capacity', 'Total Years Given', 'Consecutive Years Given',
        'Date of Last Gift', 'Date of Last Substantive Activity Report',
        'Date of Last Event Attended - Primary', 'Prospect Status', 'MG EVI',
        'Recurring Donor Ever', 'AutoPay Donor Ever', 'Do Not Solicit',
        'Primary Gift Officer',
        'Primary Volunteer Position 1', 'Primary Volunteer Position 2',
        'Primary Volunteer Position 3', 'Primary Volunteer Position 4',
        'Secondary Volunteer Position 1', 'Secondary Volunteer Position 2',
        'Secondary Volunteer Position 3', 'Secondary Volunteer Position 4',
        'Eligible for DRO? (Capacity, Giving, RM)'
    ]

    df_scoring = df[relevant_columns].copy()

    # Handle null values
    df_scoring['Prospect Status'] = df_scoring['Prospect Status'].fillna('Stewardship')
    df_scoring['Recurring Donor Ever'] = df_scoring['Recurring Donor Ever'].fillna('N')
    df_scoring['AutoPay Donor Ever'] = df_scoring['AutoPay Donor Ever'].fillna('N')
    df_scoring['Do Not Solicit'] = df_scoring['Do Not Solicit'].fillna('N')
    df_scoring['MG EVI'] = df_scoring['MG EVI'].fillna(0)
    df_scoring['Eligible for DRO? (Capacity, Giving, RM)'] = df_scoring['Eligible for DRO? (Capacity, Giving, RM)'].fillna('No')
    df_scoring['Primary Gift Officer'] = df_scoring['Primary Gift Officer'].fillna('')

    # Convert date columns and calculate recency
    date_columns = ['Date of Last Gift', 'Date of Last Substantive Activity Report', 'Date of Last Event Attended - Primary']
    current_date = pd.Timestamp.now()

    for col in date_columns:
        df_scoring[col] = pd.to_datetime(df_scoring[col], errors='coerce')
        df_scoring[f'{col} Recency'] = (current_date - df_scoring[col]).dt.days.fillna(10000)  # High value for missing dates

    # Apply scoring functions
    df_scoring['Giving Score'] = df_scoring['Total Giving Including Pre-66'].apply(score_total_giving)
    df_scoring['Capacity Score'] = df_scoring['Converted Account Capacity'].apply(score_capacity)
    df_scoring['Years Given Score'] = df_scoring['Total Years Given'].apply(score_years_given)
    df_scoring['Consecutive Years Score'] = df_scoring['Consecutive Years Given'].apply(score_consecutive_years)
    df_scoring['Last Gift Recency Score'] = df_scoring['Date of Last Gift Recency'].apply(score_recency)
    df_scoring['Last Activity Recency Score'] = df_scoring['Date of Last Substantive Activity Report Recency'].apply(score_recency_inverse)
    df_scoring['Last Event Recency Score'] = df_scoring['Date of Last Event Attended - Primary Recency'].apply(score_recency_inverse)
    df_scoring['Prospect Status Score'] = df_scoring['Prospect Status'].apply(score_prospect_status)
    df_scoring['MG EVI Score'] = df_scoring['MG EVI'].apply(score_mg_evi)
    df_scoring['Volunteer Position Score'] = df_scoring.apply(score_volunteer_positions, axis=1)
    df_scoring['Recurring Donor Score'] = df_scoring['Recurring Donor Ever'].map({'Y': 100, 'N': 0})
    df_scoring['AutoPay Donor Score'] = df_scoring['AutoPay Donor Ever'].map({'Y': 100, 'N': 0})

    # Calculate total score
    df_scoring['Total Score'] = df_scoring.apply(lambda row: calculate_total_score(row, custom_weights), axis=1)

    # Adjust score for Do Not Solicit
    df_scoring['Final Score'] = df_scoring['Total Score'] * (1 - df_scoring['Do Not Solicit'].map({'Y': 1, 'N': 0}))

    # Sort and rank donors
    df_scoring = df_scoring.sort_values('Final Score', ascending=False)
    df_scoring['Donor Rank'] = df_scoring['Final Score'].rank(method='dense', ascending=False)

    # Calculate Action Score
    df_scoring['Action Score'] = df_scoring.apply(calculate_action_score, axis=1)

    # Select columns for final output
    final_columns = [
        'Account ID', 'Account Name', 'Total Giving Including Pre-66', 
        'Overall Account Capacity', 'Converted Account Capacity', 'Total Years Given', 'Consecutive Years Given',
        'Date of Last Gift', 'Date of Last Substantive Activity Report',
        'Date of Last Event Attended - Primary', 'Prospect Status', 'MG EVI',
        'Recurring Donor Ever', 'AutoPay Donor Ever', 'Do Not Solicit',
        'Primary Gift Officer',
        'Eligible for DRO? (Capacity, Giving, RM)',
        'Final Score', 'Donor Rank', 'Action Score'
    ]

    return df_scoring[final_columns]

# Function to validate weights
def validate_weights(weights):
    return abs(sum(weights.values()) - 100) < 0.01  # Allow for small floating point errors

# Streamlit app
def main():
    st.set_page_config(page_title="Donor Scoring System", page_icon="📊", layout="wide")
    
    # Custom CSS to improve the app's appearance
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Donor Scoring System</p>', unsafe_allow_html=True)

    # Add a section for custom weights
    st.markdown('<p class="medium-font">Custom Metric Weights</p>', unsafe_allow_html=True)
    st.write("Adjust the weights for each metric. The total must add up to 100%.")

    default_weights = {
        'Giving Score': 10,
        'Capacity Score': 35,
        'Years Given Score': 10,
        'Consecutive Years Score': 10,
        'Last Gift Recency Score': 15,
        'Last Activity Recency Score': 5,
        'Last Event Recency Score': 5,
        'Prospect Status Score': 5,
        'MG EVI Score': 5,
        'Volunteer Position Score': 0,
        'Recurring Donor Score': -5,
        'AutoPay Donor Score': -5
    }

    custom_weights = {}
    col1, col2 = st.columns(2)
    for i, (metric, default_weight) in enumerate(default_weights.items()):
        with col1 if i % 2 == 0 else col2:
            custom_weights[metric] = st.number_input(f"{metric} (%)", min_value=-100, max_value=100, value=default_weight, step=1)

    total_weight = sum(custom_weights.values())
    st.write(f"Total weight: {total_weight}%")

    if not validate_weights(custom_weights):
        st.error("The total weight must equal 100%. Please adjust your weights.")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None and validate_weights(custom_weights):
        df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully. Processing...")

        # Process the data with custom weights
        results = process_data(df, custom_weights)

        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<p class="medium-font">Total Donors</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-font">{len(results):,}</p>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="medium-font">Average Final Score</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-font">{results["Final Score"].mean():.2f}</p>', unsafe_allow_html=True)
        with col3:
            st.markdown('<p class="medium-font">Median Final Score</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-font">{results["Final Score"].median():.2f}</p>', unsafe_allow_html=True)

        # Action Score Distribution
        st.markdown('<p class="medium-font">Action Score Distribution</p>', unsafe_allow_html=True)
        action_score_counts = results['Action Score'].value_counts()
        fig = px.pie(values=action_score_counts.values, names=action_score_counts.index, 
                     title='Action Score Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)

        # Final Score Distribution
        st.markdown('<p class="medium-font">Final Score Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(results, x="Final Score", nbins=50, 
                           title='Distribution of Final Scores', color_discrete_sequence=['#4F8BF9'])
        st.plotly_chart(fig)

        # Correlation between Total Giving and Final Score
        st.markdown('<p class="medium-font">Total Giving vs Final Score</p>', unsafe_allow_html=True)
        fig = px.scatter(results, x="Total Giving Including Pre-66", y="Final Score", 
                         hover_data=["Account Name"], title='Total Giving vs Final Score')
        st.plotly_chart(fig)

        # Top 10 Donors Table
        st.markdown('<p class="medium-font">Top 10 Donors</p>', unsafe_allow_html=True)
        top_10_donors = results.nlargest(10, 'Final Score')[['Account Name', 'Total Giving Including Pre-66', 'Final Score', 'Action Score']]
        st.table(top_10_donors)

        
        # Filters for data exploration
        st.markdown('<p class="medium-font">Data Exploration</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider('Minimum Final Score', 0, 100, 0)
        with col2:
            action_score_filter = st.multiselect('Action Score', options=results['Action Score'].unique(), default=results['Action Score'].unique())

        filtered_results = results[(results['Final Score'] >= min_score) & (results['Action Score'].isin(action_score_filter))]
        st.write(f"Filtered Donors: {len(filtered_results)}")
        st.dataframe(filtered_results)

        # Create a download button for the results
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results.to_excel(writer, index=False)
        output.seek(0)
        st.download_button(
            label="Download Full Results",
            data=output,
            file_name="donor_scoring_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()