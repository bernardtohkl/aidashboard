import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="AI Discovery Survey Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data():
    """Load and process the AI Discovery survey data"""
    # Load the CSV file - handle both local and deployment paths
    import os
    csv_path = 'data/AI_Discovery_Responses.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'AI_Discovery_Responses.csv')
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Rename columns for easier handling
    column_mapping = {
        'Submitted By': 'name',
        'Q1:  SCG Function - Which group are you from ?': 'function',
        'Q2. Top 3 Time-Intensive Repetitive Tasks: (Select the top 3 tasks that consumes MOST time in your typical work week)': 'top_tasks',
        'Q2a. For the identified tasks above, estimate what percentage of your time (in a week) you spend working on them.': 'time_percentage',
        'Q2b. Do you use automation/AI tools to perform the identified time-consuming tasks above?': 'uses_automation',
        'If you answered \'Yes\', please specify what automation/AI tools and the task that it is currently used for.': 'automation_tools',
        'Q3. AI Tools Familiar With: Do you use any AI tools for your work tasks? ': 'ai_tools_used',
        'Q4. AI Tool Usage: How frequently do you currently use AI tools in your work?': 'usage_frequency',
        'Q5. Current Proficiency Level: How would you rate your current proficiency level with AI tools?': 'proficiency_level',
        'Q6. Current Challenges:  When using AI tools for work, what are your biggest challenges? (Select all that apply)': 'challenges',
        'Q7. Skillset Needs: Which AI prompt engineering skills would help you most in your daily work? (Select up to 3)': 'skill_needs',
        'Q8. Future Possibilities:  Which areas of GT\'s corporate functions do you think AI can drive impact and effectiveness?': 'future_possibilities'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Clean and process the data
    df['time_percentage'] = pd.to_numeric(df['time_percentage'], errors='coerce')
    df['uses_automation'] = df['uses_automation'].fillna('No')
    
    # Clean function names
    df['function'] = df['function'].str.strip()
    
    return df

def create_overview_metrics(df):
    """Create overview metrics for the dashboard"""
    total_responses = len(df)
    avg_time_spent = df['time_percentage'].mean()
    automation_users = len(df[df['uses_automation'] == 'Yes'])
    automation_rate = (automation_users / total_responses) * 100
    
    return total_responses, avg_time_spent, automation_users, automation_rate

def create_function_breakdown(df):
    """Create function-level breakdown"""
    function_stats = df.groupby('function').agg({
        'name': 'count',
        'time_percentage': 'mean',
        'uses_automation': lambda x: (x == 'Yes').sum()
    }).reset_index()
    
    function_stats.columns = ['Function', 'Response_Count', 'Avg_Time_Percentage', 'Automation_Users']
    function_stats['Automation_Rate'] = (function_stats['Automation_Users'] / function_stats['Response_Count']) * 100
    
    return function_stats

def plot_proficiency_distribution(df, selected_function=None):
    """Plot proficiency level distribution"""
    if selected_function and selected_function != 'All Functions':
        plot_df = df[df['function'] == selected_function]
        title = f"AI Proficiency Distribution - {selected_function}"
    else:
        plot_df = df
        title = "AI Proficiency Distribution - All Functions"
    
    proficiency_counts = plot_df['proficiency_level'].value_counts()
    
    fig = px.pie(
        values=proficiency_counts.values,
        names=proficiency_counts.index,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig

def plot_usage_frequency(df, selected_function=None):
    """Plot usage frequency distribution"""
    if selected_function and selected_function != 'All Functions':
        plot_df = df[df['function'] == selected_function]
        title = f"AI Usage Frequency - {selected_function}"
    else:
        plot_df = df
        title = "AI Usage Frequency - All Functions"
    
    freq_counts = plot_df['usage_frequency'].value_counts()
    
    fig = px.bar(
        x=freq_counts.index,
        y=freq_counts.values,
        title=title,
        labels={'x': 'Usage Frequency', 'y': 'Number of Respondents'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def plot_top_challenges(df, selected_function=None):
    """Plot top challenges faced by users"""
    if selected_function and selected_function != 'All Functions':
        plot_df = df[df['function'] == selected_function]
        title = f"Top AI Challenges - {selected_function}"
    else:
        plot_df = df
        title = "Top AI Challenges - All Functions"
    
    # Process challenges (they're comma-separated)
    all_challenges = []
    for challenges in plot_df['challenges'].dropna():
        if isinstance(challenges, str):
            challenge_list = [c.strip() for c in challenges.split(',')]
            all_challenges.extend(challenge_list)
    
    challenge_counts = pd.Series(all_challenges).value_counts().head(10)
    
    fig = px.bar(
        y=challenge_counts.index,
        x=challenge_counts.values,
        title=title,
        labels={'x': 'Number of Mentions', 'y': 'Challenges'},
        orientation='h',
        color_discrete_sequence=['#ff7f0e']
    )
    
    return fig

def plot_automation_usage(df):
    """Plot automation usage by function"""
    function_stats = create_function_breakdown(df)
    
    fig = px.bar(
        function_stats,
        x='Function',
        y='Automation_Rate',
        title='Automation Usage Rate by Function',
        labels={'Automation_Rate': 'Automation Usage Rate (%)', 'Function': 'Function'},
        color='Automation_Rate',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def main():
    """Main dashboard function"""
    st.title("🤖 AI Discovery Survey Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_and_process_data()
    
    # Create tabs for each function
    functions = sorted(df['function'].unique().tolist())
    tab_names = ['📊 Overview'] + [f"📁 {func}" for func in functions]
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.header("Overall Survey Analysis")
        
        # Overview metrics
        total_responses, avg_time_spent, automation_users, automation_rate = create_overview_metrics(df)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Responses", total_responses)
        
        with col2:
            st.metric("Avg Time on Repetitive Tasks", f"{avg_time_spent:.1f}%")
        
        with col3:
            st.metric("Automation Users", automation_users)
        
        with col4:
            st.metric("Automation Rate", f"{automation_rate:.1f}%")
        
        st.markdown("---")
        
        # Function breakdown table
        st.subheader("📊 Function Breakdown")
        function_stats = create_function_breakdown(df)
        st.dataframe(function_stats, use_container_width=True)
        st.markdown("---")
        
        # Overview visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_proficiency_distribution(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_usage_frequency(df), use_container_width=True)
        
        # Additional overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_top_challenges(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_automation_usage(df), use_container_width=True)
        
        # Overall detailed data view
        st.markdown("---")
        st.subheader("📋 All Survey Responses")
        display_columns = ['name', 'function', 'time_percentage', 'uses_automation', 'proficiency_level', 'usage_frequency']
        st.dataframe(df[display_columns], use_container_width=True)
    
    # Function-specific tabs
    for i, function in enumerate(functions):
        with tabs[i + 1]:
            st.header(f"{function} Function Analysis")
            
            # Filter data for this function
            func_df = df[df['function'] == function]
            
            # Function-specific metrics
            func_responses = len(func_df)
            func_avg_time = func_df['time_percentage'].mean()
            func_automation_users = len(func_df[func_df['uses_automation'] == 'Yes'])
            func_automation_rate = (func_automation_users / func_responses) * 100 if func_responses > 0 else 0
            
            # Display function metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Function Responses", func_responses)
            
            with col2:
                st.metric("Avg Time on Tasks", f"{func_avg_time:.1f}%" if not pd.isna(func_avg_time) else "N/A")
            
            with col3:
                st.metric("Automation Users", func_automation_users)
            
            with col4:
                st.metric("Automation Rate", f"{func_automation_rate:.1f}%")
            
            st.markdown("---")
            
            # Function-specific visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plot_proficiency_distribution(df, function), use_container_width=True)
            
            with col2:
                st.plotly_chart(plot_usage_frequency(df, function), use_container_width=True)
            
            # Additional function-specific charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plot_top_challenges(df, function), use_container_width=True)
            
            with col2:
                # Time percentage distribution for this function
                if len(func_df) > 0:
                    fig = px.histogram(
                        func_df,
                        x='time_percentage',
                        title=f'Time Spent on Repetitive Tasks - {function}',
                        labels={'time_percentage': 'Time Percentage (%)', 'count': 'Number of Respondents'},
                        nbins=max(5, min(10, len(func_df)//2)),
                        color_discrete_sequence=['#17becf']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for this function")
            
            # Function-specific detailed data
            st.markdown("---")
            st.subheader(f"📋 {function} Detailed Responses")
            if len(func_df) > 0:
                display_columns = ['name', 'time_percentage', 'uses_automation', 'proficiency_level', 'usage_frequency']
                st.dataframe(func_df[display_columns], use_container_width=True)
                
                # Show specific insights for this function
                st.subheader("🔍 Function Insights")
                
                # Most common tools used
                if 'ai_tools_used' in func_df.columns:
                    tools_used = []
                    for tools in func_df['ai_tools_used'].dropna():
                        if isinstance(tools, str):
                            tool_list = [t.strip() for t in tools.split(',')]
                            tools_used.extend(tool_list)
                    
                    if tools_used:
                        tool_counts = pd.Series(tools_used).value_counts().head(5)
                        st.write("**Top AI Tools Used:**")
                        for tool, count in tool_counts.items():
                            st.write(f"• {tool}: {count} mentions")
                
                # Future possibilities mentioned
                if 'future_possibilities' in func_df.columns:
                    future_mentions = func_df['future_possibilities'].dropna()
                    if len(future_mentions) > 0:
                        st.write("**Future AI Applications Mentioned:**")
                        for i, mention in enumerate(future_mentions.head(3)):
                            if isinstance(mention, str) and len(mention.strip()) > 0:
                                st.write(f"• {mention[:200]}{'...' if len(mention) > 200 else ''}")
            else:
                st.info(f"No responses found for {function}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard generated from AI Discovery Survey responses*")

if __name__ == "__main__":
    main()