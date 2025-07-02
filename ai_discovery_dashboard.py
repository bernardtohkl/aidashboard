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
    
    # Read CSV with error handling for malformed data
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()
    
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

def calculate_time_savings_potential(df):
    """Calculate potential time savings from automation"""
    total_weekly_hours = df['time_percentage'].sum()
    current_automation_hours = df[df['uses_automation'] == 'Yes']['time_percentage'].sum()
    manual_hours = total_weekly_hours - current_automation_hours
    
    # Potential savings with 50% automation of manual tasks
    potential_savings_50 = manual_hours * 0.5
    
    return {
        'total_weekly_hours': total_weekly_hours,
        'manual_hours': manual_hours,
        'current_automation_hours': current_automation_hours,
        'potential_savings_50': potential_savings_50,
        'automation_opportunity': manual_hours / total_weekly_hours * 100 if total_weekly_hours > 0 else 0
    }

def main():
    """Main dashboard function"""
    st.title("🚀 AI Discovery Survey Dashboard")
    st.markdown("### *Unlocking Automation Potential Across Corporate Functions*")
    
    # Load data
    df = load_and_process_data()
    
    # Calculate time savings potential
    savings_data = calculate_time_savings_potential(df)
    
    # Hero section with key insights
    st.markdown("---")
    st.markdown("## 💡 **Key Insights: The Automation Opportunity**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ff6b6b, #ffa726); padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">⚠️ {:.0f}%</h2>
            <p style="color: white; margin: 5px 0;"><strong>Manual Tasks</strong></p>
            <p style="color: white; font-size: 14px; margin: 0;">Still done manually</p>
        </div>
        """.format(savings_data['automation_opportunity']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #4ecdc4, #45b7d1); padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">💰 {:.0f} hrs</h2>
            <p style="color: white; margin: 5px 0;"><strong>Weekly Savings</strong></p>
            <p style="color: white; font-size: 14px; margin: 0;">With 50% automation</p>
        </div>
        """.format(savings_data['potential_savings_50']), unsafe_allow_html=True)
    
    with col3:
        annual_savings = savings_data['potential_savings_50'] * 52
        st.markdown("""
        <div style="background: linear-gradient(90deg, #96ceb4, #ffeaa7); padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">📈 {:.0f} hrs</h2>
            <p style="color: white; margin: 5px 0;"><strong>Annual Savings</strong></p>
            <p style="color: white; font-size: 14px; margin: 0;">Potential per year</p>
        </div>
        """.format(annual_savings), unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-left: 5px solid #17a2b8; border-radius: 5px;">
        <h3 style="color: #17a2b8; margin-top: 0;">🎯 The Opportunity</h3>
        <p><strong>Repetitive manual tasks</strong> are consuming significant time across all functions. 
        By implementing AI automation for just <strong>50% of these tasks</strong>, we could free up 
        <strong>{:.0f} hours weekly</strong> for higher-value strategic work.</p>
    </div>
    """.format(savings_data['potential_savings_50']), unsafe_allow_html=True)
    
    # Create tabs for each function
    functions = sorted(df['function'].unique().tolist())
    tab_names = ['📊 Overview'] + [f"📁 {func}" for func in functions]
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.header("📈 Overall Survey Analysis")
        
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
        
        # Automation Opportunity Visualization
        st.subheader("🎯 Automation Opportunity Analysis")
        
        # Create automation potential chart
        automation_data = []
        for func in df['function'].unique():
            func_df = df[df['function'] == func]
            total_time = func_df['time_percentage'].sum()
            manual_time = func_df[func_df['uses_automation'] == 'No']['time_percentage'].sum()
            automated_time = total_time - manual_time
            potential_savings = manual_time * 0.5
            
            automation_data.append({
                'Function': func,
                'Current Manual Hours': manual_time,
                'Already Automated': automated_time,
                'Potential Savings (50%)': potential_savings
            })
        
        automation_df = pd.DataFrame(automation_data)
        
        # Create stacked bar chart
        fig_automation = px.bar(
            automation_df.melt(id_vars=['Function'], 
                             value_vars=['Already Automated', 'Current Manual Hours', 'Potential Savings (50%)']),
            x='Function',
            y='value',
            color='variable',
            title='⚡ Time Allocation & Automation Potential by Function',
            labels={'value': 'Weekly Hours', 'variable': 'Category'},
            color_discrete_map={
                'Already Automated': '#2ecc71',
                'Current Manual Hours': '#e74c3c', 
                'Potential Savings (50%)': '#f39c12'
            }
        )
        fig_automation.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_automation, use_container_width=True)
        
        # Additional overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_top_challenges(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_automation_usage(df), use_container_width=True)
        
        # Interactive Time Savings Calculator
        st.markdown("---")
        st.subheader("🧮 Interactive Time Savings Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            automation_percentage = st.slider(
                "Automation Level (%)", 
                min_value=0, 
                max_value=100, 
                value=50, 
                step=5,
                help="Percentage of manual tasks to automate"
            )
            
            total_manual_hours = savings_data['manual_hours']
            calculated_savings = total_manual_hours * (automation_percentage / 100)
            annual_calculated_savings = calculated_savings * 52
            
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
                <h4>💡 Projected Savings</h4>
                <p><strong>Weekly:</strong> {calculated_savings:.0f} hours</p>
                <p><strong>Annual:</strong> {annual_calculated_savings:.0f} hours</p>
                <p><strong>FTE Equivalent:</strong> {annual_calculated_savings/2080:.1f} positions</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ROI Message
        if automation_percentage >= 50:
            st.success(f"🎉 With {automation_percentage}% automation, you could save {calculated_savings:.0f} hours weekly - equivalent to {annual_calculated_savings/2080:.1f} full-time positions annually!")
        elif automation_percentage >= 25:
            st.info(f"💼 {automation_percentage}% automation would free up {calculated_savings:.0f} hours weekly for strategic work.")
        else:
            st.warning(f"⚠️ Only {automation_percentage}% automation leaves significant opportunity on the table.")
        
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