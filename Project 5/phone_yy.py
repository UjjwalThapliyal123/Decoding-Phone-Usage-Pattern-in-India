import streamlit as st
from PIL import Image
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pickle

@st.cache_data
def get_file(filename):
    try:
        df=pd.read_csv(filename)
        return df
    except:
        st.error(f"File name '{filename}' not found!!")
        st.stop()
    
def get_model(modelfile):
    try:
        with open (modelfile,'rb') as f:
          model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model {modelfile} not Found!!")
        st.stop()

def get_encoded_file(encodedfile):
    try:
        with open (encodedfile,'rb') as f:
          encoded = pickle.load(f)
        return encoded
    except FileNotFoundError:
        st.error(f"Encoded File {encodedfile} not Found!!")
        st.stop()
        
@st.cache_data
def preprocess_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    pipeline = make_pipeline(scaler, pca)
    pca_result = pipeline.fit_transform(numeric_df)
    return pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
 
#Loading the Files and models        
df = get_file('Csv_File.xls')
cluster_df = get_file('Cluster_data.csv')
data = get_file('Phone.xls')
model = get_model('RandomForestClassifier_pipeline.pkl')
agglo = get_model('agglo.pkl')
encode = get_encoded_file('encoded_target.pkl')
df =df.loc[:,~df.columns.str.contains('^Unnamed')]


nav = st.sidebar.radio('Go to ',['Home','Visual Representation','Prediction','Clustering'])
if nav == 'Home':

    # Title
    np.random.seed(42)
    dates = pd.date_range(end=datetime.date.today(), periods=30)
    app_usage = np.random.uniform(50, 500, size=30)  # in MB per day

    usage_data = pd.DataFrame({
        'Date': dates,
        'Data_Used_MB': app_usage
    })

    # Calculate refined stats
    total_data_gb = usage_data['Data_Used_MB'].sum() / 1024
    avg_daily_usage_gb = usage_data['Data_Used_MB'].mean() / 1024
    peak_day = usage_data.loc[usage_data['Data_Used_MB'].idxmax()]
    peak_usage_gb = peak_day['Data_Used_MB'] / 1024

    # Title Section
    st.title(" Mobile Data Usage India")
    st.caption("Track \U0001F4C8 | Analyze \U0001F4CA | Optimize \U0001F527")

    st.divider()

    # Quick Overview
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Usage (30 Days)", f"{total_data_gb:.2f} GB")
    col2.metric("Average Daily Usage", f"{avg_daily_usage_gb:.2f} GB")
    col3.metric("Peak Usage Day", peak_day['Date'].strftime('%d %b %Y'), f"{peak_usage_gb:.2f} GB")

    st.divider()

    # Navigation Section
    st.subheader("\U0001F5FA Navigation")
    navigation = st.columns(3)

    with navigation[0]:
        st.button("\U0001F4C5 Daily Breakdown", key="daily_breakdown_btn")
    with navigation[1]:
        st.button("\U0001F4CA Trend Analysis", key="trend_analysis_btn")
    with navigation[2]:
        st.button("\U0001F4D1 Reports Summary", key="report_summary_btn")

    st.divider()

    # Preview Last 30 Days
    st.subheader(" Last 30 Days Usage")
    st.line_chart(usage_data.set_index('Date')['Data_Used_MB'].tail(30) / 1024)

    st.divider()

    # About Section
    with st.expander("\U0001F4DD About This Application"):
        st.markdown("""
        Welcome to the **Mobile Data Usage Dashboard** for India! 

        **Features:**
        - Visualize your daily and monthly mobile data trends.
        - Get insights on peak usage and optimize your plans.
        - Stay informed and avoid data overuse.

        **Privacy Guaranteed:**
        - Data stays on your device.
        - No external data sharing.

        **Made for India:**
        - Local formats and data trends considered.
        """)


if nav == 'Visual Representation':
        st.title("EDA")
        # Create a dropdown menu for selecting analysis options
        eda_options = [
            "Data Check",
            "Primary Use By the Different Gender",
            "Entertainment vs Screen Time",
            "Number of Phone Brand",
            "Apps Installed vs Different Age Groups",
            "Screen Time vs Call Duration",
            "Box plot for Screen Time by different Users",
            "Correlation among them",
            "Custom Pairplot"
        ]

        selected_options = st.multiselect("Select the analysis to perform", eda_options)

        # Data Check
        if "Data Check" in selected_options:
            st.subheader("Data Preview")
            st.dataframe(df)

        # Primary Use By Different Gender
        if "Primary Use By the Different Gender" in selected_options:
            st.subheader("Primary Use by Gender")
            plt.figure(figsize=(12, 5))
            sns.countplot(x='Gender', hue='Primary Use', data=df)
            st.pyplot(plt.gcf())

        # Entertainment vs Screen Time
        if "Entertainment vs Screen Time" in selected_options:
            st.subheader("Entertainment vs Screen Time")
            plt.figure(figsize=(12, 5))
            sns.scatterplot(x='Total Entertainment', y='Screen Time (hrs/day)', data=df)
            plt.title("Entertainment vs Screen Time (hrs/day)")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())  # Show scatterplot
            
            # Linear regression plot (lmplot)
            lm = sns.lmplot(
                x='Total Entertainment',
                y='Screen Time (hrs/day)',
                hue='Gender',
                data=df,
                aspect=2.0,  # wider
                height=5
            )
            plt.xticks(rotation=45)
            st.pyplot(lm.figure)

        # Number of Phone Brands
        if "Number of Phone Brand" in selected_options:
            st.subheader("Phone Brand Distribution")
            
            # Bar plot
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.countplot(x='Phone Brand', data=df, palette='pastel')
            ax1.set_title("Count of Each Brand")
            st.pyplot(fig1)
            
            # Pie chart
            st.subheader("Phone Brand Percentage")
            brand_counts = df['Phone Brand'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.pie(brand_counts, labels=brand_counts.index, autopct='%.2f%%', startangle=140)
            ax2.axis('equal')
            st.pyplot(fig2)

        # Apps Installed vs Different Age Groups
        if "Apps Installed vs Different Age Groups" in selected_options:
            st.subheader('Apps Installed by Different Age Groups')
            grouped = df.groupby('AgeGroup')['Number of Apps Installed'].mean()
            plt.figure(figsize=(12, 5))
            plt.bar(grouped.index, grouped.values, color='purple')
            plt.xlabel("Age Groups")
            plt.ylabel('Apps')
            st.pyplot(plt.gcf())

        # Screen Time vs Call Duration
        if "Screen Time vs Call Duration" in selected_options:
            st.subheader('Screen Time vs Call Duration')
            plt.figure(figsize=(10, 5))
            sns.lineplot(x='Screen Time (hrs/day)', y='Calls Duration (mins/day)', data=df)
            plt.xlabel('Screen Time')
            plt.ylabel('Call Duration')
            plt.title('Screen Time (hrs/day) vs Call Duration (mins/day)')
            st.pyplot(plt.gcf())

        # Box plot for Screen Time by different Users
        if "Box plot for Screen Time by different Users" in selected_options:
            st.subheader('Box plot for Screen Time by Gender')
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Screen Time (hrs/day)', hue='Gender', data=df)
            st.pyplot(plt.gcf())

        # Correlation Matrix
        if "Correlation among them" in selected_options:
            st.subheader("Correlation Matrix")
            num = df.select_dtypes(include='number')
            plt.figure(figsize=(10, 6))
            sns.heatmap(num.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt.gcf())

        # Custom Pairplot
        if "Custom Pairplot" in selected_options:
            st.subheader("Custom Pairplot")
            all_num_cols = df.select_dtypes(include='number').columns.tolist()
            selected = st.multiselect("Select columns for pairplot", all_num_cols, default=all_num_cols[:3])
            
            if len(selected) >= 2:
                fig = sns.pairplot(df[selected])
                st.pyplot(fig)
            else:
                st.warning("Please select at least 2 columns.")
elif nav == 'Prediction':
        st.title('Prediction')
        if st.checkbox("Show Data Sample"):
                st.dataframe(data)
        
        gender = st.selectbox("Gender",data['Gender'].unique())
        phone_brand = st.selectbox("Phone Brand",data['Phone Brand'].unique())
        os = st.selectbox("Operating System",data['OS'].unique())
        AgeGroup = st.selectbox("AgeGroup",data['AgeGroup'].unique())
        call_dur = st.select_slider("Call Duration min",options=sorted(data['Calls Duration (mins/day)'].unique()))
        Apps = st.select_slider("Number of Apps",options=sorted(data['Number of Apps Installed'].unique()))
        Screen_Usage = st.select_slider('Screen Usage in Hours',options=sorted(data['Screen Time (hrs/day)'].unique()))
        Entertainment = st.select_slider('Total Entertainment in Hours',options=sorted(data['Total Entertainment'].unique()))
        
        input_data = pd.DataFrame([{
            'Gender':gender,
            'Phone Brand':phone_brand,
            'OS':os,
            'AgeGroup':AgeGroup,
            'Calls Duration (mins/day)':call_dur,
            'Number of Apps Installed':Apps,
            'Screen Time (hrs/day)':Screen_Usage,
            'Total Entertainment':Entertainment
        }])       
        if st.button("Predict"):
            try :
                end_prediction = model.predict(input_data)
                prediction = encode.inverse_transform(end_prediction)
                st.success(f"Predicted Values{prediction}")
            except Exception as e:
                st.error(f"Prediction Failed {e}")
                                       
elif nav == 'Clustering':
        st.title('Clustering')


        use_saved = st.checkbox(" Use saved clustering result", value=True)

        if use_saved:
            # Load saved PCA + clustering result
            pca_df = agglo.copy()
            if 'Agglo_Labels' in pca_df.columns:
                pca_df.rename(columns={'Agglo_Labels': 'Cluster'}, inplace=True)
            st.success("Using saved clustering result.")
        else:
            st.info("Recomputing clustering from current data...")
            
            # Cached preprocessed PCA data
            pca_df = preprocess_data(cluster_df)

            # Cluster selection
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

            # Perform clustering (only this part is dynamic now)
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            pca_df['Cluster'] = model.fit_predict(pca_df)

        # Show clustering plot
        st.write(" Clustered Data (2D PCA Projection):")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax)
        ax.set_title("Agglomerative Clustering Result")
        ax.grid(True)
        st.pyplot(fig)

        # Merge with original data
        clustered_df = df.copy()
        clustered_df['PC1'] = pca_df['PC1']
        clustered_df['PC2'] = pca_df['PC2']
        clustered_df['Cluster'] = pca_df['Cluster']

        # Download
        csv = clustered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Data as CSV",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv',
        )