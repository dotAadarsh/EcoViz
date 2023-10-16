import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
import plotly.graph_objects as go

st.set_page_config(
    page_title="EcoViz",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "https://github.com/dotAadarsh/EcoViz",
        'Report a bug': "https://github.com/dotAadarsh/EcoViz",
        'About': "Emission Insights Explorer"
    }
)

st.header("EcoViz")
st.write("Exploring the environmental impacts of food and agriculture?")

with st.sidebar:
    st.markdown("""
    **Environment Impact of Food Production**
    """)

    with st.expander("About", expanded=True):
        st.markdown("""
        - This project is created exclusively for [Youth Data Hack](https://youth-data-hack.devpost.com/)
        
        - The dataset is from [Kaggle](https://www.kaggle.com/datasets/selfvivek/environment-impact-of-food-production)

        - This project is inspired from the [Choose your food wisely! notebook](https://www.kaggle.com/code/selfvivek/choose-your-food-wisely/notebook) on Kaggle licensed under the [APACHE LICENSE, VERSION 2.0](https://www.apache.org/licenses/LICENSE-2.0)
        """)


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Exploring the dataset", "Food Emission", "Land usage", "Eutrophication", "Emission via transportation"])

with tab1:
    st.subheader("Context")
    st.write("As the world’s population has expanded and gotten richer, the demand for food, energy and water has seen a rapid increase. Not only has demand for all three increased, but they are also strongly interlinked: food production requires water and energy; traditional energy production demands water resources; agriculture provides a potential energy source. This project focuses on the environmental impacts of food. Ensuring everyone in the world has access to a nutritious diet in a sustainable way is one of the greatest challenges we face.")

    st.markdown("""
        This dataset contains most 43 most common foods grown across the globe and 23 columns as their respective land, water usage and carbon footprints.
        
        **Columns**

        - Land use change - Kg CO2 - equivalents per kg product
        - Animal Feed - Kg CO2 - equivalents per kg product
        - Farm - Kg CO2 - equivalents per kg product
        - Processing - Kg CO2 - equivalents per kg product
        - Transport - Kg CO2 - equivalents per kg product
        - Packaging - Kg CO2 - equivalents per kg product
        - Retail - Kg CO2 - equivalents per kg product
        
        These represent greenhouse gas emissions per kg of food product(Kg CO2 - equivalents per kg product) across different stages in the lifecycle of food production.

        Eutrophication – the pollution of water bodies and ecosystems with excess nutrients – is a major environmental problem. The runoff of nitrogen and other nutrients from agricultural production systems is a leading contributor.
    """)

    st.write("Let's explore the dataset.")

    df = pd.read_csv("Food_Production.csv")

    st.subheader("Dataset")
    st.dataframe(df)

    col1, col2 = st.columns([0.3, 0.7])
    
    with col1:
        st.subheader("Columns")
        st.dataframe(df.columns)
    
    Counter(df.dtypes.values)

    with col2:
        df_info= pd.DataFrame({"Dtype": df.dtypes, "Unique": df.nunique(), "Missing%": (df.isnull().sum()/df.shape[0])*100})
        st.subheader("Characteristics")
        st.dataframe(df_info)

    st.subheader("Overview of Missing Values")
    # Calculate the number of NA values and the percentage of NA values for each column
    n_NAvalues = df.isna().sum()
    perc_NAvalues = round(df.isna().sum()/len(df)*100,ndigits=1)

    # Create a DataFrame to store the NA information
    table_NA_info = pd.DataFrame({
        "Data Types": df.dtypes,
        "Unique Values" : df.nunique(),
        "Total NA Values": n_NAvalues,
        "%Perc NA Values": perc_NAvalues})

    # Sort the table_NA_info DataFrame by the percentage of NA values in descending order
    table_NA_info = table_NA_info.sort_values(by = "%Perc NA Values", ascending = False)

    # Display the table_NA_info DataFrame in Streamlit
    st.table(table_NA_info)

    # Mean Imputation
    st.subheader("Mean Imputation")
    df_mean = df.copy(deep=True)
    df_mean = df_mean.fillna(df_mean.mean())
    st.dataframe(df_mean.describe())


with tab2:

    st.write("Lets visualize how the emissions values are spread across the dataset, which can be useful for data analysis and understanding the data's distribution.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(df.shape[0]), np.sort(df.Total_emissions.values), s=50)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Total Emissions', fontsize=12)

    # Display the scatter plot in Streamlit
    st.pyplot(fig)

    st.write("A small selection of foods is responsible for the majority of the carbon footprint. Let's delve into identifying these major contirbutors.")

    food_df= df.groupby("Food product")['Total_emissions'].sum().reset_index()

    trace = go.Scatter(
        y = food_df.Total_emissions,
        x = food_df["Food product"],
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = food_df.Total_emissions*2,
            color = food_df.Total_emissions,
            colorscale='Portland',
            showscale=True
        )
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Total Emissions by Foods',
        hovermode= 'closest',
        xaxis= dict(
            ticklen= 5,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            title= 'Total Emissions',
            showgrid=False,
            zeroline=False,
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)

    # Display the Plotly scatter plot in Streamlit
    st.plotly_chart(fig)

    st.write("""
        The above visualization shows a scatter plot of total emissions by food product. The size and color of the markers in the scatter plot are determined by the total emissions.
    
        Some insights from the above viz is:
        - The food products with the highest total emissions are beef, lamb, and cheese.
        - The food products with the lowest total emissions are fruits, vegetables, and grains.
        - There is a large variation in total emissions between different food products. For example, beef has over 100 times the total emissions of apples.
        - There is a positive correlation between the size of the food product and its total emissions. This means that larger food products, such as beef and lamb, tend to have higher total emissions than smaller food products, such as fruits and vegetables.
        - It inform consumers about the environmental impact of different food products. For example, consumers who are concerned about their environmental impact may choose to eat less beef and lamb and more fruits, vegetables, and grains.
        - The food products with the highest total emissions are also the most resource-intensive to produce. For example, beef production requires a lot of land, water, and energy.
        - The food products with the lowest total emissions are also the most nutritious and healthy. For example, fruits, vegetables, and grains are good sources of vitamins, minerals, and fiber.
    """)

    st.subheader("Emission through various stages of supply chain")

    # Sort the DataFrame by total emissions
    temp_df = df.sort_values(by='Total_emissions', ascending=True).iloc[:, :8]

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 20))
    sns.set()
    temp_df.set_index('Food product').plot(kind='barh', stacked=True, ax=ax)
    plt.xlabel('Greenhouse gas Emissions')

    # Display the stacked bar chart in Streamlit
    st.pyplot(fig)

    st.markdown("""

    The above stacked bar chart of total emissions by food product, broken down by type of emission. The stacked bar chart shows that beef has the highest total emissions, followed by lamb and mutton.

    Some insights from the above:

    - Beef is the most environmentally impactful food product. Beef production is a major contributor to climate change, deforestation, and water pollution.
    - Lamb and mutton also have a significant environmental impact. These meats require a lot of land, water, and energy to produce.
    - Fruits, vegetables, and grains have the lowest environmental impact. These foods are also the most nutritious and healthy.

    The food products with the highest environmental impact are also the most likely to contribute to climate change. Climate change is a serious threat to the planet and its inhabitants, and it is important to reduce our greenhouse gas emissions in order to mitigate its effects.
    By choosing food products with lower environmental impact, consumers can also help to reduce their water footprint. Water scarcity is a growing problem in many parts of the world, and it is important to use water resources sustainably.

    The food products with the highest environmental impact are also the most likely to be processed foods. Processed foods are often high in unhealthy fats, sugar, and salt, and they can contribute to a number of health problems, such as obesity, heart disease, and stroke.
    By choosing food products with lower environmental impact, consumers can make a positive impact on their own health and the health of the planet.

    Here are some tips for reducing your environmental impact when choosing food:

    - Eat less meat and more plant-based foods.
    - Choose locally grown and produced foods when possible.
    - Avoid processed foods.
    - Buy food in bulk to reduce packaging waste.
    - Compost food scraps.
    - By following these tips, you can help to reduce your impact on the planet and make more sustainable food choices.
    """)


with tab3:

    st.write("The land needed for food production varies significantly depending on the type of food. Let's compare the land usage for different foods in terms of the amount of food produced per kilogram and nutritional values, such as per 100 grams or 1000 kilocalories of protein.")
    # Drop any rows with missing values
    land_df = df.dropna()

    # Sort the DataFrame by land use per 1000kcal
    land_df = land_df.sort_values(by='Land use per 1000kcal (m² per 1000kcal)', ascending=True)

    # Select the relevant columns
    land_df = land_df[['Food product','Land use per 1000kcal (m² per 1000kcal)']]

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set()
    land_df.set_index('Food product').plot(kind='barh', stacked=True, ax=ax, color="sienna")
    plt.xlabel("Land Use per 100 Kcal")
    plt.title("Land Use by Foods per 1000 Kcal\n", size=20)

    # Display the stacked bar chart in Streamlit
    st.pyplot(fig)

    st.markdown("""
        The above stacked bar chart is of land use per 1000kcal by food product. The stacked bar chart shows that beef has the highest land use per 1000kcal, followed by lamb and mutton. It also shows that the majority of land use for beef is for grazing.

        Here are some insights from the given data visualization image:

        * **Beef production is very land-intensive.** It takes a lot of land to produce beef, both for grazing and for growing feed crops.
        * **Lamb and mutton production are also land-intensive, but not to the same extent as beef production.**
        * **Plant-based foods are the least land-intensive to produce.** Fruits, vegetables, and grains have a much lower land use per 1000kcal than meat products.

        Consumers can use this information to make more informed choices about the food they eat. By choosing food products with lower land use, consumers can help to reduce their impact on the environment.

        Here are some additional insights that can be gained from the image:

        * **Beef production is a major contributor to deforestation.** Deforestation is the clearing of forests for other land uses, such as agriculture. Deforestation is a serious environmental problem because it releases carbon dioxide into the atmosphere and contributes to climate change.
        * **Beef production is also a major contributor to water pollution.** Beef production requires a lot of water, and the runoff from beef farms can pollute rivers and streams.
        * **Plant-based foods are a more sustainable choice for the environment.** Plant-based foods require less land, water, and energy to produce than meat products.

        By choosing to eat less meat and more plant-based foods, consumers can help to reduce their impact on the environment and make more sustainable food choices.

        Here are some tips for reducing your land use impact when choosing food:

        * Eat less meat and more plant-based foods.
        * Choose locally grown and produced foods when possible.
        * Avoid processed foods.
        * Buy food in bulk to reduce packaging waste.
        * Compost food scraps.

        By following these tips, you can help to reduce your impact on the environment and make more sustainable food choices.
    """)

    st.write("Beef, lamb, and mutton farming consume the majority of the land, while plant-based foods occupy the lowest end of the spectrum in terms of land usage.")


with tab4:

    st.write("The discharge of nitrogen and other nutrients from agricultural production systems is a significant driver of environmental pollution. Let's assess various foods in relation to their impact on eutrophication.")

    # Drop any rows with missing values
    eutrophication_df = df.dropna()

    # Sort the DataFrame by eutrophying emissions per 1000kcal
    eutrophication_df = eutrophication_df.sort_values(by='Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)', ascending=True)

    # Select the relevant columns
    eutrophication_df = eutrophication_df[['Food product', 'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)']]

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set()
    eutrophication_df.set_index('Food product').plot(kind='barh', stacked=True, ax=ax, color="black")
    plt.xlabel('Eutrophication emissions Per 1000kcal')
    plt.title('Eutrophication Emissions Per 1000kcal \n', size=20)

    # Display the stacked bar chart in Streamlit
    st.pyplot(fig)

    st.write("When evaluated in terms of energy per 1000 kilocalories, coffee exhibits the highest eutrophication emissions, with plant-based foods ranking at the lowest end of the scale.")

    st.write("Now lets compare eutrophication emissions of different foods required to produce 1 kg food.")
    #comparing eutrophication emissions of different foods required to produce 1 kg food

    eutrophication_df= df.dropna().sort_values(by= 'Eutrophying emissions per kilogram (gPO₄eq per kilogram)', ascending= True)[['Food product',
        'Eutrophying emissions per kilogram (gPO₄eq per kilogram)']]

    fig, ax = plt.subplots(figsize=(15,10))
    sns.set()
    eutrophication_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "black")
    plt.xlabel('Eutrophication emissions Per Kg')
    plt.title('Eutrophication Emissions Per Kg \n', size= 20)
    plt.show()
    st.pyplot(fig)

    st.markdown("""
        The  visualization shows the eutrophication emissions per kilogram for different foods. Eutrophication is a process that occurs when water bodies become over-enriched with nutrients, such as nitrogen and phosphorus. This can lead to excessive algae growth, which can deplete oxygen levels in the water and harm aquatic life.

        The image shows that beef has the highest eutrophication emissions per kilogram, followed by fish, and cheese. Beef is also the most common food in the world, so its high eutrophication emissions are a significant concern.

        Other foods with relatively high eutrophication emissions include lamb and mutton, dark chocolate, and pig meat. Foods with lower eutrophication emissions include poultry meat, rice, eggs, nuts, groundnuts, oatmeal, milk, tomatoes, berries and grapes, brassicas, potatoes, bananas, onions and leeks, citrus fruit, root vegetables, and apples.

        The insights from this data visualization image are that:

        * Beef has the highest eutrophication emissions per kilogram, followed by chicken, fish, and cheese.
        * Beef is also the most common food in the world, so its high eutrophication emissions are a significant concern.
        * Other foods with relatively high eutrophication emissions include lamb and mutton, dark chocolate, and pig meat.
        * Foods with lower eutrophication emissions include poultry meat, rice, eggs, nuts, groundnuts, oatmeal, milk, tomatoes, berries and grapes, brassicas, potatoes, bananas, onions and leeks, citrus fruit, root vegetables, and apples.

        To reduce our eutrophication footprint, we can choose to eat less beef and other foods with high eutrophication emissions, and more foods with lower eutrophication emissions. We can also support sustainable agriculture practices that reduce nutrient pollution.
    """)

    st.write("Eutrophication emissions to produce 1 kilogram are primarily attributed to animal-based foods, whereas plant-based foods make a notably smaller contribution.")


with tab5:
    
    # Group the DataFrame by food product and calculate the total land use change for each product
    temp_series = df.groupby('Food product')['Land use change'].sum()

    # Convert the index and values of the temp_series to NumPy arrays
    labels = (np.array(temp_series.index))
    sizes = (np.array((temp_series / temp_series.sum())*100))

    # Clip any negative values in the sizes array to zero
    sizes = np.clip(sizes, 0, None)

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(10,10))
    wedge, _, _ = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
    ax.set_title("Food distribution by emissions via Transport", fontsize=20)
    ax.legend(wedge, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the pie chart in Streamlit
    st.pyplot(fig)
    st.markdown("""
    
        The data visualization image shows the food distribution by emissions via transport. The pie chart shows that beef has the highest emissions, followed by lamb and mutton, dark chocolate, and pig meat. Foods with lower emissions include poultry meat, rice, eggs, nuts, groundnuts, oatmeal, milk, tomatoes, berries and grapes, brassicas, potatoes, bananas, onions and leeks, citrus fruit, root vegetables, and apples.

        **Insights**

        * Beef has the highest emissions per kilogram, followed by dark chocolate.
        * Poultry meat, rice, eggs, nuts, groundnuts, oatmeal, milk, tomatoes, berries and grapes, brassicas, potatoes, bananas, onions and leeks, citrus fruit, root vegetables, and apples have lower emissions.
        * The production and transportation of food account for a significant portion of global greenhouse gas emissions.
        * By choosing foods with lower emissions, we can reduce our impact on the environment.
    """)

    # Group the DataFrame by food product and calculate the total transport emissions for each product
    temp_series = df.groupby('Food product')['Transport'].sum()

    # Convert the index and values of the temp_series to NumPy arrays
    labels = (np.array(temp_series.index))
    sizes = (np.array((temp_series / temp_series.sum())*100))

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(10,10))
    wedge, _, _ = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
    ax.set_title("Food distribution by emissions via Transport", fontsize=20)
    ax.legend(wedge, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the pie chart in Streamlit
    st.pyplot(fig)

    st.markdown("""
    Generally Emissions via transport are uniform across different foods with cane sugar having most share.
    """)
