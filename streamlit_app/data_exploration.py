import streamlit as st
def run():
    st.title("Data exploration")

    st.write(
        """
        The first step in data exploration involved merging multiple Excel files (XLSX) into one dataset.
        After that, we created graphs to visualize the data and identify key trends.
        Finally, the dataset was exported into a CSV file for easier access and future analysis.
        Below are the plots from the data visualization.
        """
    )
    
    # Display the pie chart
    image = 'media/pie.png'
    st.image(image, use_container_width=True)

    # Display the histogram
    image = 'media/histogram.png'
    st.image(image, use_container_width=True)

    # Display the insurer standings
    image = 'media/insurerstandings.png'
    st.image(image, use_container_width=True)

    # Display the average grade plot
    image = 'media/averagegrade.png'
    st.image(image, use_container_width=True)