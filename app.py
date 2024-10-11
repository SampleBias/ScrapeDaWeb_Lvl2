import streamlit as st
from web_researcher import WebResearcher

def main():
    st.set_page_config(page_title="Web Researcher App", page_icon="📚", layout="wide")
    st.title("Web Researcher App 📚🔍")

    # Input fields
    entity_name = st.text_input("Enter the entity name to research:")
    website = st.text_input("Enter the main website of the entity:")

    if st.button("Start Research"):
        if entity_name and website:
            with st.spinner(f"Researching {entity_name}..."):
                researcher = WebResearcher(entity_name, website)
                response = researcher.website_search()

            st.success("Research completed!")

            # Display results
            st.subheader("Research Results")
            for data_point in researcher.data_points:
                st.write(f"**{data_point['name']}:** {data_point['value']}")
                if data_point['reference']:
                    st.write(f"*Source: {data_point['reference']}*")
                st.write("---")

            # Display scraped links
            st.subheader("Scraped Links")
            for link in researcher.links_scraped:
                st.write(link)

        else:
            st.error("Please enter both the entity name and website.")

if __name__ == "__main__":
    main()


