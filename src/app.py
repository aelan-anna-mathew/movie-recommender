import streamlit as st
from recommend import get_recommendations  # Import the function

st.title("🎬 Movie Recommendation System")
st.write("Type a movie name below to get similar movie suggestions:")

movie_name = st.text_input("Enter movie name:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        try:
            # 1. Get the list of recommendations
            recommendations = get_recommendations(movie_name)
            
            # 2. Display results
            if recommendations:
                st.subheader(f"Recommended Movies for '{movie_name}':")
                # Iterate correctly over the list of movie titles
                for movie in recommendations:
                    st.write(f"- **{movie}**")
            else:
                st.error(f"No similar movies found or '{movie_name}' not in database. Check spelling or try another movie.")
                
        except Exception as e:
            # This handles potential errors during setup or similarity calculation
            st.exception(e) # Use st.exception for better error logging