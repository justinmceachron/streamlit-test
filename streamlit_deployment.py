import os
from dotenv import load_dotenv
from pinecone import Pinecone
import streamlit as st
from openai import OpenAI

# 1. Setup
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not pinecone_key or not openai_key:
    st.error("Missing API Keys! Check your .env file.")
    st.stop() # Stops the app safely if keys are missing

# ... previous setup code ...
pc = Pinecone(api_key=pinecone_key)
client = OpenAI(api_key=openai_key)

#Connect to the specific index
index = pc.Index("movies")

# 2. AI Functions
def generate_blog(topic, additional_text=""):
    user_input = f"Write a detailed blog about {topic}. Additional pointers: {additional_text}"
    
    # Using GPT-5 Responses API
    response = client.responses.create(
        model="gpt-5",
        input=user_input,
        reasoning={"effort": "medium"},
        text={"verbosity": "high"},
        max_output_tokens=2000
    )
    return response.output_text

# def generate_image(prompt, number_of_images):
    
#     response = client.images.generate(
#         prompt=prompt,
#         max_images=number_of_images,
#         size="512x512",
#     )
#     return response 

# --- ADD THIS TO YOUR AI FUNCTIONS SECTION ---

def generate_image(prompt, num_images=1):
    """
    Generates multiple images using DALL-E 2.
    (Note: DALL-E 3 currently only supports n=1, so we use DALL-E 2 for multiple generations).
    Returns a list of URLs.
    """
    response = client.images.generate(
        model="dall-e-2", # Switched to DALL-E 2 to support n > 1
        prompt=prompt,
        size="1024x1024",
        n=num_images,     # Pass the slider value here
    )
    
    # The response returns a list of data objects. We need to extract all URLs.
    # Using a list comprehension here:
    image_urls = [item.url for item in response.data]
    return image_urls
# ---------------------------------------------



# 3. GUI Layout
st.set_page_config(layout="wide")
st.title("Streamlit Deployment Example")

st.sidebar.title("AI Apps")
ai_app = st.sidebar.radio("Select an AI App:", ("Blog Generator", "Image Generator", "Movie Recommender"))

# --- BLOG GENERATOR APP ---
if ai_app == "Blog Generator":
    st.header("Blog Generator")
    st.write("Input a topic to generate a blog about it using OpenAI API.")
    
    col1, col2 = st.columns(2) # Make it look nicer with columns
    
    with col1:
        st.subheader("Input")
        topic = st.text_area("Enter the blog topic here:", height=150)
        additional_text = st.text_area("Additional context (optional):", height=150)
        
        generate_btn = st.button("Generate Blog")

    with col2:
        st.subheader("Result")
        if generate_btn and topic:
            with st.spinner('Generating your blog post... (This may take a moment)'):
                try:
                    blog_content = generate_blog(topic, additional_text)
                    st.success("Blog Generated Successfully!")
                    
                    # Using markdown renders the formatting (bolding, headers) correctly
                    st.markdown(blog_content) 
                    
                    # OPTIONAL: If you want to copy-paste it, keep a raw text version too
                    with st.expander("View Raw Text"):
                        st.text_area("Raw Output", value=blog_content, height=400)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif generate_btn and not topic:
            st.warning("Please enter a topic first.")

# elif ai_app == "Image Generator":
#     st.header("Image Generator")
#     st.write("Input a description to generate an image using OpenAI API.")

#     prompt = st.text_area("Enter the image description here:", height=300, width=700)

#     number_of_images = st.slider("Select number of images to generate:", min_value=1, max_value=5, value=1)
#     if st.button("Generate Image"):
#         st.write("Image Generated:")
#         response = generate_image(prompt, 1)

#         for output in response.data:
#             st.image(output.url)

# --- IMAGE GENERATOR LOOP ---

elif ai_app == "Image Generator":
    st.header("Image Generator (DALL-E 2)")
    st.write("Input a description to generate up to 3 variations.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Settings & Input")
        
        # --- NEW: HORIZONTAL RADIO BUTTONS ---
        # This is cleaner and more compact than a slider for small integers
        number_of_images = st.radio(
            "Select number of images:", 
            [1, 2, 3], 
            horizontal=True
        )
        
        prompt = st.text_area("Describe the image(s):", height=200, help="Be specific for better results.")
        generate_btn = st.button(f"Generate {number_of_images} Image(s)")

    with col2:
        st.subheader("Results")
        if generate_btn and prompt:
            with st.spinner(f"Generating {number_of_images} image(s)..."):
                try:
                    # Calls the backend function we defined earlier
                    image_urls_list = generate_image(prompt, number_of_images)
                    
                    st.success("Images generated successfully!")
                    
                    # Display the images
                    for i, url in enumerate(image_urls_list):
                        st.image(url, caption=f"Variation {i+1}", use_column_width=True)
                        if i < len(image_urls_list) - 1:
                            st.divider()
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Tip: DALL-E blocks prompts it considers unsafe. Try softening your language if this persists.")
        
        elif generate_btn and not prompt:
            st.warning("Please enter a description first.")

elif ai_app == "Movie Recommender":
    st.header("Movie Recommender")
    st.write("Input a movie description or title to find similar films.")

    # Using a text_input is better than text_area for single titles
    favorite_movie = st.text_input("Enter your film here:", width=300)

    if st.button("Get Recommendations"):
        if favorite_movie:
            with st.spinner("Finding similar movies..."):
                try:
                    # 1. Create the embedding for the user's input
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=favorite_movie
                    )
                    
                    # Extract the vector list
                    query_vector = response.data[0].embedding

                    # 2. Query Pinecone
                    # Note: namespace is optional. If you didn't set one when uploading, leave it out.
                    search_results = index.query(
                        vector=query_vector,
                        top_k=5,
                        include_metadata=True
                    )

                    # 3. Display Results
                    st.success("Recommendations:", width=300)
                    for match in search_results.matches:
                        # Safety check: ensure 'title' exists in metadata
                        title = match.metadata.get('title', 'Unknown Title')
                        score = match.score
                        st.write(f"- **{title}** (Similarity: {score:.2f})")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a movie name first.")
