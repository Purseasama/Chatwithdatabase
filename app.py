import streamlit as st
import pandas as pd
import google.generativeai as genai
import random

# Configure Gemini using the secret key
genai.configure(api_key=st.secrets["google"]["api_key"])

# Set up the model configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Load data
@st.cache_data
def load_data():
    try:
        # Load the combined orders data from the same directory
        df = pd.read_csv('combined_orders.csv')  # Adjusted path
        # Convert date columns to datetime with flexible parsing
        date_columns = ['Delivery Date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y')
                except Exception:
                    try:
                        df[col] = pd.to_datetime(df[col], dayfirst=True)
                    except Exception:
                        try:
                            df[col] = pd.to_datetime(df[col], format='mixed')
                        except Exception as e:
                            st.warning(f"Could not parse dates in column '{col}': {str(e)}")
                            pass
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_data_dictionary():
    try:
        return pd.read_csv('data_dictionary.csv')  # Adjusted path
    except Exception as e:
        st.warning(f"Could not load data dictionary: {str(e)}")
        return pd.DataFrame(columns=['Column', 'Description'])

# Create the instruction prompt
def create_prompt(question, df_name, data_dict, example_record):
    return f"""
    You are a helpful Python code generator. Generate ONLY Python code without any explanations or markdown formatting.
    Your goal is to write Python code snippets based on the user's question and the provided DataFrame information.

    Context:
    - Question: {question}
    - DataFrame Name: {df_name}
    - DataFrame Details: {data_dict}
    - Sample Data: {example_record}

    Rules:
    1. Output ONLY executable Python code, no explanations
    2. Start your response with ANSWER = 
    3. Use only pandas operations
    4. Do not import any libraries
    5. Use the existing DataFrame named '{df_name}'
    6. Store final result in variable named 'ANSWER'

    Generate only the code, no other text:
    """

# Set up Streamlit page
st.set_page_config(page_title="Cake Shop Assistant", page_icon="🍰")
st.title("Cake Shop Data Assistant 🍰")

# Add refresh button to sidebar
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Add data view selector
data_view = st.sidebar.radio(
    "Select Data View",
    ["Orders Overview", "Financial Analysis", "Raw Data", "Add New Order"]
)

# Load data
try:
    df = load_data()
    if df.empty:
        st.error("Could not load order data. Please check your data file.")
    
    data_dict = load_data_dictionary()
    
    # Display different views based on selection
    if data_view == "Orders Overview":
        st.subheader("Orders Overview")
        st.write("Total Orders:", len(df))
        st.write("Popular Cake Types:", df['Cake Type'].value_counts())
        st.write("Delivery Options:", df['Delivery Option'].value_counts())
        
    elif data_view == "Financial Analysis":
        st.subheader("Financial Analysis")
        st.write("Total Revenue:", f"฿{df['Total sale'].sum():,.2f}")
        st.write("Average Order Value:", f"฿{df['Total sale'].mean():,.2f}")
        st.write("Revenue by Cake Type:", df.groupby('Cake Type')['Total sale'].sum())
        
    elif data_view == "Add New Order":
        st.subheader("Add New Order")
        with st.form("new_order_form"):
            # Basic Information
            col1, col2 = st.columns(2)
            with col1:
                customer_name = st.text_input("Customer Name")
                customer_phone = st.text_input("Customer Phone")
                customer_channel = st.selectbox("Contact Channel", ["Line", "FB", "Instagram"])
            
            with col2:
                cake_type = st.selectbox("Cake Type", ["Custom Cake", "Princess", "Queen", "Angel", "Floral", "Super strawberry"])
                cake_size = st.selectbox("Cake Size", ["0.5P", "1P", "1.5P", "2P"])
                cake_base = st.selectbox("Base Flavor", ["Vanilla", "Chocolate"])
                cake_filling = st.selectbox("Filling Flavor", ["Strawberry", "Chocolate", "Caramel", "Blueberry"])

            # Message and Specifications
            message = st.text_input("Message on Cake")
            specifications = st.text_area("Other Specifications")

            # Add-ons
            col3, col4, col5 = st.columns(3)
            with col3:
                candles = st.number_input("Number of Candles", min_value=0)
            with col4:
                cards = st.checkbox("Include Cards")
            with col5:
                matchbox = st.checkbox("Include Matchbox")

            # Delivery Information
            delivery_option = st.selectbox("Delivery Option", ["Pickup", "Car", "motorbike"])
            delivery_date = st.date_input("Delivery Date")
            delivery_time = st.time_input("Delivery Time")
            delivery_location = st.text_area("Delivery Location")

            # Price Information
            col6, col7, col8 = st.columns(3)
            with col6:
                cake_price = st.number_input("Cake Price", min_value=0)
            with col7:
                addon_price = st.number_input("Add-on Price", min_value=0)
            with col8:
                delivery_price = st.number_input("Delivery Price", min_value=0)

            submitted = st.form_submit_button("Add Order")

            if submitted:
                # Create new order ID (XXXX_MMDD format)
                order_id = f"{random.randint(1000,9999)}_{delivery_date.strftime('%m%d')}"
                
                # Calculate total
                total_sale = cake_price + addon_price + delivery_price

                # Create new order row
                new_order = {
                    'ID': order_id,
                    'Customer Name': customer_name,
                    'Customer Contact (Phone)': customer_phone,
                    'Customer Contact (Channel)': customer_channel,
                    'Cake Type': cake_type,
                    'Cake Size': cake_size,
                    'Cake Base Flavor': cake_base,
                    'Cake Filling Flavor': cake_filling,
                    'Message on Cake': message,
                    'Other Specifications': specifications,
                    'Candle': candles,
                    'Cards': 1 if cards else 0,
                    'Matchbox': 1 if matchbox else 0,
                    'Delivery Option': delivery_option,
                    'Delivery Date': delivery_date.strftime('%d/%m/%Y'),
                    'Delivery Time': delivery_time.strftime('%H:%M:%S'),
                    'Delivery Location': delivery_location,
                    'Cake price': cake_price,
                    'Add-on price': addon_price,
                    'Delivery price': delivery_price,
                    'Total sale': total_sale
                }

                # Add to dataframe
                df_new = pd.DataFrame([new_order])
                
                try:
                    # Ensure consistent date format in the new row
                    if 'Delivery Date' in df_new.columns:
                        df_new['Delivery Date'] = pd.to_datetime(df_new['Delivery Date'], format='%d/%m/%Y')
                    
                    # Concatenate with existing dataframe
                    df = pd.concat([df, df_new], ignore_index=True)
                    
                    # Convert back to string format for saving
                    if 'Delivery Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Delivery Date']):
                        df['Delivery Date'] = df['Delivery Date'].dt.strftime('%d/%m/%Y')
                    
                    # Save updated dataframe
                    df.to_csv('combined_orders.csv', index=False)  # Adjusted path
                    
                    st.success(f"Order {order_id} added successfully!")
                    st.cache_data.clear()  # Clear cache to reload data
                except Exception as e:
                    st.error(f"Error saving order: {str(e)}")
    
    else:  # Raw Data
        st.subheader("Raw Data")
        st.dataframe(df)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your cake shop data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create context for the model
        df_name = "cake_shop_df"
        instruction_prompt = create_prompt(
            question=prompt,
            df_name=df_name,
            data_dict=data_dict.to_string(),
            example_record=df.head(2).to_string()
        )

        # Get response from Gemini
        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-lite",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            response = model.generate_content(instruction_prompt)
            
            # Clean the response text
            code = response.text.strip()
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0].strip()
            elif '```' in code:
                code = code.split('```')[1].strip()
            
            # Execute the generated code
            try:
                local_vars = {'cake_shop_df': df, 'ANSWER': None}
                exec(code, globals(), local_vars)
                answer = local_vars['ANSWER']

                # Check if answer is defined
                if answer is not None:
                    # Display response
                    with st.chat_message("assistant"):
                        st.markdown(f"Here's what I found:")
                        st.write(answer)
                else:
                    st.error("No answer was generated.")
            
            except Exception as e:
                st.error(f"Error executing generated code: {str(e)}")
