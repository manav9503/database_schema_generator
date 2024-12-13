import streamlit as st
import pickle
import os
import hashlib
import json
import sqlite3
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# File paths for the pre-trained models and embeddings (use the paths where the models are stored)
model_path = 'logistic_model.pkl'  # Path to the pre-trained Logistic Regression model
label_encoder_path = 'label_encoder.pkl'  # Path to the pre-trained Label Encoder
vectorizer_path = 'tfidf_vectorizer.pkl'  # Path to the pre-trained TF-IDF Vectorizer
db_path = 'schema_data.db'


# SQLite database file path

  # SQLite database file

# Hashing function for admin password


# File paths for the pre-trained models and embeddings (use the paths where the models are stored)


# Hashing function for admin password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Default admin password hash
ADMIN_PASSWORD_HASH = hash_password("tony@sam")  # Set your default admin password

# Check if the entered password is correct
def check_password(input_password):
    return hash_password(input_password) == ADMIN_PASSWORD_HASH

# Load pre-trained models if they exist
if os.path.exists(model_path) and os.path.exists(label_encoder_path) and os.path.exists(vectorizer_path):
    # Load models
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    st.error("Model files not found! Please train the model first.")
    st.stop()

# Prediction function
def predict_schema(new_prompt):
    new_prompt_features = vectorizer.transform([new_prompt])
    prediction = classifier.predict(new_prompt_features)
    return label_encoder.inverse_transform(prediction)[0]

# Parse schema string into dictionary
def parse_schema(schema_str):
    return json.loads(schema_str)

# Create database based on schema
def create_database(schema):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        if isinstance(schema, dict):
            for table_name, table_schema in schema.items():
                columns = []
                for field, field_type in table_schema.items():
                    if field_type == "Text":
                        columns.append(f"{field} TEXT")
                    elif field_type == "Email":
                        columns.append(f"{field} TEXT")
                    elif field_type == "Number":
                        columns.append(f"{field} INTEGER")
                    elif field_type == "Date":
                        columns.append(f"{field} DATE")
                    elif field_type == "Currency":
                        columns.append(f"{field} REAL")
                    elif field_type == "Picklist":
                        columns.append(f"{field} TEXT")
                columns_str = ", ".join(columns)
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})")
        conn.commit()
        conn.close()
        st.success("Database created successfully!")
    except Exception as e:
        st.error(f"Error creating database: {e}")

# Fetch all table names
def fetch_all_tables():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

# Fetch table schema
def fetch_table_schema(table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        conn.close()
        return {column[1]: column[2] for column in schema}
    except Exception as e:
        st.error(f"Error fetching table schema for {table_name}: {e}")
        return {}

# Fetch table data
def fetch_table_data(table_name):
    try:
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()

# Delete a table from the database
def delete_table(table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        conn.close()
        st.success(f"Table `{table_name}` deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting table `{table_name}`: {e}")

# Add custom fields to an existing table
def add_custom_fields(table_name, custom_field_name, custom_field_type):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {custom_field_name} {custom_field_type};"
        cursor.execute(alter_sql)
        
        conn.commit()
        conn.close()
        st.success(f"Custom field `{custom_field_name}` added successfully to `{table_name}`!")
    except Exception as e:
        st.error(f"Error adding custom field to {table_name}: {e}")

# Fuzzy match columns between CSV and table columns
def fuzzy_match_columns(csv_columns, table_columns):
    matched_columns = {}
    for csv_col in csv_columns:
        match = process.extractOne(csv_col, table_columns, scorer=fuzz.ratio)
        if match and match[1] > 80:  # Threshold for fuzzy matching, can be adjusted
            matched_columns[csv_col] = match[0]
    return matched_columns

# Insert CSV data into the table with fuzzy matching
def insert_csv_data(df, table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute(f"PRAGMA table_info({table_name});")
        table_columns = [column[1] for column in cursor.fetchall()]

        # Perform fuzzy matching for CSV columns to table columns
        matched_columns = fuzzy_match_columns(df.columns, table_columns)

        # If no fuzzy matches are found, raise an error
        if not matched_columns:
            st.error("No matching columns found between CSV and table!")
            return

        # Ensure that the DataFrame only includes columns that have matches in the table
        df_matched = df[matched_columns.keys()]

        # Prepare the SQL insert query
        columns_str = ', '.join(matched_columns.values())
        placeholders = ', '.join(['?' for _ in matched_columns])

        # Iterate over the DataFrame rows and insert data
        for _, row in df_matched.iterrows():
            # Ensure all NaN values are replaced with None
            row_data = [None if pd.isna(value) else value for value in row]

            cursor.execute(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})", tuple(row_data))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        st.success("CSV data added successfully.")
    except Exception as e:
        st.error(f"Error inserting CSV data into {table_name}: {e}")

# Streamlit app
def main():
    st.sidebar.title("Database Management")
    menu = ["Predict Schema", "Add Data", "View Data", "Create Database from Schema", "Delete Table", "Add Custom Fields"]
    selection = st.sidebar.selectbox("Choose an option", menu)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if selection == "Predict Schema":
        st.title("Predict Schema from Text")
        new_prompt = st.text_area("Enter a description for your system")
        if st.button("Predict Schema"):
            if new_prompt:
                predicted_schema = predict_schema(new_prompt)
                parsed_schema = parse_schema(predicted_schema)
                st.json(parsed_schema)
                
            else:
                st.error("Please enter a description for prediction.")

    elif selection in ["Add Data", "View Data", "Create Database from Schema", "Delete Table", "Add Custom Fields"]:
        st.title("Admin Authentication")
        if not st.session_state.authenticated:
            admin_password = st.text_input("Enter Admin Password", type="password")
            if st.button("Login"):
                if check_password(admin_password):
                    st.session_state.authenticated = True
                    st.success("Authentication successful! Access granted.")
                else:
                    st.error("Invalid password. Access denied.")

        if st.session_state.authenticated:
            if selection == "Add Data":
                st.title("Add Data to Existing Database")
                tables = fetch_all_tables()
                if tables:
                    selected_table = st.selectbox("Choose a table to add data", tables)
                    if selected_table:
                        schema = fetch_table_schema(selected_table)
                        if schema:
                            st.subheader("Enter Data for the Selected Table")
                            data_list = []
                            row_count = st.session_state.get("row_count", 0)

                            for row_idx in range(row_count):
                                st.write(f"Row {row_idx + 1}")
                                form_data = {}
                                for field, field_type in schema.items():
                                    unique_key = f"{field}_{row_idx}"
                                    if field_type.upper() == "TEXT":
                                        form_data[field] = st.text_input(f"Enter {field}", key=unique_key)
                                    elif field_type.upper() == "INTEGER":
                                        form_data[field] = st.number_input(f"Enter {field}", step=1, key=unique_key)
                                    elif field_type.upper() == "REAL":
                                        form_data[field] = st.number_input(f"Enter {field}", format="%.2f", key=unique_key)
                                    elif field_type.upper() == "DATE":
                                        form_data[field] = st.date_input(f"Enter {field}", key=unique_key)
                                data_list.append(form_data)

                            if st.button("Add Another Row", key="add_row"):
                                st.session_state.row_count = row_count + 1

                            if st.button("Submit All Data", key="submit_data"):
                                try:
                                    conn = sqlite3.connect(db_path)
                                    cursor = conn.cursor()
                                    for form_data in data_list:
                                        columns = ", ".join(form_data.keys())
                                        placeholders = ", ".join(["?" for _ in form_data.values()])
                                        values = list(form_data.values())
                                        cursor.execute(f"INSERT INTO {selected_table} ({columns}) VALUES ({placeholders})", values)
                                    conn.commit()
                                    conn.close()
                                    st.success("All data added successfully!")
                                except Exception as e:
                                    st.error(f"Error inserting data: {e}")

                            # CSV upload functionality
                            st.subheader("Upload CSV Data")
                            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
                            if uploaded_file is not None:
                                df = pd.read_csv(uploaded_file)
                                st.write("CSV Data Preview:")
                                st.dataframe(df.head())

                                # Perform fuzzy matching on column names
                                if "Name" in df.columns:
                                    matches = fuzzy_match_columns(df.columns, fetch_table_schema(selected_table).keys())
                                    st.write("Fuzzy Matching Results:")
                                    st.write(matches)

                                # Insert data into the table
                                if st.button(f"Add CSV Data to `{selected_table}`", key="add_csv_data"):
                                    insert_csv_data(df, selected_table)
                                    st.success("CSV data added successfully.")

            elif selection == "View Data":
                st.title("View Data from Database")
                tables = fetch_all_tables()
                if tables:
                    selected_table = st.selectbox("Choose a table to view", tables)
                    if selected_table:
                        data = fetch_table_data(selected_table)
                        if not data.empty:
                            st.subheader(f"Data from {selected_table}")
                            st.dataframe(data)
                        else:
                            st.warning(f"No data found in {selected_table}.")
                else:
                    st.warning("No tables available in the database.")

            elif selection == "Create Database from Schema":
                st.title("Create Database from Predicted Schema")
                new_prompt = st.text_area("Enter a description for your system")
                if st.button("Predict and Create Database"):
                    if new_prompt:
                        predicted_schema = predict_schema(new_prompt)
                        parsed_schema = parse_schema(predicted_schema)
                        st.json(parsed_schema)
                        create_database(parsed_schema)
                    else:
                        st.error("Please enter a description for prediction.")

            elif selection == "Delete Table":
                st.title("Delete Table from Database")
                tables = fetch_all_tables()
                if tables:
                    selected_table = st.selectbox("Choose a table to delete", tables)
                    if selected_table:
                        st.warning(f"Are you sure you want to delete the table `{selected_table}`?")
                        confirmation = st.checkbox(f"Yes, I want to delete `{selected_table}`.")
                        if confirmation and st.button("Delete Table"):
                            delete_table(selected_table)
                        elif not confirmation:
                            st.info("You need to confirm the action before deleting.")
                else:
                    st.warning("No tables available in the database.")

            elif selection == "Add Custom Fields":
                st.title("Add Custom Fields to an Existing Table")
                tables = fetch_all_tables()
                if tables:
                    selected_table = st.selectbox("Choose a table to add custom fields", tables)
                    if selected_table:
                        st.subheader(f"Add Custom Fields to `{selected_table}`")
                        
                        custom_field_name = st.text_input("Field Name")
                        custom_field_type = st.selectbox("Field Type", ["Text", "Email", "Number", "Date", "Currency", "Picklist"])

                        if st.button("Add Custom Field"):
                            if custom_field_name and custom_field_type:
                                add_custom_fields(selected_table, custom_field_name, custom_field_type)
                            else:
                                st.warning("Please provide both field name and field type.")
                else:
                    st.warning("No tables available in the database.")
        else:
            st.warning("You need to log in to access this section.")
  
if __name__ == "__main__":
    main()

