import streamlit as st
import pickle
import os
import hashlib
import json
import sqlite3
import pandas as pd

# File paths for the pre-trained models and embeddings (use the paths where the models are stored)
model_path = 'C:\\Users\\manav\\Downloads\\logistic_model.pkl'  # Path to the pre-trained Logistic Regression model
label_encoder_path = 'C:\\Users\\manav\\Downloads\\label_encoder.pkl'  # Path to the pre-trained Label Encoder
vectorizer_path = 'C:\\Users\\manav\\Downloads\\tfidf_vectorizer.pkl'  # Path to the pre-trained TF-IDF Vectorizer
db_path = 'schema_data.db'  # SQLite database file

# Hashing function for admin password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Default admin password hash
ADMIN_PASSWORD_HASH = hash_password("manav@ram")  # Set your default admin password

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

# Fetch all table names
def fetch_all_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return [table[0] for table in tables]

# Fetch table schema
def fetch_table_schema(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    conn.close()
    return {column[1]: column[2] for column in schema}

# Fetch table data
def fetch_table_data(table_name):
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return data

# Delete a table from the database
def delete_table(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()
    conn.close()

# Normalize column names for matching (e.g., case insensitive matching)
def normalize_column_name(name):
    return name.strip().lower()

# Streamlit app
def main():
    st.sidebar.title("Database Management")
    menu = ["Predict Schema", "Add Data", "View Data", "Create Database from Schema", "Delete Table"]
    selection = st.sidebar.selectbox("Choose an option", menu)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if selection in ["Add Data", "View Data", "Create Database from Schema", "Delete Table"]:
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
                            
                            # List to store the data for multiple rows
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

                            # Add more rows
                            if st.button("Add Another Row"):
                                st.session_state.row_count = row_count + 1

                            # Submit data
                            if st.button("Submit All Data"):
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

                            # Option for CSV Upload
                            st.subheader("Upload CSV File")
                            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

                            if uploaded_file is not None:
                                # Read CSV data into a DataFrame
                                data = pd.read_csv(uploaded_file)
                                
                                # Normalize column names in the CSV
                                normalized_csv_columns = {normalize_column_name(col): col for col in data.columns}
                                
                                # Normalize table schema columns
                                normalized_schema = {normalize_column_name(col): col for col in schema.keys()}

                                # Match the CSV columns to the table schema columns
                                matched_columns = {normalized_schema[col]: normalized_csv_columns[normalized_col]
                                                   for normalized_col, col in normalized_csv_columns.items()
                                                   if normalized_col in normalized_schema}

                                if not matched_columns:
                                    st.error("No columns in CSV match the table schema. Please check the file.")
                                else:
                                    st.write("CSV data preview:")
                                    st.dataframe(data)

                                    # Option to insert the matched CSV data into the database
                                    if st.button("Insert CSV Data into Table"):
                                        conn = sqlite3.connect(db_path)
                                        cursor = conn.cursor()

                                        # Insert data row by row into the table using matched columns
                                        for _, row in data.iterrows():
                                            row_data = [row[matched_columns[col]] for col in matched_columns]
                                            columns = ", ".join(matched_columns.keys())
                                            placeholders = ", ".join(["?" for _ in row_data])
                                            cursor.execute(f"INSERT INTO {selected_table} ({columns}) VALUES ({placeholders})", tuple(row_data))
                                        
                                        conn.commit()
                                        conn.close()
                                        st.success("CSV data inserted successfully!")

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
                        st.success("Database created successfully!")
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
                            st.success(f"Table `{selected_table}` deleted successfully!")
                        elif not confirmation:
                            st.info("You need to confirm the action before deleting.")
                else:
                    st.warning("No tables available in the database.")

    elif selection == "Predict Schema":
        st.title("Schema Prediction")
        new_prompt = st.text_area("Enter a description for your system")
        if st.button("Predict Schema"):
            if new_prompt:
                predicted_schema = predict_schema(new_prompt)
                parsed_schema = parse_schema(predicted_schema)
                st.json(parsed_schema)
            else:
                st.error("Please enter a prompt for prediction.")

if __name__ == "__main__":
    main()
