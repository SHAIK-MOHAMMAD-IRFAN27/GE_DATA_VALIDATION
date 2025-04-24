import streamlit as st
import requests
import json
import pandas as pd

# Set the FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

def main():
    st.title("Data Quality Management System")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select a page",
        ["Customers", "Generate Rules", "Table Schema", "Validation Results"]
    )
    
    if page == "Customers":
        show_customers_page()
    elif page == "Generate Rules":
        show_rules_generation_page()
    elif page == "Table Schema":
        show_schema_page()
    elif page == "Validation Results":
        show_validation_results_page()

def show_customers_page():
    st.header("Customer Management")
    
    # Create new customer form
    st.subheader("Add New Customer")
    with st.form("new_customer_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        age = st.number_input("Age", min_value=0, max_value=150)
        country = st.text_input("Country")
        submit_button = st.form_submit_button("Add Customer")
        
        if submit_button:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/create_customer",
                    json={"name": name, "email": email, "age": age, "country": country}
                )
                if response.status_code == 200:
                    st.success("Customer added successfully!")
                else:
                    st.error("Failed to add customer")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display existing customers
    st.subheader("Existing Customers")
    if st.button("Refresh Customers"):
        try:
            response = requests.get(f"{BACKEND_URL}/")
            if response.status_code == 200:
                customers = response.json()
                df = pd.DataFrame(customers)
                st.dataframe(df)
            else:
                st.error("Failed to fetch customers")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_rules_generation_page():
    st.header("Generate Data Quality Rules")
    
    # Rule generation method selection
    generation_method = st.radio(
        "Select rule generation method",
        ["Generate from table", "Generate from custom prompt"]
    )
    
    if generation_method == "Generate from table":
        # Add table selection
        table_name = st.selectbox(
            "Select Table",
            ["customers", "orders", "products"]
        )
        
        if st.button("Generate Rules"):
            with st.spinner("Generating rules..."):
                try:
                    # Make request to backend to generate rules
                    response = requests.post(
                        f"{BACKEND_URL}/generate-table-rules",
                        json={"table_name": table_name}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["status"] == "success":
                            # Display generated rules
                            st.subheader("Generated Rules")
                            rules_df = pd.DataFrame(data["rule_entries"])
                            st.dataframe(rules_df)
                            
                            # Show success message
                            st.success(data["message"])
                            
                            # Display sample data if available
                            if "sample_data" in data and data["sample_data"]:
                                st.subheader("Sample Data")
                                sample_df = pd.DataFrame(data["sample_data"])
                                st.dataframe(sample_df)
                        else:
                            st.error(data["message"])
                    else:
                        st.error("Failed to generate rules")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Please make sure the MCP server is running and accessible.")
    
    else:  # Custom prompt
        prompt = st.text_area(
            "Enter your custom prompt",
            placeholder="Describe the data quality rules you want to generate..."
        )
        
        if st.button("Generate Rules"):
            with st.spinner("Generating rules..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/generate-gx-code",
                        json={"prompt": prompt}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["status"] == "success":
                            # Display generated rules
                            st.subheader("Generated Rules")
                            st.code(data["rules"], language="python")
                            
                            # Display rules in a dataframe
                            if data.get("rule_entries"):
                                display_rules = []
                                for rule in data["rule_entries"]:
                                    display_rules.append({
                                        "Table": rule["table_name"],
                                        "Column": rule["column_name"],
                                        "Rule": rule["expectation_rules"]
                                    })
                                rules_df = pd.DataFrame(display_rules)
                                st.dataframe(rules_df)
                            
                            # Show success message
                            st.success(data["message"])
                        else:
                            st.error(data["message"])
                    else:
                        st.error("Failed to generate rules")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Please make sure the MCP server is running and accessible.")

def show_schema_page():
    st.header("Table Schema")
    
    if st.button("View Schema"):
        try:
            response = requests.get(f"{BACKEND_URL}/table-schema")
            if response.status_code == 200:
                schema = response.json()
                st.json(schema)
            else:
                st.error("Failed to fetch schema")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_validation_results_page():
    st.header("Validation Results")
    
    # Add validation button at the top
    if st.button("Run Validation", type="primary"):
        with st.spinner("Running validation..."):
            try:
                response = requests.post(f"{BACKEND_URL}/validate-with-rules")
                if response.status_code == 200:
                    data = response.json()
                    
                    if data["status"] == "success":
                        validation_report = data["validation_report"]
                        
                        # Display overall statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Rules", validation_report["statistics"]["evaluated_expectations"])
                        with col2:
                            st.metric("Successful Rules", validation_report["statistics"]["successful_expectations"])
                        with col3:
                            st.metric("Failed Rules", validation_report["statistics"]["unsuccessful_expectations"])
                        with col4:
                            st.metric("Success Rate", f"{validation_report['statistics']['success_percent']}%")
                        
                        # Display detailed resultsS
                        st.subheader("Detailed Validation Results")
                        
                        # Create a DataFrame for the results
                        results_data = []
                        for result in validation_report["results"]:
                            results_data.append({
                                "Column": result["column"],
                                "Expectation Type": result["expectation_type"],
                                "Status": "✅ Pass" if result["success"] else "❌ Fail",
                                "Details": json.dumps(result["result"], indent=2)
                            })
                        
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Display failed validations in an expander
                            if validation_report["failed_validations"]:
                                with st.expander("Failed Validations Details", expanded=True):
                                    for failure in validation_report["failed_validations"]:
                                        st.markdown(f"""
                                        **Column:** {failure['column']}
                                        
                                        **Expectation Type:** {failure['expectation_type']}
                                        
                                        **Rule Details:** {json.dumps(failure['rule_details'], indent=2)}
                                        
                                        **Result:** {json.dumps(failure['result'], indent=2)}
                                        
                                        ---
                                        """)
                        else:
                            st.warning("No validation results available.")
                            
                    else:
                        st.error(f"Validation failed: {data['message']}")
                        
                else:
                    st.error("Failed to run validation")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        table_name = st.text_input("Filter by Table Name (optional)")
    with col2:
        validation_status = st.selectbox(
            "Filter by Status",
            ["All", "Success", "Failed"]
        )
    
    if st.button("View Historical Results"):
        try:
            # Make request to backend to get validation results
            response = requests.get(f"{BACKEND_URL}/validation-results")
            if response.status_code == 200:
                results = response.json()
                
                if isinstance(results, dict) and "error" in results:
                    st.error(f"Error fetching results: {results['error']}")
                    return
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(results)
                
                if len(df) > 0:
                    # Display the results
                    st.dataframe(df)
                    
                    # Show summary statistics
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Validations", len(df))
                    with col2:
                        # Count successful validations based on validation_report
                        success_count = sum(1 for row in df['validation_report'] if row.get('success', False))
                        st.metric("Successful", success_count)
                    with col3:
                        # Count failed validations based on validation_report
                        failed_count = sum(1 for row in df['validation_report'] if not row.get('success', True))
                        st.metric("Failed", failed_count)
                    
                    # Show detailed view for selected row
                    st.subheader("Detailed View")
                    selected_index = st.selectbox("Select a validation result to view details", range(len(df)))
                    if selected_index is not None:
                        selected_result = df.iloc[selected_index]
                        
                        # Display validation report details
                        validation_report = selected_result.get('validation_report', {})
                        if validation_report:
                            st.subheader("Validation Report")
                            
                            # Display statistics
                            stats = validation_report.get('statistics', {})
                            if stats:
                                st.write("**Statistics:**")
                                st.json(stats)
                            
                            # Display results
                            results = validation_report.get('results', [])
                            if results:
                                st.write("**Validation Results:**")
                                results_df = pd.DataFrame(results)
                                st.dataframe(results_df)
                            
                            # Display failed validations
                            failed = validation_report.get('failed_validations', [])
                            if failed:
                                st.write("**Failed Validations:**")
                                for failure in failed:
                                    st.markdown(f"""
                                    **Column:** {failure.get('column', 'N/A')}
                                    
                                    **Expectation Type:** {failure.get('expectation_type', 'N/A')}
                                    
                                    **Rule Details:** {json.dumps(failure.get('rule_details', {}), indent=2)}
                                    
                                    **Result:** {json.dumps(failure.get('result', {}), indent=2)}
                                    
                                    ---
                                    """)
                else:
                    st.info("No validation results found matching the criteria.")
            else:
                st.error("Failed to fetch validation results")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please make sure the MCP server is running and accessible.")

    # Add some styling
    st.markdown("""
    <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .success {
            color: green;
        }
        .failure {
            color: red;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 