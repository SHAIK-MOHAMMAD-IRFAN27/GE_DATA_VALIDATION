from supabase import create_client, Client
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import json
from typing import List
# import great_expectations as gx
import pandas as pd
from datetime import datetime
import great_expectations as ge
from great_expectations.data_context import FileDataContext
import re
# Load environment variables
load_dotenv()

class Customer(BaseModel):
    name: str
    email: str
    age: int
    country: str

class PromptRequest(BaseModel):
    prompt: str | dict

class TableRequest(BaseModel):
    table_name: str

app = FastAPI()
SUPABASE_URL = "https://rpeprbgnzgzesdzlhonh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJwZXByYmduemd6ZXNkemxob25oIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Mjk5MDAzMywiZXhwIjoyMDU4NTY2MDMzfQ.9IO4qN0Xwl_730DnjBykMKURjPJAmCgPUF--EiCnY-8"

# Initialize OpenAI client with environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Make sure to set this in your .env file

@app.get("/")
async def get_customers():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    results = supabase.table("customers").select("*").execute()
    return results.data

@app.get("/table-schema")
async def get_table_schema():
    try:
        # Return the Customer model schema
        return {
            "columns": [
                {"name": "name", "type": "str"},
                {"name": "email", "type": "str"},
                {"name": "age", "type": "int"},
                {"name": "country", "type": "str"}
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/create_customer")
async def create_customer(customer: Customer):
    try:
        new_row = customer.dict()
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        results = supabase.table("customers").insert(new_row).execute()
        print("added successfully")
        return results.data
    except Exception as e:
        print(e)
        return {"error": str(e)}

async def check_existing_rule(supabase: Client, table_name: str, column_name: str, expectation_rule: str) -> bool:
    """Check if a rule already exists in the database"""
    try:
        response = supabase.table("data_quality_rules").select("*").eq("table_name", table_name).eq("column_name", column_name).eq("expectation_rules", expectation_rule).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"Error checking for existing rule: {str(e)}")
        return False

@app.post("/generate-gx-code")
async def generate_gx_code(prompt_request: PromptRequest):
    try:
        SUPABASE_URL = "https://rpeprbgnzgzesdzlhonh.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJwZXByYmduemd6ZXNkemxob25oIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Mjk5MDAzMywiZXhwIjoyMDU4NTY2MDMzfQ.9IO4qN0Xwl_730DnjBykMKURjPJAmCgPUF--EiCnY-8"
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Create a system message that focuses on generating only GE rules
        system_message = """You are an expert in Great Expectations (GX) and data quality.
        Analyze the provided table schema and sample data to generate appropriate data quality rules.
        Generate ONLY the core expectation rules without any additional information.
        
        Consider the following aspects:
        1. Data types and format validation
        2. Null checks based on is_nullable
        3. Value ranges based on sample data
        4. Unique constraints where appropriate
        5. Pattern matching for strings (email, etc.)
        6. Business logic rules based on data patterns
        
        Format: One rule per line, only the expect_* statements.
        Use single quotes for strings and no backslashes.
        
        Examples of correct rule formats:
        expect_column_values_to_be_unique('id')
        expect_column_values_to_not_be_null('name')
        expect_column_values_to_be_of_type('id', 'integer')
        expect_column_values_to_be_of_type('name', 'string')
        expect_column_values_to_be_between('age', min_value=0, max_value=120)
        expect_column_values_to_match_regex('email', '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')
        expect_column_values_to_be_in_set('status', ['active', 'inactive', 'pending'])
        expect_column_values_to_be_in_type_list('amount', ['integer', 'float', 'decimal'])"""

        # Create the user message with the formatted prompt
        user_message = f"Generate only Great Expectations rules for this data:\n{prompt_request.prompt}"

        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract and format the rules
        generated_rules = response.choices[0].message.content.strip().split("\n")
        
        # Clean up the rules format
        cleaned_rules = []
        for rule in generated_rules:
            if rule.strip():
                # Remove backslashes before quotes
                cleaned_rule = rule.replace('\\"', '"')
                # Convert double quotes to single quotes
                cleaned_rule = cleaned_rule.replace('"', "'")
                cleaned_rules.append(cleaned_rule)

        # Extract column names from rules and prepare rule entries
        rule_entries = []
        skipped_rules = []
        table_name = "customers"  # Default to customers table, but can be modified based on prompt
        
        for rule in cleaned_rules:
            # Extract column name from the rule using regex
            import re
            match = re.search(r"'([^']*)'", rule)
            if match:
                column_name = match.group(1)
                
                # Check if rule already exists
                if await check_existing_rule(supabase, table_name, column_name, rule):
                    print(f"Rule already exists for column {column_name}: {rule}")
                    skipped_rules.append(rule)
                    continue
                    
                rule_entries.append({
                    "table_name": table_name,
                    "column_name": column_name,
                    "expectation_rules": rule,
                    "created_at": "NOW()"
                })

        # Save rules to database
        if rule_entries:
            try:
                save_result = supabase.table("data_quality_rules").insert(rule_entries).execute()
                return {
                    "table_name": table_name,
                    "rules": cleaned_rules,
                    "rule_entries": rule_entries,
                    "skipped_rules": skipped_rules,
                    "status": "success",
                    "message": f"Successfully generated and saved {len(rule_entries)} rules for table {table_name}. Skipped {len(skipped_rules)} existing rules.",
                    "saved_rules": save_result.data
                }
            except Exception as save_error:
                return {
                    "table_name": table_name,
                    "rules": cleaned_rules,
                    "rule_entries": rule_entries,
                    "skipped_rules": skipped_rules,
                    "status": "error",
                    "message": f"Generated rules but failed to save: {str(save_error)}"
                }
        else:
            return {
                "table_name": table_name,
                "rules": cleaned_rules,
                "rule_entries": [],
                "skipped_rules": skipped_rules,
                "status": "info",
                "message": f"No new rules to save. All {len(skipped_rules)} rules already exist in the database."
            }

    except Exception as e:
        return {
            "table_name": table_name,
            "rules": [],
            "rule_entries": [],
            "skipped_rules": [],
            "status": "error",
            "message": str(e)
        }

@app.post("/generate-table-rules")
async def generate_table_rules(table_request: TableRequest):
    try:
        SUPABASE_URL = "https://rpeprbgnzgzesdzlhonh.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJwZXByYmduemd6ZXNkemxob25oIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Mjk5MDAzMywiZXhwIjoyMDU4NTY2MDMzfQ.9IO4qN0Xwl_730DnjBykMKURjPJAmCgPUF--EiCnY-8"
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get sample data (first 10 rows)
        data_result = supabase.table(table_request.table_name).select("*").limit(10).execute()
        
        # Format the information for the AI
        table_info = {
            "table_name": table_request.table_name,
            "sample_data": data_result.data
        }
        
        # Create system message
        system_message = """You are an expert in Great Expectations (GX) and data quality.
        Analyze the provided table schema and sample data to generate appropriate data quality rules.
        Generate ONLY the core expectation rules without any additional information.
        
        Consider the following aspects:
        1. Data types and format validation
        2. Null checks based on is_nullable
        3. Value ranges based on sample data
        4. Unique constraints where appropriate
        5. Pattern matching for strings (email, etc.)
        6. Business logic rules based on data patterns
        
        Format: One rule per line, only the expect_* statements.
        Use single quotes for strings and no backslashes.
        
        Examples of correct rule formats:
        expect_column_values_to_be_unique('id')
        expect_column_values_to_not_be_null('name')
        expect_column_values_to_be_of_type('id', 'integer')
        expect_column_values_to_be_of_type('name', 'string')
        expect_column_values_to_be_between('age', min_value=0, max_value=120)
        expect_column_values_to_match_regex('email', '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')
        expect_column_values_to_be_in_set('status', ['active', 'inactive', 'pending'])
        expect_column_values_to_be_in_type_list('amount', ['integer', 'float', 'decimal'])"""

        # Create user message
        user_message = f"Generate Great Expectations rules for this table:\n{json.dumps(table_info, indent=2)}"

        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract and format the rules
        generated_rules = response.choices[0].message.content.strip().split("\n")
        
        # Clean up the rules format
        cleaned_rules = []
        for rule in generated_rules:
            if rule.strip():
                # Remove backslashes before quotes
                cleaned_rule = rule.replace('\\"', '"')
                # Convert double quotes to single quotes
                cleaned_rule = cleaned_rule.replace('"', "'")
                cleaned_rules.append(cleaned_rule)

        # Extract column names from rules and prepare rule entries
        rule_entries = []
        skipped_rules = []
        
        for rule in cleaned_rules:
            # Extract column name from the rule using regex
            import re
            match = re.search(r"'([^']*)'", rule)
            if match:
                column_name = match.group(1)
                
                # Check if rule already exists
                if await check_existing_rule(supabase, table_request.table_name, column_name, rule):
                    print(f"Rule already exists for column {column_name}: {rule}")
                    skipped_rules.append(rule)
                    continue
                    
                rule_entries.append({
                    "table_name": table_request.table_name,
                    "column_name": column_name,
                    "expectation_rules": rule,
                    "created_at": "NOW()"
                })

        # Save rules to database
        if rule_entries:
            try:
                save_result = supabase.table("data_quality_rules").insert(rule_entries).execute()
                return {
                    "table_name": table_request.table_name,
                    "rules": cleaned_rules,
                    "rule_entries": rule_entries,
                    "skipped_rules": skipped_rules,
                    "sample_data": data_result.data,
                    "status": "success",
                    "message": f"Successfully generated and saved {len(rule_entries)} rules for table {table_request.table_name}. Skipped {len(skipped_rules)} existing rules.",
                    "saved_rules": save_result.data
                }
            except Exception as save_error:
                return {
                    "table_name": table_request.table_name,
                    "rules": cleaned_rules,
                    "rule_entries": rule_entries,
                    "skipped_rules": skipped_rules,
                    "sample_data": data_result.data,
                    "status": "error",
                    "message": f"Generated rules but failed to save: {str(save_error)}"
                }
        else:
            return {
                "table_name": table_request.table_name,
                "rules": cleaned_rules,
                "rule_entries": [],
                "skipped_rules": skipped_rules,
                "sample_data": data_result.data,
                "status": "info",
                "message": f"No new rules to save. All {len(skipped_rules)} rules already exist in the database."
            }

    except Exception as e:
        return {
            "table_name": table_request.table_name,
            "rules": [],
            "rule_entries": [],
            "skipped_rules": [],
            "sample_data": [],
            "status": "error",
            "message": str(e)
        }

@app.post("/save-rules")
async def save_rules(rules_data: dict):
    try:
        SUPABASE_URL = "https://rpeprbgnzgzesdzlhonh.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJwZXByYmduemd6ZXNkemxob25oIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Mjk5MDAzMywiZXhwIjoyMDU4NTY2MDMzfQ.9IO4qN0Xwl_730DnjBykMKURjPJAmCgPUF--EiCnY-8"
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print("Received rules data:", rules_data)
        
        # Check for existing rules and filter out duplicates
        rule_entries = []
        skipped_rules = []
        
        for rule in rules_data["rule_entries"]:
            print(f"Processing rule: {rule}")
            if await check_existing_rule(supabase, rule["table_name"], rule["column_name"], rule["expectation_rules"]):
                print(f"Rule already exists for column {rule['column_name']}: {rule['expectation_rules']}")
                skipped_rules.append(rule)
                continue
            print(f"Adding new rule for column {rule['column_name']}: {rule['expectation_rules']}")
            rule_entries.append(rule)
        
        print(f"Total rules to save: {len(rule_entries)}")
        print(f"Total rules skipped: {len(skipped_rules)}")
        
        # Save only new rules to the database
        if rule_entries:
            try:
                print("Attempting to save rules to database...")
                save_result = supabase.table("data_quality_rules").insert(rule_entries).execute()
                print("Save result:", save_result)
                
                if save_result.data:
                    return {
                        "status": "success",
                        "message": f"Successfully saved {len(rule_entries)} rules. Skipped {len(skipped_rules)} existing rules.",
                        "saved_rules": save_result.data,
                        "skipped_rules": skipped_rules
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Failed to save rules: No data returned from database",
                        "skipped_rules": skipped_rules
                    }
            except Exception as save_error:
                print(f"Error saving rules: {str(save_error)}")
                return {
                    "status": "error",
                    "message": f"Failed to save rules: {str(save_error)}",
                    "skipped_rules": skipped_rules
                }
        else:
            return {
                "status": "info",
                "message": f"No new rules to save. All {len(skipped_rules)} rules already exist in the database.",
                "skipped_rules": skipped_rules
            }
            
    except Exception as e:
        print(f"General error in save_rules: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to save rules: {str(e)}"
        }

@app.get("/validation-results")
async def get_validation_results():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        results = supabase.table("validation_results").select("*").execute()
        return results.data
    except Exception as e:
        return {"error": str(e)}

def parse_expectation_rule(rule_str: str) -> tuple:
    """Parse the expectation rule string into method name and arguments"""
    try:
        print(f"\nParsing rule: {rule_str}")
        
        # Extract method name and arguments using regex
        match = re.match(r"(\w+)\((.*)\)", rule_str)
        if not match:
            print(f"Warning: Rule format doesn't match expected pattern: {rule_str}")
            return None, None
        
        method_name = match.group(1)
        args_str = match.group(2)
        
        print(f"Method name: {method_name}")
        print(f"Arguments string: {args_str}")
        
        # Initialize kwargs with column name
        kwargs = {}
        
        # Try to extract column name first (most rules have this)
        column_match = re.search(r"'([^']*)'", args_str)
        if column_match:
            kwargs["column"] = column_match.group(1)
            print(f"Found column: {kwargs['column']}")
        
        # Parse arguments based on method type
        if method_name == "expect_column_values_to_not_be_null":
            if not kwargs.get("column"):
                print("Warning: No column name found for null check")
                return None, None
            
        elif method_name == "expect_column_values_to_be_unique":
            if not kwargs.get("column"):
                print("Warning: No column name found for uniqueness check")
                return None, None
            
        elif method_name == "expect_column_values_to_be_between":
            if not kwargs.get("column"):
                print("Warning: No column name found for range check")
                return None, None
            
            # Extract min and max values
            remaining_args = args_str.replace(f"'{kwargs['column']}',", "").strip()
            
            # Handle both formats:
            # 1. min_value=18, max_value=99
            # 2. 18, 99
            if "min_value=" in remaining_args and "max_value=" in remaining_args:
                # Format 1: min_value=18, max_value=99
                min_match = re.search(r"min_value=(\d+)", remaining_args)
                max_match = re.search(r"max_value=(\d+)", remaining_args)
                
                if min_match and max_match:
                    kwargs["min_value"] = int(min_match.group(1))
                    kwargs["max_value"] = int(max_match.group(1))
                else:
                    print("Warning: Could not extract min_value and max_value from named parameters")
                    return None, None
            else:
                # Format 2: 18, 99
                min_max = [val.strip() for val in remaining_args.split(",")]
                
                if len(min_max) != 2:
                    print(f"Warning: Invalid number of arguments for range check: {min_max}")
                    return None, None
                
                try:
                    min_val = int(min_max[0]) if min_max[0].strip() else None
                    max_val = int(min_max[1]) if min_max[1].strip() else None
                    
                    if min_val is not None:
                        kwargs["min_value"] = min_val
                    if max_val is not None:
                        kwargs["max_value"] = max_val
                    
                    if min_val is None and max_val is None:
                        print("Warning: Both min and max values are empty in range check")
                        return None, None
                    
                except ValueError as e:
                    print(f"Warning: Could not parse min/max values: {e}")
                    return None, None
            
            print(f"Range check parameters: min_value={kwargs.get('min_value')}, max_value={kwargs.get('max_value')}")
            
        elif method_name == "expect_column_values_to_match_regex":
            if not kwargs.get("column"):
                print("Warning: No column name found for regex check")
                return None, None
            
            # Extract regex pattern
            parts = re.findall(r"'([^']*)'", args_str)
            if len(parts) >= 2:
                kwargs["regex"] = parts[1]
            else:
                print("Warning: No regex pattern found")
                return None, None
            
        elif method_name == "expect_column_values_to_be_in_set":
            if not kwargs.get("column"):
                print("Warning: No column name found for set check")
                return None, None
            
            # Extract set values
            set_match = re.search(r"\[(.*?)\]", args_str)
            if set_match:
                set_values = [val.strip().strip("'") for val in set_match.group(1).split(",")]
                kwargs["value_set"] = set_values
            else:
                print("Warning: No set values found")
                return None, None
            
        elif method_name == "expect_column_values_to_be_of_type":
            if not kwargs.get("column"):
                print("Warning: No column name found for type check")
                return None, None
            
            # Extract and map type
            parts = re.findall(r"'([^']*)'", args_str)
            if len(parts) >= 2:
                type_ = parts[1].lower()
                type_mapping = {
                    "string": "str",
                    "integer": "int64",
                    "int": "int64",
                    "float": "float64",
                    "boolean": "bool"
                }
                kwargs["type_"] = type_mapping.get(type_, type_)
            else:
                print("Warning: No type specified")
                return None, None
            
        else:
            print(f"Warning: Unsupported expectation type: {method_name}")
            if not kwargs.get("column"):
                print("Warning: No column name found for default case")
                return None, None
        
        print(f"Successfully parsed rule. Method: {method_name}, Args: {kwargs}")
        return method_name, kwargs
        
    except Exception as e:
        print(f"Error parsing rule '{rule_str}': {str(e)}")
        return None, None

async def get_validation_rules(supabase: Client, table_name: str) -> list:
    """Fetch validation rules from data_quality_rules table for the specified table"""
    try:
        response = supabase.table("data_quality_rules").select("*").eq("table_name", table_name).execute()
        rules = response.data
        
        if not rules:
            raise ValueError(f"No validation rules found for table {table_name}")
        
        # Transform rules into Great Expectations format
        formatted_rules = []
        for rule in rules:
            try:
                method_name, kwargs = parse_expectation_rule(rule["expectation_rules"])
                formatted_rule = {
                    "expectation_type": method_name,
                    "kwargs": kwargs
                }
                formatted_rules.append(formatted_rule)
            except Exception as e:
                print(f"Warning: Failed to parse rule {rule['expectation_rules']}: {str(e)}")
                continue
            
        return formatted_rules
    
    except Exception as e:
        raise Exception(f"Failed to fetch validation rules: {str(e)}")


@app.post("/validate-with-rules")
async def validate_with_rules():
    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get data from customers table
        table_name = "customers"
        response = supabase.table(table_name).select("*").execute()
        data = response.data
        
        
        if not data:
            return {
                "status": "error",
                "message": f"No data found in table {table_name}"
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"\nLoaded {len(df)} rows from {table_name} table")
        
        # Get validation rules from database
        rules_response = supabase.table("data_quality_rules").select("*").eq("table_name", table_name).execute()
        rules = rules_response.data
        print(f"\nRetrieved {len(rules)} rules from database:")
        for idx, rule in enumerate(rules, 1):
            print(f"Rule {idx}: {rule['expectation_rules']}")
        
        if not rules:
            return {
                "status": "error",
                "message": f"No validation rules found for table {table_name}"
            }
        
        # Initialize Great Expectations context
        context = ge.get_context(mode="ephemeral")
        
        # Create a Pandas DataFrame datasource
        data_source = context.data_sources.add_pandas("pandas")
        data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")
        batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
        
        # Create an expectation suite
        suite_name = f"{table_name}_suite"
        suite = ge.core.ExpectationSuite(
            name=suite_name,
            expectations=[],
            meta={}
        )
        
        # Create a validator
        validator = context.get_validator(
            batch_request=batch.batch_request,
            expectation_suite=suite
        )
        
        # Add expectations based on rules from database
        print("\nAdding expectations to validator:")
        processed_rules = 0
        skipped_rules = []
        
        for idx, rule in enumerate(rules, 1):
            try:
                print(f"\nProcessing rule {idx}:")
                print(f"Rule content: {rule['expectation_rules']}")
                
                method_name, kwargs = parse_expectation_rule(rule["expectation_rules"])
                if method_name is None or kwargs is None:
                    print(f"Skipping rule {idx} due to parsing error")
                    skipped_rules.append({
                        "index": idx,
                        "rule": rule["expectation_rules"],
                        "reason": "Failed to parse rule"
                    })
                    continue
                    
                print(f"Method name: {method_name}")
                print(f"Arguments: {kwargs}")
                
                # Get the expectation method
                expectation_method = getattr(validator, method_name)
                if expectation_method:
                    expectation_method(**kwargs)
                    print(f"Successfully added expectation: {method_name}")
                    processed_rules += 1
                else:
                    print(f"Warning: Could not find expectation method: {method_name}")
                    skipped_rules.append({
                        "index": idx,
                        "rule": rule["expectation_rules"],
                        "reason": f"Method {method_name} not found"
                    })
            except Exception as e:
                print(f"Error processing rule {idx}: {str(e)}")
                print(f"Rule content: {rule['expectation_rules']}")
                skipped_rules.append({
                    "index": idx,
                    "rule": rule["expectation_rules"],
                    "reason": str(e)
                })
                continue
        
        print("\nRule Processing Summary:")
        print(f"Total rules in database: {len(rules)}")
        print(f"Successfully processed rules: {processed_rules}")
        print(f"Skipped rules: {len(skipped_rules)}")
        if skipped_rules:
            print("\nSkipped Rules Details:")
            for skipped in skipped_rules:
                print(f"Rule {skipped['index']}:")
                print(f"Content: {skipped['rule']}")
                print(f"Reason: {skipped['reason']}")
                print("---")
        
        # Run validation
        print("\nRunning validation...")
        validation_result = validator.validate()
        
        # Extract relevant validation information
        validation_summary = {
            "success": validation_result.success,
            "statistics": validation_result.statistics,
            "results": [],
            "failed_validations": []
        }
        
        print(f"\nTotal results to process: {len(validation_result.results)}")
        
        # Process each validation result
        for idx, result in enumerate(validation_result.results):
            try:
                print(f"\nProcessing result {idx + 1}:")
                print(f"Result object: {result}")
                
                # Get the expectation configuration
                config = result.expectation_config
                print(f"Config object: {config}")
                
                # Convert to dictionary
                if hasattr(config, 'to_json_dict'):
                    config_dict = config.to_json_dict()
                else:
                    config_dict = config.__dict__
                print(f"Config dict: {config_dict}")
                
                # Extract expectation type - handle both camelCase and snake_case
                expectation_type = config_dict.get('expectation_type') or config_dict.get('expectationType')
                if not expectation_type:
                    # Try to extract from the rule string
                    rule_str = config_dict.get('kwargs', {}).get('expectation_rules', '')
                    if rule_str:
                        match = re.match(r"(\w+)\(.*\)", rule_str)
                        if match:
                            expectation_type = match.group(1)
                
                print(f"Expectation type: {expectation_type}")
                
                # Extract column and kwargs
                kwargs = config_dict.get('kwargs', {})
                column = kwargs.get('column', 'N/A')
                print(f"Column: {column}")
                print(f"Success: {result.success}")
                
                result_dict = {
                    "expectation_type": expectation_type,
                    "column": column,
                    "success": result.success,
                    "result": result.result,
                    "rule_details": kwargs
                }
                validation_summary["results"].append(result_dict)
                
                # Add to failed validations if unsuccessful
                if not result.success:
                    validation_summary["failed_validations"].append({
                        "column": column,
                        "expectation_type": expectation_type,
                        "rule_details": kwargs,
                        "result": result.result
                    })
                
            except Exception as e:
                print(f"Error processing result {idx + 1}: {str(e)}")
                print(f"Result object: {result}")
                continue
        
        validation_summary["total_failures"] = len(validation_summary["failed_validations"])
        
        print("\nValidation Summary:")
        print(f"Total rules in database: {len(rules)}")
        print(f"Total results processed: {len(validation_summary['results'])}")
        print(f"Total failures: {validation_summary['total_failures']}")
        print(f"Failed validations: {validation_summary['failed_validations']}")
        
        # Save validation results to Supabase
        try:
            supabase.table("validation_results").insert({
                "table_name": table_name,
                "validation_report": validation_summary
            }).execute()
        except Exception as e:
            print(f"Failed to save validation results: {str(e)}")
        
        return {
            "status": "success",
            "message": "Validation completed successfully",
            "validation_report": validation_summary
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred during validation: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#try:
#    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    #results = supabase.table("customers").select(count="exact").execute()
    #new_row = {"name":"mochii", "email":"mochii@gmail.com","age":20,"country":"japan"}
    #supabase.table("customers").insert(new_row).execute()
    #results = supabase.table("customers").select("*").execute()
#    print("successfully")
#    print(results)
#except Exception as e:
#    print(e)


