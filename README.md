# Data Quality Management System

A comprehensive system for managing data quality rules using Great Expectations, Supabase, and Streamlit. This system allows you to generate, store, and validate data quality rules with an intuitive web interface.

## Features

- **AI-Powered Rule Generation**
  - Generate data quality rules using GPT-4
  - Support for both table-based and custom prompt-based rule generation
  - Automatic duplicate detection and handling

- **Rule Management**
  - Store rules in Supabase database
  - View and manage existing rules
  - Automatic rule validation

- **Data Validation**
  - Validate data against defined rules
  - View validation results and statistics
  - Track validation history

- **User Interface**
  - Streamlit-based web interface
  - Easy-to-use forms for rule generation
  - Real-time validation results display

## Prerequisites

- Python 3.8 or higher
- Supabase account and project
- OpenAI API key
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data-quality-system.git
cd data-quality-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

```
data-quality-system/
├── supabase_connect.py    # FastAPI backend with rule generation and validation
├── streamlit_app.py       # Streamlit frontend for user interface
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Usage

1. Start the FastAPI backend:
```bash
uvicorn supabase_connect:app --reload
```

2. Start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Access the application:
- Open your browser and go to `http://localhost:8501`
- Use the sidebar to navigate between different features

## Features in Detail

### Rule Generation
- **Table-based Generation**: Select a table and generate rules based on its schema
- **Custom Prompt**: Generate rules using natural language prompts
- **Duplicate Prevention**: System automatically checks for existing rules

### Rule Management
- View all existing rules
- Track rule creation dates
- Monitor rule usage in validations

### Data Validation
- Run validations on demand
- View detailed validation results
- Track validation statistics

## API Endpoints

### Backend (FastAPI)
- `POST /generate-gx-code`: Generate rules from custom prompt
- `POST /generate-table-rules`: Generate rules from table schema
- `POST /validate-with-rules`: Run data validation
- `GET /validation-results`: Get validation results

## Error Handling

The system includes comprehensive error handling for:
- Database connection issues
- Rule generation failures
- Validation errors
- API communication problems

## Security

- Environment variables for sensitive data
- Input validation
- Error message sanitization
- Secure API key handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- Great Expectations for the validation framework
- Supabase for the database backend
- Streamlit for the web interface
- OpenAI for the GPT-4 integration 