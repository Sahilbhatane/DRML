# DRML Project

## Environment Setup

This project uses environment variables to securely store API keys. To set up your environment:

1. Copy the `.env.example` file to a new file named `.env`
2. Replace the placeholder values in `.env` with your actual API keys
3. Make sure never to commit your `.env` file to Git (it's already in `.gitignore`)

```bash
cp .env.example .env
# Then edit the .env file with your actual API keys
```

## Deployment

When deploying this application:

1. For services like Heroku, Vercel, or Netlify, use their environment variable settings in the dashboard
2. For cloud providers like AWS or Azure, use their secrets management services
3. For Docker deployments, pass environment variables using the `-e` flag or environment files

Never hardcode API keys directly in your code or commit them to your repository. 

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Google Chrome (recommended for screenshot functionality)

### Installation

1. Clone the repository
   ```bash
   git clone [repository-url]
   cd DRML
   ```

2. Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables as described in the "Environment Setup" section above

### Running the Application

To run the main Streamlit application:
```bash
cd demo
streamlit run app.py
```

The application will start and be accessible in your web browser at http://localhost:8501 by default. 


### Dataset and refrencences- 

Dataset is of [SALT-NLP](https://huggingface.co/datasets/SALT-NLP/Sketch2Code)