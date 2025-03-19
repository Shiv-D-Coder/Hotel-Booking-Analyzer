# Hotel Booking Data RAG API

This project implements a Retrieval-Augmented Generation (RAG) system for querying and analyzing hotel booking data. It uses FastAPI to create an API that allows users to ask questions about the data and generate analytics reports.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
  - [Environment Variables](#environment-variables)
  - [Running the API](#running-the-api)
  - [Endpoints](#endpoints)
- [API Endpoints](#api-endpoints)
  - [/ask](#ask)
  - [/analytics](#analytics)
  - [/health](#health)
- [Sample Requests](#sample-requests)
  - [Ask a Question](#ask-a-question)
  - [Get Analytics](#get-analytics)
- [RAG Implementation](#rag-implementation)
- [Data](#data)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Overview

The Hotel Booking Data RAG API is designed to provide an interface for querying and analyzing hotel booking data using natural language. It combines FastAPI for API management with a RAG system implemented using FAISS for efficient data retrieval and Groq's `llama-3.1-8b-instant` model for response generation.

## File Structure

.
├── .env # Environment variables

├── api.py # FastAPI application code

├── RAG.py # RAG code with FAISS implementation

├── RAG_with_FastAPI.py # Sample code combining RAG and FastAPI in a single file

├── Hotel_Booking_EDA.ipynb # Jupyter Notebook for Exploratory Data Analysis (EDA)

├── Data/ # Data directory

│ ├── hotel_bookings.csv # Original hotel bookings data

│ ├── processed_data.csv # Processed data with cleaning

│ └── sample_processed_data.csv # Sample data (1000 rows) for testing

└── README.md # This README file

text

## Dependencies

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `faiss-cpu`: For efficient similarity search and clustering of dense vectors.
- `scikit-learn`: For data normalization.
- `groq`: To interact with the Groq API and use the `llama-3.1-8b-instant` model.
- `fastapi`: For building the API.
- `uvicorn`: ASGI server for running the API.
- `python-dotenv`: For loading environment variables from a `.env` file.

To install all dependencies, run:

pip install pandas numpy faiss-cpu scikit-learn groq fastapi uvicorn python-dotenv

text

## Setup

1.  **Clone the Repository:**

git clone [repository_url]
cd [repository_directory]

text

2.  **Create and Activate a Virtual Environment:**

python3 -m venv venv
source venv/bin/activate # On Linux or macOS
venv\Scripts\activate # On Windows

text

3.  **Install Dependencies:**

pip install -r requirements.txt

text

## Usage

### Environment Variables

Create a `.env` file in the project root directory and add the following variables:

GROQ_API_KEY=YOUR_GROQ_API_KEY
CSV_PATH=Data/processed_data1.csv
PORT=8000

text

-   `GROQ_API_KEY`: Your Groq API key.
-   `CSV_PATH`: Path to the CSV file containing hotel booking data.
-   `PORT`: Port number for the API to listen on.

### Running the API

Run the FastAPI application using Uvicorn:

uvicorn api:app --reload --host 0.0.0.0 --port 8000

text

### Endpoints

The API provides the following endpoints:

-   `/ask`: Answers questions about hotel booking data.
-   `/analytics`: Generates analytics reports.
-   `/health`: Checks the health of the API.

## API Endpoints

### /ask

-   **Method:** `POST`
-   **Description:** Answers questions about hotel booking data using the RAG system.

**Request Body:**

{
"question": "What is the average daily rate?",
"top_k": 5
}

text

**Response:**

{
"question": "What is the average daily rate?",
"answer": "The average daily rate is $120.",
"context": [
"ADR: $120 Country: USA Lead Time: 30 days ...",
"ADR: $115 Country: UK Lead Time: 25 days ..."
],
"performance": {
"total_time": 1.234,
"retrieval_time": 0.456
}
}

text

### /analytics

-   **Method:** `POST`
-   **Description:** Generates analytics reports about hotel booking data.

**Request Body:**

{
"report_type": "summary"
}

text

**Response:**

{
"report_type": "summary",
"data": {
"total_bookings": 999,
"cancellation_rate": 15.5,
"avg_lead_time_days": 45,
"avg_total_stay_nights": 4,
"avg_weekend_nights": 2,
"avg_weekday_nights": 2,
"avg_daily_rate": 120,
"top_countries": {
"USA": 500,
"UK": 300,
"India": 199
}
},
"performance": {
"processing_time": 0.789
}
}

text

### /health

-   **Method:** `GET`
-   **Description:** Checks the health of the API.

**Response:**

{
"status": "healthy",
"rag_system_initialized": true,
"timestamp": "2025-03-19T20:12:00"
}

text

## Sample Requests

### Ask a Question

curl -X POST http://127.0.0.1:8000/ask
-H "Content-Type: application/json"
-d '{"question": "What is the average daily rate?", "top_k": 5}'

text

### Get Analytics

curl -X POST http://127.0.0.1:8000/analytics
-H "Content-Type: application/json"
-d '{"report_type": "summary"}'

text

## RAG Implementation

The RAG implementation is located in `RAG.py`. It involves the following steps:

1.  **Data Loading:** Loads the hotel booking data from a CSV file.
2.  **Data Preprocessing:** Cleans and prepares the data for embedding generation.
3.  **Embedding Generation:** Generates embeddings for each row of data using a simplified method.
4.  **FAISS Indexing:** Stores the embeddings in a FAISS index for efficient similarity search.
5.  **Query Processing:** Retrieves relevant data chunks from the FAISS index based on a user's question.
6.  **Response Generation:** Uses Groq's `llama-3.1-8b-instant` model to generate a response based on the retrieved data.

## Data

The `Data` directory contains the following files:

-   `hotel_bookings.csv`: Original hotel bookings data.
-   `processed_data.csv`: Processed data with cleaning.
-   `sample_processed_data.csv`: Sample data (1000 rows) for testing.

## Future Enhancements

-   Implement more sophisticated embedding models (e.g., Sentence Transformers).
-   Improve prompt engineering for better response generation.
-   Add more comprehensive analytics reports.
-   Implement caching mechanisms to improve API performance.
-   Incorporate user authentication and authorization.
