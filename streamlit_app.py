from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import numpy as np
import db_dtypes
import ast
from google.cloud import bigquery, storage
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings  # AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from openai import AzureOpenAI
import json
import re
import string
import nltk
import datetime
import pytz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

# Set page configuration
st.set_page_config(page_title="Talent Seeker", page_icon="👀")

service_account_file = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
# Configuration for Azure OpenAI and Azure Form Recognizer
openai_api_version = st.secrets["general"]["openai_api_version"]
llm_azure_deployment = st.secrets["general"]["llm_azure_deployment"]
azure_endpoint = st.secrets["general"]["azure_endpoint"]
openai_api_key = st.secrets["general"]["openai_api_key"]
form_recognizer_endpoint = st.secrets["general"]["form_recognizer_endpoint"]
form_recognizer_key = st.secrets["general"]["form_recognizer_key"]
model_version = st.secrets["general"]["model_version"]

# Initialize Azure OpenAI embeddings client
embeddings_model = AzureOpenAIEmbeddings(
    openai_api_key=openai_api_key,
    openai_api_version="2023-05-15",  # Ensure this matches the version you want to use
    azure_endpoint="https://recruitment-and-cultural-fit-selection.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15",
    model="text-embedding-ada-002",  # Use the appropriate model
    chunk_size=	8191  # Explicitly set the chunk size
)

# Initialize the Chat Model
llm = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_version=openai_api_version,
    azure_deployment=llm_azure_deployment,
    model_version=model_version,
    azure_endpoint=azure_endpoint,
    temperature=0
)

### Job

# Initialize BigQuery client
project_id = 'is-resume-445415'
client = bigquery.Client.from_service_account_info(service_account_file)
job_table_name = 'is-resume-445415.is_resume_dataset.job'

def fetch_effective_jobs(): 
    """
    Fetch all job data with status 'effective' from BigQuery and process list fields for better readability.
    """
    # Query to fetch job data
    query = f"""
    SELECT job_id, job_title, responsibility, qualification, technical_skill, preferred_skill, other_information, from_date 
    FROM `{job_table_name}`
    WHERE status = 'effective'
    """
    # Fetch data as a DataFrame
    job_data = client.query(query).to_dataframe()

    # List of fields to process (stored as lists in BigQuery)
    list_fields = ['responsibility', 'qualification', 'technical_skill', 'preferred_skill', 'other_information']

    # Process each list field
    for field in list_fields:
        if field in job_data.columns:
            # Convert list format to bullet points or readable strings
            job_data[field] = job_data[field].apply(
                lambda x: "\n".join([f"- {item}" for item in x]) if isinstance(x, list) else str(x)
            )
    
    return job_data

def normalize_text(text):
    """Normalize text by lowercasing, removing punctuation, numbers, stopwords,
    and applying stemming and lemmatization. Now also handles Unicode characters like en dashes."""
    if text is None:
        return "N/A"
    text = text.lower()
    unicode_replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': '\'',
        '\u2019': '\'', '\u201c': '\"', '\u201d': '\"'
    }
    for unicode_char, ascii_char in unicode_replacements.items():
        text = text.replace(unicode_char, ascii_char)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
    return ' '.join(tokens)

def normalize_info(info):
    """Recursively normalize values in nested data structures while keeping keys unchanged."""
    if isinstance(info, dict):
        return {key: normalize_info(value) for key, value in info.items()}
    elif isinstance(info, list):
        return [normalize_info(item) for item in info]
    elif isinstance(info, str):
        return normalize_text(info)
    return info

def job_details(job_title, job_description):
    """
    Extracts and structures job details using Azure OpenAI from the given job title and job description.
    """
    prompt = f"""
    You are an AI trained to structure job information. Given the job title and description, format the details into structured JSON.

    Job Title:
    {job_title}

    Job Description:
    {job_description}

    Extract and structure the following information:
    - job_title
    - responsibility
    - qualification
    - technical_skill
    - preferred_skill
    - other_information

    Return the structured information as JSON.
    """
    response = llm.predict(prompt)
    return response

def get_embeddings(text_chunks):
    """
    Generates embeddings for a list of text chunks using Azure OpenAI.
    Each text chunk must be a single string.
    """
    embeddings = []
    for chunk in text_chunks:
        try:
            # Assuming embeddings_model.embed_query exists and is functional.
            embedding = embeddings_model.embed_query(chunk)  # Generate embedding for the chunk
            embeddings.append(embedding)  # Append the embedding to the list
        except Exception as e:
            embeddings.append(f"Error: {e}")  # Append error message in place of embedding
    return embeddings

def process_job_data(job_data):
    """
    Process the job data to generate embeddings and organize them into a structured dictionary.
    Ensure each category is correctly formatted as a list of strings before embedding.
    """
    structured_embeddings = {}
    categories = ["job_title", "responsibility", "qualification", "technical_skill", "preferred_skill", "other_information"]

    # Ensure each category is treated as a list of strings
    for category in categories:
        items = job_data.get(category, [])
        if isinstance(items, str):  # Make sure job_title and any other single string entries are treated as a list
            items = [items]
        embeddings = get_embeddings(items)
        structured_embeddings[category] = {i + 1: emb for i, emb in enumerate(embeddings)}

    return {"vector": structured_embeddings}

def handle_job_data(job_df):
    # Generate auto-increment ID for Job DataFrame
    query = f"""
    SELECT MAX(CAST(SUBSTRING(job_id, 4) AS INT)) AS max_id
    FROM `{job_table_name}`
    """
    query_job = client.query(query)
    result = query_job.result()

    max_id = 0
    for row in result:
        max_id = row.max_id or 0

    job_df.insert(0, "job_id", [f"job{i}" for i in range(max_id + 1, max_id + 1 + len(job_df))])

    # Add from_date and to_date for Job DataFrame
    thailand_tz = pytz.timezone("Asia/Bangkok")
    now = datetime.datetime.now(thailand_tz)
    today_date = now.strftime("%Y-%m-%d")

    job_df["status"] = "effective"
    job_df["from_date"] = today_date
    job_df["to_date"] = None

    # Format vector and other_information
    job_df["vector"] = job_df["vector"].apply(lambda x: json.dumps(x) if x else "[]")
    job_df["other_information"] = job_df["other_information"].fillna("").astype(str)

    # Validate from_date
    if job_df["from_date"].isnull().any():
        raise ValueError("from_date contains null values!")

    # Save Job DataFrame to CSV
    formatted_datetime = now.strftime("%Y_%m_%d_%H%M%S")
    job_file_name = f"job_{formatted_datetime}.csv"
    job_df.to_csv(job_file_name, index=False)

    # Upload Job CSV to Cloud Storage
    bucket_name = 'is_job_bucket'
    bucket = storage.Client.from_service_account_info(service_account_file).bucket(bucket_name)
    remote_file_path = f'raw/{job_file_name}'

    blob = bucket.blob(remote_file_path)
    blob.upload_from_filename(job_file_name)
    print(f"Job file uploaded to gs://{bucket_name}/{remote_file_path}")

    # Load Job Data into BigQuery
    uri = f"gs://{bucket_name}/{remote_file_path}"
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.schema = [
        bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("job_title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("responsibility", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("qualification", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("technical_skill", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("preferred_skill", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("other_information", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("vector", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("from_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("to_date", "DATE", mode="NULLABLE"),
    ]
    job_config.skip_leading_rows = 1

    load_job = client.load_table_from_uri(uri, job_table_name, job_config=job_config)

    return load_job

def fetch_job_vector(job_id):
    """
    Fetch the job vector for a specific job ID from BigQuery.

    Args:
    - job_id (str): The unique identifier for the job.

    Returns:
    - dict: The job vector data.
    """
    project_id = 'is-resume-445415'
    client = bigquery.Client.from_service_account_info(service_account_file)
    job_table_name = 'is-resume-445415.is_resume_dataset.job'

    query = f"""
    SELECT vector
    FROM `{job_table_name}`
    WHERE job_id = @job_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("job_id", "STRING", job_id)
        ]
    )

    query_job = client.query(query, job_config=job_config)  # Make an API request.
    results = query_job.result()  # Waits for the query to finish.

    for row in results:
        return json.loads(row.vector)  # Assuming the vector is stored as a JSON string

    return {}  # Return an empty dict if no vector found

# ------------------------------------------------------------- Resume -----------------------------------------------------------------

def extract_text_from_file(file, endpoint, api_key):
    """
    Extracts text from a file using Azure Form Recognizer.
    Args:
    - file (UploadedFile): Streamlit UploadedFile object.
    - endpoint (str): Azure endpoint.
    - api_key (str): Azure API key.
    Returns:
    - str: Extracted text from the file.
    """
    try:
        client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        # Send the file to Azure's document analysis model
        poller = client.begin_analyze_document("prebuilt-read", document=file)

        result = poller.result()

        text = ""
        for page in result.pages:
            for line in page.lines:
                text += line.content + " "
        return text.strip()

    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def extract_resume_details(resume_text, api_key, endpoint):
    """
    Extracts detailed information from resumes using Azure OpenAI, structures the data in JSON format,
    and computes the duration of each educational and professional experience.

    Args:
    - resume_text (str): The complete text of the resume.
    - api_key (str): Azure OpenAI API key.
    - endpoint (str): Azure OpenAI endpoint URL.

    Returns:
    - str: A JSON-formatted string containing the structured resume details with computed durations.
    """
    prompt = f"""
    You are a resume parsing assistant. There are more than one resumes. Given the resume text below, extract all the relevant details,
    return them in a structured JSON format, and calculate the duration in years for each work experience entry.
    Ensure that all dates are correctly formatted and durations are accurately computed. Please ensure to order the details as specified below.

    ----
    Resume Text:
    {resume_text}
    ----

    Extract and include the following details:
    - full_name
    - contact_number
    - email
    - skill:
        - technical
        - non-technical
    - education: Include degree, institution, exact dates or year ranges, and GPA
    - activity: List each with detailed descriptions
    - project: List each with titles and detailed descriptions
    - work_experience: Include company name, role, detailed responsibilities, and exact dates or date ranges, with duration in years
    - certification
    - language: List each with proficiency levels and any applicable scores
    - url (if any)
    - other_information
    - file_name: Use keys from the resume text as file name

    Do only what I tell you to do!
    """

    response = llm.predict(prompt)
    return response

# Define the functions from your request here
def enhanced_normalize_text(text):
    if text is None or not isinstance(text, str):
        return "N/A"
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
    return ' '.join(tokens)

def is_excluded_field(key):
    exclusion_keywords = {'name', 'phonenumber', 'file', 'url', 'contact', 'email'}
    return any(keyword in key.lower().replace(' ', '') for keyword in exclusion_keywords)

def normalize_field(content, is_excluded):
    if isinstance(content, dict):
        return {key: (normalize_field(value, is_excluded) if not is_excluded(key) else value)
                for key, value in content.items()}
    elif isinstance(content, list):
        return [normalize_field(item, is_excluded) for item in content]
    elif isinstance(content, str):
        return enhanced_normalize_text(content)
    return content

def normalize_resume(resumes):
    normalized_resumes = []
    for resume in resumes:
        normalized_resume = {
            key: normalize_field(value, is_excluded_field) if not is_excluded_field(key) else value
            for key, value in resume.items()
        }
        normalized_resumes.append(normalized_resume)
    return normalized_resumes

def flatten_data(data):
    """
    Recursively flattens nested data structures into a single string for embedding.
    """
    if isinstance(data, dict):
        return ' '.join(f"{key}: {flatten_data(value)}" for key, value in data.items() if value)
    elif isinstance(data, list):
        return ' '.join(flatten_data(element) for element in data if element)
    elif isinstance(data, str):
        return data
    return str(data)

# Functions for generating embeddings
def resume_get_embeddings(text_chunks):
    """
    Generates embeddings for a list of text chunks using Azure OpenAI.
    """
    embeddings = []
    for chunk in text_chunks:
        try:
            embedding = embeddings_model.embed_query(chunk)  # Generate embedding for the chunk
            embeddings.append(embedding)  # Append the embedding to the list
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
    return embeddings

def extract_and_embed_resumes(resumes):
    """
    Extracts specified fields from resumes, prepares them for embedding, and obtains embeddings,
    specifically handling only technical skills. Returns a dictionary with numeric indices as keys.
    Each value is another dictionary containing embeddings.
    """
    embedded_resumes_dict = {}
    for index, resume in enumerate(resumes):
        key = index + 1  # Simple numeric keys starting from 1
        fields_to_embed = [
            'education', 'project', 'work_experience', 'certification', 'language'
        ]
        text_data = []

        # Handle Skills separately to focus only on technical skills
        if 'skill' in resume and 'technical' in resume['skill']:
            tech_skills_data = flatten_data(resume['skill'].get('technical', "NaN"))
            text_data.append(tech_skills_data if tech_skills_data != "NaN" else "NaN")
        else:
            text_data.append("NaN")

        # Handle other fields
        for field in fields_to_embed:
            field_data = flatten_data(resume.get(field, "NaN"))
            text_data.append(field_data)

        embeddings = resume_get_embeddings(text_data)
        fields_to_embed = ['technical_skill'] + fields_to_embed  # Update fields to include only technical skills
        embedded_resume = {'vector': dict(zip(fields_to_embed, embeddings))}
        embedded_resumes_dict[key] = embedded_resume

    return embedded_resumes_dict

def process_and_upload_resume_data(data, embedded_resumes_dict, bucket_name, project_id, table_name):
    # Normalize the data
    if 'resumes' in data:
        main_df = pd.json_normalize(data['resumes'])
    else:
        main_df = pd.json_normalize(data)

    # Serialize lists and dictionaries to JSON strings for easier handling
    main_df = main_df.applymap(lambda item: json.dumps(item) if isinstance(item, (list, dict)) else item)

    # Replace None, empty lists, and empty strings with NaN
    main_df.replace({None: np.nan, "[]": np.nan, '': np.nan}, inplace=True)

    # Rename columns explicitly
    rename_columns = {
        'skill.technical': 'technical_skill',
        'skill.non-technical': 'non_technical_skill',
        'other_information': 'other_information',
    }
    main_df.rename(columns=rename_columns, inplace=True)

    # Define columns to extract
    columns_to_extract = [
        'full_name', 'contact_number', 'email', 'file_name',
        'technical_skill', 'non_technical_skill',
        'education', 'project', 'work_experience',
        'certification', 'language', 'url', 'other_information'
    ]
    # Ensure all desired columns are in the DataFrame, filling with NaN if not present
    for col in columns_to_extract:
        if col not in main_df.columns:
            main_df[col] = np.nan

    # Filter the DataFrame to include only the specified columns
    main_df = main_df[columns_to_extract]

    # Convert the dictionary to a DataFrame
    embedded_resume_df = pd.DataFrame.from_dict(embedded_resumes_dict, orient='index')

    # Adjust the index of embedded_resume_df to start from 0
    embedded_resume_df.index = embedded_resume_df.index - 1

    # Perform a left join on the main_df using the index
    resume_df = main_df.join(embedded_resume_df, how='left')

    # Initialize BigQuery client
    client = bigquery.Client.from_service_account_info(service_account_file)

    # Generate auto-increment ID for the DataFrame
    query = f"""
    SELECT MAX(CAST(SUBSTRING(resume_id, 4) AS INT)) AS max_id
    FROM `{table_name}`
    """
    query_job = client.query(query)
    result = query_job.result()
    max_id = 0
    for row in result:
        max_id = row.max_id or 0  # If table is empty, start from 0

    resume_df.insert(0, "resume_id", [f"res{i}" for i in range(max_id + 1, max_id + 1 + len(resume_df))])

    # Add required columns for BigQuery schema
    thailand_tz = pytz.timezone("Asia/Bangkok")
    now = datetime.datetime.now(thailand_tz)
    today_date = now.strftime("%Y-%m-%d")  # Ensure DATE format

    resume_df["create_date"] = today_date
    resume_df["modify_date"] = today_date

    # Ensure all vector values are valid
    resume_df["vector"] = resume_df["vector"].apply(lambda x: json.dumps(x) if x else "[]")

    # Ensure `other_information` is treated as STRING
    resume_df["other_information"] = resume_df["other_information"].fillna("").astype(str)

    # Save DataFrame to a CSV file
    formatted_datetime = now.strftime("%Y_%m_%d_%H%M%S")
    resume_file_name = f"resume_{formatted_datetime}.csv"
    resume_df.to_csv(resume_file_name, index=False)

    # Upload the CSV file to Google Cloud Storage
    bucket = storage.Client.from_service_account_info(service_account_file).bucket(bucket_name)
    remote_file_path = f'raw/{resume_file_name}'

    blob = bucket.blob(remote_file_path)
    blob.upload_from_filename(resume_file_name)
    print(f"File uploaded to gs://{bucket_name}/{remote_file_path}")

    # Load Data into BigQuery
    uri = f"gs://{bucket_name}/{remote_file_path}"
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.schema = [
        bigquery.SchemaField("resume_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("full_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("contact_number", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("email", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("file_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("technical_skill", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("non_technical_skill", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("education", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("project", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("work_experience", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("certification", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("language", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("other_information", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("vector", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("create_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("modify_date", "DATE", mode="REQUIRED"),
    ]
    job_config.skip_leading_rows = 1  # Skip the header row

    load_job = client.load_table_from_uri(uri, table_name, job_config=job_config)
    load_job.result()
    print("Data uploaded to BigQuery successfully.")

#------------------------------------------------------- Result ---------------------------------------------------------

def get_all_tag_vectors(data, base_key=''):
    """
    Recursive function to retrieve all tag vectors and their full path keys from a nested dictionary.
    """
    tag_vectors = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{base_key}_{key}".strip('_')  # Construct a composed key for nested tags
            if isinstance(value, dict):
                # Recursively retrieve nested tags
                tag_vectors.update(get_all_tag_vectors(value, full_key))
            else:
                # Base case: store the vector if available
                if np.any(value):
                    tag_vectors[full_key] = value
    return tag_vectors

def calculate_cosine_similarity(resume_vectors, job_vectors):
    """
    Calculates cosine similarity between each pair of resume and job vectors.
    Organizes results by resume and category.
    """
    similarity_results = {}
    for resume_tag, resume_vec in resume_vectors.items():
        # Correctly split the resume_tag to separate the ID from the category
        parts = resume_tag.split('_', 2)  # Splits into ['Resume', '1', 'Technical Skills']
        resume_id = f"{parts[0]}_{parts[1]}"  # Resume_1
        category = parts[2] if len(parts) > 2 else 'General'

        if resume_id not in similarity_results:
            similarity_results[resume_id] = {}
        if category not in similarity_results[resume_id]:
            similarity_results[resume_id][category] = []

        for job_tag, job_vec in job_vectors.items():
            if np.any(resume_vec) and np.any(job_vec):
                resume_vec = np.array(resume_vec).reshape(1, -1)
                job_vec = np.array(job_vec).reshape(1, -1)
                sim = cosine_similarity(resume_vec, job_vec)[0][0]
                similarity_results[resume_id][category].append((job_tag, sim))

    return similarity_results

def process_similarity_data(embedded_job, embedded_resumes_dict, similarity_results):
    def count_fields(data):
        total = 0
        for key, value in data.items():
            if isinstance(value, dict):
                total += len(value)  # Count fields in each nested dictionary
            else:
                total += 1  # Count the field itself if not a dictionary
        return total

    def calculate_adjusted_average_scores(similarity_results):
        average_scores = {}
        for resume_id, categories in similarity_results.items():
            # Flatten list of tuples to get all scores
            all_scores = [score for category_scores in categories.values() for _, score in category_scores]

            # Filter out non-numeric values and sum up only numeric scores
            numeric_scores = [score for score in all_scores if isinstance(score, (int, float))]

            # Sum up all numeric scores for the resume
            total_score = sum(numeric_scores)
            count_numeric_scores = len(numeric_scores)  # Count of numeric scores

            # Calculate the average based on the count of numeric scores and multiply by 100 to convert it to a percentage
            if count_numeric_scores > 0:
                average_scores[resume_id] = (total_score / count_numeric_scores) * 100
            else:
                average_scores[resume_id] = 0

            # Debug: print the details of the calculation
            print(f"Processed {resume_id}:")
            print(f"  Total Numeric Scores: {count_numeric_scores}")
            print(f"  Total Score: {total_score}")
            print(f"  Average Score: {average_scores[resume_id]:.2f}%")  # Adjusted print format to show percentage

        return average_scores

    # Count fields in job descriptions and resumes
    job_field_count = count_fields(embedded_job)
    resume_field_count = 6  # Assuming all resumes have the same structure
    total_resumes = len(embedded_resumes_dict)  # Total number of resumes

    # Calculate and print adjusted average scores
    adjusted_average_scores = calculate_adjusted_average_scores(similarity_results)
    for resume_id, average_score in adjusted_average_scores.items():
        print(f"Adjusted Average Cosine Similarity for {resume_id}: {average_score:.2f}%")

    return adjusted_average_scores

def fetch_job_details(job_id):
    """
    Fetches comprehensive job details for a specific job ID from BigQuery.

    Args:
    - job_id (str): The unique identifier for the job.

    Returns:
    - dict: Dictionary containing all job details, or an empty dict if no data found.
    """
    project_id = 'is-resume-445415'
    client = bigquery.Client.from_service_account_info(service_account_file)
    job_table_name = 'is-resume-445415.is_resume_dataset.job'

    query = f"""
    SELECT job_title, responsibility, qualification, technical_skill, preferred_skill, other_information
    FROM `{job_table_name}`
    WHERE job_id = @job_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("job_id", "STRING", job_id)
        ]
    )

    query_job = client.query(query, job_config=job_config)  # Make an API request.
    results = query_job.result()  # Waits for the query to finish.

    for row in results:
        return {
            "job_title": row.job_title if row.job_title else "Not Specified",
            "responsibility": row.responsibility if row.responsibility else "Not Specified",
            "qualification": row.qualification if row.qualification else "Not Specified",
            "technical_skill": row.technical_skill if row.technical_skill else "Not Specified",
            "preferred_skill": row.preferred_skill if row.preferred_skill else "Not Specified",
            "other_information": row.other_information if row.other_information else "Not Specified"
        }

    return {}  # Return an empty dict if no job details are found

def generate_resume_comments(resume_data, job_details):
    comments_dict = {}

    # Iterate over each resume in the list using an index
    for index, resume in enumerate(resume_data, start=1):  # Start indexing from 1
        summary_parts = []
        for key, value in resume.items():
            if isinstance(value, list):
                summary_part = ', '.join([str(v) if not isinstance(v, dict) else ', '.join([f"{k}: {v}" for k, v in v.items()]) for v in value])
                summary_parts.append(f"{key.capitalize()}: {summary_part}")
            elif isinstance(value, dict):
                summary_part = ', '.join([f"{k}: {v}" for k, v in value.items()])
                summary_parts.append(f"{key.capitalize()}: {summary_part}")
            else:
                summary_parts.append(f"{key.capitalize()}: {value}")

        # Check if there is a URL and append it to the prompt if it exists
        url = resume.get('url', 'No URL provided')
        url_message = f"URL: {url}" if url != 'No URL provided' else ""

        prompt = f"""
        You are an HR professional tasked with evaluating resumes for a specific job role.

        **Job Description**:
        {job_details}

        **Resume Summary**:
        {', '.join(summary_parts)}
        {url_message}

        Please start your response with an overall suitability rating for the candidate using a fire emoji scale (🔥): 
        - The scale ranges from 1 fire (not suitable) to 5 fires (highly suitable). 
        - Example: Rating (1 - 5): 🔥🔥🔥🔥🔥 (Highly Suitable) for 5 fires.

        Then, provide a detailed and **strictly aligned** assessment of the candidate's suitability for the role by addressing the following points:

        1. **Alignment with Job Requirements**: Strictly evaluate how the candidate's background, skills, and experiences meet the specific requirements of the job description. Highlight any direct matches to the job's key criteria and explain why these aspects make the candidate a strong fit for the role.

        2. **Gaps or Missing Skills**: Identify any critical skills, experiences, or qualifications the candidate lacks based on the job description. Clearly discuss how these gaps might impact their ability to perform in the role.

        3. **Additional Information Needed**: Suggest what further information or clarifications would help HR or the hiring manager make a more informed decision about the candidate's suitability for the position (e.g., specific examples, metrics, certifications).

        Provide your insights in a professional, structured, and concise manner. Ensure your comments are actionable and adhere strictly to the job description. Your evaluation will directly influence the decision-making process regarding the candidate's potential for the role.
        """


        # Simulate an API call to the model to get a response
        response = llm.predict(prompt)  # Replace llm.predict with your actual API call

        # Directly use the response if it's already a string, adjust if your setup is different
        comment = response.strip() if response else "No comment generated."

        # Store the comment using index as the key
        comments_dict[index] = {'comment': comment}

    return comments_dict

def upload_results_to_bigquery(job_id, similarity_results, adjusted_scores, comments):

    # Step 1: Convert similarity_results into a DataFrame
    similarity_df = pd.DataFrame(
        [{"similarity_score": similarity_score} for similarity_score in similarity_results.values()]
    )

    # Add index as a column for joining
    similarity_df['index'] = range(1, len(similarity_df) + 1)

    # Step 2: Convert adjusted_scores into a DataFrame
    adjusted_df = pd.DataFrame(
        [{"adjusted_score": adjusted_score} for adjusted_score in adjusted_scores.values()]
    )
    adjusted_df['index'] = range(1, len(adjusted_df) + 1)

    # Step 3: Convert comments into a DataFrame
    comments_df = pd.DataFrame(
        [{"index": index, "comment": details.get('comment', '')}
         for index, details in comments.items()]
    )

    # Step 4: Left join DataFrames
    result_df = similarity_df.merge(adjusted_df, on="index", how="left") \
                             .merge(comments_df, on="index", how="left")

    # Drop the 'index' column if it's no longer needed
    result_df = result_df.drop(columns=['index'])

    # Initialize BigQuery client and other setup
    project_id = 'is-resume-445415'
    table_name = 'is-resume-445415.is_resume_dataset.result'
    client = bigquery.Client.from_service_account_info(service_account_file)

    # Timezone setting for timestamp
    thailand_tz = pytz.timezone("Asia/Bangkok")
    now = datetime.datetime.now(thailand_tz)
    today_date = now.date()  # Use a datetime.date object

    # Query to find the next available resume ID
    query = f"""
    SELECT MAX(CAST(SUBSTRING(resume_id, 4) AS INT)) AS max_id
    FROM `{table_name}`
    """
    query_job = client.query(query)
    results = query_job.result()
    max_id = 0
    for row in results:
        max_id = (row.max_id or 0) + 1  # Start from 1 if table is empty

    next_id = max_id

    # Assign resume_id dynamically
    result_df['job_id'] = job_id
    result_df['resume_id'] = [f"res{next_id + i}" for i in range(len(result_df))]
    result_df['similarity_score'] = result_df['similarity_score'].apply(json.dumps)


    # Rename columns to match the BigQuery schema
    result_df = result_df.rename(columns={
        'adjusted_score': 'avg_score',
        'similarity_score': 'score'
    })

    # Ensure all required columns are properly serialized
    #result_df['avg_score'] = result_df['avg_score'].astype(str)
    result_df['comment'] = result_df['comment'].astype(str)
    result_df['status'] = 'in process'
    result_df['create_date'] = today_date
    result_df['modify_date'] = today_date

    # Reorder columns to match the BigQuery schema
    result_df = result_df[[
        'job_id', 'resume_id', 'score', 'avg_score', 'comment',
        'status', 'create_date', 'modify_date'
    ]]

    # BigQuery upload configuration
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("resume_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("score", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("avg_score", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("comment", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("create_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("modify_date", "DATE", mode="REQUIRED"),
        ]
    )

    # Upload to BigQuery
    job = client.load_table_from_dataframe(result_df, table_name, job_config=job_config)
    job.result()  # Wait for the job to complete.

    if job.errors is None:
        print("Data uploaded successfully.")
    else:
        print("Errors occurred:", job.errors)

#---------------------------------------------------- Streamlit --------------------------------------------------------

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #DAF7A6 , #ffbb77);
        color: black;
    }
    .css-1d391kg {  /* Targets the sidebar */
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("👀 **Talent Seeker**")
    st.write("This app helps you quickly scan resumes and rank candidates using AI.")
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    st.write("### Navigate to:")
    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("Resume Scanner and Ranker"):
        st.session_state.page = "Resume"
    if st.button("Update Data"):
        st.session_state.page = "Update"

def home_page_view():
    st.title("👋 Welcome to Talent Seeker!")
    st.write("""
        Talent Seeker is an AI-powered platform to help you analyze resumes for your organization.
        Use the sidebar to navigate between different functionalities.
    """)

    st.header("📋 Instructions for Using Talent Seeker")

    st.subheader(":orange[Resume Scanner and Ranker]")
    st.write("""
    1. **Select a Job ID:**
       - Choose an existing job ID from the dropdown list.
       - If the desired position is not listed, follow the next step.
    
    2. **Add a New Job:**
       - Enter the **Job Title** and **Job Description** in the input fields.
       - Click **Process Job Details** to upload the job.
       - Use **Reload Job Data** to refresh and view the newly added job.
    
    3. **Upload Resumes:**
       - Upload **multiple resumes** in **PDF** or **JPG** format using the file uploader.
       - Click **Load Application Data** to process the resumes and analyze their scores and comments.
    
    4. **View Analysis Results:**
       - The app will display **Scores** and **Comments** for each candidate.
       - Expand the applicant details to view:
         - **Contact Information**
         - **Resume File Details**
         - **AI-generated comments** on the candidate's suitability for the role.
    """)

    st.subheader(":orange[Update Data]")
    st.write("""
    1. **Update Job Status:**
       - Select a job from the dropdown list.
       - Update the job's status to **Effective** or **Closed**.

    2. **Update Candidate Status:**
       - Select a job to view its associated candidates.
       - Update the status of each candidate to **Rejected** or **Accepted** using interactive tools.
       - Save changes to ensure the updated statuses are reflected in the system.
    """)

    st.info("This simple workflow enables efficient resume analysis, status tracking, and management of job positions and candidates. Happy talent hunting! 🚀")

def resume_scanner_page():
    st.title(":bookmark_tabs: Resume Scanner and Ranker")

    enter_job = st.checkbox("Do you want to enter a new position?", key="enter_job")
    if enter_job:
        job_title = st.text_input("Job Title", placeholder="Enter the job title here")
        job_description = st.text_area("Job Description", placeholder="Enter the job description here", height=150)

        if st.button("Process Job Details"):
            try:
                job_info = job_details(job_title, job_description)
                job_clean = job_info.replace("```json", "").replace("```", "").strip()
                #st.json(job_clean)  # Displaying cleaned JSON data
                job_data = json.loads(job_clean)

                # Debug print to check data structure
                #st.write("Parsed Job Data:", job_data)

                normalized_job_data = normalize_info(job_data)
                #st.write("Normalized Job Data:", normalized_job_data)  # Debug print

                normalized_job_data = normalize_info(normalized_job_data)

                embedded_job = process_job_data(normalized_job_data)
                #st.write("Embedded Job Data:", embedded_job)  # Debug print
                print(embedded_job)

                # Convert the JSON string to a DataFrame
                job_text_df = pd.read_json(job_clean, orient='index')

                # Transpose the DataFrame if necessary (optional based on your data structure)
                job_text_df = job_text_df.T

                # Replace empty dictionaries and empty lists with NaN
                job_text_df = job_text_df.applymap(lambda x: np.nan if x in [{}, []] else x)

                # Extract the "vector" key
                vector_data = embedded_job.get("vector", {})

                # Add the combined structured vector as a single row
                embedded_job_df = pd.DataFrame([{"vector": vector_data}])

                # Display the resulting DataFrame
                print(embedded_job_df)

                # Adjust the index of embedded_job_df to start from 0 (to match job_text_df)
                embedded_job_df.index = embedded_job_df.index - 0  # This ensures proper alignment with job_text_df's index

                # Perform a left join on the job_text_df using index
                job_df = job_text_df.join(embedded_job_df, how="left")

                #st.dataframe(job_df)
                handle_job_data(job_df)  # Upload to BigQuery
                st.success("Job details successfully uploaded.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.subheader("Available Jobs")
    job_data = fetch_effective_jobs()
    # Reload button to fetch data again
    if st.button("Reload Job Data"):
        job_data = fetch_effective_jobs()

    # Select a job for mapping
    selected_job_id = st.selectbox(
        "Select a job for mapping:", 
        job_data['job_id'],
        format_func=lambda x: f"{x} - {job_data.loc[job_data['job_id'] == x, 'job_title'].values[0]} (date: {job_data.loc[job_data['job_id'] == x, 'from_date'].values[0]})"
    )

    # Filter the selected job
    selected_job = job_data[job_data["job_id"] == selected_job_id]

    # Display selected job details as a table
    #st.dataframe(selected_job)

    # Alternatively, display job details in a user-friendly format
    for _, row in selected_job.iterrows():
        st.write(f"### Job Title: {row['job_title']}")

        # Display responsibilities with bullet points
        st.write("#### Responsibilities:")
        try:
            responsibilities = ast.literal_eval(row['responsibility'])  # Convert string to list if needed
            if isinstance(responsibilities, list):
                st.markdown("\n".join([f"- {item}" for item in responsibilities]))
            else:
                st.markdown(row['responsibility'])
        except (ValueError, SyntaxError):
            st.markdown(row['responsibility'])  # Display as-is if parsing fails

        # Display qualifications
        st.write("#### Qualifications:")
        try:
            qualifications = ast.literal_eval(row['qualification'])  # Convert string to list if needed
            if isinstance(qualifications, list):
                st.markdown("\n".join([f"- {item}" for item in qualifications]))
            else:
                st.markdown(row['qualification'])
        except (ValueError, SyntaxError):
            st.markdown(row['qualification'])  # Display as-is if parsing fails

        # Display technical skills
        st.write("#### Technical Skills:")
        try:
            technical_skills = ast.literal_eval(row['technical_skill'])  # Convert string to list if needed
            if isinstance(technical_skills, list):
                st.markdown("\n".join([f"- {item}" for item in technical_skills]))
            else:
                st.markdown(row['technical_skill'])
        except (ValueError, SyntaxError):
            st.markdown(row['technical_skill'])  # Display as-is if parsing fails

        # Display preferred skills
        st.write("#### Preferred Skills:")
        try:
            preferred_skills = ast.literal_eval(row['preferred_skill'])  # Convert string to list if needed
            if isinstance(preferred_skills, list):
                st.markdown("\n".join([f"- {item}" for item in preferred_skills]))
            else:
                st.markdown(row['preferred_skill'])
        except (ValueError, SyntaxError):
            st.markdown(row['preferred_skill'])  # Display as-is if parsing fails

        # Display other information
        st.write("#### Other Information:")
        try:
            other_info = ast.literal_eval(row['other_information'])  # Convert string to list if needed
            if isinstance(other_info, list):
                st.markdown("\n".join([f"- {item}" for item in other_info]))
            else:
                st.markdown(row['other_information'])
        except (ValueError, SyntaxError):
            st.markdown(row['other_information'])  # Display as-is if parsing fails

    # Fetch and display vector for the selected job
    selected_job_vector = fetch_job_vector(selected_job_id)
    #job_vectors = get_all_tag_vectors(selected_job_vector)

    print(selected_job_vector)
    #print("job_vectors:", job_vectors)

    st.divider()

    st.markdown("Upload **resumes** here (PDF or JPG only)", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["pdf", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Extract and Analyze Resumes"):
            with st.spinner('Processing resumes... Please wait.'):
                extracted_texts = {}

                # Text extraction step
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.lower().endswith((".pdf", ".jpg")):
                        text = extract_text_from_file(uploaded_file, form_recognizer_endpoint, form_recognizer_key)
                        extracted_texts[uploaded_file.name] = text
                        #st.write(f"Text extracted from {uploaded_file.name}:\n{text}\n")
                        print(f"Text extracted from {uploaded_file.name}:\n{text}\n")

                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}")

                print(extracted_texts)

                # Check if texts were successfully extracted
                if extracted_texts:
                    # Analysis step
                        resume_details = extract_resume_details(extracted_texts, openai_api_key, azure_endpoint)
                        #st.write(f"Resume details from {filename}:\n{resume_details}\n")
                        print(resume_details)

                        # Ensure JSON format and remove any unexpected text formatting
                        resume_clean = resume_details.replace("```json", "").replace("```", "").strip() # ออกแค่ resume ล่าสุด ต้อง loop
                        print(resume_clean)

                        data = json.loads(resume_clean)
                        #st.subheader(f"Analyzed Resume for {filename}:")
                        #st.json(data)  # Display cleaned and parsed JSON
                        print(data)

                        # Ensure data structure compatibility
                        resumes_data = data.get("resumes", [data])  # Use 'resumes' if it exists, otherwise treat 'data' as a single resume.

                        # Normalize and process resumes
                        normalized_resumes = normalize_resume(resumes_data)

                        # Process each normalized resume
                        for resume in normalized_resumes:
                            print(resume)

                        embedded_resumes_dict = extract_and_embed_resumes(normalized_resumes)
                        print(embedded_resumes_dict)

                        # Process and upload the data (Assuming these functions are defined)
                        bucket_name = 'is_resume_bucket'
                        project_id = 'is-resume-445415'
                        table_name = 'is-resume-445415.is_resume_dataset.resume'

                        # Further processing and uploading logic
                        process_and_upload_resume_data(data, embedded_resumes_dict, bucket_name, project_id, table_name)

                        # Cosine similarity calculations
                        resume_vectors = get_all_tag_vectors(embedded_resumes_dict)
                        job_vectors = get_all_tag_vectors(selected_job_vector)

                        print("job_vectors:", job_vectors)

                        similarity_results = calculate_cosine_similarity(resume_vectors, job_vectors)
                        adjusted_scores = process_similarity_data(selected_job_vector, embedded_resumes_dict, similarity_results)
                        #st.write("Adjusted Scores:", adjusted_scores)
                        print(similarity_results)
                        print(adjusted_scores)

                        # Generate comments and other post-analysis tasks
                        fetched_job_data = fetch_job_details(selected_job_id)
                        print(fetched_job_data)
                        #st.dataframe(fetched_job_data)

                        # Generate comments for the resumes
                        comments = generate_resume_comments(resumes_data, job_details)
                        print(comments)

                        # for full_name, comment in comments.items():
                        #     st.write(f"Comment for {full_name}: {comment}")
                        # st.success("Successfully processed all resumes.")

                        upload_results_to_bigquery(selected_job_id, similarity_results, adjusted_scores, comments)
                        st.success("Results uploaded to BigQuery.")
                        print("Data uploaded to BigQuery. Please click 'Load Candidate Data' to view the result")

                else:
                        st.warning("No resumes were successfully extracted.")

    st.divider()

    # Fetch and display joined data from BigQuery for the selected job_id
    if selected_job_id:
        st.subheader(f"Applicants for Job ID: :orange[{selected_job_id}] - :orange[{job_data.loc[job_data['job_id'] == selected_job_id, 'job_title'].values[0]}]")
        st.write("Below is the list of applicants ranked by their average score:")

        # "Reload Data" Button
        if st.button("Load Candidate Data"):
            try:
                project_id = 'is-resume-445415'
                client = bigquery.Client.from_service_account_info(service_account_file)
                query = f"""
                SELECT
                    r.resume_id,
                    r.full_name,
                    r.contact_number,
                    r.email,
                    r.url,
                    r.file_name,
                    res.avg_score,
                    res.comment,
                    res.status
                FROM `is-resume-445415.is_resume_dataset.resume` r
                JOIN `is-resume-445415.is_resume_dataset.result` res
                ON r.resume_id = res.resume_id
                WHERE res.job_id = '{selected_job_id}'
                AND res.status IN ('rejected', 'in process')
                ORDER BY CAST(res.avg_score AS FLOAT64) DESC
                """

                query_job = client.query(query)
                results = query_job.result()
                applicants_data = pd.DataFrame([dict(row) for row in results])

                # Rename columns for better readability
                applicants_data = applicants_data.rename(columns={
                    'resume_id': 'Resume ID',
                    'full_name': 'Full Name',
                    'contact_number': 'Contact Number',
                    'email': 'Email',
                    'url': 'URL',
                    'file_name': 'File Name',
                    'avg_score': 'Average Score',
                    'comment': 'Comment'
                })

            except Exception as e:
                st.error(f"An error occurred while fetching applicant data: {e}")
                applicants_data = pd.DataFrame()

            # Display applicants data
            if not applicants_data.empty:
                for _, row in applicants_data.iterrows():
                    st.markdown(
                        f"""
                        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #333; font-size: 18px;">{row['Resume ID']}: {row['Full Name']} (Average Score: {float(row['Average Score']):.2f})</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.expander("View Details"):
                        st.markdown(
                            f"""
                            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                                <p style="margin: 5px 0; font-size: 16px;"><strong>Contact Number:</strong> {row['Contact Number']}</p>
                                <p style="margin: 5px 0; font-size: 16px;"><strong>Email:</strong> {row['Email']}</p>
                                <p style="margin: 5px 0; font-size: 16px;"><strong>URL:</strong> <a href="{row['URL']}" target="_blank">{row['URL']}</a></p>
                                <p style="margin: 5px 0; font-size: 16px;"><strong>File Name:</strong> {row['File Name']}</p>
                                <p style="margin: 10px 0; font-size: 16px;">{row['Comment']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.write("No applicants data available for the selected job.")

    st.divider()    

def update_page():
    st.title(":wrench: Update Data")

    project_id = 'is-resume-445415'
    client = bigquery.Client.from_service_account_info(service_account_file)

    # Select Job to Update
    st.subheader("Update Job Status")
    jobs_query = """
        SELECT job_id, job_title, status, from_date 
        FROM `is-resume-445415.is_resume_dataset.job`
    """
    jobs = client.query(jobs_query).to_dataframe()
    if not jobs.empty:
        selected_job_id = st.selectbox(
        "Select a Job to Update Status",
        jobs["job_id"],
        format_func=lambda x: f"{x} - {jobs[jobs['job_id'] == x]['job_title'].values[0]} (date: {jobs[jobs['job_id'] == x]['from_date'].values[0]})",
        key="select_job_status"  # Unique key for this selectbox
        )
        current_status = jobs[jobs["job_id"] == selected_job_id]["status"].values[0]
        st.write(f"Current Status: **{current_status}**")

        # Update Job Status
        new_job_status = st.selectbox("New Job Status", ["effective", "closed"])
        if st.button("Update Job Status"):
            try:
                update_job_query = f"""
                UPDATE `is-resume-445415.is_resume_dataset.job`
                SET status = '{new_job_status}'
                WHERE job_id = '{selected_job_id}'
                """
                client.query(update_job_query)
                st.success(f"Job status updated to '{new_job_status}' for Job ID: {selected_job_id}")
            except Exception as e:
                st.error(f"An error occurred while updating job status: {e}")
    else:
        st.write("No jobs available to update.")

    st.divider()

    # Select Candidate to Update
    st.subheader("Update Candidate Status")
    job_query = "SELECT job_id, job_title, from_date FROM `is-resume-445415.is_resume_dataset.job`"
    job_list = client.query(job_query).to_dataframe()

    if not job_list.empty:
        # Select a job to view candidates
        job_to_update = st.selectbox(
            "Select a Job to View Candidates", job_list["job_id"],
        format_func=lambda x: f"{x} - {job_list[job_list['job_id'] == x]['job_title'].values[0]} (date: {job_list[job_list['job_id'] == x]['from_date'].values[0]})",
        key="select_job_candidates"  # Unique key for this selectbox
        )

        if job_to_update:
            # Fetch candidates for the selected job
            candidates_query = f"""
                SELECT r.resume_id, r.full_name, res.avg_score, res.status 
                FROM `is-resume-445415.is_resume_dataset.resume` r
                JOIN `is-resume-445415.is_resume_dataset.result` res
                ON r.resume_id = res.resume_id
                WHERE res.job_id = '{job_to_update}'
            """
            candidates = client.query(candidates_query).to_dataframe()

            if not candidates.empty:
                # Map statuses to checkbox values
                candidates["Accepted"] = candidates["status"] == "accepted"

                # Display candidates in an editable table with checkboxes
                edited_candidates = st.data_editor(
                    candidates,
                    column_config={
                        "resume_id": st.column_config.Column(
                            disabled=True, label="Resume ID"
                        ),
                        "full_name": st.column_config.Column(
                            disabled=True, label="Full Name"
                        ),
                        "avg_score": st.column_config.Column(
                            disabled=True, label="Score (%)"
                        ),
                        "Accepted": st.column_config.CheckboxColumn(label="Accepted"),
                    },
                    use_container_width=True
                )

                # Save changes to BigQuery
                if st.button("Update Candidate Status"):
                    try:
                        # Iterate through the edited rows
                        for _, row in edited_candidates.iterrows():
                            # Determine the new status based on the checkbox
                            new_status = "accepted" if row["Accepted"] else "rejected"

                            update_query = f"""
                            UPDATE `is-resume-445415.is_resume_dataset.result`
                            SET status = '{new_status}'
                            WHERE resume_id = '{row["resume_id"]}' AND job_id = '{job_to_update}'
                            """
                            client.query(update_query)

                        st.success("Candidate statuses successfully updated.")
                    except Exception as e:
                        st.error(f"An error occurred while updating statuses: {e}")
            else:
                st.write("No candidates available for the selected job.")
    else:
        st.write("No jobs available to view candidates.")

    st.divider()



def main():
    if st.session_state.page == "Home":
        home_page_view()
    elif st.session_state.page == "Resume":
        resume_scanner_page()
    elif st.session_state.page == "Update":
        update_page()

if __name__ == '__main__':
    main()