"""
SQL Query Writer Agent
======================

A natural language to SQL query translator for the MindBridge AI Competition.

This agent uses an LLM (via Ollama) to convert natural language questions
into executable SQL queries for a bike store database.

Features:
    - Schema-aware query generation
    - Sample data injection for accurate value matching
    - Few-shot examples for consistent output format
    - Self-correction with retry on SQL errors

Author: Amin Nabavi
Competition: SQL Query Writer Agent Competition (Carleton University)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import duckdb
from db.bike_store import get_schema_info


def get_ollama_client():
    """
    Get Ollama client configured for either Carleton server or local instance.

    Environment Variables:
        OLLAMA_HOST: Server URL (default: http://localhost:11434)
        OLLAMA_API_KEY: API key for Carleton server authentication

    Returns:
        ollama.Client: Configured Ollama client instance.
    """
    import ollama
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    api_key = os.getenv('OLLAMA_API_KEY')
    
    if api_key:
        custom_header = {"x-api-key": api_key}
        return ollama.Client(host=host, headers=custom_header)
    else:
        return ollama.Client(host=host)


def get_model_name() -> str:
    """
    Get the model name from environment or use default.

    Environment Variables:
        OLLAMA_MODEL: Model to use (default: llama3.2)

    Returns:
        str: Name of the model to use.
    """
    return os.getenv('OLLAMA_MODEL', 'llama3.2')


class QueryWriter:
    """
    SQL Query Writer Agent that converts natural language to SQL queries.

    This class is the main interface for the competition evaluation.
    It uses an LLM to translate natural language questions into SQL,
    with built-in validation and self-correction capabilities.

    Attributes:
        db_path (str): Path to the DuckDB database file.
        schema (dict): Database schema information.
        client: Ollama client instance.
        model (str): Name of the LLM model to use.

    Example:
        >>> agent = QueryWriter(db_path='bike_store.db')
        >>> sql = agent.generate_query("How many customers are there?")
        >>> print(sql)
        SELECT COUNT(*) FROM customers;
    """

    def __init__(self, db_path: str = 'bike_store.db'):
        """
        Initialize the QueryWriter.

        Args:
            db_path: Path to the DuckDB database file.
        """
        self.db_path = db_path
        self.schema = get_schema_info(db_path=db_path)
        self.client = get_ollama_client()
        self.model = get_model_name()

    def generate_query(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate a SQL query from a natural language prompt.

        Uses the LLM to translate the question into SQL, validates the result,
        and retries with error feedback if the query is invalid.

        Args:
            prompt: Natural language question from the user.
            max_retries: Maximum number of retry attempts on error.

        Returns:
            A valid SQL query string that answers the question.

        Example:
            >>> agent.generate_query("What are the top 5 most expensive products?")
            'SELECT product_name, list_price FROM products ORDER BY list_price DESC LIMIT 5'
        """
        # Input validation
        if not prompt or not prompt.strip():
            return "SELECT 'Please provide a valid question' AS error;"
        
        prompt = prompt.strip()
        
        # Relevance check
        if not self._is_relevant_question(prompt):
            return "SELECT 'I can only answer questions about the bike store database (products, customers, orders, stores, staff, inventory)' AS message;"
        schema_text = self._format_schema()
        samples = self._get_sample_values(limit=3)
        samples_text = self._format_samples(samples)
        examples = self._get_few_shot_examples()
        
        system_prompt = f"""You are a SQL expert working with a DuckDB database.

Database schema:
{schema_text}

Sample values from each column (to understand the data format):
{samples_text}

Here are some examples of questions and their SQL queries:
{examples}

Generate a SQL query to answer the user's question.

Rules:
- Return ONLY a single SQL query
- Do not wrap the query in markdown code blocks
- Do not include any explanations
- Use the sample values to understand data formats (e.g., state codes like 'CA' not 'California')
- Never return multiple SQL statements - combine into one query if needed
"""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages
                )
            except Exception as e:
                error_msg = str(e)
                if '413' in error_msg or 'too large' in error_msg.lower():
                    return "SELECT 'Your question is too broad. Please ask a more specific question.' AS message;"
                raise e
            
            raw_response = response['message']['content']
            sql = self._extract_sql(raw_response)
            
            # Validate by trying to execute
            error = self._validate_sql(sql)
            
            if error is None:
                return sql.strip()
            
            # Add the failed attempt and error to messages for retry
            messages.append({'role': 'assistant', 'content': sql})
            messages.append({'role': 'user', 'content': f"That query failed with error: {error}\n\nPlease fix the query."})
        # Return last attempt even if it failed
        return sql.strip()

    def _format_schema(self) -> str:
        """
        Format the database schema as a string for the LLM prompt.

        Returns:
            Formatted string representation of all tables and columns.
        """
        schema_parts = []
        for table_name, columns in self.schema.items():
            cols = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            schema_parts.append(f"Table {table_name}: {cols}")
        return "\n".join(schema_parts)

    def _extract_sql(self, response: str) -> str:
        """
        Extract SQL from LLM response, handling thinking models and markdown.

        Removes <think> blocks from models like qwen3 and strips markdown
        code block formatting if present. Also handles multi-statement
        responses by returning only the first statement.

        Args:
            response: Raw response text from the LLM.

        Returns:
            Cleaned SQL query string (single statement).
        """
        # Remove <think>...</think> blocks (for qwen3 and similar models)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```sql\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        cleaned = cleaned.strip()
        
        # Handle multi-statement responses: take only the first statement
        if ';\n' in cleaned:
            statements = [s.strip() for s in cleaned.split(';\n') if s.strip()]
            if statements:
                # Return first statement with semicolon
                cleaned = statements[0]
                if not cleaned.endswith(';'):
                    cleaned += ';'
        
        return cleaned

    def _get_sample_values(self, limit: int = 3) -> dict:
        """
        Get sample values from each table to help the LLM understand the data.

        Args:
            limit: Maximum number of sample values per column.

        Returns:
            Dictionary mapping table names to column samples.
        """
        samples = {}
        con = duckdb.connect(database=self.db_path, read_only=True)
        
        try:
            for table_name, columns in self.schema.items():
                samples[table_name] = {}
                for col in columns:
                    col_name = col['name']
                    query = f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT {limit}"
                    try:
                        result = con.execute(query).fetchall()
                        samples[table_name][col_name] = [row[0] for row in result]
                    except Exception:
                        samples[table_name][col_name] = []
        finally:
            con.close()
        
        return samples

    def _format_samples(self, samples: dict) -> str:
        """
        Format sample values into a readable string for the prompt.

        Args:
            samples: Dictionary of table -> column -> sample values.

        Returns:
            Formatted string showing sample data for each column.
        """
        parts = []
        for table_name, columns in samples.items():
            col_samples = []
            for col_name, values in columns.items():
                if values:
                    formatted_values = [repr(v) for v in values]
                    col_samples.append(f"  {col_name}: {', '.join(formatted_values)}")
            if col_samples:
                parts.append(f"{table_name}:\n" + "\n".join(col_samples))
        
        return "\n\n".join(parts)

    def _get_few_shot_examples(self) -> str:
        """
        Return few-shot examples to guide the model.

        These examples demonstrate the expected input/output format and
        cover various query types: simple selects, filters, joins, and aggregations.

        Returns:
            String containing example question/SQL pairs.
        """
        examples = """
Example 1:
Question: How many products do we have?
SQL: SELECT COUNT(*) FROM products;

Example 2:
Question: Show me all customers from Texas
SQL: SELECT * FROM customers WHERE state = 'TX';

Example 3:
Question: What is the total revenue per store?
SQL: SELECT s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN stores s ON o.store_id = s.store_id
GROUP BY s.store_id, s.store_name
ORDER BY total_revenue DESC;

Example 4:
Question: Which staff members work at Baldwin Bikes?
SQL: SELECT st.first_name, st.last_name
FROM staffs st
JOIN stores s ON st.store_id = s.store_id
WHERE s.store_name = 'Baldwin Bikes';

Example 5:
Question: What are the top 3 best-selling product categories?
SQL: SELECT c.category_name, SUM(oi.quantity) AS total_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN categories c ON p.category_id = c.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_sold DESC
LIMIT 3;
"""
        return examples.strip()

    def _validate_sql(self, sql: str) -> str | None:
        """
        Validate SQL by attempting to execute it with EXPLAIN.

        Args:
            sql: SQL query string to validate.

        Returns:
            None if valid, error message string if invalid.
        """
        con = duckdb.connect(database=self.db_path, read_only=True)
        try:
            con.execute(f"EXPLAIN {sql}")
            return None
        except Exception as e:
            return str(e)
        finally:
            con.close()

    def _is_relevant_question(self, prompt: str) -> bool:
        """
        Check if the question is relevant to the database.

        Uses the LLM to determine if the question can be answered
        with SQL queries against the bike store database.

        Args:
            prompt: The user's question.

        Returns:
            True if relevant, False otherwise.
        """
        check_prompt = f"""You are a classifier. Determine if the following question can be answered by querying a bike store database.

The database contains information about:
- Products (bikes and accessories)
- Customers
- Orders and order items
- Stores and staff
- Brands and categories
- Inventory/stocks

Question: {prompt}

Answer with ONLY 'YES' or 'NO'. Nothing else."""

        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': check_prompt}]
        )
        
        answer = self._extract_sql(response['message']['content']).upper().strip()
        return 'YES' in answer