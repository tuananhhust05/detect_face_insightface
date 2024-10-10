from elasticsearch import Elasticsearch

# Connect to the Elasticsearch instance
es = Elasticsearch("http://localhost:9200")

# Define the index name
index_name = "images"

# Define the mapping
mapping = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text"
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 512  # Number of dimensions for your vectors
            }
        }
    }
}

# Create the index with the defined mapping
try:
    es.indices.delete(index=index_name, ignore=[400, 404])
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")
