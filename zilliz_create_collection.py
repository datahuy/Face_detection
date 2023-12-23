from pymilvus import MilvusClient
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


if __name__ == "__main__":
    
    with open("./authen.txt", "r") as f:
        lines = f.read()
        
    api_endpoint, token = lines.split("\n")
    print(api_endpoint, token)
    
    # Connect to cluster
    client = MilvusClient(uri=api_endpoint, token=token)
    
    print(client.list_collections())
    
    #client.drop_collection(collection_name=client.list_collections()[0])
    
    fields = [FieldSchema(name="id", dtype=DataType.INT64, auto_id=True, is_primary=True),
               FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
               FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512)]
    
    schema = CollectionSchema(fields=fields, 
                              description="Vector Database for Face Recognition", 
                              enable_dynamic_field=False)
    # collection = Collection(name="face_recognition_ip", 
    #                         schema=schema)
    
    collection_name = "face_recognition_ip"
    
    
    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": "L2",
        "params": {}
    }
    
    # collection.create_index(
    #     field_name="vector", 
    #     index_params=index_params,
    #     index_name="vector_index"
    # )
    # client._create_index(collection_name=collection_name,
    #                      vec_field_name="vector",
    #                      index_params=index_params)
    client.create_collection_with_schema(collection_name=collection_name,
                                         schema=schema,
                                         index_param=index_params)
    
    # 6. Load collection
    #collection.load()
    client._load(collection_name=collection_name)

    # Get loading progress
    #progress = utility.loading_progress(collection_name=collection_name)

    client.close()