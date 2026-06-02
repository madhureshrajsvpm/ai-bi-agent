from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd
import os

load_dotenv()

DATA_PATH = "data/itunes"
FAISS_PATH = "faiss_index"

def load_and_merge():
    album      = pd.read_csv(f"{DATA_PATH}/album.csv")
    artist     = pd.read_csv(f"{DATA_PATH}/artist.csv")
    customer   = pd.read_csv(f"{DATA_PATH}/customer.csv")
    employee   = pd.read_csv(f"{DATA_PATH}/employee.csv")
    genre      = pd.read_csv(f"{DATA_PATH}/genre.csv")
    invoice    = pd.read_csv(f"{DATA_PATH}/invoice.csv")
    inv_line   = pd.read_csv(f"{DATA_PATH}/invoice_line.csv")
    media_type = pd.read_csv(f"{DATA_PATH}/media_type.csv")
    playlist   = pd.read_csv(f"{DATA_PATH}/playlist.csv")
    pl_track   = pd.read_csv(f"{DATA_PATH}/playlist_track.csv")
    track      = pd.read_csv(f"{DATA_PATH}/track.csv")

    print("All 11 tables loaded.")

    # Rename duplicate 'name' columns before merging
    artist.rename(columns={"name": "artist_name"}, inplace=True)
    genre.rename(columns={"name": "genre_name"}, inplace=True)
    media_type.rename(columns={"name": "media_type_name"}, inplace=True)
    track.rename(columns={"name": "track_name", "unit_price": "track_price"}, inplace=True)
    album.rename(columns={"title": "album_title"}, inplace=True)

    sales = (
        inv_line
        .merge(invoice,    on="invoice_id")
        .merge(customer,   on="customer_id")
        .merge(track,      on="track_id")
        .merge(album,      on="album_id")
        .merge(artist,     on="artist_id")
        .merge(genre,      on="genre_id")
        .merge(media_type, on="media_type_id")
    )

    print(f"Sales fact table: {len(sales)} rows, {len(sales.columns)} columns")
    return sales, customer, employee, playlist, pl_track, track

def create_documents(sales, customer, employee):
    docs = []

    for _, row in sales.iterrows():
        text = (
            f"Sale: Customer {row['first_name']} {row['last_name']} "
            f"from {row['billing_city']} {row['billing_country']} "
            f"bought track '{row['track_name']}' "
            f"by artist '{row['artist_name']}' "
            f"in genre '{row['genre_name']}' "
            f"for ${row['track_price']}. "
            f"Invoice date: {row['invoice_date']}. "
            f"Invoice total: ${row['total']}."
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": "sales",
                "country": str(row["billing_country"]),
                "genre": str(row["genre_name"]),
            }
        ))

    for _, row in customer.iterrows():
        text = (
            f"Customer: {row['first_name']} {row['last_name']} "
            f"from {row['city']} {row['state']} {row['country']}. "
            f"Email: {row['email']}. "
            f"Support rep ID: {row['support_rep_id']}."
        )
        docs.append(Document(
            page_content=text,
            metadata={"source": "customer", "country": str(row["country"])}
        ))

    for _, row in employee.iterrows():
        text = (
            f"Employee: {row['first_name']} {row['last_name']} "
            f"title '{row['title']}' "
            f"located in {row['city']} {row['country']}."
        )
        docs.append(Document(
            page_content=text,
            metadata={"source": "employee"}
        ))

    print(f"Created {len(docs)} documents.")
    return docs

def build_faiss_index(docs):
    print("Loading embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding documents into FAISS...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"FAISS index saved to '{FAISS_PATH}/'")
    return vectorstore

if __name__ == "__main__":
    sales, customer, employee, playlist, pl_track, track = load_and_merge()
    docs = create_documents(sales, customer, employee)
    build_faiss_index(docs)
    print("Phase 2 ingestion complete.")

