import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate



"""
Netflix Movie Recommendation Agent
====================================
A LangChain tool-calling agent that recommends Netflix titles
by genre and optional rating filter using Chroma + OpenAI embeddings.
 
Requirements:
    pip install langchain langchain-openai langchain-chroma chromadb pandas
 
Usage:
    export OPENAI_API_KEY="sk-..."
    python movie_agent.py
"""
 

 
# ─────────────────────────────────────────────
# 1. LOAD & PREPARE THE DATASET
# ─────────────────────────────────────────────
 
def load_documents(csv_path: str) -> list[Document]:
    """
    Load the Netflix CSV and convert each row into a LangChain Document.
    - page_content: human-readable text the embedding model will encode
    - metadata:     structured fields used for filtering
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["description", "listed_in"])   # require these fields
    df = df.fillna("Unknown")
 
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Title: {row['title']}\n"
            f"Type: {row['type']}\n"
            f"Genres: {row['listed_in']}\n"
            f"Description: {row['description']}"
        )
        metadata = {
            "title":        row["title"],
            "type":         row["type"],          # "Movie" | "TV Show"
            "rating":       row["rating"],         # PG, PG-13, TV-MA, etc.
            "release_year": int(row["release_year"]) if str(row["release_year"]).isdigit() else 0,
            "genres":       row["listed_in"],
            "director":     row["director"],
            "duration":     row["duration"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
 
    print(f"✅ Loaded {len(docs)} titles from {csv_path}")
    return docs
 
 
# ─────────────────────────────────────────────
# 2. BUILD (OR LOAD) THE CHROMA VECTOR STORE
# ─────────────────────────────────────────────
 
CHROMA_DIR  = "./chroma_netflix_db"
CSV_PATH    = "netflix_titles.csv"   # update path if needed
 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
 
def get_vector_store() -> Chroma:
    """Return existing Chroma DB if already built, otherwise create it."""
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("📂 Loading existing Chroma vector store...")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
 
    print("🔨 Building Chroma vector store (this may take a minute)...")
    docs = load_documents(CSV_PATH)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("✅ Vector store built and persisted.")
    return db
 
vector_store = get_vector_store()
 
 
# ─────────────────────────────────────────────
# 3. DEFINE THE TOOL
# ─────────────────────────────────────────────
 
# Valid Netflix rating values for reference
VALID_RATINGS = {
    "G", "PG", "PG-13", "R", "NC-17",           # movie ratings
    "TV-Y", "TV-Y7", "TV-G", "TV-PG",           # kids TV
    "TV-14", "TV-MA",                            # teen / mature TV
}
 
@tool
def get_movie_recommendation(genre: str, rating: str = "") -> str:
    """
    Recommends Netflix titles matching the requested genre.
    Optionally filter by content rating (e.g. PG, PG-13, R, TV-MA).
 
    Args:
        genre:  The genre or mood the user wants (e.g. "horror", "romantic comedy").
        rating: Optional content rating to filter by (e.g. "PG", "R", "TV-MA").
                Leave blank to return results of any rating.
 
    Returns:
        A formatted list of up to 5 recommended titles with descriptions.
    """
    # Build Chroma metadata filter
    where_filter = None
    if rating:
        rating_upper = rating.upper().strip()
        if rating_upper not in VALID_RATINGS:
            return (
                f"'{rating}' is not a recognised Netflix rating. "
                f"Valid options are: {', '.join(sorted(VALID_RATINGS))}."
            )
        where_filter = {"rating": {"$eq": rating_upper}}
 
    # Semantic search
    results = vector_store.similarity_search(
        query=genre,
        k=5,
        filter=where_filter,
    )
 
    if not results:
        msg = f"No titles found for genre '{genre}'"
        if rating:
            msg += f" with rating '{rating}'"
        return msg + "."
 
    # Format the response
    lines = [f"🎬 Here are some {genre} recommendations" + (f" rated {rating.upper()}" if rating else "") + ":\n"]
    for i, doc in enumerate(results, start=1):
        m = doc.metadata
        lines.append(
            f"{i}. **{m['title']}** ({m['type']}, {m['release_year']}) — {m['rating']}\n"
            f"   Genres: {m['genres']}\n"
            f"   {doc.page_content.split('Description: ')[-1]}\n"
        )
 
    return "\n".join(lines)
 
 
# ─────────────────────────────────────────────
# 4. BUILD THE TOOL-CALLING AGENT
# ─────────────────────────────────────────────
 
tools = [get_movie_recommendation]
 
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)
 
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly Netflix recommendation assistant. "
        "When the user asks for a movie or show, use the get_movie_recommendation tool "
        "to find matching titles. Always mention the title, year, and a brief description. "
        "If the user mentions a rating preference (e.g. 'family-friendly', 'for kids', 'R-rated'), "
        "map it to the appropriate Netflix rating before calling the tool:\n"
        "  - 'family-friendly' or 'for kids' → PG or TV-G\n"
        "  - 'teen'            → PG-13 or TV-14\n"
        "  - 'mature' or 'adult' → R or TV-MA\n"
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
 
agent        = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
 
 
# ─────────────────────────────────────────────
# 5. RUN THE AGENT  (interactive loop)
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n🎥 Netflix Recommendation Agent ready! (type 'quit' to exit)\n")
 
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in {"quit", "exit"}:
            print("Goodbye! 🍿")
            break
 
        response = agent_executor.invoke({"input": user_input})
        print(f"\nAgent: {response['output']}\n")