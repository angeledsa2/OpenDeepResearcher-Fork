import nest_asyncio
nest_asyncio.apply()
import asyncio
import aiohttp
import json
import os

# ===========================
# CONFIGURATION
# ===========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search"
JINA_BASE_URL = "https://r.jina.ai/"

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
MAX_ITERATIONS = 7

# Dynamic topic memory
QUERY_HISTORY = {}
CONTEXT_CACHE = {}

# ============================
# ASYNCHRONOUS HELPERS
# ============================
async def call_openrouter_async(session, messages):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": DEFAULT_MODEL, "messages": messages}
    async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
        return await resp.json() if resp.status == 200 else None


async def generate_topic_tree_async(session, high_level_goal):
    """Generates a research roadmap as an ontology tree and single query paragraph."""
    prompt = (
        f"Given the high-level goal: '{high_level_goal}', generate:\n"
        "- A hierarchical topic tree with 3 levels.\n"
        "- A single paragraph combining the most critical research questions for the tree.\n"
        "Format output as JSON with keys 'topic_tree' and 'question_paragraph'."
    )
    messages = [{"role": "user", "content": prompt}]
    response = await call_openrouter_async(session, messages)
    if response:
        return json.loads(response['choices'][0]['message']['content'])
    return {"topic_tree": {}, "question_paragraph": ""}


async def perform_search_async(session, query):
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google"}
    async with session.get(SERPAPI_URL, params=params) as resp:
        results = await resp.json() if resp.status == 200 else {}
        return [r['link'] for r in results.get('organic_results', []) if 'link' in r]


async def fetch_and_score_page_async(session, url, research_goal):
    """Fetch page and evaluate relevance."""
    full_url = f"{JINA_BASE_URL}{url}"
    async with session.get(full_url, headers={"Authorization": f"Bearer {JINA_API_KEY}"}) as resp:
        text = await resp.text() if resp.status == 200 else ""
    if not text:
        return None

    # Evaluate relevance
    eval_prompt = (
        f"Does this text contain relevant information for: '{research_goal}'? "
        "Score relevance from 1 to 10, then extract core points if >=6."
    )
    messages = [{"role": "user", "content": f"{eval_prompt}\n\nContent:\n{text[:3000]}"}]
    response = await call_openrouter_async(session, messages)
    return response['choices'][0]['message']['content'].strip() if response else None


# ============================
# MAIN RESEARCH LOGIC
# ============================
async def conduct_adaptive_research(goal):
    async with aiohttp.ClientSession() as session:
        tree = await generate_topic_tree_async(session, goal)
        paragraph = tree.get("question_paragraph", "")
        topic_tree = tree.get("topic_tree", {})

        if not paragraph:
            print("Failed to generate query paragraph.")
            return

        # Generate queries dynamically from paragraph
        queries = paragraph.split("?")
        queries = [q.strip() + "?" for q in queries if len(q) > 5]

        all_contexts = []
        iteration = 0

        while iteration < MAX_ITERATIONS:
            print(f"\n🔄 Iteration {iteration+1}")
            tasks = [perform_search_async(session, q) for q in queries]
            search_results = await asyncio.gather(*tasks)

            unique_links = {link for results in search_results for link in results}
            print(f"🌐 Found {len(unique_links)} unique links.")

            # Fetch and score pages
            tasks = [fetch_and_score_page_async(session, link, goal) for link in unique_links]
            results = await asyncio.gather(*tasks)

            # Filter relevant content
            relevant_content = [r for r in results if r and "Score:" in r and int(r.split("Score:")[1][:1]) >= 6]
            all_contexts.extend(relevant_content)

            # Adaptive query refinement
            feedback_prompt = (
                "Based on the following extracted information, suggest 3 refined queries to deepen understanding.\n\n"
                f"{all_contexts[:3000]}"
            )
            messages = [{"role": "user", "content": feedback_prompt}]
            refinement = await call_openrouter_async(session, messages)
            new_queries = refinement['choices'][0]['message']['content'].strip() if refinement else ""
            queries = [q.strip() for q in new_queries.split("\n") if len(q) > 5]

            if len(queries) == 0:
                break

            iteration += 1

        # Final report synthesis
        report_prompt = (
            "Synthesize a research report based on the following context.\n\n"
            f"{all_contexts[:10000]}"
        )
        messages = [{"role": "user", "content": report_prompt}]
        final_report = await call_openrouter_async(session, messages)

        print("\n🧠 FINAL REPORT:\n", final_report['choices'][0]['message']['content'] if final_report else "No report generated.")


# ========================
# RESEARCH EXECUTION
# ========================
if __name__ == "__main__":
    research_goal = input("Enter your high-level research goal: ")
    asyncio.run(conduct_adaptive_research(research_goal))
