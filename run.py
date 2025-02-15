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
MAX_CONCURRENT_REQUESTS = 5

async def call_openrouter_async(session, messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:3000",
        "X-Title": "Research Assistant"
    }
    payload = {"model": DEFAULT_MODEL, "messages": messages}
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            print(f"OpenRouter API error: Status {resp.status}")
            return None
    except Exception as e:
        print(f"Error calling OpenRouter: {str(e)}")
        return None

async def generate_initial_queries(session, goal):
    prompt = (
        f"For this topic: '{goal}'\n"
        "Generate 3 academic search queries optimized for finding quantitative research. Format:\n"
        "- Start with research type (meta-analysis, longitudinal study, clinical trial)\n"
        "- Include measurement terms (correlation, effect size, statistical significance)\n"
        "- Specify variables and outcomes\n"
        "- Use academic databases syntax (AND, OR, quotation marks)\n"
        'Examples:\n'
        '"meta-analysis" AND "schumann resonance" AND "cognitive performance" measurement\n'
        '"longitudinal study" AND "geomagnetic activity" AND "physiological effects"\n'
        '"clinical trial" AND "electromagnetic field exposure" AND outcomes\n'
        "Return ONLY 3 queries, one per line."
    )
    messages = [{"role": "user", "content": prompt}]
    response = await call_openrouter_async(session, messages)
    
    if not response or 'choices' not in response:
        return []
        
    content = response['choices'][0]['message']['content'].strip()
    return [q.strip() for q in content.split('\n') if len(q.strip()) > 5]

async def perform_search_async(session, query):
    try:
        params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google"}
        async with session.get(SERPAPI_URL, params=params) as resp:
            if resp.status == 200:
                results = await resp.json()
                return [r['link'] for r in results.get('organic_results', [])[:5] if 'link' in r]
            return []
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

async def fetch_and_score_page_async(session, url, research_goal):
    try:
        full_url = f"{JINA_BASE_URL}{url}"
        print(f"📄 Analyzing: {url}")
        
        async with session.get(full_url, headers={"Authorization": f"Bearer {JINA_API_KEY}"}) as resp:
            if resp.status != 200:
                print(f"❌ Failed to fetch: {url}")
                return None
            text = await resp.text()

        if not text.strip():
            return None

        eval_prompt = (
            "Evaluate this text and return ONLY a single-line JSON object:\n"
            '{"score":N,"points":["point1","point2"],"source":"CITATION","url":"URL"}\n'
            f"Rate relevance 1-10 for: {research_goal}\n"
            f"URL: {url}\n"
            "Text:\n" + text[:3000]
        )
        messages = [{"role": "user", "content": eval_prompt}]
        response = await call_openrouter_async(session, messages)
        
        if not response or 'choices' not in response:
            return None
            
        try:
            content = response['choices'][0]['message']['content'].strip()
            evaluation = json.loads(content)
            if evaluation['score'] >= 6:
                print(f"✅ Found relevant content from: {url}")
                return evaluation
            return None
        except json.JSONDecodeError as e:
            print(f"⚠️ Parse error for {url}: {str(e)}")
            return None
            
    except Exception as e:
        print(f"⚠️ Error processing {url}: {str(e)}")
        return None

async def process_urls_with_semaphore(session, urls, research_goal, semaphore):
    async def process_single_url(url):
        async with semaphore:
            return await fetch_and_score_page_async(session, url, research_goal)
    
    tasks = [process_single_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]

async def refine_queries(session, context_data):
    prompt = (
        "Based on the research findings, generate 3 new academic search queries to explore gaps.\n"
        "Use same format as initial queries - return ONLY 3 queries, one per line.\n\n"
        "Research findings:\n" + json.dumps(context_data, indent=2)
    )
    messages = [{"role": "user", "content": prompt}]
    response = await call_openrouter_async(session, messages)
    
    if not response or 'choices' not in response:
        return []
        
    content = response['choices'][0]['message']['content'].strip()
    return [q.strip() for q in content.split('\n') if len(q.strip()) > 5]

async def conduct_adaptive_research(goal):
    async with aiohttp.ClientSession() as session:
        print("🔍 Generating initial queries...")
        queries = await generate_initial_queries(session, goal)
        if not queries:
            print("❌ Failed to generate initial queries")
            return

        all_findings = []
        iteration = 0
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        while iteration < MAX_ITERATIONS and queries:
            print(f"\n🔄 Iteration {iteration + 1}")
            print("📝 Current queries:")
            for i, q in enumerate(queries, 1):
                print(f"{i}. {q}")
            
            all_links = []
            for query in queries:
                links = await perform_search_async(session, query)
                all_links.extend(links)
            unique_links = list(set(all_links))
            print(f"🌐 Found {len(unique_links)} unique links")

            results = await process_urls_with_semaphore(session, unique_links, goal, semaphore)
            print(f"📚 Found {len(results)} relevant pages")
            
            if results:
                all_findings.extend(results)
                queries = await refine_queries(session, results)
            else:
                break
                
            iteration += 1

        if all_findings:
            print("\n📊 Generating final report...")
            report_prompt = (
                "Create an executive research report addressing the original research goal. Include:\n\n"
                "1. EXECUTIVE SUMMARY (2-3 paragraphs synthesizing key findings)\n"
                "2. METHODOLOGY\n"
                "   - Search strategy\n"
                "   - Quality assessment criteria\n"
                "   - Limitations\n"
                "3. KEY FINDINGS\n"
                "   - Major trends and patterns\n"
                "   - Statistical significance\n"
                "   - Contradictory evidence\n"
                "4. ACTIONABLE INSIGHTS\n"
                "   - Practical applications\n"
                "   - Implementation considerations\n"
                "   - Risk factors\n"
                "5. FUTURE DIRECTIONS\n"
                "   - Research gaps\n"
                "   - Emerging hypotheses\n"
                "   - Recommended studies\n"
                "6. APPENDIX\n"
                "   - Data sources\n"
                "   - Quality ratings\n"
                "   - Methodology details\n\n"
                "Use [citation] format and provide complete source list.\n"
                "Focus on quantitative data and actionable insights.\n\n"
                "Research data:\n" + json.dumps(all_findings, indent=2)
            )
            messages = [{"role": "user", "content": report_prompt}]
            final_report = await call_openrouter_async(session, messages)
            
            if final_report and 'choices' in final_report:
                print("\n🧠 FINAL REPORT:")
                print(final_report['choices'][0]['message']['content'])
            else:
                print("❌ Failed to generate final report")
        else:
            print("❌ No relevant information found")

if __name__ == "__main__":
    research_goal = input("Enter your high-level research goal: ")
    asyncio.run(conduct_adaptive_research(research_goal))
