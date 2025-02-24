>> # 🧠 AI-Driven Research Assistant 2.1
>> 
>> An advanced autonomous research platform leveraging large language models (LLMs) to decompose complex queries, conduct structured investigations, and synthesize findings from academic and scientific sources. Now includes enhanced source prioritization, LLM-driven search query generation, and improved research failure analysis.
>> 
>> ## 🎯 Core Value
>> 
>> The Research Assistant autonomously:
>> - Decomposes queries into detailed research aspects
>> - Generates targeted search queries using LLMs
>> - Prioritizes sources based on relevance and quality
>> - Evaluates source credibility and synthesizes findings
>> - Identifies contradictions, gaps, and biases
>> - Recovers from failed research attempts with LLM-driven analysis
>> 
>> ## 🔍 How It Works
>> 
>> ### 1. Research Decomposition
>> The assistant analyzes your research query and structures it into investigatable aspects using LLMs.
>> 
>> ### 2. Query Generation & Source Search
>> For each aspect, the assistant:
>> - Uses the LLM to generate concise, high-relevance search queries
>> - Conducts academic searches via SERPAPI
>> - Retrieves and prioritizes sources using LLM-based evaluation
>> 
>> ### 3. Coverage and Quality Tracking
>> The assistant:
>> - Measures research coverage per aspect
>> - Assesses evidence quality and identifies gaps
>> - Adjusts queries based on results
>> - Evaluates research failures and suggests alternative approaches
>> 
>> ### 4. Report Generation
>> The assistant produces a comprehensive markdown report including:
>> - Executive summary
>> - Key findings by aspect
>> - Evidence quality assessments
>> - Identified contradictions and gaps
>> - Top-priority studies with citations
>> 
>> ## ⚙️ Setup in Google Colab
>> 
>> ### Requirements
>> - A Google Colab notebook (free tier works fine)
>> - API Keys:
>>   - OpenRouter (for LLM queries)
>>   - SERPAPI (for academic search)
>>   - JINA (for content extraction)
>> 
>> ### Step-by-Step Setup
>> 1. **Open a New Colab Notebook**  
>>    Go to [Google Colab](https://colab.research.google.com/), sign in with your Google account, and create a new notebook.
>> 
>> 2. **Install Dependencies**  
>>    In a code cell, run the following to install required packages:
>>    ```bash
>>    !pip install aiohttp nest_asyncio
>>    ```
>> 
>> 3. **Set Up API Keys**  
>>    In a new code cell, securely define your API keys (avoid sharing them publicly):
>>    ```python
>>    import os
>> 
>>    os.environ['OPENROUTER_API_KEY'] = 'your_openrouter_key_here'
>>    os.environ['SERPAPI_API_KEY'] = 'your_serpapi_key_here'
>>    os.environ['JINA_API_KEY'] = 'your_jina_key_here'
>>    ```
>>    Replace `'your_openrouter_key_here'`, `'your_serpapi_key_here'`, and `'your_jina_key_here'` with your actual API keys. See below for how to obtain these.
>> 
>> 4. **Paste the ResearchAssistant Code**  
>>    Copy the entire code (from your provided script) into a new code cell. This includes all imports, the `ResearchAssistant` class, and the `conduct_research` function.
>> 
>> ### Obtaining API Keys
>> Below are instructions for getting the necessary API keys and what each service does.
>> 
>> #### OpenRouter API Key
>> - **What It’s For**: OpenRouter provides access to various large language models (e.g., Claude, GPT-3.5) through a single API. The Research Assistant uses it to analyze queries, generate search terms, and process findings.
>> - **How to Get It**:
>>   1. Visit [openrouter.ai](https://openrouter.ai) and sign up with your email or wallet.
>>   2. Log in, then go to [openrouter.ai/keys](https://openrouter.ai/keys).
>>   3. Click "Create Key", name it (e.g., "ResearchAssistant"), and optionally set a credit limit.
>>   4. Copy the key immediately—it won’t be shown again. Add credits via the account page if needed.
>> 
>> #### SerpAPI API Key
>> - **What It’s For**: SerpAPI enables searching Google Scholar and other engines programmatically. The Research Assistant uses it to find academic sources relevant to your query.
>> - **How to Get It**:
>>   1. Go to [serpapi.com](https://serpapi.com) and create an account.
>>   2. After logging in, navigate to the "Dashboard" or "Your Account" section.
>>   3. Find your API key under the "API Key" or "Account" tab and copy it.
>>   4. SerpAPI offers a free tier with limited searches; upgrade for more capacity if needed.
>> 
>> #### Jina AI API Key
>> - **What It’s For**: Jina AI extracts clean, LLM-friendly text from web pages (e.g., converting HTML to Markdown). The Research Assistant uses it as a fallback to fetch content from paywalled or complex sites.
>> - **How to Get It**:
>>   1. Sign up at [jina.ai](https://jina.ai).
>>   2. Log in, then go to the "API" or "Developer" section (check the dashboard).
>>   3. Generate an API key—look for a "Create API Key" or similar option.
>>   4. Copy the key. Jina offers a free tier; check their pricing for higher limits.
>> 
>> ## 💻 Usage in Colab
>> 
>> ### Basic Research
>> In a new code cell, run the research assistant:
>> ```python
>> import asyncio
>> 
>> # Define your research question
>> question = "Analyze the impact of remote work adoption on organizational productivity and employee well-being."
>> 
>> # Run the research (Colab handles async automatically with nest_asyncio)
>> report = asyncio.run(conduct_research(question))
>> 
>> # Display the report
>> print(report)
>> ```
>> 
>> ### Advanced Configuration
>> For custom settings, configure the assistant before running:
>> ```python
>> import asyncio
>> 
>> # Advanced configuration
>> async def run_research():
>>     assistant = ResearchAssistant()
>>     # Optionally adjust settings in the code (e.g., MAX_CONCURRENT_REQUESTS=5)
>>     question = "Examine the effectiveness of ML approaches in disease detection using medical imaging."
>>     return await conduct_research(question)
>> 
>> # Execute and display
>> report = asyncio.run(run_research())
>> print(report)
>> ```
>> 
>> ### Notes for Colab
>> - **Async Support**: The code uses `nest_asyncio` to handle Colab’s existing event loop. No additional setup is needed.
>> - **Execution**: Run cells sequentially. If you get a "Runtime disconnected" error, reduce `MAX_CONCURRENT_REQUESTS` (e.g., to 3) in the code to lower resource usage.
>> - **Output**: Results appear in the cell output below your code.
>> 
>> ## 🔧 Key Enhancements (v2.1)
>> - **LLM-Driven Query Generation:** Improved search query generation using OpenRouter LLM.
>> - **Source Prioritization:** Source ranking with LLM-based relevance and quality assessment.
>> - **Research Failure Handling:** LLM-guided analysis of failed queries with alternative suggestions.
>> - **Improved Coverage Metrics:** Enhanced tracking of coverage per research aspect.
>> - **Paywall Detection:** Integrated paywall detection and fallback to JINA API.
>> 
>> ## 📊 Output Format
>> ```markdown
>> # Research Report
>> ## Executive Summary
>> [Summary of key findings]
>> ## Findings by Research Aspect
>> [Detailed findings and evidence quality]
>> ## Contradictory Evidence
>> [Conflicting results]
>> ## Research Gaps
>> [Areas requiring further investigation]
>> ## Top-Impact Studies
>> [Key citations]
>> ```
>> 
>> ## ⚠️ Limitations
>> - Requires valid API keys
>> - Limited to English sources
>> - API rate limits apply
>> - Maximum content length of 5000 characters per source
>> 
>> ## 🤝 Contributing
>> Visit [github.com/angeledsa2/OpenDeepResearcher-Fork](https://github.com/angeledsa2/OpenDeepResearcher-Fork) for contributions.
>> 
>> ## 📜 License
>> MIT License - See LICENSE.md
>> 
>> ---
>> 
>> **Original Concept:** Based on OpenDeepResearcher by Matt Shumer  
>> **Fork Maintainer:** @angeledsa2  
>> 
>> **Note:** This is an autonomous research tool. Always verify critical findings through additional sources.