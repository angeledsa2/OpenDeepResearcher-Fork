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
>> ## ⚙️ Setup
>> ### Requirements
>> - Python 3.8+
>> - API Keys:
>>   * OpenRouter (LLM queries)
>>   * SERPAPI (Academic search)
>>   * JINA (Content extraction)
>> 
>> ### Installation
>> ```bash
>> git clone https://github.com/angeledsa2/OpenDeepResearcher-Fork.git
>> cd OpenDeepResearcher-Fork
>> ```
>> 
>> ### Configure API Keys
>> ```bash
>> export OPENROUTER_API_KEY="your_key"
>> export SERPAPI_API_KEY="your_key"
>> export JINA_API_KEY="your_key"
>> ```
>> 
>> ## 💻 Usage
>> ### Basic Research:
>> ```python
>> from research_assistant import ResearchAssistant
>> 
>> assistant = ResearchAssistant()
>> 
>> goal = """Analyze the impact of remote work adoption on organizational productivity and employee well-being."""
>> 
>> result = await assistant.conduct_research(goal)
>> print(result)
>> ```
>> 
>> ### Advanced Configuration:
>> ```python
>> assistant = ResearchAssistant(
>>     max_concurrent_requests=5,
>>     default_model="anthropic/claude-3.5-sonnet",
>>     min_evidence_quality=7
>> )
>> 
>> goal = """Examine the effectiveness of ML approaches in disease detection using medical imaging."""
>> result = await assistant.conduct_research(goal)
>> ```
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
>> Visit github.com/angeledsa2/OpenDeepResearcher-Fork for contributions.
>> 
>> ## 📜 License
>> MIT License - See LICENSE.md
>> ```

 ---

 Original Concept: Based on OpenDeepResearcher by Matt Shumer
 Fork Maintainer: @angeledsa2

 > Note: This is an autonomous research tool. Always verify critical findings through additional sources.