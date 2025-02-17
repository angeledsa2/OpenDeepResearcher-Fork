  # 🧠 AI-Driven Research Assistant 2.0

 An autonomous research platform that leverages large language models in agentic fashion to decompose complex research queries into structured investigations, automatically gathering and synthesizing findings from academic and scientific sources.

 ## 🎯 Core Value

 This system autonomously:
 - Decomposes research queries into investigatable aspects
 - Executes concurrent searches across academic sources
 - Evaluates source credibility and evidence quality 
 - Synthesizes findings into comprehensive reports
 - Identifies contradictions and knowledge gaps

 ## 🔍 How It Works

 ### 1. Research Decomposition
 The assistant analyzes your research query and breaks it into distinct aspects:

 ```python
 query = """Investigate the relationship between environmental factors 
           (temperature, air quality) and public health metrics in 
           urban environments over the past decade."""

 ## System automatically identifies aspects like:
 # - Environmental measurements
 # - Public health statistics
 # - Urban demographics
 # - Temporal patterns

 ### 2. Concurrent Investigation
 For each aspect, the system:
 - Generates targeted search queries
 - Retrieves and evaluates academic sources
 - Extracts relevant findings
 - Tracks evidence quality
 - Identifies contradictions

 ### 3. Dynamic Coverage Analysis
 The assistant:
 - Monitors research progress per aspect
 - Adjusts queries based on findings
 - Identifies knowledge gaps
 - Ensures comprehensive coverage
 - Validates findings across sources

 ### 4. Report Generation
 Produces a structured report with:
 - Executive summary
 - Key findings by aspect
 - Evidence quality analysis
 - Contradictory evidence
 - Critical gaps
 - High-impact studies

 ## ⚙️ Setup

 ### Requirements
 - Python 3.8+
 - API Keys:
   * OpenRouter (LLM queries)
   * SERPAPI (Academic search)
   * JINA (Content extraction)

 ### Installation
 ```bash
 git clone https://github.com/angeledsa2/OpenDeepResearcher-Fork.git
 cd OpenDeepResearcher-Fork
 ```

 ### Configure API Keys
 ```bash
 export OPENROUTER_API_KEY="your_key"
 export SERPAPI_API_KEY="your_key"
 export JINA_API_KEY="your_key"
 ```


 ## 💻 Usage

 # Basic research:
 ```python
 from research_assistant import ResearchAssistant

 # Initialize assistant
 assistant = ResearchAssistant()

 # Define research goal
 goal = """Analyze the impact of remote work adoption on 
           organizational productivity and employee well-being 
           across different industries."""

 # Execute research
 result = await assistant.conduct_research(goal)
 print(result)
 ```

 # Advanced configuration:
 ```python
 assistant = ResearchAssistant(
     max_concurrent_requests=5,  # Parallel search limit
     default_model="anthropic/claude-3.5-sonnet",  # LLM model
     min_evidence_quality=7  # Minimum source quality (1-10)
 )

 # Example complex query
 goal = """Examine the effectiveness of different machine learning 
           approaches in early disease detection using medical imaging, 
           including validation studies and performance metrics."""

 result = await assistant.conduct_research(goal)
 ```
 ## Advanced configuration:
 ```python
 assistant = ResearchAssistant(
     max_concurrent_requests=5,  # Parallel search limit
     default_model="anthropic/claude-3.5-sonnet",  # LLM model
     min_evidence_quality=7  # Minimum source quality (1-10)
 )
 ```
 ## 🔧 Key Parameters

 - MAX_CONCURRENT_REQUESTS: Controls parallel processing (default: 5)
 - DEFAULT_MODEL: OpenRouter model selection
 - min_evidence_quality: Minimum source quality threshold (1-10)
 - coverage_threshold: Required coverage per aspect (0-100%)

 ## 📊 Output Format

 The assistant generates a structured markdown report:
 ```markdown
 # Research Report

 ## Executive Summary
 [Key findings across all aspects]

 ## Findings by Research Aspect
 [Detailed findings for each aspect]

 ## Evidence Quality Analysis
 [Source credibility assessment]

 ## Contradictory Evidence
 [Conflicting findings]

 ## Critical Gaps
 [Areas needing further research]

 ## High-Impact Studies
 [Most significant sources]
 ```

 ## ⚠️ Limitations

 - Requires valid API keys
 - 5000 character limit per source
 - English language sources only
 - Output length constraints
 - API rate limits apply

 ## 🤝 Contributing

 Visit github.com/angeledsa2/OpenDeepResearcher-Fork to:
 - Report issues
 - Submit feature requests
 - Contribute improvements

 ## 📜 License

 MIT License - See LICENSE.md

 ---

 Original Concept: Based on OpenDeepResearcher by Matt Shumer
 Fork Maintainer: @angeledsa2

 > Note: This is an autonomous research tool. Always verify critical findings through additional sources.