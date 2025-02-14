# 🧠 **Adaptive Research Assistant** (OpenDeepResearcher) 🚀  
**Version:** 1.0  
Based on Original OpenDeepResearcher by Matt Shumer  

---

## 📖 **Introduction**  

The **Adaptive Research Assistant** is an advanced, self-optimizing research tool designed to streamline, accelerate, and refine the process of gathering, analyzing, and synthesizing information from the web.  

Unlike traditional research automation scripts, this assistant:  
- **Dynamically generates research questions** based on high-level goals.  
- **Refines queries iteratively** to maximize information yield.  
- **Scores content relevance** before extraction to save time and resources.  
- **Synthesizes a comprehensive final report** based on extracted context.  
- **Learns from feedback** to improve performance over time.  
---

## ⚙️ **How It Works**  

1. **Input:** You provide a single high-level research goal.  
2. **Ontology Builder:** The assistant generates a structured topic tree and a single-paragraph query input.  
3. **Query Execution:** It sends generated queries to SERPAPI and retrieves webpage text using Jina AI.  
4. **Relevance Scoring:** The assistant evaluates webpage content for relevance and extracts key information if deemed useful.  
5. **Adaptive Refinement:** The assistant asks the LLM to generate refined queries if more information is needed.  
6. **Report Generation:** It synthesizes a clear, concise, and comprehensive research report.  
7. **Feedback Loop:** Query performance is tracked, enabling long-term optimization.  

---

## 🔍 **Example Use Case**  

**Research Goal:** *"Investigate the relationship between cosmic activity and human behavior."*  

The assistant will:  
- Create a research roadmap with subtopics like geomagnetic influences, lunar cycles, and biological rhythms.  
- Generate search queries, retrieve relevant articles, and synthesize insights.  
- Provide a final report detailing correlations, mechanisms, and practical applications.  

---

## 🛠️ **Setup Guide**  

### 1️⃣ **Install Requirements**  

```bash
pip install nest_asyncio aiohttp
```

*(Additional dependencies like OpenRouter, SERPAPI, and Jina are API-based and don't require separate installations.)*  

---

### 2️⃣ **Environment Variables**  

Before running the assistant, set the following environment variables:  

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export SERPAPI_API_KEY="your_serpapi_api_key"
export JINA_API_KEY="your_jina_api_key"
```

---

### 3️⃣ **API Key Acquisition**  

| **API**       | **Website**                             | **Notes**                          |
|----------------|----------------------------------------|------------------------------------|
| OpenRouter    | [openrouter.ai](https://openrouter.ai/)  | GPT-based query generation         |
| SERPAPI       | [serpapi.com](https://serpapi.com/)      | Google Search API                  |
| Jina AI       | [jina.ai](https://jina.ai/)              | Webpage content retrieval          |

---

### 4️⃣ **Running the Assistant**  

Run the script by executing:  

```bash
python adaptive_research_assistant.py
```

---

## 🎯 **Usage Instructions**  

The assistant will prompt you for:  

1. **High-Level Research Goal:** *(e.g., "Investigate the effects of geomagnetic activity on human health.")*  
2. **Maximum Iterations:** *(Limits how long the assistant will search for new insights.)*  

The assistant will then:  
- Generate queries  
- Conduct searches  
- Analyze and extract content  
- Produce a final report  

---

## 🧠 **Customization & Advanced Configuration**  

### 🧩 **Adaptive Query Refinement**  

The assistant refines its queries based on context extraction. You can tweak this behavior via:  

```python
MAX_ITERATIONS = 7  # Adjust for longer/shorter research loops
```

### 🔧 **Topic Tree Depth**  

The initial topic tree is generated with 3 levels by default. To adjust, modify:  

```python
"generate:\n- A hierarchical topic tree with 3 levels.\n"
```

### 🎚️ **Relevance Threshold**  

The assistant filters content with a relevance score ≥ 6. Adjust by modifying:  

```python
if "Score:" in r and int(r.split("Score:")[1][:1]) >= 6:
```

---

## 🚧 **Known Limitations**  

- **API Dependency:** Requires consistent availability of OpenRouter, SERPAPI, and Jina services.  
- **Relevance Evaluation Noise:** Relevance scoring may occasionally misclassify pages; manual review is advised for critical research.  
- **Text Length Constraints:** Extracted content is truncated at 20,000 characters for processing efficiency.  

---

## 🔍 **Potential Improvements** *(Future Scope)*  

- 🧠 **Machine Learning Integration:** Track past query performance to optimize new query generation.  
- 🔄 **Recurrent Query Looping:** Introduce a semi-supervised feedback mechanism for higher accuracy.  
- 🌐 **Multi-Language Support:** Adapt for research in multiple languages.  

---

## 📊 **Practical Applications**  

1. **Scientific Literature Reviews:** Discover patterns across publications.  
2. **Market Intelligence:** Analyze trends, competitors, and customer sentiment.  
4. **Technological Scouting:** Track emerging innovations and patents.  

---

## 🎯 **Tips for Maximizing Research Efficiency**  

1. **Be Specific:** The more precise the goal, the better the results.  
2. **Monitor Query Quality:** Adjust maximum iterations based on desired depth.  
3. **Regularly Rotate APIs:** Avoid hitting rate limits by rotating keys across multiple accounts if needed.  
4. **Cross-Validate Findings:** Use the assistant as a starting point, not a final decision-making tool.  

---

## 🤝 **Collaborative Potential**  

The modular architecture allows for:  
- Plugging in new APIs  
- Domain-specific research enhancements  
- Integration with analytics platforms  

This flexibility ensures that the Adaptive Research Assistant remains valuable even as research goals evolve.  

---

💡 **Happy Researching!** 🌐
