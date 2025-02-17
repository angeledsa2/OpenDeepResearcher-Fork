import asyncio
import aiohttp
import json
from bs4 import BeautifulSoup
import re



#Configuration

OPENROUTER_API_KEY = "sk-or-v1-"
SERPAPI_API_KEY = ""
JINA_API_KEY = ""



OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search"
JINA_BASE_URL = "https://r.jina.ai/"



DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
MAX_CONCURRENT_REQUESTS = 5


class ResearchAssistant:
    def __init__(self):
       # Immediately initialize the research_state attribute.
       self.research_state = {
           "original_request": None,
           "research_aspects": {},
           "findings_by_aspect": {},
           "aspect_coverage": {},
           "aspect_queries": {},
           "findings": [],
           "queries_attempted": set(),
           "successful_queries": [],
           "coverage_analysis": {},
           "gaps_identified": set(),
           "papers_found": [],
           "null_results": [],
           "source_credibility_scores": {},
           "keyword_cache": {},
           "significant_studies": {
               "high_relevance": [],
               "medium_relevance": [],
               "contradictory": [],
               "null_results": []
           },
           "citations": set(),
           "completed_aspects": set(),
           "aspect_reports": {},
           "failed_queries": [],
           "error_log": [],
           "emergency_summaries": {}
       }


    async def call_llm(self, session, messages, max_retries=3):
        """Make an async call to the LLM with enhanced error handling."""
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json", 
                    "HTTP-Referer": "https://localhost:3000",
                    "X-Title": "Research Assistant"
                }
                payload = {"model": DEFAULT_MODEL, "messages": messages}
                
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    error_msg = f"OpenRouter API error: Status {resp.status}"
                    self.research_state["error_log"].append(error_msg)
                    print(error_msg)
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
            except Exception as e:
                error_msg = f"Error calling OpenRouter (attempt {attempt + 1}/{max_retries}): {str(e)}"
                self.research_state["error_log"].append(error_msg)
                print(error_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
        return None



    async def _get_llm_response(self, session, prompt, max_retries=3):
        """Enhanced helper method to get LLM response and parse JSON with retries."""
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.call_llm(session, messages)
                
                if response and 'choices' in response:
                    content = response['choices'][0]['message']['content']
                    parsed = await self._safe_json_parse(content)
                    if parsed:
                        return parsed
                    print(f"Failed to parse LLM response (attempt {attempt + 1}/{max_retries})")
                    print("Raw response:", content[:200])
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
            except Exception as e:
                error_msg = f"Error processing LLM response (attempt {attempt + 1}/{max_retries}): {e}"
                self.research_state["error_log"].append(error_msg)
                print(error_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
        
        return None



    async def _safe_json_parse(self, content):
        """Enhanced JSON parsing with multiple fallback strategies."""
        if not content:
            return None
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks
                matches = re.findall(r'```(?:json)?\s([\s\S]?)\s*```', content)
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
                
                # Try to find content between curly braces
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    return json.loads(match.group(0))
                    
                # Try to extract arrays
                match = re.search(r'\[[\s\S]*\]', content)
                if match:
                    return json.loads(match.group(0))
            except Exception as nested_e:
                error_msg = f"Nested JSON parsing error: {nested_e}"
                self.research_state["error_log"].append(error_msg)
                print(error_msg)
                
        return None



    def _track_significant_study(self, evaluation, url):
        """Track and categorize significant studies."""
        try:
            # Prepare study metadata
            study_info = {
                "url": url,
                "citation": evaluation.get("citation", ""),
                "evidence_quality": evaluation.get("source_credibility", 0),
                "key_findings": [],
                "limitations": evaluation.get("limitations", []),
                "methodology": evaluation.get("methodology", "Not specified"),
                "relevance_scores": {}
            }



            # Extract key findings and relevance scores
            for aspect, details in evaluation.get("relevance_by_aspect", {}).items():
                if details["score"] >= 6:
                    study_info["key_findings"].extend(details["key_findings"])
                    study_info["relevance_scores"][aspect] = details["score"]



            # Categorize study based on characteristics
            if evaluation.get("contradictory_evidence", False):
                self.research_state["significant_studies"]["contradictory"].append(study_info)
            elif evaluation.get("null_result", False):
                self.research_state["significant_studies"]["null_results"].append(study_info)
            else:
                # Categorize based on average relevance score
                avg_score = sum(study_info["relevance_scores"].values()) / len(study_info["relevance_scores"])
                if avg_score >= 8:
                    self.research_state["significant_studies"]["high_relevance"].append(study_info)
                elif avg_score >= 6:
                    self.research_state["significant_studies"]["medium_relevance"].append(study_info)



        except Exception as e:
            error_msg = f"Error tracking significant study: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)



    async def _cache_aspect_keywords(self, session, aspects):
        """Extract and cache keywords for each aspect."""
        try:
            for aspect, details in aspects.items():
                keywords = set()
                # Add key concepts as keywords
                keywords.update(details.get("key_concepts", []))
                
                # Generate additional keywords using LLM
                prompt = (
                    f"Generate relevant search keywords for research aspect: {aspect}\n"
                    f"Context: {json.dumps(details, indent=2)}\n"
                    "Return JSON array of keywords"
                )
                
                response = await self._get_llm_response(session, prompt)
                if isinstance(response, list):
                    keywords.update(response)
                
                self.research_state["keyword_cache"][aspect] = keywords
                
        except Exception as e:
            error_msg = f"Error caching keywords: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)



    async def perform_search(self, session, query):
        """Perform academic-focused search with enhanced error handling."""
        try:
            # Try Google Scholar first
            scholar_params = {
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "engine": "google_scholar",
                "num": 10
            }
            
            async with session.get(SERPAPI_URL, params=scholar_params) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    if "organic_results" in results:
                        return [r['link'] for r in results["organic_results"][:5]]
            
            # Fallback to regular Google with academic sites
            sites = "site:edu OR site:org OR site:gov OR site:ac.uk"
            params = {
                "q": f"({query}) ({sites})",
                "api_key": SERPAPI_API_KEY,
                "engine": "google",
                "num": 10
            }
            
            async with session.get(SERPAPI_URL, params=params) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    if "organic_results" in results:
                        return [r['link'] for r in results.get("organic_results", [])[:5]]
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
        return []



    async def process_urls(self, session, urls, semaphore):
        """Process URLs concurrently with enhanced error handling."""
        async def process_url(url):
            try:
                async with semaphore:
                    content = await self.fetch_content(session, url)
                    if content:
                        return await self.evaluate_content(session, content, url)
            except Exception as e:
                error_msg = f"Error processing URL {url}: {str(e)}"
                self.research_state["error_log"].append(error_msg)
                print(error_msg)
            return None
        
        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]



    async def fetch_content(self, session, url):
        """Fetch webpage content with enhanced error handling."""
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    text = await resp.text(encoding='utf-8', errors='replace')
                    return self.clean_content(text)
            
            # Fallback to JINA
            full_url = f"{JINA_BASE_URL}{url}"
            async with session.get(full_url, headers={"Authorization": f"Bearer {JINA_API_KEY}"}) as resp:
                if resp.status == 200:
                    text = await resp.text(encoding='utf-8', errors='replace')
                    return self.clean_content(text)
        except Exception as e:
            error_msg = f"Error fetching {url}: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
        return None



    def clean_content(self, text):
        """Clean and normalize content."""
        if not text:
            return None
        try:
            soup = BeautifulSoup(text, 'html.parser')
            content = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            content = re.sub(r'\s+', ' ', content).strip()
            return content[:5000]
        except Exception as e:
            error_msg = f"Error cleaning content: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return None
        
    async def initialize_research(self, session, research_request):
        """Initialize research with enhanced error handling and validation."""
        self.research_state["original_request"] = research_request
        
        prompt = (
            f"Analyze this research request thoroughly: {research_request}\n\n"
            "1. Break down the request into distinct research aspects\n"
            "2. For each aspect, identify:\n"
            "   - Key concepts and terminology\n"
            "   - Required evidence types\n"
            "   - Success criteria\n"
            "   - Potential data sources\n"
            "   - Methodological requirements\n"
            "3. Specify relationships between aspects\n"
            "4. Define completion criteria for each aspect\n\n"
            "Return a structured JSON:\n"
            "{\n"
            '  "aspects": {\n'
            '    "aspect_name": {\n'
            '      "key_concepts": ["concept1", "concept2"],\n'
            '      "evidence_types": ["type1", "type2"],\n'
            '      "success_criteria": ["criterion1", "criterion2"],\n'
            '      "data_sources": ["source1", "source2"],\n'
            '      "methodology": ["method1", "method2"],\n'
            '      "completion_criteria": ["criteria1", "criteria2"],\n'
            '      "related_aspects": ["aspect1", "aspect2"]\n'
            '    }\n'
            '  },\n'
            '  "cross_cutting_themes": ["theme1", "theme2"],\n'
            '  "priority_order": ["aspect1", "aspect2"]\n'
            "}"
        )
        
        analysis = await self._get_llm_response(session, prompt)
        if analysis:
            try:
                if "aspects" not in analysis or not analysis["aspects"]:
                    raise ValueError("Invalid analysis structure: missing or empty aspects")
                
                self.research_state["research_aspects"] = analysis
                
                for aspect in analysis["aspects"]:
                    self.research_state["findings_by_aspect"][aspect] = []
                    self.research_state["aspect_coverage"][aspect] = 0
                    self.research_state["aspect_queries"][aspect] = set()
                    self.research_state["emergency_summaries"][aspect] = ""
                
                await self._cache_aspect_keywords(session, analysis["aspects"])
                return analysis
            except Exception as e:
                error_msg = f"Error initializing research: {str(e)}"
                self.research_state["error_log"].append(error_msg)
                print(error_msg)
                return None
        return None



    async def generate_targeted_queries(self, session):
        """Generate targeted queries for uncovered aspects."""
        try:
            queries = []
            coverage = self.research_state.get("coverage_analysis", {}).get("coverage_by_aspect", {})
            
            aspects_by_coverage = sorted(
                self.research_state["aspect_coverage"].items(),
                key=lambda x: x[1]
            )
            
            for aspect, current_coverage in aspects_by_coverage:
                if current_coverage < 90:
                    keywords = self.research_state["keyword_cache"].get(aspect, set())
                    if keywords:
                        keyword_queries = self._generate_keyword_queries(keywords)
                        for query in keyword_queries:
                            if query not in self.research_state["queries_attempted"]:
                                queries.append({
                                    "query": query,
                                    "aspect": aspect,
                                    "type": "keyword",
                                    "targets": [aspect]
                                })
            
            final_queries = queries[:5]
            
            for query in final_queries:
                self.research_state["queries_attempted"].add(query["query"])
                self.research_state["aspect_queries"][query["aspect"]].add(query["query"])
            
            return final_queries
        except Exception as e:
            error_msg = f"Error generating targeted queries: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return []



    def _generate_keyword_queries(self, keywords):
        """Generate queries based on keyword combinations."""
        queries = []
        if len(keywords) >= 2:
            for kw1 in keywords:
                for kw2 in keywords:
                    if kw1 != kw2:
                        queries.append(f"{kw1} AND {kw2}")
        return queries



    async def assess_coverage(self, session):
        """Enhanced coverage assessment with consistent structure."""
        try:
            prompt = (
                "Analyze current research coverage:\n\n"
                f"Original Request: {self.research_state['original_request']}\n\n"
                f"Research Aspects: {json.dumps(self.research_state['research_aspects'], indent=2)}\n\n"
                f"Findings by Aspect: {json.dumps(self.research_state['findings_by_aspect'], indent=2)}\n\n"
                f"Current Coverage: {json.dumps(self.research_state['aspect_coverage'], indent=2)}\n\n"
                "Return a detailed JSON analysis:\n"
                "{\n"
                '  "coverage_by_aspect": {\n'
                '    "aspect_name": {\n'
                '      "coverage_percentage": 0-100,\n'
                '      "evidence_quality": 1-10,\n'
                '      "missing_elements": ["element1", "element2"],\n'
                '      "contradictions": ["contradiction1", "contradiction2"],\n'
                '      "confidence_level": 1-10\n'
                '    }\n'
                '  },\n'
                '  "overall_completion": 0-100,\n'
                '  "research_complete": false,\n'
                '  "recommended_focus": ["focus1", "focus2"]\n'
                "}"
            )
            
            coverage = await self._get_llm_response(session, prompt)
            if not coverage:
                print("⚠️ Using fallback coverage assessment")
                coverage = self._generate_fallback_coverage()
            
            self.research_state["coverage_analysis"] = coverage
            return coverage
            
        except Exception as e:
            error_msg = f"Error in assess_coverage: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return self._generate_fallback_coverage()



    def _generate_fallback_coverage(self):
        """Generate fallback coverage analysis when LLM fails."""
        try:
            return {
                "coverage_by_aspect": {
                    aspect: {
                        "coverage_percentage": self.research_state["aspect_coverage"].get(aspect, 0),
                        "evidence_quality": 0,
                        "missing_elements": list(details.get("key_concepts", [])),
                        "contradictions": [],
                        "confidence_level": 0
                    } for aspect, details in self.research_state["research_aspects"].get("aspects", {}).items()
                },
                "overall_completion": min(
                    sum(self.research_state["aspect_coverage"].values()) / 
                    len(self.research_state["aspect_coverage"]) if self.research_state["aspect_coverage"] else 0,
                    100
                ),
                "research_complete": False,
                "recommended_focus": list(self.research_state["research_aspects"].get("aspects", {}).keys())
            }
        except Exception as e:
            error_msg = f"Error generating fallback coverage: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return {
                "coverage_by_aspect": {},
                "overall_completion": 0,
                "research_complete": False,
                "recommended_focus": []
            }



    async def evaluate_content(self, session, content, url):
        """Enhanced content evaluation with comprehensive study tracking."""
        prompt = (
            f"Evaluate this content for all research aspects:\n\n"
            f"Content: {content[:3000]}\n\n"
            f"Source: {url}\n\n"
            f"Research Aspects: {json.dumps(self.research_state['research_aspects'], indent=2)}\n\n"
            "Provide a detailed evaluation JSON:\n"
            "{\n"
            '  "relevance_by_aspect": {\n'
            '    "aspect_name": {\n'
            '      "score": 0-10,\n'
            '      "key_findings": ["finding1", "finding2"],\n'
            '      "evidence_quality": 1-10,\n'
            '      "contribution_to_coverage": 0-100\n'
            '    }\n'
            '  },\n'
            '  "source_credibility": 1-10,\n'
            '  "new_gaps_identified": ["gap1", "gap2"],\n'
            '  "citation": "formatted citation",\n'
            '  "methodology": "description of study methodology",\n'
            '  "limitations": ["limitation1", "limitation2"],\n'
            '  "contradictory_evidence": false,\n'
            '  "null_result": false,\n'
            '  "study_type": "primary research/meta-analysis/review/etc",\n'
            '  "publication_info": {\n'
            '    "year": "YYYY",\n'
            '    "journal": "journal name",\n'
            '    "peer_reviewed": true/false\n'
            '  }\n'
            "}"
        )
        
        try:
            evaluation = await self._get_llm_response(session, prompt)
            if evaluation:
                # Update research state with new findings
                for aspect, details in evaluation["relevance_by_aspect"].items():
                    if details["score"] >= 6:  # Relevant content threshold
                        self.research_state["findings_by_aspect"][aspect].extend(details["key_findings"])
                        self._update_aspect_coverage(aspect, details["contribution_to_coverage"])
                
                # Update source credibility
                self.research_state["source_credibility_scores"][url] = evaluation["source_credibility"]
                
                # Track paper with enhanced metadata
                paper_info = {
                    "url": url,
                    "relevance_scores": {k: v["score"] for k, v in evaluation["relevance_by_aspect"].items()},
                    "citation": evaluation["citation"],
                    "methodology": evaluation.get("methodology", "Not specified"),
                    "study_type": evaluation.get("study_type", "Not specified"),
                    "publication_info": evaluation.get("publication_info", {}),
                    "contradictory_evidence": evaluation.get("contradictory_evidence", False),
                    "null_result": evaluation.get("null_result", False)
                }



                self.research_state["papers_found"].append(paper_info)
                
                # Track citation
                if paper_info["citation"]:
                    self.research_state["citations"].add(paper_info["citation"])
                
                # Update gaps
                self.research_state["gaps_identified"].update(evaluation["new_gaps_identified"])
                
                # Track significant study
                self._track_significant_study(evaluation, url)
                
                # Display significant studies periodically
                if len(self.research_state["papers_found"]) % 5 == 0:
                    self._display_significant_studies()
                
                return evaluation
        except Exception as e:
            error_msg = f"Error evaluating content from {url}: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return None



    def _update_aspect_coverage(self, aspect, contribution):
        """Update coverage metrics for a specific aspect."""
        try:
            current_coverage = self.research_state["aspect_coverage"][aspect]
            # Update coverage using a weighted approach
            new_coverage = min(100, current_coverage + (contribution * 0.1))
            self.research_state["aspect_coverage"][aspect] = new_coverage
        except Exception as e:
            error_msg = f"Error updating coverage for aspect {aspect}: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)



    async def generate_final_report(self, session):
        """Generate comprehensive final report with enhanced analysis and citations."""
        try:
            prompt = (
                "Generate a comprehensive research report in the following JSON structure:\n\n"
                "{\n"
                '  "executive_summary": "text",\n'
                '  "methodology": "text",\n'
                '  "findings_by_aspect": {\n'
                '    "aspect_name": {\n'
                '      "key_findings": ["finding1", "finding2"],\n'
                '      "evidence_quality": "text",\n'
                '      "significant_studies": ["study1", "study2"]\n'
                '    }\n'
                '  },\n'
                '  "evidence_quality_analysis": "text",\n'
                '  "coverage_analysis": "text",\n'
                '  "contradictory_evidence": ["evidence1", "evidence2"],\n'
                '  "null_results": ["result1", "result2"],\n'
                '  "gaps_and_limitations": ["gap1", "gap2"],\n'
                '  "recommendations": ["rec1", "rec2"],\n'
                '  "references": ["ref1", "ref2"]\n'
                "}\n\n"
                f"Original Request: {self.research_state['original_request']}\n\n"
                f"Research Aspects: {json.dumps(self.research_state['research_aspects'], indent=2)}\n\n"
                f"Findings by Aspect: {json.dumps(self.research_state['findings_by_aspect'], indent=2)}\n\n"
                f"Coverage Analysis: {json.dumps(self.research_state['coverage_analysis'], indent=2)}\n\n"
                f"Significant Studies: {json.dumps(self.research_state['significant_studies'], indent=2)}\n\n"
                f"Citations: {json.dumps(list(self.research_state['citations']), indent=2)}\n\n"
                "Ensure all findings are properly cited using the provided citations.\n"
                "Include contradictory evidence and null results.\n"
                "Format references in APA style."
            )
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.call_llm(session, messages)
            
            if response and 'choices' in response:
                # Get the structured response
                structured_report = await self._safe_json_parse(response['choices'][0]['message']['content'])
                if structured_report:
                    # Convert structured report to markdown
                    markdown_report = self._format_report_as_markdown(structured_report)
                    return markdown_report
                else:
                    # Fallback: use raw response if JSON parsing fails
                    return response['choices'][0]['message']['content']
            return None
        except Exception as e:
            error_msg = f"Error generating final report: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return None



    def _format_report_as_markdown(self, report):
        """Convert structured report to markdown format."""
        try:
            markdown = [
                "# Research Report\n\n",
                "## Executive Summary\n\n",
                f"{report.get('executive_summary', 'No executive summary available.')}\n\n",
                "## Methodology\n\n",
                f"{report.get('methodology', 'No methodology available.')}\n\n",
                "## Findings by Research Aspect\n\n"
            ]



            for aspect, details in report.get('findings_by_aspect', {}).items():
                markdown.extend([
                    f"### {aspect}\n\n",
                    "#### Key Findings\n\n"
                ])
                for finding in details.get('key_findings', []):
                    markdown.append(f"- {finding}\n")
                markdown.extend([
                    f"\n#### Evidence Quality\n\n{details.get('evidence_quality', 'Not available.')}\n\n",
                    "#### Significant Studies\n\n"
                ])
                for study in details.get('significant_studies', []):
                    markdown.append(f"- {study}\n")
                markdown.append("\n")



            markdown.extend([
                "## Evidence Quality Analysis\n\n",
                f"{report.get('evidence_quality_analysis', 'No analysis available.')}\n\n",
                "## Coverage Analysis\n\n",
                f"{report.get('coverage_analysis', 'No analysis available.')}\n\n",
                "## Contradictory Evidence\n\n"
            ])



            for evidence in report.get('contradictory_evidence', []):
                markdown.append(f"- {evidence}\n")



            markdown.extend([
                "\n## Null Results\n\n"
            ])



            for result in report.get('null_results', []):
                markdown.append(f"- {result}\n")



            markdown.extend([
                "\n## Gaps and Limitations\n\n"
            ])



            for gap in report.get('gaps_and_limitations', []):
                markdown.append(f"- {gap}\n")



            markdown.extend([
                "\n## Recommendations\n\n"
            ])



            for rec in report.get('recommendations', []):
                markdown.append(f"- {rec}\n")



            markdown.extend([
                "\n## References\n\n"
            ])



            for ref in report.get('references', []):
                markdown.append(f"{ref}\n")



            return "".join(markdown)
        except Exception as e:
            error_msg = f"Error formatting report as markdown: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return "Error generating formatted report."
        
    async def ensure_final_report(self, session):
        """Ensure final report generation even after errors."""
        try:
            # Generate mini-reports for any uncompleted aspects
            for aspect in self.research_state["research_aspects"]["aspects"]:
                if aspect not in self.research_state["completed_aspects"]:
                    await self.generate_aspect_report(session, aspect)
            
            # Display significant studies found
            self._display_significant_studies()
            
            # Generate final report with error summary
            report = await self.generate_final_report(session)
            if report:
                print("\n🧠 FINAL REPORT:")
                print(report)
            else:
                print("\n⚠️ Failed to generate detailed final report. Generating emergency summary...")
                
                emergency_summary = (
                    "EMERGENCY RESEARCH SUMMARY\n\n"
                    f"Original Request: {self.research_state['original_request']}\n\n"
                    "Research Progress:\n"
                )
                
                for aspect, coverage in self.research_state["aspect_coverage"].items():
                    emergency_summary += f"\n{aspect}: {coverage}% complete"
                
                emergency_summary += "\n\nSignificant Findings:\n"
                for aspect, findings in self.research_state["findings_by_aspect"].items():
                    if findings:
                        emergency_summary += f"\n{aspect}:\n"
                        for finding in findings:
                            emergency_summary += f"- {finding}\n"
                
                print(emergency_summary)
            
            return report or emergency_summary
        except Exception as e:
            error_msg = f"Error ensuring final report: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return "Failed to generate any report due to critical errors."



    def _display_significant_studies(self):
        """Display tracked significant studies."""
        print("\n📚 Significant Studies Found:")
        
        categories = {
            "High Relevance Studies": "high_relevance",
            "Medium Relevance Studies": "medium_relevance",
            "Contradictory Evidence": "contradictory",
            "Null Results": "null_results"
        }



        for category_name, category_key in categories.items():
            studies = self.research_state["significant_studies"][category_key]
            if studies:
                print(f"\n{category_name}:")
                for i, study in enumerate(studies, 1):
                    print(f"\n{i}. {study['citation']}")
                    print(f"   URL: {study['url']}")
                    print(f"   Evidence Quality: {study['evidence_quality']}/10")
                    if study['key_findings']:
                        print("   Key Findings:")
                        for finding in study['key_findings']:
                            print(f"   - {finding}")
                    if study['limitations']:
                        print("   Limitations:")
                        for limitation in study['limitations']:
                            print(f"   - {limitation}")
                    print()



    # --- NEW METHODS ADDED TO SUPPORT ASPECT AND QUERY PROCESSING ---
    async def check_aspect_completion(self, session, aspect):
        """Check if an individual research aspect has reached completion (e.g., 90% coverage)."""
        current = self.research_state["aspect_coverage"].get(aspect, 0)
        if current >= 90:
            self.research_state["completed_aspects"].add(aspect)
            print(f"Aspect '{aspect}' marked as completed (coverage: {current}%).")
        else:
            print(f"Aspect '{aspect}' not yet complete (coverage: {current}%).")



    async def generate_aspect_report(self, session, aspect):
        """Generate a simplified report for the given research aspect."""
        report = f"Report for aspect '{aspect}':\n"
        findings = self.research_state["findings_by_aspect"].get(aspect, [])
        if findings:
            report += "\n".join(f"- {finding}" for finding in findings)
        else:
            report += "No significant findings available."
        self.research_state["aspect_reports"][aspect] = report
        self.research_state["completed_aspects"].add(aspect)
        print(f"Generated report for aspect '{aspect}'.")



    async def process_query_safely(self, session, query_info):
        """Perform a targeted query and process returned URLs for a given research aspect."""
        query = query_info.get("query")
        aspect = query_info.get("aspect")
        print(f"Processing query for aspect '{aspect}': {query}")
        urls = await self.perform_search(session, query)
        if urls:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            evaluations = await self.process_urls(session, urls, semaphore)
            return len(evaluations) > 0
        return False



    async def handle_failed_query(self, session, query_info):
        """Log and handle a failed query."""
        error_msg = f"Query failed for aspect '{query_info.get('aspect')}' with query: {query_info.get('query')}"
        self.research_state["failed_queries"].append(query_info)
        self.research_state["error_log"].append(error_msg)
        print(error_msg)
    # --- END OF NEW METHODS ---



async def conduct_research(goal):
    """Main research orchestration function with enhanced error handling."""
    assistant = None
    session = None
    try:
        assistant = ResearchAssistant()
        async with aiohttp.ClientSession() as session:
            # Initialize research
            print("🔍 Analyzing research request...")
            analysis = await assistant.initialize_research(session, goal)
            if not analysis:
                print("❌ Failed to analyze research request")
                return await assistant.ensure_final_report(session)



            print("\n📊 Research Plan:")
            print(json.dumps(analysis, indent=2))



            iteration = 0
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)



            while iteration < 20:  # Limit maximum iterations
                try:
                    iteration += 1
                    print(f"\n🔄 Iteration {iteration}")
                    
                    # Assess coverage
                    coverage = await assistant.assess_coverage(session)
                    if not coverage:
                        print("⚠️ Failed to assess coverage, using fallback...")
                        coverage = assistant._generate_fallback_coverage()



                    print("\n📊 Coverage Analysis:")
                    print(f"Overall completion: {coverage.get('overall_completion', 0)}%")
                    
                    for aspect, details in coverage.get('coverage_by_aspect', {}).items():
                        print(f"\n{aspect}:")
                        print(f"- Coverage: {details.get('coverage_percentage', 0)}%")
                        print(f"- Evidence Quality: {details.get('evidence_quality', 0)}/10")
                        missing = details.get('missing_elements', [])
                        if missing:
                            print("- Missing:", ", ".join(missing))
                        
                        # Check for aspect completion using the new method
                        await assistant.check_aspect_completion(session, aspect)



                    # Check for overall completion and break if done
                    if coverage.get('research_complete', False) and coverage.get('overall_completion', 0) >= 90:
                        print("\n✅ Research complete!")
                        break



                    # Generate and process queries
                    queries = await assistant.generate_targeted_queries(session)
                    if not queries:
                        print("⚠️ No new queries generated, trying to complete research...")
                        if iteration >= 5:  # Minimum iterations before giving up
                            break
                        continue



                    print("\n📝 Executing targeted queries:")
                    for query_info in queries:
                        success = await assistant.process_query_safely(session, query_info)
                        if not success:
                            await assistant.handle_failed_query(session, query_info)



                    # Display interim findings periodically
                    if iteration % 2 == 0:
                        assistant._display_significant_studies()



                except Exception as e:
                    error_msg = f"Error in iteration {iteration}: {str(e)}"
                    if assistant:
                        assistant.research_state["error_log"].append(error_msg)
                    print(error_msg)
                    if iteration >= 10:  # Prevent infinite loops
                        break
                    continue



            # Ensure final report generation
            return await assistant.ensure_final_report(session)



    except Exception as e:
        error_msg = f"Critical error in research process: {str(e)}"
        if assistant:
            assistant.research_state["error_log"].append(error_msg)
        print(error_msg)
        if assistant and session and not session.closed:
            return await assistant.ensure_final_report(session)
        return "Critical error occurred, unable to generate report."



    finally:
        if session and not session.closed:
            await session.close()



if __name__ == "__main__":
    try:
        goal = input("Enter your research goal: ")
        asyncio.run(conduct_research(goal))
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
    except Exception as e:
        print(f"\n\nCritical error: {str(e)}")
