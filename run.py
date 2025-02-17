import asyncio
import aiohttp
import json
from bs4 import BeautifulSoup
import re

# Configuration
OPENROUTER_API_KEY = ""
SERPAPI_API_KEY = ""
JINA_API_KEY = ""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search"
JINA_BASE_URL = "https://r.jina.ai/"

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
MAX_CONCURRENT_REQUESTS = 5

class ResearchAssistant:
    def __init__(self):
        # Initialize research state with additional mapping for query results.
        self.research_state = {
            "original_request": None,
            "research_aspects": {},
            "findings_by_aspect": {},
            "aspect_coverage": {},
            "aspect_queries": {},
            "findings": [],
            "queries_attempted": set(),
            "successful_queries": [],
            "query_results": {},  # New mapping: query -> list of evaluations
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
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
            except Exception as e:
                error_msg = f"Error calling OpenRouter (attempt {attempt + 1}/{max_retries}): {str(e)}"
                self.research_state["error_log"].append(error_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
        return None

    async def _get_llm_response(self, session, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.call_llm(session, messages)
                if response and 'choices' in response:
                    content = response['choices'][0]['message']['content']
                    parsed = await self._safe_json_parse(content)
                    if parsed:
                        return parsed
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
            except Exception as e:
                error_msg = f"Error processing LLM response (attempt {attempt + 1}/{max_retries}): {e}"
                self.research_state["error_log"].append(error_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
        return None

    async def _safe_json_parse(self, content):
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                matches = re.findall(r'```(?:json)?\s([\s\S]*?)\s*```', content)
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    return json.loads(match.group(0))
                match = re.search(r'\[[\s\S]*\]', content)
                if match:
                    return json.loads(match.group(0))
            except Exception as nested_e:
                error_msg = f"Nested JSON parsing error: {nested_e}"
                self.research_state["error_log"].append(error_msg)
        return None

    def _track_significant_study(self, evaluation, url):
        """Track and categorize significant studies with deduplication."""
        study_hash = hash(f"{url}_{evaluation.get('citation', '')}")
        if study_hash in self.research_state.get("processed_studies", set()):
            return
        
        self.research_state.setdefault("processed_studies", set()).add(study_hash)
        try:
            study_info = {
                "url": url,
                "citation": evaluation.get("citation", ""),
                "evidence_quality": evaluation.get("source_credibility", 0),
                "key_findings": [],
                "limitations": evaluation.get("limitations", []),
                "methodology": evaluation.get("methodology", "Not specified"),
                "relevance_scores": {}
            }
            for aspect, details in evaluation.get("relevance_by_aspect", {}).items():
                if details["score"] >= 6:
                    study_info["key_findings"].extend(details["key_findings"])
                    study_info["relevance_scores"][aspect] = details["score"]
            if evaluation.get("contradictory_evidence", False):
                self.research_state["significant_studies"]["contradictory"].append(study_info)
            elif evaluation.get("null_result", False):
                self.research_state["significant_studies"]["null_results"].append(study_info)
            else:
                avg_score = sum(study_info["relevance_scores"].values()) / len(study_info["relevance_scores"])
                if avg_score >= 8:
                    self.research_state["significant_studies"]["high_relevance"].append(study_info)
                elif avg_score >= 6:
                    self.research_state["significant_studies"]["medium_relevance"].append(study_info)
        except Exception as e:
            error_msg = f"Error tracking significant study: {str(e)}"
            self.research_state["error_log"].append(error_msg)

    async def _cache_aspect_keywords(self, session, aspects):
        try:
            for aspect, details in aspects.items():
                keywords = set(details.get("key_concepts", []))
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

    async def perform_search(self, session, query):
        try:
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
        return []

    async def process_urls(self, session, urls, semaphore):
        async def process_url(url):
            try:
                async with semaphore:
                    content = await self.fetch_content(session, url)
                    if content:
                        return await self.evaluate_content(session, content, url)
            except Exception as e:
                error_msg = f"Error processing URL {url}: {str(e)}"
                self.research_state["error_log"].append(error_msg)
            return None
        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    async def fetch_content(self, session, url):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    text = await resp.text(encoding='utf-8', errors='replace')
                    return self.clean_content(text)
            full_url = f"{JINA_BASE_URL}{url}"
            async with session.get(full_url, headers={"Authorization": f"Bearer {JINA_API_KEY}"}) as resp:
                if resp.status == 200:
                    text = await resp.text(encoding='utf-8', errors='replace')
                    return self.clean_content(text)
        except Exception as e:
            error_msg = f"Error fetching {url}: {str(e)}"
            self.research_state["error_log"].append(error_msg)
        return None

    def clean_content(self, text):
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
            return None

    async def initialize_research(self, session, research_request):
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
                return None
        return None

    async def generate_targeted_queries(self, session):
        """Generate targeted queries with reduced redundancy."""
        try:
            queries = []
            coverage = self.research_state.get("coverage_analysis", {}).get("coverage_by_aspect", {})
            
            # Track query combinations to avoid redundancy
            used_combinations = set()
            
            for aspect, current_coverage in sorted(
                self.research_state["aspect_coverage"].items(),
                key=lambda x: x[1]
            ):
                if current_coverage < 90:
                    keywords = self.research_state["keyword_cache"].get(aspect, set())
                    if keywords:
                        for kw1, kw2 in itertools.combinations(keywords, 2):
                            query_combo = frozenset([kw1, kw2])
                            if query_combo not in used_combinations:
                                used_combinations.add(query_combo)
                                query = f"{kw1} AND {kw2}"
                                if query not in self.research_state["queries_attempted"]:
                                    queries.append({
                                        "query": query,
                                        "aspect": aspect,
                                        "type": "keyword",
                                        "targets": [aspect]
                                    })
            
            return queries[:5]  # Limit to 5 most promising queries
            
        except Exception as e:
            error_msg = f"Error generating targeted queries: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return []

    async def assess_coverage(self, session):
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
                coverage = self._generate_fallback_coverage()
            self.research_state["coverage_analysis"] = coverage
            return coverage
        except Exception as e:
            error_msg = f"Error in assess_coverage: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            return self._generate_fallback_coverage()

    def _generate_fallback_coverage(self):
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
            return {
                "coverage_by_aspect": {},
                "overall_completion": 0,
                "research_complete": False,
                "recommended_focus": []
            }

    async def evaluate_content(self, session, content, url):
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
                for aspect, details in evaluation["relevance_by_aspect"].items():
                    if details["score"] >= 6:
                        self.research_state["findings_by_aspect"][aspect].extend(details["key_findings"])
                        self._update_aspect_coverage(aspect, details["contribution_to_coverage"])
                self.research_state["source_credibility_scores"][url] = evaluation["source_credibility"]
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
                if paper_info["citation"]:
                    self.research_state["citations"].add(paper_info["citation"])
                self.research_state["gaps_identified"].update(evaluation["new_gaps_identified"])
                self._track_significant_study(evaluation, url)
                # Save query mapping if available (assume the last attempted query is stored in a temporary field)
                if hasattr(self, "_last_query_info"):
                    q_info = self._last_query_info
                    self.research_state["query_results"].setdefault(q_info["query"], []).append({
                        "aspect": q_info["aspect"],
                        "evaluation": evaluation
                    })
                return evaluation
        except Exception as e:
            error_msg = f"Error evaluating content from {url}: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            return None

    def _update_aspect_coverage(self, aspect, contribution):
        try:
            current_coverage = self.research_state["aspect_coverage"][aspect]
            new_coverage = min(100, current_coverage + (contribution * 0.1))
            self.research_state["aspect_coverage"][aspect] = new_coverage
        except Exception as e:
            error_msg = f"Error updating coverage for aspect {aspect}: {str(e)}"
            self.research_state["error_log"].append(error_msg)

    async def generate_final_report(self, session):
        """Generate comprehensive final report with intelligent chunking and summary prioritization."""
        try:
            # First generate executive summary of key findings
            executive_summary = await self._generate_executive_summary(session)
            
            # Generate aspect-specific reports with prioritized findings
            aspect_reports = []
            for aspect, coverage in self.research_state["aspect_coverage"].items():
                if coverage > 0:  # Only include aspects with findings
                    aspect_report = await self._generate_aspect_report(session, aspect)
                    aspect_reports.append(aspect_report)

            # Compile final report with intelligent length management
            report_sections = [
                "# Research Report\n\n",
                "## Executive Summary\n\n",
                f"{executive_summary}\n\n",
                "## Methodology\n\n",
                await self._generate_methodology_section(session),
                "\n## Key Findings by Research Aspect\n\n"
            ]

            # Add prioritized aspect reports
            for aspect_report in aspect_reports:
                report_sections.append(aspect_report)

            # Add analysis sections with length management
            report_sections.extend([
                "\n## Evidence Quality Analysis\n\n",
                await self._generate_evidence_analysis(session),
                "\n## Critical Gaps and Limitations\n\n",
                await self._generate_gaps_analysis(session),
                "\n## High-Impact Studies\n\n",
                await self._generate_significant_studies_summary(session)
            ])

            # Combine sections with intelligent truncation
            return self._combine_report_sections(report_sections)

        except Exception as e:
            error_msg = f"Error generating final report: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            print(error_msg)
            return await self._generate_emergency_report(session)

    async def _generate_executive_summary(self, session):
        """Generate concise executive summary focusing on key findings."""
        try:
            # Prioritize findings by evidence quality and coverage
            key_findings = []
            for aspect, data in self.research_state["findings_by_aspect"].items():
                quality = self.research_state["coverage_analysis"].get("coverage_by_aspect", {}).get(aspect, {}).get("evidence_quality", 0)
                findings = sorted(data, key=lambda x: x.get("significance", 0), reverse=True)[:3]
                key_findings.extend([(aspect, finding, quality) for finding in findings])

            # Sort by evidence quality
            key_findings.sort(key=lambda x: x[2], reverse=True)

            # Generate summary
            summary = f"Analysis of {len(self.research_state['papers_found'])} studies across {len(self.research_state['research_aspects'])} aspects reveals:\n\n"
            for aspect, finding, quality in key_findings[:10]:  # Top 10 findings
                summary += f"- {finding} ({aspect}, Evidence Quality: {quality}/10)\n"

            return summary

        except Exception as e:
            return f"Error generating executive summary: {str(e)}"

    def _combine_report_sections(self, sections):
        """Combine report sections with intelligent length management."""
        total_length = sum(len(s) for s in sections)
        if total_length <= 5000:
            return "".join(sections)

        # If too long, prioritize sections
        essential_sections = sections[:7]  # Executive summary and methodology
        remaining_space = 5000 - sum(len(s) for s in essential_sections)
        
        # Distribute remaining space across other sections
        other_sections = sections[7:]
        if remaining_space > 0 and other_sections:
            space_per_section = remaining_space // len(other_sections)
            truncated_sections = [
                section[:space_per_section] + "\n[Section truncated for length]\n"
                for section in other_sections
            ]
            return "".join(essential_sections + truncated_sections)
        
        return "".join(essential_sections) + "\n[Report truncated for length]\n"

    def _format_report_as_markdown(self, report):
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
            markdown.extend(["\n## Null Results\n\n"])
            for result in report.get('null_results', []):
                markdown.append(f"- {result}\n")
            markdown.extend(["\n## Gaps and Limitations\n\n"])
            for gap in report.get('gaps_and_limitations', []):
                markdown.append(f"- {gap}\n")
            markdown.extend(["\n## Recommendations\n\n"])
            for rec in report.get('recommendations', []):
                markdown.append(f"- {rec}\n")
            markdown.extend(["\n## References\n\n"])
            for ref in report.get('references', []):
                markdown.append(f"{ref}\n")
            markdown.extend(["\n## Query Results\n\n"])
            for query, evaluations in report.get('query_results', {}).items():
                markdown.append(f"**Query:** {query}\n")
                for eval_item in evaluations:
                    markdown.append(f"- Aspect: {eval_item.get('aspect')}, Summary: {json.dumps(eval_item.get('evaluation'), indent=2)}\n")
                markdown.append("\n")
            return "".join(markdown)
        except Exception as e:
            error_msg = f"Error formatting report as markdown: {str(e)}"
            self.research_state["error_log"].append(error_msg)
            return "Error generating formatted report."

    async def ensure_final_report(self, session):
        try:
            for aspect in self.research_state["research_aspects"]["aspects"]:
                if aspect not in self.research_state["completed_aspects"]:
                    await self.generate_aspect_report(session, aspect)
            # Print significant studies only once at the end.
            self._display_significant_studies()
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
            return "Failed to generate any report due to critical errors."

    def _display_significant_studies(self):
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

    async def check_aspect_completion(self, session, aspect):
        current = self.research_state["aspect_coverage"].get(aspect, 0)
        if current >= 90:
            self.research_state["completed_aspects"].add(aspect)
            print(f"Aspect '{aspect}' marked as completed (coverage: {current}%).")
        else:
            print(f"Aspect '{aspect}' not yet complete (coverage: {current}%).")

    async def generate_aspect_report(self, session, aspect):
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
        query = query_info.get("query")
        aspect = query_info.get("aspect")
        print(f"Processing query for aspect '{aspect}': {query}")
        self._last_query_info = query_info  # Save query info for mapping
        urls = await self.perform_search(session, query)
        success = False
        if urls:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            evaluations = await self.process_urls(session, urls, semaphore)
            if evaluations:
                success = True
        if success:
            self.research_state["successful_queries"].append(query)
        else:
            await self.handle_failed_query(session, query_info)
        return success

    async def handle_failed_query(self, session, query_info):
        error_msg = f"Query failed for aspect '{query_info.get('aspect')}' with query: {query_info.get('query')}"
        self.research_state["failed_queries"].append(query_info)
        self.research_state["error_log"].append(error_msg)
        print(error_msg)

async def conduct_research(goal):
    assistant = None
    session = None
    try:
        assistant = ResearchAssistant()
        async with aiohttp.ClientSession() as session:
            print("🔍 Analyzing research request...")
            analysis = await assistant.initialize_research(session, goal)
            if not analysis:
                print("❌ Failed to analyze research request")
                return await assistant.ensure_final_report(session)

            print("\n📊 Research Plan:")
            print(json.dumps(analysis, indent=2))

            iteration = 0
            previous_overall = 0
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            while iteration < 20:
                iteration += 1
                print(f"\n🔄 Iteration {iteration}")
                coverage = await assistant.assess_coverage(session)
                overall = coverage.get('overall_completion', 0)
                print(f"Overall completion: {overall}%")
                # Print summary per aspect
                for aspect, details in coverage.get('coverage_by_aspect', {}).items():
                    print(f"- {aspect}: {details.get('coverage_percentage', 0)}% complete")
                    await assistant.check_aspect_completion(session, aspect)
                # Check for improvement
                if overall - previous_overall < 2 and iteration > 3:
                    print("No significant improvement detected; breaking loop early.")
                    break
                previous_overall = overall

                queries = await assistant.generate_targeted_queries(session)
                if not queries:
                    print("No new queries generated. Proceeding to final report.")
                    break

                print("\n📝 Executing targeted queries:")
                for query_info in queries:
                    await assistant.process_query_safely(session, query_info)

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
