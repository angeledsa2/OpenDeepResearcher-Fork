import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Optional
import os
from datetime import datetime

# If running in an environment with an already-running event loop (e.g. Jupyter/Colab), patch it.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', "")
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY', "")
JINA_API_KEY = os.getenv('JINA_API_KEY', "")
JINA_BASE_URL = "https://r.jina.ai/"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
MAX_CONCURRENT_REQUESTS = 5

# Endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search"
JINA_BASE_URL = "https://r.jina.ai/"

class PaywallException(Exception):
    """Custom exception for paywall detection"""
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        self.message = f"Paywall detected at {url}: {reason}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"🔒 Paywall at {self.url} (reason: {self.reason})"

    def get_details(self) -> Dict:
        return {
            "url": self.url,
            "reason": self.reason,
            "timestamp": datetime.now().isoformat()
        }

class ResearchAssistant:
    def __init__(self):
        """Initialize research assistant state."""
        self.state = {
            "request": None,
            "aspects": {},
            "findings": {},
            "coverage": {},
            "studies": {
                "high_relevance": [],
                "medium_relevance": [],
                "contradictory": [],
                "null_results": []
            },
            "citations": set(),
            "processed_urls": set(),
            "errors": [],
            "research_quality": {
                "aspect_scores": {},
                "gaps": set(),
                "quality_issues": []
            },
            "progress": {
                "queries_attempted": set(),
                "successful_queries": [],
                "failed_queries": []
            }
        }


    async def initialize_research(self, session: aiohttp.ClientSession, request: str) -> bool:
        """Initialize research with LLM analysis."""
        self.state["request"] = request
        prompt = (
            f"Analyze this research request thoroughly:\n{request}\n\n"
            "Break it down into research aspects, each with:\n"
            "- Key concepts and search terms\n"
            "- Required evidence types\n"
            "- Success criteria\n"
            "- Credibility requirements\n\n"
            "Format as JSON with clear aspect names as keys.\n"
            "Each aspect should have: concepts, evidence_types, criteria, credibility_requirements"
        )
        print("DEBUG: Sending the following prompt to LLM for research analysis:")
        print(prompt)
        try:
            analysis = await self._call_llm(session, prompt)
            if not analysis:
                print("DEBUG: Received no output from LLM.")
                return False
            print("DEBUG: Raw LLM output:")
            print(analysis)
            try:
                plan = json.loads(analysis)
            except Exception as e:
                print("DEBUG: JSON parsing failed. Raw response:")
                print(analysis)
                raise e
            self.state["aspects"] = plan
            for aspect in plan.keys():
                self.state["findings"][aspect] = []
                self.state["coverage"][aspect] = 0
                self.state["research_quality"]["aspect_scores"][aspect] = []
            return True
        except Exception as e:
            self.state["errors"].append(f"Initialization failed: {str(e)}")
            print(f"❌ Initialization error: {str(e)}")
            return False

    async def _call_llm(self, session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make LLM API call with robust response handling and retries for overloaded conditions."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Research Assistant",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise research assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        
        for attempt in range(max_retries):
            try:
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'error' in data and data['error'].get('code') == 502:
                            print(f"⚠️ API overloaded (attempt {attempt + 1}/{max_retries})")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return None
                        if ('choices' in data and len(data['choices']) > 0 and
                            'message' in data['choices'][0] and 'content' in data['choices'][0]['message']):
                            return data['choices'][0]['message']['content']
                        print("⚠️ Unexpected LLM response structure:", data)
                        return None
                    elif resp.status == 429:  # Rate limit
                        print(f"⚠️ Rate limited (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                    else:
                        error_text = await resp.text()
                        print(f"❌ LLM API error: {resp.status} - {error_text}")
                        return None
            except Exception as e:
                print(f"❌ LLM call failed: {str(e)}")
                self.state["errors"].append(f"LLM call failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
        return None

    async def generate_search_query(self, session: aiohttp.ClientSession, aspect: str, findings: List[str]) -> Optional[str]:
        """
        Generate focused search queries using LLM. Allows one complex operator and handles query validation.
        Returns a query string optimized for academic search.
        """
        prompt = (
            f"Create ONE search query for research on:\n"
            f"Topic: {aspect}\n"
            f"Research Area: {json.dumps(self.state['aspects'].get(aspect, {}))}\n\n"
            "Requirements:\n"
            "1. Use 2-4 key concepts\n"
            "2. You may use ONE Boolean operator (AND, OR) if needed\n"
            "3. Include essential synonyms in parentheses if needed\n"
            "4. Focus on academic/scientific sources\n\n"
            "Return ONLY the search query string without quotes or explanations."
        )
        try:
            query = await self._call_llm(session, prompt)
            if query:
                query = query.strip().strip('"\'')
                words = query.split()
                # Allow 2-10 words to accommodate Boolean operators
                if 2 <= len(words) <= 10 and len(query) >= 3:
                    # Check if query has at most one Boolean operator
                    upper_query = query.upper()
                    boolean_count = upper_query.count(" AND ") + upper_query.count(" OR ")
                    if boolean_count <= 1:
                        print(f"🔍 Generated query: {query}")
                        self.state['progress']['queries_attempted'].add(query)
                        return query
                    else:
                        print(f"⚠️ Too many Boolean operators ({boolean_count})")
                else:
                    print(f"⚠️ Query length outside bounds ({len(words)} words)")
                
                # Retry with simpler prompt if needed
                retry_prompt = (
                    f"Generate ONE simple search query for: {aspect}\n"
                    "Use 2-4 key terms and at most one Boolean operator (AND/OR).\n"
                    "Return only the search terms."
                )
                retry_query = await self._call_llm(session, retry_prompt)
                if retry_query:
                    retry_query = retry_query.strip().strip('"\'')
                    if 2 <= len(retry_query.split()) <= 10:
                        return retry_query
        except Exception as e:
            self.state["errors"].append(f"Query generation failed: {str(e)}")
            print(f"❌ Query error: {str(e)}")
        return None
    
    async def process_research_results(self, session: aiohttp.ClientSession, urls: List[str], aspect: str) -> List[Dict]:
        """
        Process research results with improved methodology. Focuses on extracting key findings
        while maintaining relevancy tracking for further analysis.
        """
        # Limit to 5 results per research area
        urls = urls[:5]
        results = []
        
        for url in urls:
            if url in self.state["processed_urls"]:
                continue
                
            try:
                content = await self.fetch_content(session, url)
                if not content:
                    continue
                    
                analysis_prompt = (
                    f"Analyze this content for {aspect}:\n"
                    f"{content[:3000]}\n\n"
                    "Provide JSON with:\n"
                    "1. relevance_score (0-10)\n"
                    "2. key_findings (array of relevant findings)\n"
                    "3. citation_info (title, authors, year, journal)\n"
                    "Only include findings directly relevant to our research aspect."
                )
                
                analysis = await self._call_llm(session, analysis_prompt)
                if analysis:
                    try:
                        data = json.loads(analysis)
                        result = {
                            "url": url,
                            "relevance_score": data.get("relevance_score", 0),
                            "key_findings": data.get("key_findings", []),
                            "citation_info": data.get("citation_info", {})
                        }
                        
                        # Track citation for later use
                        if result["citation_info"]:
                            self._track_citation(result["citation_info"])
                        
                        # Store findings if relevance score is above threshold
                        if result["relevance_score"] >= 6:
                            self.state["findings"].setdefault(aspect, []).extend(result["key_findings"])
                            
                        results.append(result)
                        self.state["processed_urls"].add(url)
                        
                        # Log progress
                        print(f"\n📄 Processed: {url}")
                        print(f"Relevance: {result['relevance_score']}/10")
                        if result["key_findings"]:
                            print("Key Findings:")
                            for finding in result["key_findings"][:2]:  # Show first 2 findings
                                print(f"- {finding}")
                            if len(result["key_findings"]) > 2:
                                print(f"... and {len(result['key_findings'])-2} more findings")
                                
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse analysis for {url}")
                        
            except Exception as e:
                print(f"❌ Processing error for {url}: {str(e)}")
                self.state["errors"].append(f"Result processing failed: {str(e)}")
                
        # Sort by relevance and return top results
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)

    async def perform_deep_analysis(self, session: aiohttp.ClientSession, aspect: str, top_results: List[Dict], max_results: int = 3) -> List[Dict]:
        """
        Perform deeper analysis on the most relevant results for each research aspect.
        """
        top_results = top_results[:max_results]  # Limit to top 3 most relevant
        detailed_findings = []
        
        for result in top_results:
            try:
                content = await self.fetch_content(session, result["url"])
                if not content:
                    continue
                    
                detailed_prompt = (
                    f"Perform detailed analysis for {aspect}:\n"
                    f"{content[:4000]}\n\n"
                    "Provide JSON with:\n"
                    "1. comprehensive_findings (detailed findings with evidence)\n"
                    "2. methodology_quality (0-10)\n"
                    "3. evidence_strength (0-10)\n"
                    "4. limitations\n"
                    "5. implications\n"
                    "Focus on evidence quality and research implications."
                )
                
                analysis = await self._call_llm(session, detailed_prompt)
                if analysis:
                    data = json.loads(analysis)
                    detailed_findings.append({
                        "url": result["url"],
                        "citation_info": result["citation_info"],
                        "analysis": data
                    })
                    
                    print(f"\n🔍 Deep analysis completed for: {result['url']}")
                    print(f"Quality: {data.get('methodology_quality', 0)}/10")
                    print(f"Evidence Strength: {data.get('evidence_strength', 0)}/10")
                    
            except Exception as e:
                print(f"❌ Deep analysis failed for {result['url']}: {str(e)}")
                
        return detailed_findings

    async def search_sources(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """Search for potential sources using the SERP API with enhanced error handling."""
        print(f"\n🔎 Executing search query: {query}")
        try:
            params = {
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "engine": "google_scholar",
                "num": 20
            }
            async with session.get("https://serpapi.com/search", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "organic_results" in data:
                        urls = []
                        titles = []
                        for result in data["organic_results"]:
                            if not isinstance(result, dict):
                                continue
                            url = result.get('link')
                            title = result.get('title', 'No title')
                            if url:  # Only add if URL exists
                                urls.append(url)
                                titles.append(title)
                        
                        if urls:
                            print(f"✅ Found {len(urls)} potential sources:")
                            for i, (url, title) in enumerate(zip(urls, titles), 1):
                                print(f"{i}. {title}\n   {url}")
                            return urls
                        else:
                            print("⚠️ No valid URLs found in results")
                    else:
                        print("⚠️ No organic_results found in SERP API response")
                        if "error" in data:
                            print(f"API Error: {data['error']}")
                else:
                    error_text = await resp.text()
                    print(f"❌ SERP API failed with status: {resp.status}")
                    print(f"Error details: {error_text}")
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.state["errors"].append(error_msg)
            print(f"❌ Search error: {error_msg}")
        return []

    async def prioritize_sources(self, session: aiohttp.ClientSession, urls: List[str], aspect: str) -> List[Dict]:
        """Prioritize sources using comprehensive LLM evaluation with robust priority parsing."""
        prioritized = []
        aspect_details = self.state['aspects'].get(aspect, {})
        
        for url in urls:
            if url in self.state["processed_urls"]:
                continue
            try:
                preview = await self._get_source_preview(session, url)
                prompt = (
                    f"Research Aspect: {aspect}\n"
                    f"Required Evidence Types: {aspect_details.get('evidence_types', [])}\n"
                    f"Quality Criteria: {aspect_details.get('criteria', [])}\n\n"
                    f"Source URL: {url}\n"
                    f"Preview: {preview}\n\n"
                    "Rate source priority (1-10) based on:\n"
                    "1. Relevance to research aspect\n"
                    "2. Match with required evidence types\n"
                    "3. Potential to meet quality criteria\n"
                    "4. Scientific credibility indicators\n"
                    "5. Data/methodology rigor signals\n\n"
                    "Respond with: [priority number] | [one-sentence reasoning]"
                )
                
                response = await self._call_llm(session, prompt)
                if response:
                    try:
                        # First try to find a number at the start of the response
                        import re
                        number_match = re.search(r'^[^\d]*(\d+(?:\.\d+)?)', response)
                        if not number_match:
                            # If no number at start, look for any number
                            number_match = re.search(r'(\d+(?:\.\d+)?)', response)
                        
                        if number_match:
                            try:
                                priority = float(number_match.group(1))
                                priority = int(max(1, min(10, priority)))
                                # Get everything after the number as reasoning
                                reasoning = response[number_match.end():].strip()
                                if '|' in reasoning:  # If there's a pipe, take everything after it
                                    reasoning = reasoning.split('|', 1)[1].strip()
                                
                                prioritized.append({
                                    "url": url,
                                    "priority": priority,
                                    "reasoning": reasoning or "No explicit reasoning provided",
                                    "evidence_types": [et for et in aspect_details.get('evidence_types', []) 
                                                    if et.lower() in preview.lower()],
                                    "criteria_matched": [c for c in aspect_details.get('criteria', []) 
                                                    if c.lower() in preview.lower()]
                                })
                                
                                if priority >= 7:
                                    print(f"🌟 High Priority ({priority}/10): {url}")
                                    print(f"📋 Reason: {reasoning}")
                                elif priority <= 3:
                                    print(f"⚠️ Low Priority ({priority}/10): {url}")
                            except ValueError:
                                print(f"⚠️ Could not convert priority to number: {number_match.group(1)}")
                                self.state["errors"].append(f"Priority conversion failed for {url}")
                        else:
                            print(f"⚠️ No priority number found in response: {response[:100]}...")
                            self.state["errors"].append(f"No priority number found for {url}")
                    except Exception as e:
                        print(f"⚠️ Priority parsing error: {str(e)}")
                        self.state["errors"].append(f"Priority parsing failed for {url}: {str(e)}")
                        
            except Exception as e:
                self.state["errors"].append(f"Source prioritization failed for {url}: {str(e)}")
                continue
                
        return sorted(prioritized,
                    key=lambda x: (x['priority'], len(x['evidence_types']), len(x['criteria_matched'])),
                    reverse=True)

    async def _get_source_preview(self, session: aiohttp.ClientSession, url: str) -> str:
        """Get a preview of source content using fetch_preview."""
        try:
            preview = await self.fetch_preview(session, url)
            return preview if preview else ""
        except Exception as e:
            print(f"⚠️ Preview fetch failed: {str(e)}")
            return ""

    async def analyze_failure(self, session: aiohttp.ClientSession, query_info: Dict, error: str) -> Dict:
        """Use LLM to analyze and recover from research failures."""
        prompt = (
            f"Analyze this research failure:\n"
            f"Query: {query_info['query']}\n"
            f"Error: {error}\n"
            f"Aspect: {query_info['aspect']}\n"
            f"Research Goal: {self.state['request']}\n\n"
            "Provide a JSON response with:\n"
            "1. analysis: What went wrong\n"
            "2. alternative_queries: Array of 3 alternative search queries\n"
            "3. different_approaches: Array of other research strategies\n"
            "4. recovery_plan: Step-by-step recovery strategy"
        )
        try:
            response = await self._call_llm(session, prompt)
            if response:
                return json.loads(response)
        except Exception as e:
            self.state["errors"].append(f"Failure analysis failed: {str(e)}")
        return {
            "analysis": "Analysis failed",
            "alternative_queries": [],
            "different_approaches": [],
            "recovery_plan": ["Retry with modified query"]
        }

    async def handle_failed_query(self, session: aiohttp.ClientSession, query_info: Dict, error_msg: str) -> None:
        """Handle failed search queries with analysis and recovery."""
        try:
            failure_analysis = await self.analyze_failure(session, query_info, error_msg)
            self.state['progress']['failed_queries'].append({
                "query": query_info['query'],
                "aspect": query_info['aspect'],
                "error": error_msg,
                "analysis": failure_analysis
            })
            print(f"\n⚠️ Search failed for query: {query_info['query']}")
            if failure_analysis.get('analysis'):
                print(f"Analysis: {failure_analysis['analysis']}")
            if failure_analysis.get('alternative_queries'):
                print("Suggested alternatives:", failure_analysis['alternative_queries'])
        except Exception as e:
            self.state["errors"].append(f"Failed to handle failed query: {str(e)}")
            print(f"❌ Error handling failed query: {str(e)}")

    async def _analyze_content(self, session: aiohttp.ClientSession, content: str, aspect: str) -> Dict:
        """Perform a single LLM call that both evaluates relevance and extracts full research analysis."""
        prompt = (
            f"Analyze this content thoroughly:\n"
            f"Research Aspect: {aspect}\n"
            f"Content: {content[:3000]}\n\n"
            "Extract the following:\n"
            "1. Relevance score (0-10) for this content regarding the research aspect\n"
            "2. Key findings relevant to the research (each with: finding, confidence (0-10), evidence_strength (0-10), novelty (0-10))\n"
            "3. Quality assessment: {{ methodology: 0-10, evidence: 0-10, novelty: 0-10, rigor: 0-10 }}\n"
            "4. Citation info: {{ title, authors, year, journal, doi }}\n\n"
            "Return JSON with the fields: relevance_score, key_findings, quality_assessment, citation_info."
        )
        try:
            analysis = await self._call_llm(session, prompt)
            return json.loads(analysis) if analysis else {}
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return {}

    async def fetch_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch content with enhanced error handling and retry logic."""
        max_retries = 3
        backoff_delay = 1
        for attempt in range(max_retries):
            print(f"\n🔍 Attempting to fetch: {url} (Attempt {attempt + 1}/{max_retries})")
            print("📡 Attempting direct access...")
            try:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 200:
                        content = await resp.text(errors='ignore')
                        if content:
                            print("✅ Direct access successful")
                            cleaned_content = self._clean_content(content)
                            if cleaned_content:
                                return cleaned_content
                            print("⚠️ Content cleaning failed, retrying...")
                        else:
                            print("⚠️ Empty content received")
                    else:
                        print(f"⚠️ Direct access failed with status: {resp.status}")
                        if resp.status in [403, 404, 410]:
                            break
                if attempt < max_retries - 1:
                    delay = backoff_delay * (2 ** attempt)
                    print(f"⏳ Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
            except asyncio.TimeoutError:
                print("⚠️ Request timed out")
                self.state["errors"].append(f"Direct access timeout: {url}")
            except Exception as e:
                print(f"❌ Direct access error: {str(e)}")
                self.state["errors"].append(f"Direct access failed: {str(e)}")
                if isinstance(e, (aiohttp.ClientPayloadError, aiohttp.ClientOSError)):
                    break
        print("🔄 Attempting JINA API access...")
        try:
            jina_url = f"{JINA_BASE_URL}{url}"
            headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
            async with session.get(jina_url, headers=headers, timeout=30) as resp:
                if resp.status == 200:
                    content = await resp.text(errors='ignore')
                    if content:
                        print("✅ JINA API access successful")
                        cleaned_content = self._clean_content(content)
                        if cleaned_content:
                            return cleaned_content
                        print("⚠️ JINA content cleaning failed")
                    else:
                        print("⚠️ Empty content from JINA API")
                else:
                    print(f"❌ JINA API failed with status: {resp.status}")
        except asyncio.TimeoutError:
            print("⚠️ JINA API request timed out")
            self.state["errors"].append(f"JINA API timeout: {url}")
        except Exception as e:
            print(f"❌ JINA API error: {str(e)}")
            self.state["errors"].append(f"JINA access failed: {str(e)}")
        print("❌ Failed to fetch content via both direct access and JINA API")
        return None

    def _clean_content(self, text: str) -> Optional[str]:
        """
        Clean content with enhanced HTML handling and better text normalization.
        Removes scripts, styles, and other non-content elements.
        """
        if not text:
            return None
        
        try:
            # First remove common HTML elements that don't contain useful content
            html_patterns = [
                r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Remove scripts
                r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>',     # Remove styles
                r'<nav\b[^>]*>.*?<\/nav>',                             # Remove navigation
                r'<header\b[^>]*>.*?<\/header>',                       # Remove headers
                r'<footer\b[^>]*>.*?<\/footer>',                       # Remove footers
                r'<!--.*?-->',                                         # Remove comments
                r'<!\[CDATA\[.*?\]\]>',                               # Remove CDATA
                r'<select\b[^>]*>.*?<\/select>'                       # Remove select elements
            ]
            
            cleaned = text
            for pattern in html_patterns:
                cleaned = re.sub(pattern, ' ', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove remaining HTML tags while preserving their content
            cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
            
            # Clean special characters and normalize whitespace
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', cleaned)
            cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)  # Remove HTML entities
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Remove common noise
            noise_patterns = [
                r'\[PDF\]|\[HTML\]|\[CITATION\]',
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                r'[""'']',
                r'[–—]',
                r'cookie[s]?\s+policy',
                r'privacy\s+policy',
                r'terms\s+of\s+service',
                r'javascript\s+must\s+be\s+enabled'
            ]
            
            for pattern in noise_patterns:
                cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
            
            # Normalize quotes and dashes
            cleaned = re.sub(r'[""'']', '"', cleaned)
            cleaned = re.sub(r'[–—]', '-', cleaned)
            
            # Final cleanup
            cleaned = cleaned.strip()
            
            # Verify minimum content length and meaningful content
            if len(cleaned) < 50 or not re.search(r'[a-zA-Z]{3,}', cleaned):
                print("⚠️ Cleaned content too short or lacks meaningful text")
                return None
                
            return cleaned
            
        except Exception as e:
            print(f"❌ Content cleaning error: {str(e)}")
            self.state["errors"].append(f"Content cleaning failed: {str(e)}")
            return None

    def _track_citation(self, citation_info: Dict) -> None:
        """Track citation with proper formatting."""
        if not citation_info:
            return
        authors = citation_info.get('authors')
        if not authors or len(authors) == 0:
            authors = ['Unknown']
        formatted_authors = authors[0] + " et al." if len(authors) > 1 else authors[0]
        citation = " ".join(filter(None, [
            formatted_authors,
            f"({citation_info.get('year', 'n.d.')})",
            citation_info.get('title', ''),
            citation_info.get('journal', ''),
            citation_info.get('doi', '')
        ]))
        self.state["citations"].add(citation)

    async def fetch_preview(self, session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Fetch a preview of the content with enhanced error handling and retry logic.
        Handles various HTTP status codes and implements exponential backoff.
        """
        headers = {
            "Range": "bytes=0-2000",
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +http://example.com/bot)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status in (200, 206):
                        content = await resp.text(errors='ignore')
                        preview = self._clean_content(content)
                        if preview:
                            print(f"Preview for {url} (first 100 chars): {preview[:100]}...")
                            return preview
                        print(f"⚠️ No usable content found in preview for {url}")
                    elif resp.status == 403:
                        print(f"⚠️ Access forbidden for {url}")
                        # Try alternative preview method using JINA
                        try:
                            jina_url = f"{JINA_BASE_URL}{url}"
                            jina_headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
                            async with session.get(jina_url, headers=jina_headers, timeout=10) as jina_resp:
                                if jina_resp.status == 200:
                                    content = await jina_resp.text(errors='ignore')
                                    preview = self._clean_content(content)
                                    if preview:
                                        print(f"✅ JINA preview successful for {url}")
                                        return preview
                        except Exception as je:
                            print(f"⚠️ JINA preview failed: {str(je)}")
                    elif resp.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        print(f"⚠️ Rate limited, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"⚠️ Preview fetch failed with status {resp.status} for {url}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    
            except asyncio.TimeoutError:
                print(f"⚠️ Timeout fetching preview for {url}")
                if attempt < max_retries - 1:
                    continue
            except Exception as e:
                error_msg = f"Preview fetch failed for {url}: {str(e)}"
                self.state["errors"].append(error_msg)
                print(f"⚠️ {error_msg}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break
        
        return None

    async def research_aspect(self, session: aiohttp.ClientSession, aspect: str, iteration: int) -> bool:
        """
        Perform one iteration of researching a given aspect using improved methodology.
        Maintains asynchronous processing with semaphore control.
        """
        print(f"\n📊 Research Progress for {aspect}")
        print(f"Current Coverage: {self.state['coverage'].get(aspect, 0)}%")
        print(f"Successful Findings: {len(self.state['findings'].get(aspect, []))}")
        print(f"Failed Attempts: {len([q for q in self.state['progress']['failed_queries'] if q['aspect'] == aspect])}")

        try:
            # Generate and execute search query
            query = await self.generate_search_query(session, aspect, self.state['findings'].get(aspect, []))
            if not query:
                return False

            self.state['progress']['queries_attempted'].add(query)
            print(f"\n📚 Researching {aspect} (Iteration {iteration})")
            print(f"🔍 Query: {query}")
            
            # Get search results
            urls = await self.search_sources(session, query)
            if not urls:
                await self.handle_failed_query(session, {"query": query, "aspect": aspect}, "No sources found")
                return False

            # Process results concurrently with semaphore control
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            async def process_with_semaphore(url):
                async with semaphore:
                    return await self.process_research_results(session, [url], aspect)

            tasks = [process_with_semaphore(url) for url in urls[:5]]  # Limit to 5 results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and flatten results
            valid_results = [
                result[0] for result in results 
                if isinstance(result, list) and result and not isinstance(result, Exception)
            ]

            if not valid_results:
                return False

            # Perform deep analysis concurrently on top 3
            sorted_results = sorted(valid_results, key=lambda x: x.get('relevance_score', 0), reverse=True)
            top_results = sorted_results[:3]

            async def analyze_with_semaphore(result):
                async with semaphore:
                    return await self.perform_deep_analysis(session, aspect, [result])

            analysis_tasks = [analyze_with_semaphore(result) for result in top_results]
            detailed_findings = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process successful analyses
            valid_findings = [
                finding[0] for finding in detailed_findings 
                if isinstance(finding, list) and finding and not isinstance(finding, Exception)
            ]

            if valid_findings:
                # Calculate quality metrics
                quality_scores = []
                for finding in valid_findings:
                    analysis = finding.get('analysis', {})
                    quality = (
                        analysis.get('methodology_quality', 0) +
                        analysis.get('evidence_strength', 0)
                    ) / 2
                    quality_scores.append(quality)

                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    coverage_increase = min(avg_quality * 10, 25)
                    current = self.state['coverage'].get(aspect, 0)
                    self.state['coverage'][aspect] = min(100, current + coverage_increase)
                    
                    print(f"\n📈 Updated Coverage: {self.state['coverage'][aspect]}%")
                    print(f"📊 Average Quality Score: {avg_quality:.1f}/10")
                    
                    if query not in self.state['progress']['successful_queries']:
                        self.state['progress']['successful_queries'].append(query)
                    
                    return True

            # Handle unsuccessful research attempt
            failure_analysis = await self.analyze_failure(
                session,
                {"query": query, "aspect": aspect},
                "No relevant findings extracted"
            )
            print("\n⚠️ Research attempt did not yield quality findings")
            print("Analysis:", failure_analysis.get('analysis', 'No analysis available'))
            self.state['progress']['failed_queries'].append({
                "query": query,
                "aspect": aspect,
                "analysis": failure_analysis
            })
            return False
            
        except Exception as e:
            self.state["errors"].append(f"Research failed for {aspect}: {str(e)}")
            print(f"❌ Research error: {str(e)}")
            return False
        
    async def process_url(self, session: aiohttp.ClientSession, url: str, aspect: str) -> Optional[Dict]:
        """Process URL with emphasis on thorough review."""
        if url in self.state["processed_urls"]:
            print(f"⏭️ Already processed: {url}")
            return None
        print(f"\n📖 Processing: {url}")
        try:
            content = await self.fetch_content(session, url)
            if not content:
                print("❌ Could not fetch content")
                return None
            # In this merged approach, we call _analyze_content directly,
            # which evaluates relevance and extracts findings.
            analysis = await self._analyze_content(session, content, aspect)
            if analysis and 'key_findings' in analysis:
                self.state['findings'].setdefault(aspect, []).extend(analysis['key_findings'])
                print(f"✨ Added {len(analysis['key_findings'])} findings")
            self._track_citation({
                "url": url,
                "title": analysis.get('title', ''),
                "authors": analysis.get('authors', []),
                "year": analysis.get('year'),
                "journal": analysis.get('journal')
            })
            self.state["processed_urls"].add(url)
            return analysis
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            self.state["errors"].append(f"URL processing failed: {str(e)}")
        return None

    async def generate_interim_report(self, session: aiohttp.ClientSession, aspect: str) -> str:
        """Generate detailed interim report for a research aspect."""
        analysis_prompt = (
            f"Analyze research progress for aspect: {aspect}\n"
            f"Findings: {json.dumps(self.state['findings'].get(aspect, []))}\n"
            f"Coverage: {self.state['coverage'].get(aspect, 0)}%\n"
            f"Quality Scores: {json.dumps(self.state['research_quality']['aspect_scores'].get(aspect, []))}\n"
            f"Studies: {json.dumps([s for s in self.state['studies']['high_relevance'] if aspect in s.get('relevance_scores', {})])}\n\n"
            "Provide detailed JSON analysis with:\n"
            "1. key_findings_synthesis\n"
            "2. evidence_quality_assessment\n"
            "3. remaining_gaps\n"
            "4. research_recommendations\n"
            "5. potential_biases"
        )
        try:
            analysis = await self._call_llm(session, analysis_prompt)
            if not analysis:
                raise ValueError("Analysis failed")
            analyzed = json.loads(analysis)
            report_prompt = (
                f"Generate interim research report using this analysis:\n"
                f"{json.dumps(analyzed)}\n\n"
                "Format as markdown with sections:\n"
                "1. Current Progress\n"
                "2. Key Findings\n"
                "3. Evidence Quality\n"
                "4. Gaps and Limitations\n"
                "5. Next Steps\n\n"
                "Include relevant citations and quality metrics."
            )
            report = await self._call_llm(session, report_prompt)
            if report:
                return report
        except Exception as e:
            self.state["errors"].append(f"Interim report generation failed for {aspect}: {str(e)}")
        return (
            f"# Interim Report: {aspect}\n\n"
            f"## Current Progress\nCoverage: {self.state['coverage'].get(aspect, 0)}%\n\n"
            f"## Key Findings\n" + "\n".join(f"- {finding}" for finding in self.state['findings'].get(aspect, [])) +
            "\n\n## Evidence Quality\nBased on {len(self.state['research_quality']['aspect_scores'].get(aspect, []))} evaluated sources\n\n"
            "## Gaps and Limitations\n- Report generation failed, showing raw findings\n\n"
            "## Next Steps\n- Continue research to improve coverage\n- Attempt report generation again"
        )

    async def generate_academic_report(self, session: aiohttp.ClientSession) -> str:
        """Generate comprehensive executive report with citations in one step."""
        try:
            findings = {
                aspect: [
                    finding if isinstance(finding, dict) else {"finding": finding, "confidence": "N/A", "evidence_strength": "N/A"}
                    for finding in findings_list
                ]
                for aspect, findings_list in self.state.get('findings', {}).items() if findings_list
            }
            citations = list(self.state.get('citations', set()))
            coverage = {aspect: score for aspect, score in self.state.get('coverage', {}).items() if isinstance(score, (int, float))}
            quality_metrics = {
                aspect: scores for aspect, scores in self.state.get('research_quality', {}).get('aspect_scores', {}).items() if scores
            }
            if not findings:
                print("⚠️ No findings available for report generation")
                return self._format_raw_findings()
            report_prompt = (
                f"Generate an executive research report for:\n"
                f"Research Question: {self.state.get('request', 'Research question not available')}\n"
                f"Findings: {json.dumps(findings, ensure_ascii=False)}\n"
                f"Citations: {json.dumps(citations, ensure_ascii=False)}\n"
                f"Coverage: {json.dumps(coverage, ensure_ascii=False)}\n"
                f"Quality Metrics: {json.dumps(quality_metrics, ensure_ascii=False)}\n\n"
                "Create a professional executive report with:\n"
                "1. Executive Summary (key findings and implications)\n"
                "2. Research Overview\n"
                "3. Key Findings by Research Aspect\n"
                "4. Evidence Quality and Limitations\n"
                "5. References\n\n"
                "Requirements:\n"
                "- Use clear, professional language\n"
                "- Include relevant citations [Author, Year]\n"
                "- Highlight strength of evidence\n"
                "- Address research gaps\n"
                "Format in markdown with proper styling."
            )
            report = await self._call_llm(session, report_prompt)
            if report and len(report.strip()) > 100:
                try:
                    if all(section in report for section in ['# ', '## ', 'Summary', 'Findings']):
                        return report
                    else:
                        print("⚠️ Generated report missing required sections")
                except Exception as e:
                    print(f"⚠️ Report validation failed: {str(e)}")
            print("⚠️ Falling back to raw findings format")
            return self._format_raw_findings()
        except Exception as e:
            self.state["errors"].append(f"Report generation failed: {str(e)}")
            return self._format_raw_findings()

    def _format_raw_findings(self) -> str:
        """Format raw findings in a clear, structured way."""
        sections = [
            "# Research Findings Summary\n",
            f"## Research Question\n{self.state['request']}\n",
            "## Key Findings by Aspect\n"
        ]
        for aspect, findings in self.state['findings'].items():
            if findings:
                sections.append(f"\n### {aspect} (Coverage: {self.state['coverage'].get(aspect, 0)}%)\n")
                for finding in findings:
                    if isinstance(finding, dict):
                        confidence = finding.get('confidence', 'N/A')
                        evidence = finding.get('evidence_strength', 'N/A')
                        sections.append(f"- Finding: {finding.get('finding', 'Unknown')}\n")
                        sections.append(f"  Confidence: {confidence}/10\n")
                        sections.append(f"  Evidence Strength: {evidence}/10\n")
                    else:
                        sections.append(f"- {finding}\n")
        if self.state['citations']:
            sections.extend([
                "\n## Sources\n",
                *[f"- {citation}\n" for citation in sorted(self.state['citations'])]
            ])
        return "\n".join(sections)

    def display_relevant_studies(self, aspect: str, studies: List[Dict]) -> None:
        """Display relevant studies for an aspect."""
        relevant = [study for study in studies if aspect.lower() in study.get('title', '').lower()]
        if relevant:
            print(f"\n📚 Relevant Studies for {aspect}:")
            for study in relevant[:3]:
                print(f"- {study['title']}")
                print(f"  Quality: {study['quality_score']:.1f}/10")
            print(f"  Relevance: {study['relevance_score']:.1f}/10")

    def _safe_json_parse(self, text: str) -> Dict:
        """Safely parse JSON with error handling."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.state["errors"].append(f"JSON parsing failed: {str(e)}")
            return {}

async def conduct_research(question: str) -> Optional[str]:
    """
    Orchestrates research by processing aspects concurrently while maintaining controlled concurrency.
    """
    assistant = ResearchAssistant()
    async with aiohttp.ClientSession() as session:
        try:
            print("🔍 Analyzing research request...")
            init_success = await assistant.initialize_research(session, question)
            if not init_success:
                print("❌ Failed to analyze research request. Captured errors:")
                for err in assistant.state.get("errors", []):
                    print(f"   - {err}")
                return None

            print("\n📊 Research Plan:")
            print(json.dumps(assistant.state["aspects"], indent=2))

            # Track the most relevant sources for each aspect
            top_sources_by_aspect = {aspect: [] for aspect in assistant.state["aspects"].keys()}
            
            iteration = 1
            while iteration <= 10:
                print(f"\n🔄 Iteration {iteration}")
                overall_progress = (sum(assistant.state['coverage'].values()) / 
                                  len(assistant.state['coverage']) if assistant.state['coverage'] else 0)
                print(f"\nResearch Progress: {overall_progress}%")
                
                print("\nCurrent Coverage:")
                for aspect, coverage in assistant.state['coverage'].items():
                    aspect_scores = assistant.state['research_quality'].get('aspect_scores', {}).get(aspect, [])
                    quality = 0
                    if aspect_scores:
                        total_scores = sum(float(v) for score_dict in aspect_scores 
                                        for v in score_dict.values() if isinstance(v, (int, float)))
                        score_count = sum(1 for score_dict in aspect_scores 
                                        for v in score_dict.values() if isinstance(v, (int, float)))
                        quality = total_scores / score_count if score_count > 0 else 0
                    print(f"- {aspect}: {coverage}% (Quality: {quality:.1f}/10)")

                if overall_progress >= 90:
                    print("\n✨ All research aspects have reached 90% coverage!")
                    break

                # Process aspects concurrently
                incomplete_aspects = [
                    aspect for aspect in assistant.state["aspects"].keys()
                    if assistant.state["coverage"].get(aspect, 0) < 90
                ]

                # Create tasks for all incomplete aspects
                async def research_with_semaphore(aspect):
                    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
                    async with semaphore:
                        return await assistant.research_aspect(session, aspect, iteration)

                tasks = [research_with_semaphore(aspect) for aspect in incomplete_aspects]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and update top sources
                for aspect, result in zip(incomplete_aspects, results):
                    if isinstance(result, bool) and result:
                        # Update top sources tracking
                        aspect_sources = [s for s in assistant.state.get('processed_urls', set())
                                       if s not in top_sources_by_aspect[aspect]]
                        if aspect_sources:
                            top_sources_by_aspect[aspect].extend(aspect_sources)
                            top_sources_by_aspect[aspect] = top_sources_by_aspect[aspect][:5]

                iteration += 1

            # After main research, perform deep analysis on top sources concurrently
            print("\n🔍 Performing deep analysis on most relevant sources...")
            analysis_tasks = []
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

            for aspect, sources in top_sources_by_aspect.items():
                if sources:
                    async def analyze_aspect_sources():
                        async with semaphore:
                            print(f"\nAnalyzing top sources for {aspect}...")
                            return await assistant.perform_deep_analysis(
                                session, 
                                aspect, 
                                [{"url": url} for url in sources[:3]]
                            )
                    analysis_tasks.append(analyze_aspect_sources())

            detailed_findings_list = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process successful analyses
            for aspect, findings in zip(top_sources_by_aspect.keys(), detailed_findings_list):
                if isinstance(findings, list) and findings:
                    assistant.state['findings'][aspect].extend(
                        finding.get('analysis', {}).get('comprehensive_findings', [])
                        for finding in findings
                    )

            print("\n📚 Generating comprehensive academic report...")
            return await assistant.generate_academic_report(session)

        except Exception as e:
            print(f"❌ Critical error: {str(e)}")
            for err in assistant.state.get("errors", []):
                print(f"   - {err}")
            if assistant.state.get('findings'):
                return assistant._format_raw_findings()
            return f"Research failed: {str(e)}"

if __name__ == "__main__":
    try:
        question = input("Enter your research question: ")
        report = asyncio.run(conduct_research(question))
        if report:
            print("\n🎯 Final Research Report:")
            print(report)
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
    except Exception as e:
        print(f"\n\nCritical error: {str(e)}")
