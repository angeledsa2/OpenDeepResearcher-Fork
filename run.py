import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Optional
import os

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
            },
            "aspect_queries": {}  # New addition
        }


    async def initialize_research(self, session: aiohttp.ClientSession, request: str) -> bool:
        """Initialize research with LLM analysis."""
        self.state["request"] = request
        prompt = (
            f"Analyze this research request thoroughly:\n{request}\n\n"
            "Break it down into targeted research aspects for academic review, each with:\n"
            "- Key concepts and search terms\n"
            "- Required evidence type\n"
            "- Success criteria\n"
            "- Credibility requirements\n\n"
            "Format as JSON with clear aspect names as keys. The aspect names should clearly capture and communicate the core questions we're researching.\n"
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
        """
        Makes an API call to the configured LLM endpoint with the given prompt.
        Handles retries, rate limits, and returns the LLM's text response if successful.
        """
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Research Assistant",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise research assistant performing critical academic research."},
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

    async def generate_search_query(self, session, aspect, findings):
        # Step 1: Extract key entities from the original question
        entity_prompt = (
            f"Extract the 3-5 most critical concepts or entities (nouns, key terms) from this research question:\n"
            f"{self.state['request']}\n\n"
            "Return a JSON list of terms, e.g., ['term1', 'term2', 'term3']."
        )
        entities = await self._call_llm(session, entity_prompt)
        try:
            key_terms = json.loads(entities) if entities else []
        except Exception:
            key_terms = ["unknown"]  # Fallback if parsing fails

        # Step 2: Build the main query prompt
        prompt = (
            f"Create ONE focused search query for research on:\n"
            f"Full Research Question: {self.state['request']}\n"
            f"Specific Aspect: {aspect}\n"
            f"Aspect Details: {json.dumps(self.state['aspects'].get(aspect, {}))}\n"
            f"Key Terms from Question (must include at least two): {json.dumps(key_terms)}\n"
            f"Prior Findings (refine based on these if present): {json.dumps(findings[:3]) if findings else 'None'}\n\n"
            "Requirements:\n"
            "1. Use 3-5 key terms, including at least two from the key terms list\n"
            "2. At most one Boolean operator (AND/OR)\n"
            "3. Keep it simple, scientific, and directly tied to the full question and aspect\n"
            "4. Put multi-word terms in quotes\n"
            "5. Avoid generic terms unless paired with key question concepts\n\n"
            "Return ONLY the search query string."
        )
        try:
            query = await self._call_llm(session, prompt)
            if query and self._validate_query(query, key_terms):
                print(f"🔍 Generated query: {query}")
                self.state['progress']['queries_attempted'].add(query)
                return query.strip().strip('"\'')
            
            # Fallback with stricter guidance
            retry_prompt = (
                f"Generate ONE simple 2-3 word search query for: {aspect}\n"
                f"Full Research Question: {self.state['request']}\n"
                f"Must include at least one of: {json.dumps(key_terms)}\n"
                "Keep it tied to the question."
            )
            retry_query = await self._call_llm(session, retry_prompt)
            if retry_query and self._validate_query(retry_query, key_terms):
                print(f"🔍 Retry query: {retry_query}")
                return retry_query.strip().strip('"\'')
        except Exception as e:
            print(f"❌ Query generation error: {str(e)}")
        return None
        
    def _validate_query(self, query: str, key_terms: List[str]) -> bool:
        """
        Validates a generated query against certain constraints:

        1) Tokenizes the query to identify terms and quoted phrases for counting.
        2) Ensures we do not exceed eight total terms.
        3) Permits at most one Boolean operator (AND/OR).
        4) Requires at least one key term match (case-insensitive partial match).
        5) Prints a rejection message and returns False if the query fails any constraint.

        Returns:
        True if the query passes all checks, otherwise False.
        """
        import re

        # --- Step 1: Split into tokens and track quotes ---
        terms = []
        current_term = []
        in_quotes = False

        for word in query.split():
            if '"' in word:
                quotes_count = word.count('"')
                # Toggle in_quotes if we see an odd number of quotes in this token
                if quotes_count % 2 == 1:
                    in_quotes = not in_quotes

            if in_quotes:
                # Accumulate words until closing quote
                current_term.append(word)
            else:
                # If we exit quotes, store the accumulated phrase
                if current_term:
                    terms.append(" ".join(current_term))
                    current_term = []
                # Exclude Boolean operators from being counted as 'terms'
                if word.upper() not in ("AND", "OR"):
                    terms.append(word)

        # If something remains in current_term, push it as a separate token
        if current_term:
            terms.append(" ".join(current_term))

        # Clean up leading/trailing whitespace and count Boolean operators
        terms = [term.strip() for term in terms if term.strip()]
        upper_query = query.upper()
        boolean_count = upper_query.count(" AND ") + upper_query.count(" OR ")

        # --- Step 2: Check partial key term matches in the entire cleaned query ---
        # Remove punctuation and uppercase everything for easier substring matching
        cleaned_query = re.sub(r"[^\w\s]", "", upper_query)

        key_terms_upper = [kt.upper() for kt in key_terms]
        matched_terms = 0
        for kt in key_terms_upper:
            # Strip punctuation from the key term as well
            cleaned_kt = re.sub(r"[^\w\s]", "", kt)
            if cleaned_kt and cleaned_kt in cleaned_query:
                matched_terms += 1

        # --- Step 3: Verify constraints ---
        # - Up to 8 terms
        # - At most 1 Boolean operator
        # - At least 1 key term match
        is_valid = (
            len(terms) <= 8 and
            boolean_count <= 1 and
            matched_terms >= 1
        )

        if not is_valid:
            print(f"⚠️ Query rejected: {query} (Key terms matched: {matched_terms}/1 required)")
        return is_valid


    
    async def process_research_results(self, session: aiohttp.ClientSession, urls: List[str], aspect: str) -> List[Dict]:
        query = self.state.get('aspect_queries', {}).get(aspect, None)  # Retrieve the query for this aspect
        results = []

        for url in urls:
            if url in self.state["processed_urls"]:
                continue

            try:
                content = await self.fetch_content(session, url)
                if not content:
                    continue

                # Modify the prompt to include the query, handling cases where query might be None
                query_str = f" with the search query: {query}" if query else ""
                analysis_prompt = (
                    f"Review this content for its relevancy to {aspect} {query_str}\n"
                    f"{content[:3000]}\n\n"
                    "Your entire response must be valid JSON. Return only JSON, with no additional text outside the JSON.\n"
                    "Include the following fields:\n"
                    "1. relevance_score (0-10)\n"
                    "2. key_findings (array of relevant findings)\n"
                    "3. citation_info (title, authors, year, journal)\n"
                    f"Only include findings directly relevant to research on {aspect} {(' and the search query' if query else '')}. key_findings should be a comprehensive answer to, or significant insight for, our research. Unless the content is highly relevant to our  {aspect} {(' and the search query' if query else '')}, keep key_findings to a maximum of 3 items. If the key_findings do not directly answer or have strong significance to our research {aspect} {(' and the search query' if query else '')}, do not score relevance higher than 3.\n"
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
                        if result["relevance_score"] >= 5:
                            self.state["findings"].setdefault(aspect, []).extend(result["key_findings"])

                        results.append(result)
                        self.state["processed_urls"].add(url)

                        # Log progress
                        print(f"\n📄 Processed: {url}")
                        print(f"Relevance: {result['relevance_score']}/10")
                  #      if result["key_findings"]:
                  #          print("Key Findings:")
                  #          for finding in result["key_findings"][:4]:  # Show first 4 findings
                  #              print(f"- {finding}")
                  #          if len(result["key_findings"]) > 2:
                  #              print(f"... and {len(result['key_findings'])-2} more findings")

                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse analysis for {url}")

            except Exception as e:
                print(f"❌ Processing error for {url}: {str(e)}")
                self.state["errors"].append(f"Result processing failed: {str(e)}")

        # Sort by relevance and return top results
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)

    async def perform_deep_analysis(self, session: aiohttp.ClientSession, aspect: str, top_results: List[Dict], max_results: int = 5) -> List[Dict]:
        """
        Performs a deeper analysis of the most relevant URLs for a given aspect.
        Instructs the LLM to return valid JSON with the following numeric fields:
        - methodology_quality (0–10)
        - evidence_strength (0–10)
        If the article has limited data, the LLM is prompted to estimate these values rather
        than default to zero. This helps ensure coverage and quality stats get updated.
        """
        query = self.state.get('aspect_queries', {}).get(aspect, None)
        top_results = top_results[:max_results]  # Limit to top 5 most relevant
        detailed_findings = []
        
        for result in top_results:
            try:
                content = await self.fetch_content(session, result["url"])
                if not content:
                    continue
                
                # Modified the prompt to force non-zero numeric fields in a 0–10 range
                query_str = f" with the search query: {query}" if query else ""
                detailed_prompt = (
                    f"Perform detailed, academic-grade research analysis on the subject {aspect}{query_str}:\n"
                    f"{content[:40000]}\n\n"
                    "Your entire response must be valid JSON. Return only JSON, with no extra text.\n"
                    "Include these fields:\n"
                    "1. comprehensive_findings (string)\n"
                    "2. methodology_quality (0–10 numeric)\n"
                    "3. evidence_strength (0–10 numeric)\n"
                    "4. limitations (array of strings)\n"
                    "5. implications (array of strings)\n"
                    "If you cannot find enough detail, estimate or infer numeric values.\n"
                    "DO NOT add data, findings, or presumptions not present in the content.\n"
                    "Do NOT wrap the JSON in backticks or disclaimers. Output JSON only."
                )
                
                analysis = await self._call_llm(session, detailed_prompt)
                if analysis:
                    try:
                        # NEW: debug print to see raw
                       # print("\nDEBUG: Raw LLM deep-analysis text:\n", analysis)

                        cleaned_json = self._clean_json_response(analysis)

                        # NEW: debug print to see extracted JSON
                        # print("DEBUG: JSON extracted:\n", cleaned_json)

                        data = json.loads(cleaned_json)

                        cleaned_json = self._clean_json_response(analysis)
                        data = json.loads(cleaned_json)
                        detailed_findings.append({
                            "url": result["url"],
                            "citation_info": result.get("citation_info", {}),
                            "analysis": data
                        })
                        
                        print(f"\n🔍 Deep analysis completed for: {result['url']}")
                       # print(f"Quality: {data.get('methodology_quality', 0)}/10")
                       # print(f"Evidence Strength: {data.get('evidence_strength', 0)}/10")
                    except json.JSONDecodeError as je:
                        print(f"⚠️ JSON parsing error in deep analysis: {str(je)}")
                        # Create a minimal valid analysis to prevent research failure
                        fallback_analysis = {
                            "comprehensive_findings": ["Could not extract detailed findings"],
                            "methodology_quality": 0,
                            "evidence_strength": 0,
                            "limitations": ["Analysis failed due to response format issues"],
                            "implications": ["Further manual review recommended"]
                        }
                        detailed_findings.append({
                            "url": result["url"],
                            "citation_info": result.get("citation_info", {}),
                            "analysis": fallback_analysis
                        })
                        print(f"⚠️ Using fallback analysis for: {result['url']}")
                        
            except Exception as e:
                print(f"❌ Deep analysis failed for {result.get('url', 'unknown URL')}: {str(e)}")
                self.state["errors"].append(f"Deep analysis failed for {result.get('url', 'unknown URL')}: {str(e)}")
                    
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


    def _clean_json_response(self, response: str) -> str:
        """
        Attempts to extract valid JSON from an LLM response by searching
        for the first '{' and last '}' only. Then we do minimal cleanup:
        - Remove triple backticks if present
        - Strip leading/trailing whitespace
        We do NOT re-quote keys or do heavy regex replacements. We rely on
        the LLM to provide correct JSON.

        Returns a JSON string or '{}' if it cannot be properly extracted.
        """
        # Debug print to see raw LLM text
        # print("\nDEBUG: Raw LLM response for JSON:\n", response)

        # Remove triple backticks if present
        response = response.replace("```json", "").replace("```", "").strip()

        start = response.find('{')
        end = response.rfind('}')

        if start == -1 or end == -1 or end <= start:
            # If we never find a real JSON block, return empty
            return "{}"

        json_str = response[start:end+1].strip()
        return json_str

    async def fetch_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch content with enhanced error handling and retry logic."""
        max_retries = 3
        backoff_delay = 1
        for attempt in range(max_retries):
            print(f"\n🔍 Attempting to fetch: {url} (Attempt {attempt + 1}/{max_retries})")
            # print("📡 Attempting direct access...")
            try:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 200:
                        content = await resp.text(errors='ignore')
                        if content:
                            # print("✅ Direct access successful")
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
                    # print(f"⏳ Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
            except asyncio.TimeoutError:
                print("⚠️ Request timed out")
                self.state["errors"].append(f"Direct access timeout: {url}")
            except Exception as e:
                print(f"❌ Direct access error: {str(e)}")
                self.state["errors"].append(f"Direct access failed: {str(e)}")
                if isinstance(e, (aiohttp.ClientPayloadError, aiohttp.ClientOSError)):
                    break
        # print("🔄 Attempting JINA API access...")
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
        Clean content with enhanced text extraction and validation.
        Handles HTML, PDFs, and plain text with improved content preservation.
        """
        if not text:
            return None
            
        try:
            # First detect if content is HTML-like
            is_html = bool(re.search(r'<[^>]+>', text))
            
            if is_html:
                # Extract text from HTML while preserving important content
                cleaned = self._extract_html_content(text)
            else:
                cleaned = text
                
            # Clean special characters and normalize whitespace
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', cleaned)
            cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)  # Remove HTML entities
            
            # Normalize whitespace while preserving paragraph breaks
            cleaned = re.sub(r'\s*\n\s*\n\s*', '\n\n', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Remove common noise
            noise_patterns = [
                r'\[PDF\]|\[HTML\]|\[CITATION\]',
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
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
            
            # Verify content quality
            if self._validate_content_quality(cleaned):
                return cleaned
            else:
                print("⚠️ Content failed quality validation")
                return None
                
        except Exception as e:
            print(f"❌ Content cleaning error: {str(e)}")
            self.state["errors"].append(f"Content cleaning failed: {str(e)}")
            return None

    def _extract_html_content(self, html: str) -> str:
        """Extract meaningful content from HTML while preserving structure."""
        # Remove script and style elements
        html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html)
        html = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', html)
        
        # Remove navigation, headers, footers
        html = re.sub(r'<nav\b[^>]*>.*?<\/nav>', '', html)
        html = re.sub(r'<header\b[^>]*>.*?<\/header>', '', html)
        html = re.sub(r'<footer\b[^>]*>.*?<\/footer>', '', html)
        
        # Preserve paragraph breaks
        html = re.sub(r'</p>\s*<p>', '\n\n', html)
        html = re.sub(r'<br\s*/?\s*>', '\n', html)
        
        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        return text

    def _validate_content_quality(self, text: str) -> bool:
        """
        Validate cleaned content meets quality standards.
        Returns True if content is valid for analysis.
        """
        if not text or len(text) < 50:
            return False
            
        # Check for meaningful text (at least 3 letter words)
        if not re.search(r'[a-zA-Z]{3,}', text):
            return False
            
        # Check for reasonable text structure
        words = text.split()
        if len(words) < 10:  # Need at least 10 words
            return False
            
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.split()) > 3]
        if not valid_sentences:
            return False
            
        return True

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

    
    async def research_aspect(self, session: aiohttp.ClientSession, aspect: str, iteration: int) -> bool:
        """
        Perform one iteration of researching a given aspect using improved methodology.
        Maintains asynchronous processing with semaphore control.
        """
        print(f"\n📊 Research Progress for {aspect}")
        print(f"Current Coverage: {self.state['coverage'].get(aspect, 0)}%")
        print(f"Successful Findings: {len(self.state['findings'].get(aspect, []))}")
        # print(f"Failed Attempts: {len([q for q in self.state['progress']['failed_queries'] if q['aspect'] == aspect])}")

        try:
            # Generate and execute search query
            query = await self.generate_search_query(session, aspect, self.state['findings'].get(aspect, []))
            if not query:
                return False

            self.state['progress']['queries_attempted'].add(query)
            self.state['aspect_queries'][aspect] = query  # Store the query for this aspect
            print(f"\n📚 Researching {aspect} ---- (Iteration {iteration})")
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
                    results = await self.process_research_results(session, [url], aspect)
                    return results[0] if results else None  # Return first result or None

            tasks = [process_with_semaphore(url) for url in urls[:5]]  # Limit to 5 results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_results = [
                result for result in results 
                if result and not isinstance(result, Exception)  # Changed from result[0]
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
                "Create a detailed, comprehensive academic report with:\n"
                "1. Executive Summary (key findings and implications)\n"
                "2. Research Overview\n"
                "3. Key Findings by Research Aspect\n"
                "4. Evidence Quality and Limitations\n"
                "5. References\n\n"
                "Requirements:\n"
                "- DO NOT make assumptions or add your own knowledge to the report - solely use the findings made available to you here. \n"
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



        
    def calculate_quality_score(self, aspect_scores: List[Dict]) -> float:
        """Calculate normalized quality score from aspect scores."""
        try:
            # Flatten and validate scores
            all_scores = [
                float(v) for score_dict in aspect_scores 
                for v in score_dict.values() 
                if isinstance(v, (int, float)) and 0 <= float(v) <= 10
            ]
            return sum(all_scores) / len(all_scores) if all_scores else 0
        except Exception:
            return 0

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
                coverage_values = [v for v in assistant.state['coverage'].values() if isinstance(v, (int, float))]
                overall_progress = (sum(coverage_values) / len(coverage_values)) if coverage_values else 0
                print(f"\nResearch Progress: {overall_progress}%")
                
                print("\nCurrent Coverage:")
                for aspect, coverage in assistant.state['coverage'].items():
                    aspect_scores = assistant.state['research_quality'].get('aspect_scores', {}).get(aspect, [])
                    quality = assistant.calculate_quality_score(aspect_scores)  # Use the class method
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
                semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Create once at top level
                async def research_with_semaphore(aspect):
                    async with semaphore:
                        return await assistant.research_aspect(session, aspect, iteration)

                tasks = [research_with_semaphore(aspect) for aspect in incomplete_aspects]
                try:
                    results = await asyncio.gather(*tasks)
                except Exception as e:
                    print(f"❌ Task error: {str(e)}")
                    results = []

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
