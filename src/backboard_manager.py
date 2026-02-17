"""
BackboardManager - Wrapper for Backboard.io SDK
Handles all API interactions for ModelForge data generation.
"""

import os
import asyncio
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from backboard import BackboardClient


@dataclass
class GeneratedSample:
    """Represents a single generated training sample."""
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any] = None
    
    def to_alpaca(self) -> Dict[str, str]:
        """Convert to Alpaca format for Unsloth training."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }
    
    def to_sharegpt(self) -> Dict[str, Any]:
        """Convert to ShareGPT format for conversational training."""
        conversations = [
            {"from": "human", "value": self.instruction if not self.input else f"{self.instruction}\n\n{self.input}"},
            {"from": "gpt", "value": self.output}
        ]
        return {"conversations": conversations}


class BackboardManager:
    """
    Manages Backboard.io API interactions for synthetic data generation.
    
    Features used:
    - Assistants: Create specialized assistants for each generation mode
    - Threads: Stateful conversation threads
    - Memory: Auto mode for deduplication and context retention
    - Model Routing: Select different LLM providers/models
    - Documents: RAG with uploaded files (Mode 2)
    - Tools: Function calling schema (Mode 4)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Backboard client.
        
        Args:
            api_key: Backboard API key. If not provided, reads from BACKBOARD_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("BACKBOARD_API_KEY")
        if not self.api_key:
            raise ValueError("BACKBOARD_API_KEY not found. Set it in .env or pass directly.")
        
        self.client = BackboardClient(api_key=self.api_key)
        self._assistants: Dict[str, str] = {}  # mode -> assistant_id cache
        self._threads: Dict[str, str] = {}     # session_key -> thread_id cache
    
    # =========================================================================
    # Assistant Management
    # =========================================================================
    
    async def get_or_create_assistant(
        self,
        mode: str,
        name: str,
        description: str,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Get existing assistant for a mode or create a new one.
        
        Args:
            mode: Generation mode identifier (e.g., "general_chat", "code")
            name: Display name for the assistant
            description: System description/personality
            tools: Optional tool definitions for agent mode
            
        Returns:
            assistant_id: The Backboard assistant ID
        """
        if mode in self._assistants:
            return self._assistants[mode]
        
        kwargs = {
            "name": name,
            "description": description
        }
        if tools:
            kwargs["tools"] = tools
            
        assistant = await self.client.create_assistant(**kwargs)
        self._assistants[mode] = assistant.assistant_id
        return assistant.assistant_id
    
    # =========================================================================
    # Thread Management
    # =========================================================================
    
    async def create_thread(self, assistant_id: str, session_key: Optional[str] = None) -> str:
        """
        Create a new conversation thread.
        
        Args:
            assistant_id: The assistant to attach this thread to
            session_key: Optional key to cache this thread for reuse
            
        Returns:
            thread_id: The Backboard thread ID
        """
        thread = await self.client.create_thread(assistant_id)
        
        if session_key:
            self._threads[session_key] = thread.thread_id
            
        return thread.thread_id
    
    def get_cached_thread(self, session_key: str) -> Optional[str]:
        """Get a cached thread ID by session key."""
        return self._threads.get(session_key)
    
    # =========================================================================
    # Mode 1: General Chat (Memory-Driven)
    # =========================================================================
    
    async def generate_chat_sample(
        self,
        thread_id: str,
        prompt: str,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        use_memory: bool = True
    ) -> GeneratedSample:
        """
        Generate a single chat training sample using memory to avoid duplicates.
        
        The memory="Auto" setting ensures:
        1. Previous generations are stored in long-term memory
        2. New prompts are contextualized against past outputs
        3. The model avoids repeating similar responses
        
        Args:
            thread_id: The thread to use for generation
            prompt: The generation prompt (e.g., "Generate a unique Q&A pair about Python")
            llm_provider: LLM provider (openai, anthropic, etc.)
            model_name: Specific model to use
            use_memory: Whether to enable memory for deduplication
            
        Returns:
            GeneratedSample with instruction/input/output fields
        """
        # Send the prompt
        await self.client.add_message(
            thread_id=thread_id,
            content=prompt,
            llm_provider=llm_provider,
            model_name=model_name,
            memory="Auto" if use_memory else None,
            stream=False
        )
        
        # Fetch the assistant's response from the thread
        content = await self._get_assistant_response(thread_id)
        
        # Try to parse as JSON, fallback to raw text
        sample = self._parse_generation_response(content, prompt)
        return sample
    
    def _parse_generation_response(self, content: str, original_prompt: str) -> GeneratedSample:
        """
        Parse LLM response into a structured sample.
        
        Handles multiple formats:
        - JSON with instruction/input/output
        - JSON with question/answer
        - JSON with challenge/solution (code mode)
        - Plain text (treated as output to the prompt)
        """
        import json
        
        # Try JSON parsing first
        try:
            # Clean potential markdown code blocks
            clean_content = content.strip()
            if clean_content.startswith("```"):
                # Remove markdown code block
                lines = clean_content.split("\n")
                clean_content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            
            data = json.loads(clean_content)
            
            # Handle various JSON structures with multiple key name conventions
            if isinstance(data, dict):
                instruction = (
                    data.get("instruction") or 
                    data.get("question") or 
                    data.get("challenge") or 
                    data.get("problem") or 
                    data.get("prompt") or
                    data.get("task") or
                    ""
                )
                input_text = (
                    data.get("input") or 
                    data.get("context") or 
                    data.get("constraints") or
                    ""
                )
                output = (
                    data.get("output") or 
                    data.get("answer") or 
                    data.get("solution") or 
                    data.get("code") or
                    data.get("response") or
                    ""
                )
                
                return GeneratedSample(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    metadata={"raw": content, "format": "json"}
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback: treat the entire response as output
        return GeneratedSample(
            instruction=original_prompt,
            input="",
            output=content,
            metadata={"raw": content, "format": "text"}
        )
    
    async def _get_assistant_response(self, thread_id: str) -> str:
        """
        Fetch the thread and get the last assistant response.
        
        After calling add_message, Backboard processes the request and adds
        the assistant's response to the thread. We need to fetch it.
        """
        thread = await self.client.get_thread(thread_id)
        
        # Get the last assistant message
        assistant_messages = [
            m for m in thread.messages 
            if hasattr(m.role, 'value') and m.role.value == "assistant" or str(m.role).lower() == "assistant"
        ]
        
        if not assistant_messages:
            return ""
        
        # Get the most recent assistant message
        last_message = assistant_messages[-1]
        return last_message.content or ""
    
    # =========================================================================
    # Mode 2: Knowledge Injection (RAG with Documents)
    # =========================================================================
    
    async def upload_document(self, file_path: str, file_content: bytes, file_name: str) -> str:
        """
        Upload a document to Backboard for RAG capabilities.
        
        Note: If Backboard SDK doesn't have upload_document, we'll read content
        and inject it into the prompt instead.
        
        Args:
            file_path: Path to the file
            file_content: Binary content of the file
            file_name: Name of the file
            
        Returns:
            document_id: A pseudo-ID (document content will be used in prompts)
        """
        # Since upload_document doesn't exist in the SDK, we'll store content
        # and inject it into prompts instead
        import hashlib
        doc_id = hashlib.md5(file_content).hexdigest()[:16]
        
        # Store document content for later use
        if not hasattr(self, '_document_cache'):
            self._document_cache = {}
        
        content_str = ""
        file_lower = file_name.lower()
        
        # Handle different file types
        if file_lower.endswith('.pdf'):
            # Extract text from PDF
            try:
                import io
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(io.BytesIO(file_content))
                text_parts = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                content_str = "\n\n".join(text_parts)
                if not content_str.strip():
                    content_str = f"[PDF with no extractable text: {file_name}]"
            except Exception as e:
                content_str = f"[Error reading PDF {file_name}: {str(e)}]"
        elif file_lower.endswith(('.txt', '.md', '.py', '.js', '.json', '.csv', '.xml', '.html')):
            # Text-based files
            try:
                content_str = file_content.decode('utf-8')
            except:
                try:
                    content_str = file_content.decode('latin-1')
                except:
                    content_str = f"[Could not decode text file: {file_name}]"
        elif file_lower.endswith('.docx'):
            # Word documents
            try:
                import io
                import zipfile
                import xml.etree.ElementTree as ET
                # DOCX is a zip file with XML content
                with zipfile.ZipFile(io.BytesIO(file_content)) as z:
                    xml_content = z.read('word/document.xml')
                    tree = ET.fromstring(xml_content)
                    # Extract text from w:t elements
                    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    texts = [node.text for node in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                    content_str = ' '.join(texts)
            except Exception as e:
                content_str = f"[Error reading DOCX {file_name}: {str(e)}]"
        else:
            # Try to decode as text, fallback to binary note
            try:
                content_str = file_content.decode('utf-8')
            except:
                content_str = f"[Binary file not supported: {file_name}]"
        
        self._document_cache[doc_id] = {
            'name': file_name,
            'content': content_str
        }
        
        print(f"Uploaded document {file_name}: {len(content_str)} characters extracted")
        
        return doc_id
    
    async def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a web search using Perplexity API and return results as context.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (1-10)
            
        Returns:
            Dict with 'content' (formatted search results), 'sources' (list of URLs), 
            and 'summary' (brief summary)
        """
        import os
        import httpx
        
        # Get Perplexity API key from environment
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_key:
            raise ValueError(
                "PERPLEXITY_API_KEY not found. Set it in your .env file. "
                "Get an API key at https://www.perplexity.ai/settings/api"
            )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a research assistant. Provide comprehensive, factual information with sources. Format your response clearly with key facts and details."
                            },
                            {
                                "role": "user", 
                                "content": query
                            }
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.2,
                        "return_citations": True
                    }
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Perplexity API error: {response.status_code} - {response.text}")
                
                data = response.json()
                
                # Extract content and citations
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                citations = data.get("citations", [])
                
                # Format sources
                sources = []
                for i, citation in enumerate(citations[:max_results], 1):
                    if isinstance(citation, str):
                        sources.append(citation)
                    elif isinstance(citation, dict):
                        sources.append(citation.get("url", citation.get("source", f"Source {i}")))
                
                # Create a formatted context block
                formatted_content = f"""=== WEB SEARCH RESULTS ===
Query: {query}

{content}

Sources:
""" + "\n".join([f"[{i+1}] {src}" for i, src in enumerate(sources)])
                
                return {
                    'content': formatted_content,
                    'sources': sources,
                    'summary': content[:500] + "..." if len(content) > 500 else content,
                    'query': query
                }
                
        except httpx.TimeoutException:
            raise TimeoutError(f"Web search timed out for query: {query}")
        except Exception as e:
            raise RuntimeError(f"Web search failed: {str(e)}")

    def get_cached_documents(self) -> List[Dict[str, Any]]:
        """
        Get all cached documents (files and URLs).
        
        Returns:
            List of document info dicts with id, name, type, and content preview
        """
        if not hasattr(self, '_document_cache'):
            return []
        
        docs = []
        for doc_id, doc in self._document_cache.items():
            doc_type = doc.get('type', 'file')
            preview = doc['content'][:150] + '...' if len(doc['content']) > 150 else doc['content']
            docs.append({
                'id': doc_id,
                'name': doc['name'],
                'type': doc_type,
                'url': doc.get('url', ''),
                'preview': preview,
                'char_count': len(doc['content'])
            })
        return docs
    
    def clear_document_cache(self):
        """Clear all cached documents and URLs."""
        self._document_cache = {}
        print("Document cache cleared")
    
    async def generate_rag_sample(
        self,
        thread_id: str,
        prompt: str,
        document_ids: List[str],
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
    ) -> GeneratedSample:
        """
        Generate a sample grounded in uploaded documents.
        
        Since Backboard SDK may not have document upload, we inject
        document content directly into the prompt.
        
        Args:
            thread_id: The thread to use
            prompt: Generation prompt
            document_ids: List of document IDs to use as context
            llm_provider: LLM provider
            model_name: Model to use
            
        Returns:
            GeneratedSample grounded in the documents
        """
        # Retrieve document content - use more chars for better context
        doc_cache = getattr(self, '_document_cache', {})
        
        # Build document context with up to 8000 chars per doc
        document_parts = []
        for doc_id in document_ids:
            if doc_id in doc_cache:
                doc = doc_cache[doc_id]
                # Truncate very long documents
                content = doc['content'][:8000]
                if len(doc['content']) > 8000:
                    content += f"\n... [Document truncated, {len(doc['content'])} total characters]"
                document_parts.append(f"=== Document: {doc['name']} ===\n{content}")
        
        document_context = "\n\n".join(document_parts)
        
        if not document_context.strip() or "[Binary" in document_context or "[Error" in document_context:
            # No valid document content
            return GeneratedSample(
                instruction="Document processing error",
                input="No readable document content available",
                output="Please upload text-based documents (PDF, TXT, MD) for RAG generation.",
                metadata={"error": "no_document_content"}
            )
        
        # Inject document context into prompt
        enhanced_prompt = f"""You have access to the following document content:

{document_context}

Based on the document content above, create a training example for topic: {prompt}

Generate a question that can be answered using SPECIFIC information from the documents, then provide the answer.

You MUST output valid JSON in this exact format:
{{
  "instruction": "A specific question about the document content",
  "input": "Relevant excerpt or context from the document",
  "output": "Detailed answer based on the document"
}}

Important: Use actual facts, quotes, and information from the document content provided above."""
        
        await self.client.add_message(
            thread_id=thread_id,
            content=enhanced_prompt,
            llm_provider=llm_provider,
            model_name=model_name,
            memory="Auto",
            stream=False
        )
        
        # Fetch the assistant's response
        content = await self._get_assistant_response(thread_id)
        return self._parse_generation_response(content, prompt)
    
    # =========================================================================
    # Mode 3: Code Specialist
    # =========================================================================
    
    async def generate_code_sample(
        self,
        thread_id: str,
        prompt: str,
        code_language: str = "python",
        llm_provider: str = "qwen",
        model_name: str = "qwen-2.5-coder-32b-instruct"
    ) -> GeneratedSample:
        """
        Generate code-focused training samples using specialized code models.
        
        Routes to Qwen or other code-specialized models by default.
        
        Args:
            thread_id: Thread to use
            prompt: Code generation prompt
            code_language: Programming language
            llm_provider: Provider (defaults to qwen for code)
            model_name: Code model to use
            
        Returns:
            GeneratedSample with code instruction/solution pairs
        """
        # Enhanced prompt for code generation
        enhanced_prompt = f"""
Generate a {code_language} coding challenge and solution in JSON format:
{{
  "instruction": "The coding problem or task description",
  "input": "Example input or constraints",
  "output": "Complete working code solution with comments"
}}

Topic: {prompt}
"""
        
        await self.client.add_message(
            thread_id=thread_id,
            content=enhanced_prompt,
            llm_provider=llm_provider,
            model_name=model_name,
            memory="Auto",
            stream=False
        )
        
        # Fetch the assistant's response
        content = await self._get_assistant_response(thread_id)
        return self._parse_generation_response(content, enhanced_prompt)
    
    # =========================================================================
    # Mode 4: Agent / Tool Use
    # =========================================================================
    
    async def generate_tool_use_sample(
        self,
        thread_id: str,
        prompt: str,
        tools: List[Dict[str, Any]],
        llm_provider: str = "openai",
        model_name: str = "gpt-4o"
    ) -> GeneratedSample:
        """
        Generate training samples for teaching models to use tools/functions.
        
        Uses Backboard's tools parameter to demonstrate function calling.
        
        Args:
            thread_id: Thread to use
            prompt: Scenario requiring tool use
            tools: List of tool definitions (OpenAI function format)
            llm_provider: Provider
            model_name: Model
            
        Returns:
            GeneratedSample with tool usage patterns
        """
        # Create enhanced prompt for tool use training
        tools_description = json.dumps(tools, indent=2) if tools else "[]"
        enhanced_prompt = f"""You have access to the following tools:

{tools_description}

User request: {prompt}

Generate a training example showing how to use the appropriate tool(s).
Output as JSON:
{{
  "instruction": "user's request",
  "input": "list of available tools",
  "output": "tool call with arguments and explanation"
}}"""
        
        await self.client.add_message(
            thread_id=thread_id,
            content=enhanced_prompt,
            llm_provider=llm_provider,
            model_name=model_name,
            memory="Auto",
            stream=False
        )
        
        # Fetch the assistant's response
        content = await self._get_assistant_response(thread_id)
        return self._parse_generation_response(content, prompt)
    
    def _parse_tool_response(
        self, 
        response: Any, 
        prompt: str, 
        tools: List[Dict]
    ) -> GeneratedSample:
        """
        Parse a response that may contain tool calls.
        
        Formats it as training data for teaching function calling.
        """
        content = response.message or response.content or ""
        tool_calls = getattr(response, 'tool_calls', [])
        
        if tool_calls:
            # Format tool calls as structured training data
            tool_calls_str = json.dumps([
                {
                    "function": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                }
                for tc in tool_calls
            ], indent=2)
            
            return GeneratedSample(
                instruction=prompt,
                input=f"Available tools: {json.dumps([t['function']['name'] for t in tools])}",
                output=f"Tool calls:\n{tool_calls_str}\n\nFinal response:\n{content}",
                metadata={"tools": tools, "tool_calls": tool_calls}
            )
        
        return self._parse_generation_response(content, prompt)
    
    # =========================================================================
    # Mode 5: Reasoning (Chain of Thought)
    # =========================================================================
    
    async def generate_reasoning_sample(
        self,
        thread_id: str,
        prompt: str,
        llm_provider: str = "deepseek",
        model_name: str = "deepseek-r1"
    ) -> GeneratedSample:
        """
        Generate reasoning traces with chain-of-thought.
        
        Routes to reasoning models like DeepSeek R1 or OpenAI O1 that produce
        <think>...</think> tags for internal reasoning.
        
        Args:
            thread_id: Thread to use
            prompt: Problem requiring reasoning
            llm_provider: Provider (defaults to deepseek)
            model_name: Reasoning model
            
        Returns:
            GeneratedSample with reasoning traces
        """
        # Enhanced prompt to encourage reasoning
        enhanced_prompt = f"""
Generate a complex reasoning problem and solution with step-by-step thought process.

Format your response as JSON:
{{
  "instruction": "The problem or question",
  "input": "Additional context or constraints",
  "output": "<think>Step-by-step reasoning process</think>\n\nFinal answer with explanation"
}}

Topic: {prompt}

Ensure the output includes detailed reasoning in <think> tags before the final answer.
"""
        
        await self.client.add_message(
            thread_id=thread_id,
            content=enhanced_prompt,
            llm_provider=llm_provider,
            model_name=model_name,
            memory="Auto",
            stream=False
        )
        
        # Fetch the assistant's response
        content = await self._get_assistant_response(thread_id)
        return self._parse_generation_response(content, enhanced_prompt)
    
    # =========================================================================
    # Batch Generation
    # =========================================================================
    
    async def generate_batch(
        self,
        mode: str,
        num_samples: int,
        base_prompt: str,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        topic: Optional[str] = None,
        batch_size: int = 5,  # Generate this many samples per API call
        existing_thread_id: Optional[str] = None,  # For appending to existing datasets
        existing_assistant_id: Optional[str] = None,  # For appending to existing datasets
        web_search_context: Optional[str] = None,  # Web search results to include
        **kwargs  # Mode-specific parameters
    ) -> tuple[List[GeneratedSample], str, str]:
        """
        Generate multiple samples for a given mode using batch prompting.
        
        Uses memory="Auto" to ensure samples don't repeat across batches.
        Generates multiple samples per API call for efficiency.
        
        Args:
            mode: Generation mode (general_chat, code, rag, agent, reasoning)
            num_samples: Total number of samples to generate
            base_prompt: The prompt template for generation
            llm_provider: LLM provider
            model_name: Model to use
            topic: Optional topic to inject into prompts
            batch_size: Number of samples to request per API call (default: 5)
            existing_thread_id: Optional thread ID to reuse for memory continuity
            existing_assistant_id: Optional assistant ID to reuse
            web_search_context: Optional web search results to inject as context
            **kwargs: Mode-specific parameters (document_ids, tools, code_language, etc.)
            
        Returns:
            Tuple of (List of GeneratedSample objects, thread_id, assistant_id)
        """
        # Get or create assistant for this mode
        tools = kwargs.get('tools', None)
        
        if existing_assistant_id:
            assistant_id = existing_assistant_id
        else:
            assistant_id = await self.get_or_create_assistant(
                mode=mode,
                name=f"ModelForge {mode.replace('_', ' ').title()} Generator",
                description=f"Synthetic data generator for {mode} training scenarios",
                tools=tools
            )
        
        # Use existing thread (for appending) or create new one
        # Memory persists across calls within the same thread
        if existing_thread_id:
            thread_id = existing_thread_id
            print(f"Using existing thread {thread_id} for memory continuity")
        else:
            thread_id = await self.create_thread(assistant_id)
            print(f"Created new thread {thread_id}")
        
        samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_num in range(num_batches):
            remaining = num_samples - len(samples)
            current_batch_size = min(batch_size, remaining)
            
            if current_batch_size <= 0:
                break
            
            try:
                # Generate batch prompt requesting multiple samples at once
                batch_samples = await self._generate_batch_samples(
                    thread_id=thread_id,
                    mode=mode,
                    base_prompt=base_prompt,
                    topic=topic,
                    batch_size=current_batch_size,
                    batch_num=batch_num + 1,
                    llm_provider=llm_provider,
                    model_name=model_name,
                    web_search_context=web_search_context,
                    **kwargs
                )
                samples.extend(batch_samples)
                print(f"Batch {batch_num + 1}/{num_batches}: Generated {len(batch_samples)} samples (total: {len(samples)})")
                
            except Exception as e:
                print(f"Error in batch {batch_num + 1}: {e}")
                # Fall back to individual generation for this batch
                for i in range(current_batch_size):
                    try:
                        sample = await self._generate_single_sample(
                            thread_id=thread_id,
                            mode=mode,
                            prompt=base_prompt.replace("{topic}", topic or ""),
                            llm_provider=llm_provider,
                            model_name=model_name,
                            **kwargs
                        )
                        samples.append(sample)
                    except Exception as e2:
                        print(f"Error generating sample: {e2}")
                        continue
        
        return samples, thread_id, assistant_id
    
    async def _generate_batch_samples(
        self,
        thread_id: str,
        mode: str,
        base_prompt: str,
        topic: Optional[str],
        batch_size: int,
        batch_num: int,
        llm_provider: str,
        model_name: str,
        web_search_context: Optional[str] = None,
        **kwargs
    ) -> List[GeneratedSample]:
        """
        Generate multiple samples in a single API call.
        
        Memory is used to ensure uniqueness across batches.
        """
        # Build batch prompt based on mode
        if mode == "rag" or mode == "knowledge_injection":
            batch_prompt = self._build_rag_batch_prompt(
                topic=topic,
                batch_size=batch_size,
                document_ids=kwargs.get('document_ids', []),
                web_search_context=web_search_context,
                custom_instructions=base_prompt
            )
        elif mode == "code" or mode == "code_specialist":
            batch_prompt = self._build_code_batch_prompt(
                topic=topic,
                batch_size=batch_size,
                code_language=kwargs.get('code_language', 'python'),
                web_search_context=web_search_context,
                custom_instructions=base_prompt
            )
        elif mode == "agent" or mode == "tool_use":
            batch_prompt = self._build_tool_batch_prompt(
                topic=topic,
                batch_size=batch_size,
                tools=kwargs.get('tools', []),
                web_search_context=web_search_context,
                custom_instructions=base_prompt
            )
        elif mode == "reasoning" or mode == "chain_of_thought":
            batch_prompt = self._build_reasoning_batch_prompt(
                topic=topic,
                batch_size=batch_size,
                web_search_context=web_search_context,
                custom_instructions=base_prompt
            )
        else:  # general_chat
            batch_prompt = self._build_chat_batch_prompt(
                topic=topic,
                batch_size=batch_size,
                base_prompt=base_prompt,
                web_search_context=web_search_context
            )
        
        # Make single API call with memory for deduplication
        await self.client.add_message(
            thread_id=thread_id,
            content=batch_prompt,
            llm_provider=llm_provider,
            model_name=model_name,
            memory="Auto",  # Ensures uniqueness across batches
            stream=False
        )
        
        # Get the response
        content = await self._get_assistant_response(thread_id)
        
        # Parse multiple samples from response
        return self._parse_batch_response(content, batch_size)
    
    def _build_chat_batch_prompt(self, topic: str, batch_size: int, base_prompt: str, web_search_context: Optional[str] = None) -> str:
        """Build batch prompt for general chat mode."""
        web_context_section = ""
        if web_search_context:
            web_context_section = f"""
You have access to the following web search results for reference:

{web_search_context}

Use this information to make your examples more accurate and up-to-date.
"""
        
        # Include custom instructions if provided
        custom_instructions_section = ""
        if base_prompt and base_prompt.strip():
            custom_instructions_section = f"""

Additional Instructions:
{base_prompt.strip()}
"""
        
        return f"""Generate {batch_size} UNIQUE and DIVERSE training examples for an AI assistant.

Topic/Domain: {topic or 'general knowledge'}
{web_context_section}
Requirements:
- Each example must be completely different from the others
- Each example must be different from any you've generated before in this conversation
- Cover different aspects, difficulty levels, and question types
- Make them realistic and useful for training
{custom_instructions_section}
CRITICAL: Output ONLY a raw JSON array. NO introduction, NO explanation, NO markdown code blocks. Start directly with [ and end with ].

Format:
[{{"instruction": "task 1", "input": "context", "output": "response"}}, ...]

["""

    def _build_rag_batch_prompt(self, topic: str, batch_size: int, document_ids: List[str], web_search_context: Optional[str] = None, custom_instructions: Optional[str] = None) -> str:
        """Build batch prompt for RAG mode with document context."""
        doc_cache = getattr(self, '_document_cache', {})
        
        # Build document context
        document_parts = []
        for doc_id in document_ids:
            if doc_id in doc_cache:
                doc = doc_cache[doc_id]
                content = doc['content'][:8000]
                document_parts.append(f"=== {doc['name']} ===\n{content}")
        
        document_context = "\n\n".join(document_parts)
        
        # Add web search context if available
        web_context_section = ""
        if web_search_context:
            web_context_section = f"""\n=== WEB SEARCH RESULTS ===\n{web_search_context}\n"""
        
        combined_context = document_context + web_context_section
        
        if not combined_context.strip():
            return self._build_chat_batch_prompt(topic, batch_size, custom_instructions or "")
        
        # Include custom instructions if provided
        custom_instructions_section = ""
        if custom_instructions and custom_instructions.strip():
            custom_instructions_section = f"""\nAdditional Instructions:\n{custom_instructions.strip()}\n"""
        
        return f"""You have access to the following documents and web search results:

{combined_context}

Generate {batch_size} UNIQUE question-answer training examples based on information in the context above.

Topic focus: {topic or 'document content'}

Requirements:
- Each question must be answerable from the document/web content
- Use specific facts, quotes, and details from the sources
- Each example must be completely different
- Cover different sections and aspects of the content
- Do NOT repeat any questions from earlier in this conversation
{custom_instructions_section}
CRITICAL: Output ONLY a raw JSON array with exactly {batch_size} objects. NO introduction text, NO explanation, NO markdown. Start directly with [ and end with ].

Each object MUST have this exact format:
{{"instruction": "A specific question about the content", "input": "", "output": "The detailed answer based on the sources"}}

["""

    def _build_code_batch_prompt(self, topic: str, batch_size: int, code_language: str, web_search_context: Optional[str] = None, custom_instructions: Optional[str] = None) -> str:
        """Build batch prompt for code generation mode."""
        web_context_section = ""
        if web_search_context:
            web_context_section = f"""\nUse the following web search results as reference for best practices and modern approaches:\n\n{web_search_context}\n"""
        
        # Include custom instructions if provided
        custom_instructions_section = ""
        if custom_instructions and custom_instructions.strip():
            custom_instructions_section = f"""\nAdditional Instructions:\n{custom_instructions.strip()}\n"""
        
        return f"""Generate {batch_size} UNIQUE {code_language} programming challenges and solutions.

Topic/Domain: {topic or 'programming fundamentals'}
{web_context_section}
Requirements:
- Each challenge must be completely different in concept
- Include varying difficulty levels (easy, medium, hard)
- Solutions must be complete, working code with comments
- Cover different programming concepts
- Do NOT repeat any challenges from earlier in this conversation
{custom_instructions_section}
CRITICAL: Output ONLY a raw JSON array. NO introduction, NO explanation, NO markdown. Start directly with [ and end with ].

["""

    def _build_tool_batch_prompt(self, topic: str, batch_size: int, tools: List[Dict], web_search_context: Optional[str] = None, custom_instructions: Optional[str] = None) -> str:
        """Build batch prompt for tool/agent mode."""
        tools_json = json.dumps(tools, indent=2) if tools else "[]"
        
        web_context_section = ""
        if web_search_context:
            web_context_section = f"""\nAdditional context from web search:\n{web_search_context}\n"""
        
        # Include custom instructions if provided
        custom_instructions_section = ""
        if custom_instructions and custom_instructions.strip():
            custom_instructions_section = f"""\nAdditional Instructions:\n{custom_instructions.strip()}\n"""
        
        return f"""You have access to these tools:

{tools_json}
{web_context_section}
Generate {batch_size} UNIQUE scenarios where a user needs help and the AI must use the appropriate tool(s).

Topic: {topic or 'general tasks'}

Requirements:
- Each scenario must require different tool usage
- Show realistic user requests
- Include the tool call with proper arguments in the output
- Vary the complexity and combinations of tools
- Do NOT repeat scenarios from earlier in this conversation
{custom_instructions_section}
CRITICAL: Output ONLY a raw JSON array with exactly {batch_size} objects. NO introduction text, NO explanation, NO markdown. Start directly with [ and end with ].

Format each object as:
{{"instruction": "user request requiring tools", "input": "context or details", "output": "tool call JSON and response"}}

["""

    def _build_reasoning_batch_prompt(self, topic: str, batch_size: int, web_search_context: Optional[str] = None, custom_instructions: Optional[str] = None) -> str:
        """Build batch prompt for reasoning/CoT mode."""
        web_context_section = ""
        if web_search_context:
            web_context_section = f"""\nUse the following web search results as reference for factual accuracy:\n\n{web_search_context}\n"""
        
        # Include custom instructions if provided
        custom_instructions_section = ""
        if custom_instructions and custom_instructions.strip():
            custom_instructions_section = f"""\nAdditional Instructions:\n{custom_instructions.strip()}\n"""
        
        return f"""Generate {batch_size} UNIQUE complex reasoning problems with step-by-step solutions.

Topic/Domain: {topic or 'logical reasoning, math, and analytical puzzles'}
{web_context_section}
Requirements:
- Each problem must require multi-step reasoning
- Include the thinking process in <think>...</think> tags in the output
- Vary problem types: logic puzzles, math problems, analysis, deduction
- Each problem must be completely different
- Do NOT repeat problems from earlier in this conversation
{custom_instructions_section}
CRITICAL: Output ONLY a raw JSON array with exactly {batch_size} objects. NO introduction text, NO explanation, NO markdown. Start directly with [ and end with ].

Format each object as:
{{"instruction": "The question or problem to solve", "input": "Additional context if needed, or empty string", "output": "<think>Step 1: ...\\nStep 2: ...\\n...</think>\\n\\nFinal Answer: ..."}}

["""

    def _parse_batch_response(self, content: str, expected_count: int) -> List[GeneratedSample]:
        """Parse a batch response containing multiple samples as JSON array."""
        samples = []
        
        # Clean the content
        clean_content = content.strip()
        
        # Handle empty response
        if not clean_content:
            print("Warning: Empty response received")
            return samples
        
        # Remove markdown code blocks if present
        if clean_content.startswith("```"):
            lines = clean_content.split("\n")
            # Find the end of code block
            start_idx = 1
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if i > 0 and line.strip().startswith("```"):
                    end_idx = i
                    break
            clean_content = "\n".join(lines[start_idx:end_idx])
        
        # Strip any leading text before the JSON array (common LLM issue)
        # Find the first '[' which starts the array
        first_bracket = clean_content.find('[')
        if first_bracket > 0:
            clean_content = clean_content[first_bracket:]
        
        # Find matching closing bracket
        last_bracket = clean_content.rfind(']')
        if last_bracket > 0:
            clean_content = clean_content[:last_bracket + 1]
        
        # Fix common JSON issues: escape unescaped newlines in strings
        # This handles cases where LLMs produce actual newlines inside JSON string values
        import re
        
        def fix_json_newlines(json_str: str) -> str:
            """Fix unescaped newlines inside JSON string values."""
            # Replace actual newlines that appear inside JSON strings with escaped version
            result = []
            in_string = False
            escape_next = False
            
            for char in json_str:
                if escape_next:
                    result.append(char)
                    escape_next = False
                elif char == '\\':
                    result.append(char)
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                    result.append(char)
                elif char == '\n' and in_string:
                    result.append('\\n')
                elif char == '\r' and in_string:
                    result.append('\\r')
                elif char == '\t' and in_string:
                    result.append('\\t')
                else:
                    result.append(char)
            
            return ''.join(result)
        
        clean_content = fix_json_newlines(clean_content)
        
        def is_valid_sample(instruction: str, input_text: str, output: str) -> bool:
            """Check if a sample has meaningful content."""
            # Must have at least instruction or output with real content
            has_instruction = bool(instruction and instruction.strip() and instruction.strip() not in ["", "N/A", "None"])
            has_output = bool(output and output.strip() and output.strip() not in ["", "N/A", "None"])
            return has_instruction or has_output
        
        # Try to parse as JSON array
        try:
            data = json.loads(clean_content)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Handle various key naming conventions
                        # For instruction/question/challenge/problem
                        instruction = (
                            item.get("instruction") or 
                            item.get("question") or 
                            item.get("challenge") or 
                            item.get("problem") or 
                            item.get("prompt") or
                            item.get("task") or
                            ""
                        )
                        
                        # For input/context
                        input_text = (
                            item.get("input") or 
                            item.get("context") or 
                            item.get("constraints") or
                            ""
                        )
                        
                        # For output/answer/solution/code/response
                        output = (
                            item.get("output") or 
                            item.get("answer") or 
                            item.get("solution") or 
                            item.get("code") or
                            item.get("response") or
                            ""
                        )
                        
                        if isinstance(output, dict):
                            output = json.dumps(output, indent=2)
                        
                        instruction = str(instruction)
                        input_text = str(input_text)
                        output = str(output)
                        
                        # Only add if sample has valid content
                        if is_valid_sample(instruction, input_text, output):
                            samples.append(GeneratedSample(
                                instruction=instruction,
                                input=input_text,
                                output=output,
                                metadata={"format": "json_batch"}
                            ))
                        else:
                            print(f"Warning: Skipping empty sample: {item}")
                            
            elif isinstance(data, dict):
                # Check if this is a malformed response where samples are nested inside "output"
                # This happens when LLM returns: {"instruction": "...", "output": [{...}, {...}]}
                output_value = data.get("output") or data.get("answer") or data.get("response")
                
                # Check if output contains a JSON array (either as list or string)
                nested_samples = None
                if isinstance(output_value, list) and len(output_value) > 0:
                    # Output is already a list - check if it contains sample dicts
                    if isinstance(output_value[0], dict) and ("instruction" in output_value[0] or "output" in output_value[0]):
                        nested_samples = output_value
                        print(f"Detected nested samples array in 'output' field ({len(nested_samples)} items)")
                elif isinstance(output_value, str) and output_value.strip().startswith("["):
                    # Output is a JSON string that looks like an array
                    try:
                        parsed_output = json.loads(output_value)
                        if isinstance(parsed_output, list) and len(parsed_output) > 0:
                            if isinstance(parsed_output[0], dict) and ("instruction" in parsed_output[0] or "output" in parsed_output[0]):
                                nested_samples = parsed_output
                                print(f"Detected nested samples JSON string in 'output' field ({len(nested_samples)} items)")
                    except json.JSONDecodeError:
                        pass
                
                if nested_samples:
                    # Extract samples from the nested array
                    for item in nested_samples:
                        if isinstance(item, dict):
                            instruction = (
                                item.get("instruction") or 
                                item.get("question") or 
                                item.get("challenge") or 
                                item.get("problem") or 
                                item.get("prompt") or
                                item.get("task") or
                                ""
                            )
                            input_text = (
                                item.get("input") or 
                                item.get("context") or 
                                item.get("constraints") or
                                ""
                            )
                            output = (
                                item.get("output") or 
                                item.get("answer") or 
                                item.get("solution") or 
                                item.get("code") or
                                item.get("response") or
                                ""
                            )
                            
                            if isinstance(output, dict):
                                output = json.dumps(output, indent=2)
                            
                            instruction = str(instruction)
                            input_text = str(input_text)
                            output = str(output)
                            
                            if is_valid_sample(instruction, input_text, output):
                                samples.append(GeneratedSample(
                                    instruction=instruction,
                                    input=input_text,
                                    output=output,
                                    metadata={"format": "json_nested_extracted"}
                                ))
                            else:
                                print(f"Warning: Skipping empty nested sample: {item}")
                else:
                    # Single object returned - handle various key naming conventions
                    instruction = (
                        data.get("instruction") or 
                        data.get("question") or 
                        data.get("challenge") or 
                        data.get("problem") or 
                        data.get("prompt") or
                        data.get("task") or
                        ""
                    )
                    
                    input_text = (
                        data.get("input") or 
                        data.get("context") or 
                        data.get("constraints") or
                        ""
                    )
                    
                    output = (
                        data.get("output") or 
                        data.get("answer") or 
                        data.get("solution") or 
                        data.get("code") or
                        data.get("response") or
                        ""
                    )
                    
                    if isinstance(output, dict):
                        output = json.dumps(output, indent=2)
                    
                    instruction = str(instruction)
                    input_text = str(input_text)
                    output = str(output)
                    
                    if is_valid_sample(instruction, input_text, output):
                        samples.append(GeneratedSample(
                            instruction=instruction,
                            input=input_text,
                            output=output,
                            metadata={"format": "json_single"}
                        ))
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content preview: {clean_content[:500]}...")
            
            # Try to find JSON array in the response using regex
            import re
            array_match = re.search(r'\[[\s\S]*?\](?=\s*$|\s*[^\[\{])', clean_content)
            if array_match:
                try:
                    data = json.loads(array_match.group())
                    for item in data:
                        if isinstance(item, dict):
                            output = item.get("output", "")
                            if isinstance(output, dict):
                                output = json.dumps(output, indent=2)
                            
                            instruction = str(item.get("instruction", ""))
                            input_text = str(item.get("input", ""))
                            output = str(output)
                            
                            if is_valid_sample(instruction, input_text, output):
                                samples.append(GeneratedSample(
                                    instruction=instruction,
                                    input=input_text,
                                    output=output,
                                    metadata={"format": "json_extracted"}
                                ))
                except Exception as ex:
                    print(f"Regex extraction failed: {ex}")
            
            # If still no samples, try to find individual JSON objects more aggressively
            if not samples:
                # Try to extract objects by finding balanced braces
                def find_json_objects(text):
                    """Find JSON objects with balanced braces."""
                    objects = []
                    i = 0
                    while i < len(text):
                        if text[i] == '{':
                            depth = 1
                            start = i
                            i += 1
                            in_string = False
                            escape_next = False
                            while i < len(text) and depth > 0:
                                c = text[i]
                                if escape_next:
                                    escape_next = False
                                elif c == '\\':
                                    escape_next = True
                                elif c == '"':
                                    in_string = not in_string
                                elif not in_string:
                                    if c == '{':
                                        depth += 1
                                    elif c == '}':
                                        depth -= 1
                                i += 1
                            if depth == 0:
                                obj_str = text[start:i]
                                if '"instruction"' in obj_str:
                                    objects.append(obj_str)
                        else:
                            i += 1
                    return objects
                
                obj_strings = find_json_objects(fix_json_newlines(content))
                for obj_str in obj_strings:
                    try:
                        item = json.loads(obj_str)
                        instruction = str(item.get("instruction", ""))
                        input_text = str(item.get("input", ""))
                        output = str(item.get("output", ""))
                        
                        if is_valid_sample(instruction, input_text, output):
                            samples.append(GeneratedSample(
                                instruction=instruction,
                                input=input_text,
                                output=output,
                                metadata={"format": "json_individual"}
                            ))
                    except:
                        pass
            
            # If still no samples, create one from the raw content
            if not samples and content.strip():
                samples.append(GeneratedSample(
                    instruction="Generated content",
                    input="",
                    output=content,
                    metadata={"format": "raw_text"}
                ))
        
        print(f"Parsed {len(samples)} valid samples from response (expected {expected_count})")
        return samples
    
    async def _generate_single_sample(
        self,
        thread_id: str,
        mode: str,
        prompt: str,
        llm_provider: str,
        model_name: str,
        **kwargs
    ) -> GeneratedSample:
        """Fallback: Generate a single sample (used if batch fails)."""
        if mode == "rag" or mode == "knowledge_injection":
            return await self.generate_rag_sample(
                thread_id=thread_id,
                prompt=prompt,
                document_ids=kwargs.get('document_ids', []),
                llm_provider=llm_provider,
                model_name=model_name
            )
        elif mode == "code" or mode == "code_specialist":
            return await self.generate_code_sample(
                thread_id=thread_id,
                prompt=prompt,
                code_language=kwargs.get('code_language', 'python'),
                llm_provider=llm_provider,
                model_name=model_name
            )
        elif mode == "agent" or mode == "tool_use":
            return await self.generate_tool_use_sample(
                thread_id=thread_id,
                prompt=prompt,
                tools=kwargs.get('tools', []),
                llm_provider=llm_provider,
                model_name=model_name
            )
        elif mode == "reasoning" or mode == "chain_of_thought":
            return await self.generate_reasoning_sample(
                thread_id=thread_id,
                prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name
            )
        else:
            return await self.generate_chat_sample(
                thread_id=thread_id,
                prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name,
                use_memory=True
            )
    
    # =========================================================================
    # Memory Methods
    # =========================================================================

    async def get_all_memories(self, assistant_id: str) -> Dict[str, Any]:
        """Fetch all memories for an assistant via GET /assistants/{assistant_id}/memories.
        Returns {"memories": [...], "total_count": int}."""
        import httpx
        url = f"https://app.backboard.io/api/assistants/{assistant_id}/memories"
        headers = {"X-API-Key": self.api_key}
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    async def test_connection(self) -> bool:
        """Test the Backboard API connection."""
        try:
            assistant = await self.client.create_assistant(
                name="Connection Test",
                description="Temporary assistant for connection testing"
            )
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Synchronous wrapper for Streamlit compatibility
def run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
