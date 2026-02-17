"""
Data Generator - All 5 Generation Modes
Generates synthetic training data using Backboard's full feature set.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .backboard_manager import BackboardManager, GeneratedSample, run_async


class DataGenerator:
    """
    Orchestrates synthetic data generation for fine-tuning across 5 modes:
    
    Mode 1: General Chat (Memory-Driven)
    Mode 2: Knowledge Injection (RAG with Documents)
    Mode 3: Code Specialist (Model Routing to Qwen)
    Mode 4: Agent / Tool Use (Function Calling)
    Mode 5: Reasoning (Chain of Thought)
    """
    
    # Default prompts for Mode 1
    MODE1_PROMPTS = {
        "general_qa": """Generate a unique and creative question-answer pair for training a helpful AI assistant.

Topic focus: {topic}

Requirements:
- The question should be specific and interesting
- The answer should be detailed, helpful, and accurate
- Do NOT repeat any previous questions or answers from our conversation
- Be creative and vary the complexity

Output as JSON:
{{"instruction": "the question", "input": "", "output": "the detailed answer"}}""",

        "conversation": """Generate a unique multi-turn conversation snippet between a human and an AI assistant.

Topic: {topic}

Requirements:
- 2-4 turns of natural dialogue
- The AI should be helpful, accurate, and friendly
- Include variety in question types (how-to, explanation, opinion, etc.)
- Do NOT repeat patterns from previous conversations

Output as JSON:
{{"conversations": [
    {{"from": "human", "value": "user message"}},
    {{"from": "gpt", "value": "assistant response"}},
    ...
]}}""",

        "instruction_following": """Generate a unique instruction-following example for AI training.

Domain: {topic}

Requirements:
- Create a clear instruction/task
- Optionally include input context if the task requires it
- Provide a high-quality response demonstrating the task completion
- Avoid repeating any previous examples

Output as JSON:
{{"instruction": "what the user wants done", "input": "optional context or data", "output": "the completed task result"}}"""
    }
    
    # Mode 2: RAG prompts
    MODE2_PROMPTS = {
        "qa_from_docs": """Based on the attached documents, generate a question that can be answered using information from the documents, and provide the answer.

Topic: {topic}

Output as JSON:
{{"instruction": "question about the documents", "input": "relevant context from docs", "output": "answer grounded in the documents"}}""",
        
        "summarization": """Create a summarization task based on the attached documents.

Output as JSON:
{{"instruction": "Summarize the key points about [topic]", "input": "excerpt from documents", "output": "comprehensive summary"}}"""
    }
    
    # Mode 3: Code prompts
    MODE3_PROMPTS = {
        "algorithm": """Generate a {code_language} coding challenge focused on algorithms or data structures.

Topic: {topic}

Output as JSON:
{{"instruction": "coding problem description", "input": "example input/constraints", "output": "complete working solution with comments and explanation"}}""",
        
        "debugging": """Create a debugging exercise in {code_language}.

Topic: {topic}

Output as JSON:
{{"instruction": "fix the bug in this code", "input": "buggy code snippet", "output": "corrected code with explanation of the fix"}}""",
        
        "implementation": """Generate an implementation task in {code_language}.

Topic: {topic}

Output as JSON:
{{"instruction": "implement this feature/function", "input": "requirements and specifications", "output": "full implementation with docstrings and tests"}}"""
    }
    
    # Mode 4: Agent/Tool use prompts
    MODE4_PROMPTS = {
        "tool_selection": """Generate a scenario where an AI assistant needs to use the provided tools to help the user.

Scenario: {topic}

The assistant should:
1. Understand the user's request
2. Select appropriate tool(s) to use
3. Make tool calls with correct parameters
4. Synthesize the tool results into a helpful response

Output as JSON showing the reasoning and tool usage.""",
        
        "multi_step": """Create a complex task requiring multiple tool calls in sequence.

Task: {topic}

The assistant should demonstrate:
- Breaking down complex tasks
- Chaining tool calls
- Handling tool outputs
- Providing final synthesis"""
    }
    
    # Mode 5: Reasoning prompts
    MODE5_PROMPTS = {
        "logical_reasoning": """Generate a logical reasoning problem that requires step-by-step thinking.

Topic: {topic}

Output as JSON:
{{"instruction": "the reasoning problem", "input": "given information", "output": "<think>detailed step-by-step reasoning</think>\\n\\nFinal answer with explanation"}}""",
        
        "math_reasoning": """Create a math word problem requiring multi-step reasoning.

Topic: {topic}

Output as JSON with reasoning traces in <think> tags.""",
        
        "analysis": """Generate an analysis task requiring careful reasoning.

Topic: {topic}

Output as JSON with detailed reasoning in <think> tags before the final answer."""
    }
    
    DEFAULT_PROMPTS = MODE1_PROMPTS  # Backward compatibility
    
    def __init__(self, manager: Optional[BackboardManager] = None):
        """
        Initialize the data generator.
        
        Args:
            manager: BackboardManager instance. Creates new one if not provided.
        """
        self.manager = manager or BackboardManager()
        self.generated_samples: List[GeneratedSample] = []
        self.last_thread_id: Optional[str] = None
        self.last_assistant_id: Optional[str] = None
    
    def generate_mode1_samples(
        self,
        num_samples: int,
        topic: str = "general knowledge",
        style: str = "general_qa",
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        custom_prompt: Optional[str] = None,
        existing_thread_id: Optional[str] = None,
        existing_assistant_id: Optional[str] = None,
        web_search_context: Optional[str] = None
    ) -> List[GeneratedSample]:
        """
        Generate samples using Mode 1: General Chat with Memory.
        
        Args:
            num_samples: Number of samples to generate
            topic: Topic to focus generation on
            style: Generation style (general_qa, conversation, instruction_following)
            llm_provider: LLM provider (openai, anthropic, etc.)
            model_name: Specific model to use
            custom_prompt: Optional custom prompt template (use {topic} placeholder)
            existing_thread_id: Thread ID for appending to existing dataset
            existing_assistant_id: Assistant ID for appending to existing dataset
            web_search_context: Optional web search results to include as context
            
        Returns:
            List of generated samples
        """
        # Get the prompt template
        prompt = custom_prompt or self.MODE1_PROMPTS.get(style, self.MODE1_PROMPTS["general_qa"])
        
        # Generate batch using async manager
        samples, thread_id, assistant_id = run_async(
            self.manager.generate_batch(
                mode="general_chat",
                num_samples=num_samples,
                base_prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name,
                topic=topic,
                existing_thread_id=existing_thread_id,
                existing_assistant_id=existing_assistant_id,
                web_search_context=web_search_context
            )
        )
        
        self.last_thread_id = thread_id
        self.last_assistant_id = assistant_id
        self.generated_samples.extend(samples)
        return samples
    
    def generate_mode2_samples(
        self,
        num_samples: int,
        document_ids: List[str],
        topic: str = "document analysis",
        style: str = "qa_from_docs",
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        custom_prompt: Optional[str] = None,
        existing_thread_id: Optional[str] = None,
        existing_assistant_id: Optional[str] = None,
        web_search_context: Optional[str] = None
    ) -> List[GeneratedSample]:
        """
        Generate samples using Mode 2: Knowledge Injection (RAG).
        
        Args:
            num_samples: Number of samples
            document_ids: List of uploaded document IDs
            topic: Topic focus
            style: Generation style
            llm_provider: LLM provider
            model_name: Model to use
            custom_prompt: Optional custom prompt
            existing_thread_id: Thread ID for appending
            existing_assistant_id: Assistant ID for appending
            web_search_context: Optional web search results to include as context
            
        Returns:
            List of generated samples grounded in documents
        """
        prompt = custom_prompt or self.MODE2_PROMPTS.get(style, self.MODE2_PROMPTS["qa_from_docs"])
        
        samples, thread_id, assistant_id = run_async(
            self.manager.generate_batch(
                mode="rag",
                num_samples=num_samples,
                base_prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name,
                topic=topic,
                document_ids=document_ids,
                existing_thread_id=existing_thread_id,
                existing_assistant_id=existing_assistant_id,
                web_search_context=web_search_context
            )
        )
        
        self.last_thread_id = thread_id
        self.last_assistant_id = assistant_id
        self.generated_samples.extend(samples)
        return samples
    
    def generate_mode3_samples(
        self,
        num_samples: int,
        topic: str = "algorithms",
        style: str = "algorithm",
        code_language: str = "python",
        llm_provider: str = "qwen",
        model_name: str = "qwen-2.5-coder-32b-instruct",
        custom_prompt: Optional[str] = None,
        existing_thread_id: Optional[str] = None,
        existing_assistant_id: Optional[str] = None,
        web_search_context: Optional[str] = None
    ) -> List[GeneratedSample]:
        """
        Generate samples using Mode 3: Code Specialist.
        
        Args:
            num_samples: Number of samples
            topic: Coding topic
            style: Generation style (algorithm, debugging, implementation)
            code_language: Programming language
            llm_provider: Provider (defaults to qwen)
            model_name: Code model
            custom_prompt: Optional custom prompt
            existing_thread_id: Thread ID for appending
            existing_assistant_id: Assistant ID for appending
            web_search_context: Optional web search results to include as context
            
        Returns:
            List of code training samples
        """
        prompt = custom_prompt or self.MODE3_PROMPTS.get(style, self.MODE3_PROMPTS["algorithm"])
        prompt = prompt.replace("{code_language}", code_language)
        
        samples, thread_id, assistant_id = run_async(
            self.manager.generate_batch(
                mode="code",
                num_samples=num_samples,
                base_prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name,
                topic=topic,
                code_language=code_language,
                existing_thread_id=existing_thread_id,
                existing_assistant_id=existing_assistant_id,
                web_search_context=web_search_context
            )
        )
        
        self.last_thread_id = thread_id
        self.last_assistant_id = assistant_id
        self.generated_samples.extend(samples)
        return samples
    
    def generate_mode4_samples(
        self,
        num_samples: int,
        tools: List[Dict[str, Any]],
        topic: str = "task automation",
        style: str = "tool_selection",
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        custom_prompt: Optional[str] = None,
        existing_thread_id: Optional[str] = None,
        existing_assistant_id: Optional[str] = None,
        web_search_context: Optional[str] = None
    ) -> List[GeneratedSample]:
        """
        Generate samples using Mode 4: Agent / Tool Use.
        
        Args:
            num_samples: Number of samples
            tools: Tool definitions (OpenAI function format)
            topic: Scenario topic
            style: Generation style
            llm_provider: Provider
            model_name: Model
            custom_prompt: Optional custom prompt
            existing_thread_id: Thread ID for appending
            existing_assistant_id: Assistant ID for appending
            web_search_context: Optional web search results to include as context
            
        Returns:
            List of tool-use training samples
        """
        prompt = custom_prompt or self.MODE4_PROMPTS.get(style, self.MODE4_PROMPTS["tool_selection"])
        
        samples, thread_id, assistant_id = run_async(
            self.manager.generate_batch(
                mode="agent",
                num_samples=num_samples,
                base_prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name,
                topic=topic,
                tools=tools,
                existing_thread_id=existing_thread_id,
                existing_assistant_id=existing_assistant_id,
                web_search_context=web_search_context
            )
        )
        
        self.last_thread_id = thread_id
        self.last_assistant_id = assistant_id
        self.generated_samples.extend(samples)
        return samples
    
    def generate_mode5_samples(
        self,
        num_samples: int,
        topic: str = "logic puzzles",
        style: str = "logical_reasoning",
        llm_provider: str = "deepseek",
        model_name: str = "deepseek-r1",
        custom_prompt: Optional[str] = None,
        existing_thread_id: Optional[str] = None,
        existing_assistant_id: Optional[str] = None,
        web_search_context: Optional[str] = None
    ) -> List[GeneratedSample]:
        """
        Generate samples using Mode 5: Reasoning (Chain of Thought).
        
        Args:
            num_samples: Number of samples
            topic: Reasoning topic
            style: Generation style (logical_reasoning, math_reasoning, analysis)
            llm_provider: Provider (defaults to deepseek)
            model_name: Reasoning model
            custom_prompt: Optional custom prompt
            existing_thread_id: Thread ID for appending
            existing_assistant_id: Assistant ID for appending
            web_search_context: Optional web search results to include as context
            
        Returns:
            List of reasoning training samples with <think> tags
        """
        prompt = custom_prompt or self.MODE5_PROMPTS.get(style, self.MODE5_PROMPTS["logical_reasoning"])
        
        samples, thread_id, assistant_id = run_async(
            self.manager.generate_batch(
                mode="reasoning",
                num_samples=num_samples,
                base_prompt=prompt,
                llm_provider=llm_provider,
                model_name=model_name,
                topic=topic,
                existing_thread_id=existing_thread_id,
                existing_assistant_id=existing_assistant_id,
                web_search_context=web_search_context
            )
        )
        
        self.last_thread_id = thread_id
        self.last_assistant_id = assistant_id
        self.generated_samples.extend(samples)
        return samples
    
    def _is_valid_sample(self, sample: GeneratedSample) -> bool:
        """Check if a sample has meaningful content (not empty)."""
        has_instruction = bool(sample.instruction and sample.instruction.strip())
        has_output = bool(sample.output and sample.output.strip())
        return has_instruction or has_output
    
    def export_to_jsonl(
        self,
        samples: Optional[List[GeneratedSample]] = None,
        output_path: Optional[str] = None,
        format_type: str = "alpaca"
    ) -> str:
        """
        Export generated samples to JSONL file for Unsloth training.
        
        Args:
            samples: Samples to export. Uses all generated samples if not provided.
            output_path: Output file path. Auto-generates if not provided.
            format_type: Output format - "alpaca" or "sharegpt"
            
        Returns:
            Path to the exported file
        """
        samples = samples or self.generated_samples
        
        # Filter out empty samples
        valid_samples = [s for s in samples if self._is_valid_sample(s)]
        
        if not valid_samples:
            raise ValueError("No valid samples to export. All samples were empty or invalid.")
        
        if len(valid_samples) < len(samples):
            print(f"Warning: Filtered out {len(samples) - len(valid_samples)} empty samples")
        
        # Auto-generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(__file__).parent.parent / "data"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"mode1_{format_type}_{timestamp}.jsonl")
        
        # Export valid samples only
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in valid_samples:
                if format_type == "alpaca":
                    data = sample.to_alpaca()
                else:
                    data = sample.to_sharegpt()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"Exported {len(valid_samples)} valid samples to {output_path}")
        return output_path
    
    def get_statistics(self, samples: Optional[List[GeneratedSample]] = None) -> Dict[str, Any]:
        """Get statistics about generated samples."""
        samples = samples or self.generated_samples
        
        if not samples:
            return {"total": 0}
        
        # Calculate stats
        total_chars = sum(
            len(s.instruction) + len(s.input) + len(s.output) 
            for s in samples
        )
        avg_output_len = sum(len(s.output) for s in samples) / len(samples)
        
        return {
            "total": len(samples),
            "total_characters": total_chars,
            "avg_output_length": round(avg_output_len, 2),
            "formats": {
                "json": sum(1 for s in samples if s.metadata and s.metadata.get("format") == "json"),
                "text": sum(1 for s in samples if s.metadata and s.metadata.get("format") == "text")
            }
        }
    
    def clear_samples(self):
        """Clear all generated samples."""
        self.generated_samples = []
