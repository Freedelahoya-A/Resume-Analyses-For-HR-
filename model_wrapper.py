import json
import urllib.request
import urllib.error
from typing import Iterator, List, Optional
from . import config


class LLM:
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.host = (host or getattr(config, "OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or getattr(config, "MODEL_NAME", "qwen2.5:1.5b-instruct-q4_K_M")
        self.timeout = timeout or getattr(config, "REQUEST_TIMEOUT", 120)

    def _build_prompts(
        self,
        prompt: str = "",
        context_chunks: Optional[List[str]] = None,
        query: str = "",
    ) -> tuple[str, str]:
        effective_query = query.strip() or prompt.strip()
        context_text = "\n\n".join(context_chunks or [])

        system_prompt = (
            "You are an expert recruitment assistant analyzing candidate resumes.\n"
            "Answer the user's question accurately, directly, and concisely using ONLY the provided resume context.\n"
            "- Always attribute facts, skills, or experience to the specific candidate name.\n"
            "- If multiple candidates match, compare or list them clearly.\n"
            "- If the question cannot be answered from the provided resume text, say: "
            "\"This information is not mentioned in the provided resumes.\"\n"
            "- Do not make up facts or extrapolate beyond what is stated."
        )

        user_prompt = (
            f"Resume Excerpts:\n"
            f"--------------------\n"
            f"{context_text}\n"
            f"--------------------\n\n"
            f"Question: {effective_query}\n\n"
            f"Answer:"
        )
        return system_prompt, user_prompt

    def generate_stream(
        self,
        prompt: str = "",
        context_chunks: Optional[List[str]] = None,
        query: str = "",
    ) -> Iterator[str]:
        """Stream response tokens in real-time from local Ollama."""
        effective_query = query.strip() or prompt.strip()
        if not effective_query:
            yield "Please provide a question."
            return

        if not context_chunks:
            yield "No relevant context found in the indexed resumes."
            return

        system_prompt, user_prompt = self._build_prompts(prompt, context_chunks, query)

        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {
                "temperature": 0.2,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                yield (
                    f"\n[Ollama Error 404]: Model '{self.model}' not found.\n"
                    f"Please run in your terminal: ollama pull {self.model}\n"
                )
            else:
                yield f"\n[Ollama HTTP Error {e.code}]: {e.reason}\n"
        except urllib.error.URLError as e:
            yield (
                f"\n[Ollama Connection Error]: Unable to connect to Ollama at {self.host}.\n"
                f"Make sure Ollama is running ('ollama serve') in your terminal.\n"
                f"Error details: {e.reason}\n"
            )
        except TimeoutError:
            yield f"\n[Ollama Timeout]: Model '{self.model}' timed out after {self.timeout}s.\n"
        except Exception as e:
            yield f"\n[Ollama Error]: {str(e)}\n"

    def generate(
        self,
        prompt: str = "",
        context_chunks: Optional[List[str]] = None,
        query: str = "",
    ) -> str:
        """Non-streaming generation returning the full string."""
        return "".join(self.generate_stream(prompt, context_chunks, query))
