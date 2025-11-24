import os
import google.generativeai as genai
from ddgs import DDGS
from fastapi import FastAPI
from mcp.server.fastapi import MCPServer

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise RuntimeError('Set GEMINI_API_KEY')

genai.configure(api_key=api_key)
MODEL = 'models/gemini-2.5-flash'

def call_llm(prompt: str) -> str:
    return genai.GenerativeModel(MODEL).generate_content(prompt).text

def web_search(query: str, k: int = 3) -> str:
    hits = DDGS().text(query, max_results=k)
    return '
'.join(f"- {h.get('title')}: {h.get('href')}" for h in hits)

app = FastAPI()
mcp = MCPServer(app, 'classroom-tools')

@mcp.tool()
def search(query: str) -> str:
    '''Search the web and return top results.'''
    return web_search(query)

@mcp.tool()
def answer(question: str) -> str:
    '''Answer concisely with citations using web context.'''
    obs = web_search(question)
    prompt = f"You are a teaching assistant. Use the observation to answer in 2-3 bullets with URLs.
Question: {question}
Observation:
{obs}"
    return call_llm(prompt)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
