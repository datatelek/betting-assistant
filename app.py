import os
import sys
import base64
import httpx
from openai import OpenAI
from PIL import Image
import io
import gradio as gr
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
AGENT_MODEL = "gpt-4o"

# ─────────────────────────────────────────────
# TOOLS (function definitions for GPT)
# ─────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_team_matches",
            "description": "Search for the last 5 matches of a football team. Returns results, scorelines, match flow and performance indicators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "Name of the football team"
                    }
                },
                "required": ["team_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_team_news",
            "description": "Search for the latest news about a football team: injuries, suspensions, manager changes, press quotes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "Name of the football team"
                    }
                },
                "required": ["team_name"]
            }
        }
    }
]


# ─────────────────────────────────────────────
# WEB SEARCH HELPER (via Serper / fallback to DuckDuckGo)
# ─────────────────────────────────────────────
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web using Serper API or DuckDuckGo as fallback."""
    serper_key = os.environ.get("SERPER_API_KEY", "")
    if serper_key:
        try:
            resp = httpx.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": num_results, "gl": "ru", "hl": "ru"},
                timeout=10
            )
            resp.raise_for_status()
            results = resp.json().get("organic", [])
            return [{"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")} for r in results]
        except Exception as e:
            print(f"Serper error: {e}")

    # DuckDuckGo fallback
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10
        )
        data = resp.json()
        results = []
        for r in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in r:
                results.append({"title": r.get("Text", "")[:80], "snippet": r.get("Text", ""), "url": r.get("FirstURL", "")})
        return results
    except Exception as e:
        return [{"title": "Error", "snippet": str(e), "url": ""}]


# ─────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────
def search_team_matches(team_name: str) -> str:
    today = datetime.now().strftime("%Y")
    query = f"{team_name} last 5 matches results {today} football"
    results = web_search(query, 6)
    if not results:
        return f"No match data found for {team_name}."
    lines = [f"LAST 5 MATCHES — {team_name.upper()}\n"]
    for r in results:
        lines.append(f"• {r['title']}")
        lines.append(f"  {r['snippet']}")
        lines.append(f"  Source: {r['url']}\n")
    return "\n".join(lines)


def search_team_news(team_name: str) -> str:
    query = f"{team_name} injuries suspensions news coach 2025 football"
    results = web_search(query, 6)
    if not results:
        return f"No news found for {team_name}."
    lines = [f"LATEST NEWS — {team_name.upper()}\n"]
    for r in results:
        lines.append(f"• {r['title']}")
        lines.append(f"  {r['snippet']}")
        lines.append(f"  Source: {r['url']}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL DISPATCHER
# ─────────────────────────────────────────────
def dispatch_tool(name: str, args: dict) -> str:
    if name == "search_team_matches":
        return search_team_matches(**args)
    elif name == "search_team_news":
        return search_team_news(**args)
    return "Unknown tool."


# ─────────────────────────────────────────────
# VISION: extract team names from screenshot
# ─────────────────────────────────────────────
def extract_teams_from_image(image: Image.Image) -> str:
    """Use GPT-4o vision to extract team names from a bookmaker screenshot."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": "This is a screenshot from a bookmaker/betting site. Extract the names of the two teams playing. Return ONLY: 'Team A vs Team B', nothing else."
                    }
                ]
            }
        ],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────
def run_agent(team1: str, team2: str) -> str:
    """Run the agentic loop for two teams."""
    system_prompt = """You are an expert football betting analyst. Your job:
1. Get the last 5 matches for EACH team (use search_team_matches tool for each)
2. Get latest news for EACH team (use search_team_news tool for each)
3. Write a structured scouting report in RUSSIAN with:
   - Last 5 matches for each team: scores, opponent quality, match flow, how convincing they were
   - Confidence/form rating for each team (1-10)
   - Latest news: injuries, suspensions, manager changes, notable quotes
   - Head-to-head context if available
   - Brief betting insight (what to watch, key factors)
Be factual, concrete, and analyst-like. Use markdown formatting."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this upcoming match: {team1} vs {team2}. Collect data for both teams."}
    ]

    max_iterations = 10
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                import json
                args = json.loads(tc.function.arguments)
                result = dispatch_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            return msg.content or "No analysis generated."

    return "Max iterations reached."


# ─────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────
def analyze_text(match_text: str) -> str:
    if " vs " in match_text.lower():
        parts = match_text.lower().split(" vs ")
        team1, team2 = parts[0].strip(), parts[1].strip()
    elif " - " in match_text:
        parts = match_text.split(" - ")
        team1, team2 = parts[0].strip(), parts[1].strip()
    elif "\n" in match_text:
        parts = match_text.split("\n")
        team1, team2 = parts[0].strip(), parts[1].strip()
    else:
        return "Введите матч в формате: Команда1 vs Команда2"
    return run_agent(team1, team2)


def analyze_image(image) -> str:
    if image is None:
        return "Загрузите скриншот из букмекерской конторы."
    match_str = extract_teams_from_image(image)
    if " vs " not in match_str.lower() and " - " not in match_str:
        return f"Не удалось определить команды. GPT ответил: {match_str}"
    return f"**Определены команды:** {match_str}\n\n" + analyze_text(match_str)


def analyze_combined(match_text: str, image) -> str:
    if image is not None:
        return analyze_image(image)
    if match_text and match_text.strip():
        return analyze_text(match_text)
    return "Введите название матча или загрузите скриншот."


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
with gr.Blocks(title="AI Betting Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI-ассистент беттора
    Введите матч текстом (например `Manchester City vs Arsenal`) или загрузите скриншот из букмекерской конторы.
    Ассистент соберёт:
    - Последние 5 матчей обеих команд с ходом игр и уверенностью
    - Свежие новости: травмы, дисквалификации, смены тренеров, цитаты
    - Краткий беттинговый разбор ключевых факторов
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Матч (текст)",
                placeholder="Manchester City vs Arsenal",
                lines=2
            )
            image_input = gr.Image(
                label="Скриншот из букмекерской (опционально)",
                type="pil"
            )
            analyze_btn = gr.Button("Анализировать", variant="primary", size="lg")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Анализ матча")

    analyze_btn.click(
        fn=analyze_combined,
        inputs=[text_input, image_input],
        outputs=output
    )

    gr.Examples(
        examples=[
            ["Manchester City vs Arsenal", None],
            ["Real Madrid vs Barcelona", None],
            ["Liverpool vs Chelsea", None],
            ["Bayern Munich vs Borussia Dortmund", None],
        ],
        inputs=[text_input, image_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
