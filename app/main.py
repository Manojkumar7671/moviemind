import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.agent import MovieMindAgent
from src.baselines import ALL_GENRES
from src.coldstart import ColdStartRecommender

app = FastAPI(title="MovieMind", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
agent = MovieMindAgent()
coldstart = ColdStartRecommender()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "demo_user"

class OnboardingRequest(BaseModel):
    user_id: Optional[str] = "demo_user"
    genres: List[str]

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = "demo_user"
    movie_id: int
    feedback_type: str

@app.get("/api/health")
def healthcheck():
    return {"status": "ok", "app": "MovieMind", "version": "1.0.0"}

@app.get("/api/genres")
def get_genres():
    return {"genres": ALL_GENRES}

@app.post("/api/onboarding")
def onboarding(req: OnboardingRequest):
    uid = req.user_id or "demo_user"
    if not req.genres:
        raise HTTPException(status_code=400, detail="Must select at least one genre")
    recs = coldstart.recommend(req.genres, top_n=6)
    summary = f"User onboarded with favorite genres: {', '.join(req.genres)}."
    agent.memory.set_taste_summary(uid, summary, favorite_genres=",".join(req.genres))
    return {
        "status": "success",
        "user_id": uid,
        "selected_genres": req.genres,
        "recommendations": recs,
    }

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    uid = req.user_id or "demo_user"
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    reply = agent.chat(req.message, user_id=uid)
    return {"response": reply, "user_id": uid}

@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest):
    res = agent.record_feedback(req.user_id or "demo_user", req.movie_id, req.feedback_type)
    return res

@app.get("/api/ab")
def get_ab_metrics():
    ab_file = ARTIFACTS_DIR / "ab_results.json"
    if ab_file.exists():
        with open(ab_file) as f:
            return json.load(f)
    from src.ab_sim import run_ab_simulation
    return run_ab_simulation()

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MovieMind - AI Movie Recommender</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
      --bot-msg: #334155;
      --user-msg: #4338ca;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .logo { font-size: 20px; font-weight: 700; color: #a5b4fc; display: flex; align-items: center; gap: 8px; }
    .nav-links a {
      color: var(--muted);
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
      padding: 6px 14px;
      border-radius: 6px;
      background: #0f172a;
      border: 1px solid var(--border);
      transition: all 0.2s;
    }
    .nav-links a:hover { color: #fff; border-color: var(--accent); }
    .main-container {
      display: flex;
      flex: 1;
      overflow: hidden;
      max-width: 1200px;
      margin: 0 auto;
      width: 100%;
      padding: 16px;
      gap: 16px;
    }
    .sidebar {
      width: 320px;
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
    }
    .chat-panel {
      flex: 1;
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .genre-title { font-size: 15px; font-weight: 600; color: #e2e8f0; }
    .genre-grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .genre-pill {
      font-size: 12px;
      padding: 6px 12px;
      border-radius: 16px;
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
    }
    .genre-pill.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
      transition: background 0.2s;
    }
    .btn:hover { background: var(--accent-hover); }
    .quick-chips {
      display: flex;
      gap: 6px;
      padding: 10px 16px;
      overflow-x: auto;
      border-bottom: 1px solid var(--border);
      background: #172033;
    }
    .chip {
      white-space: nowrap;
      font-size: 12px;
      padding: 4px 10px;
      background: #1e293b;
      border: 1px solid var(--border);
      border-radius: 12px;
      color: #cbd5e1;
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: #fff; }
    .messages-container {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: 80%;
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.5;
      word-break: break-word;
    }
    .msg.bot {
      background: var(--bot-msg);
      align-self: flex-start;
      border-bottom-left-radius: 2px;
      border: 1px solid #475569;
    }
    .msg.user {
      background: var(--user-msg);
      align-self: flex-end;
      border-bottom-right-radius: 2px;
    }
    .input-row {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      background: #172033;
      border-top: 1px solid var(--border);
    }
    .input-box {
      flex: 1;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: #fff;
      font-size: 14px;
      outline: none;
    }
    .input-box:focus { border-color: var(--accent); }
    .rec-item {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 12px;
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .rec-info { font-size: 13px; }
    .rec-meta { font-size: 11px; color: var(--muted); }
    .fb-btns { display: flex; gap: 4px; }
    .fb-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      cursor: pointer;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
    }
    .fb-btn:hover { color: #fff; border-color: var(--accent); }
    @media (max-width: 768px) {
      .main-container { flex-direction: column; }
      .sidebar { width: 100%; max-height: 180px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="logo">🎬 MovieMind</div>
    <div class="nav-links">
      <a href="/ab">📊 A/B Simulation Dashboard</a>
    </div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <div>
        <div class="genre-title">⚡ Quick Onboarding</div>
        <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">Pick 3-5 favorite genres:</div>
        <div class="genre-grid" id="genreGrid"></div>
      </div>
      <button class="btn" onclick="submitOnboarding()">Apply Genre Profile</button>
      <div id="onboardingResults" style="display:none; flex-direction: column; gap: 6px;">
        <div class="genre-title" style="font-size: 13px;">Cold-Start Picks:</div>
        <div id="coldRecsList"></div>
      </div>
    </div>

    <div class="chat-panel">
      <div class="quick-chips">
        <span class="chip" onclick="sendQuick('Recommend movies')">✨ Recommend</span>
        <span class="chip" onclick="sendQuick('Search for Matrix')">🔍 Search Matrix</span>
        <span class="chip" onclick="sendQuick('Why should I watch movie ID 2571?')">💡 Explain 2571</span>
        <span class="chip" onclick="sendQuick('Show Sci-Fi movies from 1999')">📅 Sci-Fi 1999</span>
        <span class="chip" onclick="sendQuick('What is my current taste profile?')">👤 Taste Profile</span>
      </div>

      <div class="messages-container" id="messagesContainer">
        <div class="msg bot">
          👋 Welcome to <strong>MovieMind</strong>! I am your AI movie assistant powered by a Two-Tower neural retriever, Wide&Deep ranker, and MMR diversity reranker.<br><br>
          You can pick your favorite genres on the left, ask for recommendations, search films, or get factual explanations!
        </div>
      </div>

      <div class="input-row">
        <input type="text" id="userInput" class="input-box" placeholder="Ask for a movie, genre, or explanation..." onkeydown="if(event.key==='Enter') sendMessage()" />
        <button class="btn" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </div>

  <script>
    const genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"];
    const selectedGenres = new Set(["Action", "Sci-Fi", "Thriller"]);
    const currentUserId = "demo_user";

    function renderGenres() {
      const grid = document.getElementById("genreGrid");
      grid.innerHTML = "";
      genres.forEach(g => {
        const pill = document.createElement("span");
        pill.className = "genre-pill" + (selectedGenres.has(g) ? " active" : "");
        pill.innerText = g;
        pill.onclick = () => {
          if (selectedGenres.has(g)) selectedGenres.delete(g);
          else selectedGenres.add(g);
          renderGenres();
        };
        grid.appendChild(pill);
      });
    }

    async function submitOnboarding() {
      if (selectedGenres.size === 0) return alert("Please select at least 1 genre.");
      const res = await fetch("/api/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: currentUserId, genres: Array.from(selectedGenres) })
      });
      const data = await res.json();
      const container = document.getElementById("onboardingResults");
      const list = document.getElementById("coldRecsList");
      list.innerHTML = "";
      data.recommendations.slice(0, 4).forEach(r => {
        const item = document.createElement("div");
        item.className = "rec-item";
        item.innerHTML = `
          <div class="rec-info">
            <strong>${r.title}</strong>
            <div class="rec-meta">${r.genres} (${r.year})</div>
          </div>
          <div class="fb-btns">
            <button class="fb-btn" onclick="giveFeedback(${r.movie_id}, 'like')">👍</button>
            <button class="fb-btn" onclick="giveFeedback(${r.movie_id}, 'dislike')">👎</button>
          </div>
        `;
        list.appendChild(item);
      });
      container.style.display = "flex";
      appendMsg("bot", `✅ Saved your genre preferences: <em>${Array.from(selectedGenres).join(", ")}</em>. Added cold-start picks to your profile!`);
    }

    async function giveFeedback(mid, type) {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: currentUserId, movie_id: mid, feedback_type: type })
      });
      appendMsg("bot", `Recorded <strong>${type}</strong> for movie ID ${mid}. Your taste profile is updated!`);
    }

    function appendMsg(sender, htmlContent) {
      const container = document.getElementById("messagesContainer");
      const div = document.createElement("div");
      div.className = "msg " + sender;
      div.innerHTML = htmlContent;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    }

    async function sendMessage() {
      const input = document.getElementById("userInput");
      const text = input.value.trim();
      if (!text) return;
      appendMsg("user", text);
      input.value = "";

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, user_id: currentUserId })
        });
        const data = await res.json();
        // Convert simple markdown bullets and bold
        let formatted = data.response
          .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
          .replace(/^-\\s+(.*)$/gm, '<li>$1</li>')
          .replace(/^\\d+\\.\\s+(.*)$/gm, '<li>$1</li>')
          .replace(/\\n/g, '<br>');
        appendMsg("bot", formatted);
      } catch (err) {
        appendMsg("bot", "⚠️ Error contacting assistant: " + err);
      }
    }

    function sendQuick(txt) {
      document.getElementById("userInput").value = txt;
      sendMessage();
    }

    renderGenres();
  </script>
</body>
</html>
"""

@app.get("/ab", response_class=HTMLResponse)
def ab_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MovieMind - A/B Testing Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --accent: #6366f1;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
    }
    .container { max-width: 1000px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .back-btn {
      color: var(--muted);
      text-decoration: none;
      font-size: 14px;
      padding: 6px 14px;
      border-radius: 6px;
      background: var(--card);
      border: 1px solid var(--border);
    }
    .back-btn:hover { color: #fff; border-color: var(--accent); }
    .badge {
      background: rgba(16, 185, 129, 0.2);
      color: var(--success);
      border: 1px solid var(--success);
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 13px;
      font-weight: 600;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .card-title { font-size: 16px; font-weight: 600; color: #cbd5e1; margin-bottom: 12px; }
    .metric-value { font-size: 32px; font-weight: 700; color: #fff; }
    .subtext { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .lift-banner {
      background: linear-gradient(135deg, #312e81, #1e1b4b);
      border: 1px solid var(--accent);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .lift-val { font-size: 40px; font-weight: 800; color: #a5b4fc; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }
    th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1 style="font-size: 24px;">🔬 Recommender A/B Test Results</h1>
        <p class="subtext">Simulated user interactions against held-out test data (N=5,000 sessions)</p>
      </div>
      <a href="/" class="back-btn">← Back to MovieMind Chat</a>
    </header>

    <div class="lift-banner">
      <div>
        <div style="font-size: 14px; color: #c7d2fe; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Relative CTR Lift</div>
        <div class="lift-val" id="liftVal">+11.18%</div>
        <div class="subtext" style="color: #cbd5e1;" id="ciVal">95% Confidence Interval: [+0.0045, +0.0145]</div>
      </div>
      <div>
        <span class="badge" id="statBadge">✓ Statistically Significant (p = 0.0002)</span>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Group A: Control (Popularity)</div>
        <div class="metric-value" id="ctrA">8.48%</div>
        <div class="subtext" id="subA">2,120 clicks / 25,000 impressions</div>
        <div style="margin-top: 12px; font-size: 13px; color: var(--muted);">Session-level CTR: <strong style="color: #fff;" id="sCtrA">60.48%</strong></div>
      </div>
      <div class="card" style="border-color: var(--accent);">
        <div class="card-title">Group B: Treatment (Two-Stage + MMR)</div>
        <div class="metric-value" style="color: #a5b4fc;" id="ctrB">9.42%</div>
        <div class="subtext" id="subB">2,355 clicks / 25,000 impressions</div>
        <div style="margin-top: 12px; font-size: 13px; color: var(--muted);">Session-level CTR: <strong style="color: #fff;" id="sCtrB">62.88%</strong></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Statistical Test Breakdown</div>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Control (Popularity)</th>
            <th>Treatment (MovieMind)</th>
            <th>Difference / Lift</th>
          </tr>
        </thead>
        <tbody id="tableBody">
          <tr>
            <td>Item-Level CTR</td>
            <td id="tCtrA">8.48%</td>
            <td id="tCtrB">9.42%</td>
            <td style="color: var(--success); font-weight: 600;" id="tLift">+11.18%</td>
          </tr>
          <tr>
            <td>Session-Level CTR</td>
            <td id="tSCtrA">60.48%</td>
            <td id="tSCtrB">62.88%</td>
            <td style="color: var(--success); font-weight: 600;" id="tSLift">+3.97%</td>
          </tr>
          <tr>
            <td>p-value</td>
            <td colspan="2">Two-Proportion Z-Test</td>
            <td id="tPval" style="color: #a5b4fc;">2.05e-04 (p &lt; 0.001)</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    async function loadMetrics() {
      try {
        const res = await fetch("/api/ab");
        const d = await res.json();
        document.getElementById("liftVal").innerText = (d.relative_ctr_lift_pct > 0 ? "+" : "") + d.relative_ctr_lift_pct + "%";
        document.getElementById("ciVal").innerText = `95% Confidence Interval: [${d.ci_95[0]}, ${d.ci_95[1]}]`;
        document.getElementById("statBadge").innerText = d.is_significant ? `✓ Statistically Significant (p = ${d.p_value})` : `p = ${d.p_value}`;
        document.getElementById("ctrA").innerText = (d.control.ctr * 100).toFixed(2) + "%";
        document.getElementById("subA").innerText = `${d.control.clicks.toLocaleString()} clicks / ${d.control.impressions.toLocaleString()} impressions`;
        document.getElementById("sCtrA").innerText = (d.control.session_ctr * 100).toFixed(2) + "%";
        document.getElementById("ctrB").innerText = (d.treatment.ctr * 100).toFixed(2) + "%";
        document.getElementById("subB").innerText = `${d.treatment.clicks.toLocaleString()} clicks / ${d.treatment.impressions.toLocaleString()} impressions`;
        document.getElementById("sCtrB").innerText = (d.treatment.session_ctr * 100).toFixed(2) + "%";
        document.getElementById("tCtrA").innerText = (d.control.ctr * 100).toFixed(2) + "%";
        document.getElementById("tCtrB").innerText = (d.treatment.ctr * 100).toFixed(2) + "%";
        document.getElementById("tLift").innerText = "+" + d.relative_ctr_lift_pct + "%";
        document.getElementById("tSCtrA").innerText = (d.control.session_ctr * 100).toFixed(2) + "%";
        document.getElementById("tSCtrB").innerText = (d.treatment.session_ctr * 100).toFixed(2) + "%";
        document.getElementById("tSLift").innerText = "+" + d.session_ctr_lift_pct + "%";
        document.getElementById("tPval").innerText = d.p_value;
      } catch (e) {
        console.error(e);
      }
    }
    loadMetrics();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
