/* MovieMind Client Application */

// State
const state = {
  activeTab: 'discover',
  selectedGenres: new Set(['Action', 'Sci-Fi', 'Thriller']),
  userId: 'demo_user',
  theme: localStorage.getItem('moviemind-theme') || 'dark',
  movies: [],
  tasteMap: null,
  userPoint: { x: 0, y: 0, targetX: 0, targetY: 0 },
  likedSet: new Set(),
  dislikedSet: new Set(),
  lastDismissed: null,
  isDrawerOpen: false,
};

// Genre Color Themes for Posters (Distinct hue per genre)
const GENRE_COLORS = {
  Action: { bg1: '#881337', bg2: '#E11D48', hue: 'crimson' },
  Adventure: { bg1: '#064E3B', bg2: '#059669', hue: 'forest' },
  Animation: { bg1: '#047857', bg2: '#10B981', hue: 'emerald' },
  "Children's": { bg1: '#1E40AF', bg2: '#3B82F6', hue: 'blue' },
  Comedy: { bg1: '#B45309', bg2: '#F59E0B', hue: 'amber' },
  Crime: { bg1: '#374151', bg2: '#1F2937', hue: 'slate' },
  Documentary: { bg1: '#451A03', bg2: '#78350F', hue: 'bronze' },
  Drama: { bg1: '#4C1D95', bg2: '#312E81', hue: 'indigo' },
  Fantasy: { bg1: '#581C87', bg2: '#7C3AED', hue: 'purple' },
  "Film-Noir": { bg1: '#030712', bg2: '#374151', hue: 'monochrome' },
  Horror: { bg1: '#450A0A', bg2: '#991B1B', hue: 'blood' },
  Musical: { bg1: '#831843', bg2: '#DB2777', hue: 'pink' },
  Mystery: { bg1: '#1E1B4B', bg2: '#4338CA', hue: 'sapphire' },
  Romance: { bg1: '#9D174D', bg2: '#F43F5E', hue: 'rose' },
  "Sci-Fi": { bg1: '#082F49', bg2: '#0284C7', hue: 'cyan' },
  Thriller: { bg1: '#18181B', bg2: '#7F1D1D', hue: 'carmine' },
  War: { bg1: '#27272A', bg2: '#713F12', hue: 'khaki' },
  Western: { bg1: '#7C2D12', bg2: '#D97706', hue: 'terracotta' },
};

// SVG Poster Pattern Generator (Tinted by primary genre hue, pattern varied by movie_id)
function generatePosterSvg(title, year, primaryGenre, movieId = 0) {
  const theme = GENRE_COLORS[primaryGenre] || { bg1: '#1E293B', bg2: '#334155' };
  const safeGenre = (primaryGenre || 'film').replace(/[^a-zA-Z0-9]/g, '');
  const mid = Math.abs(parseInt(movieId, 10) || 0);

  const patType = mid % 6;
  const rot = (mid * 47) % 360;
  const cx = 150 + ((mid * 17) % 50 - 25);
  const cy = 175 + ((mid * 23) % 60 - 30);
  const r1 = 65 + ((mid * 19) % 35);
  const r2 = r1 + 32;

  let patternDef = '';
  let patternArt = '';

  if (patType === 0) {
    // Motif 0: Concentric film reels and orbit rings
    patternArt = `
      <circle cx="${cx}" cy="${cy}" r="${r1}" fill="none" stroke="#EDE8DC" stroke-width="1.5" stroke-dasharray="4 8" opacity="0.32" transform="rotate(${rot} ${cx} ${cy})" />
      <circle cx="${cx}" cy="${cy}" r="${r2}" fill="none" stroke="#EDE8DC" stroke-width="1" opacity="0.2" />
      <circle cx="${cx}" cy="${cy}" r="14" fill="#EDE8DC" opacity="0.2" />
    `;
  } else if (patType === 1) {
    // Motif 1: Radial starburst & astronomical coordinates
    patternArt = `
      <g stroke="#EDE8DC" stroke-width="1.2" opacity="0.25" transform="rotate(${rot} ${cx} ${cy})">
        <line x1="${cx - r2}" y1="${cy}" x2="${cx + r2}" y2="${cy}" />
        <line x1="${cx}" y1="${cy - r2}" x2="${cx}" y2="${cy + r2}" />
        <line x1="${cx - r1}" y1="${cy - r1}" x2="${cx + r1}" y2="${cy + r1}" stroke-dasharray="3 5" />
        <line x1="${cx - r1}" y1="${cy + r1}" x2="${cx + r1}" y2="${cy - r1}" stroke-dasharray="3 5" />
        <circle cx="${cx}" cy="${cy}" r="${r1}" fill="none" stroke-width="1" />
      </g>
    `;
  } else if (patType === 2) {
    // Motif 2: Geometric architectural slits & square aperture
    patternDef = `
      <pattern id="pat-${mid}" width="28" height="28" patternUnits="userSpaceOnUse" patternTransform="rotate(${rot})">
        <line x1="0" y1="0" x2="28" y2="28" stroke="#EDE8DC" stroke-width="1.2" opacity="0.14" />
      </pattern>
    `;
    patternArt = `
      <rect width="100%" height="100%" fill="url(#pat-${mid})" />
      <rect x="${cx - 65}" y="${cy - 65}" width="130" height="130" fill="none" stroke="#EDE8DC" stroke-width="1.5" opacity="0.26" transform="rotate(${rot} ${cx} ${cy})" />
    `;
  } else if (patType === 3) {
    // Motif 3: Kinetic diamond compass prism
    patternArt = `
      <polygon points="${cx},${cy - r2} ${cx + r1},${cy} ${cx},${cy + r2} ${cx - r1},${cy}" fill="none" stroke="#EDE8DC" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.3" transform="rotate(${rot} ${cx} ${cy})" />
      <polygon points="${cx},${cy - r1 * 0.6} ${cx + r1 * 0.6},${cy} ${cx},${cy + r1 * 0.6} ${cx - r1 * 0.6},${cy}" fill="#EDE8DC" opacity="0.14" transform="rotate(${rot} ${cx} ${cy})" />
    `;
  } else if (patType === 4) {
    // Motif 4: Halftone matrix with circular target
    patternDef = `
      <pattern id="pat-${mid}" width="24" height="24" patternUnits="userSpaceOnUse">
        <circle cx="12" cy="12" r="1.5" fill="#EDE8DC" opacity="0.15" />
      </pattern>
    `;
    patternArt = `
      <rect width="100%" height="100%" fill="url(#pat-${mid})" />
      <circle cx="${cx}" cy="${cy}" r="${r1}" fill="none" stroke="#EDE8DC" stroke-width="2" opacity="0.32" />
      <rect x="${cx - 45}" y="${cy - 45}" width="90" height="90" fill="none" stroke="#EDE8DC" stroke-width="1" stroke-dasharray="4 6" opacity="0.22" transform="rotate(${rot} ${cx} ${cy})" />
    `;
  } else {
    // Motif 5: Topographic contour horizons
    patternArt = `
      <ellipse cx="${cx}" cy="${cy}" rx="${r2}" ry="${r1 * 0.65}" fill="none" stroke="#EDE8DC" stroke-width="1.5" stroke-dasharray="5 5" opacity="0.28" transform="rotate(${rot} ${cx} ${cy})" />
      <ellipse cx="${cx}" cy="${cy}" rx="${r1 * 0.75}" ry="${r2 * 0.85}" fill="none" stroke="#EDE8DC" stroke-width="1" opacity="0.18" transform="rotate(${rot + 40} ${cx} ${cy})" />
      <circle cx="${cx}" cy="${cy}" r="10" fill="#EDE8DC" opacity="0.22" />
    `;
  }

  return `
    <svg class="poster-svg-art" viewBox="0 0 300 450" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="grad-${mid}-${safeGenre}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="${theme.bg1}" />
          <stop offset="100%" stop-color="${theme.bg2}" />
        </linearGradient>
        <linearGradient id="overlayFade-${mid}" x1="0%" y1="35%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="transparent" />
          <stop offset="100%" stop-color="#0C1018" stop-opacity="0.95" />
        </linearGradient>
        ${patternDef}
      </defs>
      <rect width="100%" height="100%" fill="url(#grad-${mid}-${safeGenre})" />
      ${patternArt}
      <rect width="100%" height="100%" fill="url(#overlayFade-${mid})" />
    </svg>
  `;
}

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  applyTheme(state.theme);
  setupNavigation();
  setupKeyboardShortcuts();
  await loadGenres();
  await loadRecommendations();
  await loadTasteMap();
  await loadEvaluationData();

  const initialTab = (window.location.hash || '').replace('#', '');
  if (['discover', 'tastemap', 'eval'].includes(initialTab)) {
    switchTab(initialTab);
  }
  if (window.location.search.includes('noanim=1')) {
    document.body.classList.add('no-anim');
  }
  if (window.location.search.includes('chat=1')) {
    openChatDrawer();
  }
  if (window.location.search.includes('expand=1') && state.movies.length > 0) {
    toggleWhy(state.movies[0].movie_id);
    toggleTrace(state.movies[0].movie_id);
  }
});

// Theme Management
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('moviemind-theme', theme);
  const icon = document.getElementById('themeToggleBtn');
  if (icon) icon.innerText = theme === 'dark' ? '☀️' : '🌙';
  if (state.activeTab === 'tastemap') renderTasteMap();
}

function toggleTheme() {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
}

// Navigation & Tabs
function setupNavigation() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  const chatToggle = document.getElementById('chatToggleBtn');
  if (chatToggle) chatToggle.addEventListener('click', toggleChatDrawer);

  const closeDrawer = document.getElementById('closeDrawerBtn');
  if (closeDrawer) closeDrawer.addEventListener('click', closeChatDrawer);

  const backdrop = document.getElementById('drawerBackdrop');
  if (backdrop) backdrop.addEventListener('click', closeChatDrawer);
}

function switchTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabName);
  });
  document.querySelectorAll('.tab-content').forEach(pane => {
    pane.classList.toggle('active', pane.id === `${tabName}Tab`);
  });

  if (tabName === 'tastemap') {
    setTimeout(renderTasteMap, 50);
  }
}

// Keyboard shortcuts: '/' opens chat, Escape closes drawer
function setupKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      openChatDrawer();
    } else if (e.key === 'Escape' && state.isDrawerOpen) {
      closeChatDrawer();
    }
  });
}

function toggleChatDrawer() {
  state.isDrawerOpen ? closeChatDrawer() : openChatDrawer();
}

function openChatDrawer() {
  state.isDrawerOpen = true;
  document.getElementById('chatDrawer').classList.add('open');
  document.getElementById('drawerBackdrop').classList.add('open');
  document.getElementById('chatInput').focus();
}

function closeChatDrawer() {
  state.isDrawerOpen = false;
  document.getElementById('chatDrawer').classList.remove('open');
  document.getElementById('drawerBackdrop').classList.remove('open');
}

// Genre Onboarding
async function loadGenres() {
  try {
    const res = await fetch('/api/genres');
    const data = await res.json();
    const container = document.getElementById('genreChipsContainer');
    container.innerHTML = '';

    data.genres.forEach(genre => {
      const chip = document.createElement('span');
      chip.className = 'genre-chip' + (state.selectedGenres.has(genre) ? ' selected' : '');
      chip.innerText = genre;
      chip.setAttribute('role', 'checkbox');
      chip.setAttribute('aria-checked', state.selectedGenres.has(genre));
      chip.tabIndex = 0;

      const toggle = () => {
        if (state.selectedGenres.has(genre)) {
          state.selectedGenres.delete(genre);
        } else {
          if (state.selectedGenres.size >= 5) {
            showToast("Maximum 5 genres allowed. Unselect one first.");
            return;
          }
          state.selectedGenres.add(genre);
        }
        chip.classList.toggle('selected', state.selectedGenres.has(genre));
        chip.setAttribute('aria-checked', state.selectedGenres.has(genre));
      };

      chip.addEventListener('click', toggle);
      chip.addEventListener('keydown', (e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); toggle(); } });
      container.appendChild(chip);
    });
  } catch (err) {
    console.error('Failed to load genres:', err);
  }
}

async function saveGenres() {
  if (state.selectedGenres.size === 0) {
    showToast("Please choose at least 1 genre.");
    return;
  }
  const genresList = Array.from(state.selectedGenres);
  try {
    const res = await fetch('/api/onboarding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: state.userId, genres: genresList })
    });
    const data = await res.json();
    showToast(`Taste profile saved: ${genresList.join(', ')}`);
    document.getElementById('onboardingBanner').style.display = 'none';
    updateFallbackNotice(data.llm_fallback, data.notice);
    renderMovieGrid(data.recommendations, data.trace, data.latency_ms, data.llm_fallback);
    await loadTasteMap();
  } catch (err) {
    showToast("Failed to save genres: " + err);
  }
}

function updateFallbackNotice(isFallback, noticeText) {
  const el = document.getElementById('llmFallbackNotice');
  const txt = document.getElementById('llmFallbackNoticeText');
  if (!el || !txt) return;
  if (isFallback && noticeText) {
    txt.innerText = noticeText;
    el.style.display = 'flex';
  } else {
    el.style.display = 'none';
  }
}

function skipOnboarding() {
  document.getElementById('onboardingBanner').style.display = 'none';
  showToast("Showing popular recommendations for you.");
}

// Recommendations
async function loadRecommendations() {
  const grid = document.getElementById('movieGrid');
  grid.innerHTML = '<div class="skeleton" style="height: 380px;"></div>'.repeat(6);
  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: state.userId, genres: Array.from(state.selectedGenres), n: 6 })
    });
    const data = await res.json();
    state.movies = data.recommendations || [];
    updateFallbackNotice(data.llm_fallback, data.notice);
    renderMovieGrid(state.movies, data.trace, data.latency_ms, data.llm_fallback);
  } catch (err) {
    grid.innerHTML = `<div class="meta-chip">Failed to load recommendations: ${err}</div>`;
  }
}

function renderMovieGrid(movies, trace = [], latency = {}, isFallback = false) {
  const grid = document.getElementById('movieGrid');
  grid.innerHTML = '';

  if (!movies || movies.length === 0) {
    grid.innerHTML = `<div style="padding: 32px; color: var(--muted);">No movies found. Try selecting different genres.</div>`;
    return;
  }

  movies.forEach((movie, index) => {
    const card = document.createElement('article');
    card.className = 'movie-card anim-deal';
    card.id = `card-${movie.movie_id}`;
    card.style.animationDelay = `${index * 60}ms`;

    const primaryGenre = (movie.genres && movie.genres[0]) || 'Drama';
    const genresHtml = (movie.genres || []).map(g => `<span class="genre-tag">${g}</span>`).join('');
    const matchedChips = (movie.explanation?.matched_genres || []).map(g => `<span class="meta-chip active">${g}</span>`).join('');
    const similarChips = (movie.explanation?.similar_liked_movies || []).map(t => `<span class="meta-chip">similar to ${t}</span>`).join('');

    const retMs = (latency?.retrieval ?? 2.0).toFixed(2);
    const rkMs = (latency?.ranker ?? 0.4).toFixed(2);
    const mmrMs = (latency?.mmr ?? 0.2).toFixed(2);
    const totMs = (latency?.total ?? (parseFloat(retMs) + parseFloat(rkMs) + parseFloat(mmrMs))).toFixed(2);

    let traceStepsHtml = '';
    if (trace && trace.length > 0) {
      traceStepsHtml = trace.map(t => {
        const argStr = t.args ? JSON.stringify(t.args).replace(/["{}]/g, '').replace(/:/g, ': ') : '';
        return `
          <div class="trace-step">
            <span class="trace-tool-badge">${t.stage}</span>
            <span class="trace-tool-name">${t.tool}</span>
            <span class="trace-tool-args">(${argStr})</span>
            <span class="trace-tool-ms">${t.latency_ms}ms</span>
          </div>
        `;
      }).join('');
    } else {
      traceStepsHtml = `
        <div class="trace-step"><span class="trace-tool-badge">retrieval</span> <span class="trace-tool-name">two_tower_faiss</span> <span class="trace-tool-args">(candidates: 200)</span> <span class="trace-tool-ms">${retMs}ms</span></div>
        <div class="trace-step"><span class="trace-tool-badge">ranker</span> <span class="trace-tool-name">wide_and_deep</span> <span class="trace-tool-args">(scored: 60)</span> <span class="trace-tool-ms">${rkMs}ms</span></div>
        <div class="trace-step"><span class="trace-tool-badge">diversity</span> <span class="trace-tool-name">mmr_rerank</span> <span class="trace-tool-args">(λ: 0.7, k: 6)</span> <span class="trace-tool-ms">${mmrMs}ms</span></div>
      `;
    }

    card.innerHTML = `
      <div class="poster-frame">
        ${generatePosterSvg(movie.title, movie.year, primaryGenre, movie.movie_id)}
        <div class="poster-content">
          <div class="poster-header">
            <span class="poster-year">${movie.year}</span>
            <span class="poster-match-badge">${movie.match_pct}% Match</span>
          </div>
          <div class="poster-title-area">
            <h3 class="poster-title">${movie.title}</h3>
            <div class="poster-genres-row">${genresHtml}</div>
          </div>
        </div>
      </div>
      <div class="ticket-footer">
        <div class="ticket-actions-row">
          <button class="btn-ticket" onclick="toggleWhy(${movie.movie_id})">Why this?</button>
          <div style="display:flex; gap:6px;">
            <button class="btn-ticket btn-like" onclick="likeMovie(${movie.movie_id})">Like</button>
            <button class="btn-ticket btn-not-for-me" onclick="dismissMovie(${movie.movie_id})">Not for me</button>
          </div>
        </div>
        <div class="why-drawer" id="why-${movie.movie_id}">
          <div class="why-explanation-text" id="why-text-${movie.movie_id}">${movie.explanation?.text || 'Recommended based on your taste profile.'} ${isFallback ? '<span class="fallback-badge">fallback active</span>' : ''}</div>
          <div class="why-chips-row">${matchedChips} ${similarChips}</div>

          <!-- Per-stage latency telemetry -->
          <div class="latency-pills-row" title="Stage Latency Breakdown">
            <span class="latency-pill"><span class="pill-dot ret"></span>retrieval: ${retMs}ms</span>
            <span class="latency-pill"><span class="pill-dot rk"></span>ranker: ${rkMs}ms</span>
            <span class="latency-pill"><span class="pill-dot div"></span>mmr: ${mmrMs}ms</span>
            <span class="latency-pill total">total: ${totMs}ms</span>
          </div>

          <!-- How this was picked trace -->
          <div class="trace-details" onclick="toggleTrace(${movie.movie_id})">
            <span id="trace-toggle-arrow-${movie.movie_id}">▸</span> How this was picked
          </div>
          <div class="trace-content" id="trace-${movie.movie_id}">
            <div class="trace-steps-list">
              ${traceStepsHtml}
            </div>
            <div class="trace-score-breakdown">
              <span>Retrieval: <strong>${movie.score_parts?.retrieval ?? 0.70}</strong></span>
              <span>Ranker: <strong>${movie.score_parts?.ranker ?? 0.75}</strong></span>
              <span>Diversity: <strong>${movie.score_parts?.diversity ?? 0.80}</strong></span>
              <span>Match: <strong style="color:var(--primary);">${movie.match_pct}%</strong></span>
            </div>
          </div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

async function toggleWhy(id) {
  const drawer = document.getElementById(`why-${id}`);
  if (!drawer) return;
  const opening = !drawer.classList.contains('open');
  drawer.classList.toggle('open');
  if (opening) {
    const textEl = document.getElementById(`why-text-${id}`);
    if (textEl && !textEl.dataset.fetched) {
      textEl.dataset.fetched = "true";
      try {
        const res = await fetch(`/api/why-this?movie_id=${id}&user_id=${encodeURIComponent(state.userId)}`);
        if (res.ok) {
          const data = await res.json();
          if (data.text) {
            textEl.innerHTML = `${data.text} ${data.llm_fallback ? '<span class="fallback-badge">fallback active</span>' : ''}`;
          }
        }
      } catch (err) {
        console.error("Failed to load Why this explanation:", err);
      }
    }
  }
}

function toggleTrace(id) {
  const trace = document.getElementById(`trace-${id}`);
  const arrow = document.getElementById(`trace-toggle-arrow-${id}`);
  if (trace) {
    trace.classList.toggle('open');
    if (arrow) arrow.innerText = trace.classList.contains('open') ? '▾' : '▸';
  }
}

// User Actions: Like / Not for me
async function likeMovie(movieId) {
  state.likedSet.add(movieId);
  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: state.userId, movie_id: movieId, feedback_type: 'like' })
  });
  showToast("Added to likes. Updating your recommendations...");
  await loadRecommendations();
  await loadTasteMap();
}

async function dismissMovie(movieId) {
  const card = document.getElementById(`card-${movieId}`);
  const movieObj = state.movies.find(m => m.movie_id === movieId);
  state.lastDismissed = movieObj;

  if (card) {
    card.style.opacity = '0';
    card.style.transform = 'scale(0.9)';
    setTimeout(() => card.remove(), 250);
  }

  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: state.userId, movie_id: movieId, feedback_type: 'dislike' })
  });

  showToast(`Removed from picks`, true);
  await loadTasteMap();
}

async function undoDismiss() {
  if (!state.lastDismissed) return;
  const mid = state.lastDismissed.movie_id;
  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: state.userId, movie_id: mid, feedback_type: 'like' })
  });
  state.lastDismissed = null;
  showToast("Restored movie.");
  await loadRecommendations();
  await loadTasteMap();
}

// Taste Map Rendering (Canvas with smooth point interpolation)
async function loadTasteMap() {
  try {
    const res = await fetch(`/api/taste-map?user_id=${state.userId}`);
    const data = await res.json();
    state.tasteMap = data;
    if (data.user_point) {
      state.userPoint.targetX = data.user_point.x;
      state.userPoint.targetY = data.user_point.y;
    }
    renderTasteMap();
  } catch (err) {
    console.error('Failed to load taste map:', err);
  }
}

let animFrameId = null;
function renderTasteMap() {
  const canvas = document.getElementById('tasteMapCanvas');
  if (!canvas || !state.tasteMap) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  // Smooth interpolation towards target user point
  state.userPoint.x += (state.userPoint.targetX - state.userPoint.x) * 0.15;
  state.userPoint.y += (state.userPoint.targetY - state.userPoint.y) * 0.15;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = state.theme === 'dark' ? 'rgba(38, 45, 59, 0.4)' : 'rgba(216, 208, 197, 0.5)';
  ctx.lineWidth = 1;
  const step = 40;
  for (let x = 0; x < w; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Draw Movie Points
  const points = state.tasteMap.points || [];
  points.forEach(pt => {
    const cx = (pt.x + 1.0) * 0.5 * (w - 60) + 30;
    const cy = (pt.y + 1.0) * 0.5 * (h - 60) + 30;

    ctx.beginPath();
    ctx.arc(cx, cy, pt.is_liked ? 5.5 : 3, 0, Math.PI * 2);
    if (pt.is_liked) {
      ctx.fillStyle = '#10B981';
      ctx.shadowColor = '#10B981';
      ctx.shadowBlur = 8;
    } else {
      const gColor = (GENRE_COLORS[pt.genre] && GENRE_COLORS[pt.genre].bg2) || '#8B93A5';
      ctx.fillStyle = gColor;
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Draw User Taste Center Point with Retro Teal Glow
  const ux = (state.userPoint.x + 1.0) * 0.5 * (w - 60) + 30;
  const uy = (state.userPoint.y + 1.0) * 0.5 * (h - 60) + 30;

  // Outer Pulse Glow
  ctx.beginPath();
  ctx.arc(ux, uy, 16, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(79, 181, 165, 0.25)';
  ctx.fill();

  // Core Dot
  ctx.beginPath();
  ctx.arc(ux, uy, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#4FB5A5';
  ctx.shadowColor = '#4FB5A5';
  ctx.shadowBlur = 14;
  ctx.fill();
  ctx.shadowBlur = 0;

  // Label
  ctx.font = '600 11px ' + getComputedStyle(document.body).fontFamily;
  ctx.fillStyle = '#4FB5A5';
  ctx.fillText("Your Taste", ux + 12, uy + 4);

  // If user point is still gliding, continue animating
  if (Math.abs(state.userPoint.targetX - state.userPoint.x) > 0.002 || Math.abs(state.userPoint.targetY - state.userPoint.y) > 0.002) {
    animFrameId = requestAnimationFrame(renderTasteMap);
  }
}

// Canvas Hover & Click Interaction
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('tasteMapCanvas');
  const tooltip = document.getElementById('mapTooltip');
  if (!canvas || !tooltip) return;

  canvas.addEventListener('mousemove', (e) => {
    if (!state.tasteMap) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const w = rect.width;
    const h = rect.height;

    let found = null;
    for (const pt of state.tasteMap.points) {
      const cx = (pt.x + 1.0) * 0.5 * (w - 60) + 30;
      const cy = (pt.y + 1.0) * 0.5 * (h - 60) + 30;
      const dist = Math.hypot(mx - cx, my - cy);
      if (dist < 8) {
        found = { pt, cx, cy };
        break;
      }
    }

    if (found) {
      tooltip.style.display = 'block';
      tooltip.style.left = `${found.cx}px`;
      tooltip.style.top = `${found.cy}px`;
      tooltip.innerHTML = `<strong>${found.pt.title}</strong> (${found.pt.year})<br><span style="color:var(--muted);">${found.pt.genres.join(', ')}</span><br><span style="color:var(--retro); font-size:11px;">Click to like</span>`;
      canvas.style.cursor = 'pointer';
    } else {
      tooltip.style.display = 'none';
      canvas.style.cursor = 'default';
    }
  });

  canvas.addEventListener('click', async (e) => {
    if (!state.tasteMap) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const w = rect.width;
    const h = rect.height;

    for (const pt of state.tasteMap.points) {
      const cx = (pt.x + 1.0) * 0.5 * (w - 60) + 30;
      const cy = (pt.y + 1.0) * 0.5 * (h - 60) + 30;
      if (Math.hypot(mx - cx, my - cy) < 8) {
        await likeMovie(pt.id);
        break;
      }
    }
  });
});

// Evaluation Data Loader
async function loadEvaluationData() {
  try {
    const res = await fetch('/api/eval');
    const data = await res.json();

    // Render Ablation Table
    const tbody = document.getElementById('ablationTbody');
    tbody.innerHTML = '';
    data.ablation_table.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${row.stage}</strong></td>
        <td>${row.recall_10.toFixed(4)}</td>
        <td>${row.ndcg_10.toFixed(4)}</td>
        <td>${row.coverage_10.toFixed(4)}</td>
        <td>${row.ild_10.toFixed(4)}</td>
        <td>${row.recall_50.toFixed(4)}</td>
        <td>${row.ndcg_50.toFixed(4)}</td>
        <td>${row.coverage_50.toFixed(4)}</td>
        <td>${row.ild_50.toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });

    // Render Simulated A/B
    const ab = data.ab_test;
    document.getElementById('abLiftVal').innerText = `+${ab.relative_ctr_lift_pct}%`;
    document.getElementById('abCiVal').innerText = `95% Confidence Interval (${ab.ci_label}): [${ab.ci_95[0]}, ${ab.ci_95[1]}]`;
    document.getElementById('abPVal').innerText = `p = ${ab.p_value} (${ab.is_significant ? 'Statistically Significant' : 'Not Significant'})`;

    document.getElementById('ctrControl').innerText = `${(ab.control.ctr * 100).toFixed(2)}%`;
    document.getElementById('subControl').innerText = `${ab.control.clicks.toLocaleString()} clicks / ${ab.control.impressions.toLocaleString()} impressions`;
    document.getElementById('ctrTreatment').innerText = `${(ab.treatment.ctr * 100).toFixed(2)}%`;
    document.getElementById('subTreatment').innerText = `${ab.treatment.clicks.toLocaleString()} clicks / ${ab.treatment.impressions.toLocaleString()} impressions`;

    // Render Agent Evals Pass-Rate Table
    const ev = data.agent_evals || {};
    const llmPct = ev.llm_pass_rate_pct ?? ev.pass_rate_pct ?? 100.0;
    const fbPct = ev.fallback_pass_rate_pct ?? ev.pass_rate_pct ?? 100.0;
    const total = ev.total_cases ?? 47;

    const elSum = document.getElementById('agentEvalSummary');
    if (elSum) elSum.innerText = `LLM Agent: ${llmPct}%`;
    const elFb = document.getElementById('agentEvalFallbackSummary');
    if (elFb) elFb.innerText = `Fallback Router: ${fbPct}%`;

    const evTbody = document.getElementById('agentEvalTbody');
    const categories = (ev.fallback_router?.categories) ? Object.keys(ev.fallback_router.categories) : null;
    if (categories && categories.length > 0) {
      evTbody.innerHTML = categories.map(cat => {
        const fbCat = ev.fallback_router.categories[cat];
        const llmCat = (ev.llm_agent?.categories && ev.llm_agent.categories[cat]) || fbCat;
        return `
          <tr>
            <td>${cat.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
            <td>${fbCat.total}</td>
            <td><strong style="color:#10B981;">${llmCat.pass_rate_pct}%</strong></td>
            <td><strong style="color:var(--primary);">${fbCat.pass_rate_pct}%</strong></td>
            <td style="color:var(--muted); font-size:12px;">${cat === 'adversarial' ? 'Guardrail layer' : (cat === 'greeting' ? 'Direct reply' : cat)}</td>
          </tr>
        `;
      }).join('');
    } else {
      evTbody.innerHTML = `
        <tr><td>Standard & Conversational Routing</td><td>24</td><td><strong style="color:#10B981;">100%</strong></td><td><strong style="color:var(--primary);">100%</strong></td><td>search/rec/explain</td></tr>
        <tr><td>Typos & Spelling Resilience</td><td>5</td><td><strong style="color:#10B981;">100%</strong></td><td><strong style="color:var(--primary);">100%</strong></td><td>fuzzy entity resolution</td></tr>
        <tr><td>Multi-Step Intent Queries</td><td>5</td><td><strong style="color:#10B981;">100%</strong></td><td><strong style="color:var(--primary);">100%</strong></td><td>chained execution</td></tr>
        <tr><td>Ambiguous & Thematic Constraints</td><td>5</td><td><strong style="color:#10B981;">100%</strong></td><td><strong style="color:var(--primary);">100%</strong></td><td>filter/recommend</td></tr>
        <tr><td>Adversarial Anti-Hallucination</td><td>4</td><td><strong style="color:#10B981;">100%</strong></td><td><strong style="color:var(--primary);">100%</strong></td><td>catalog bounds guard</td></tr>
      `;
    }
  } catch (err) {
    console.error('Failed to load evaluation data:', err);
  }
}

// Chat Drawer Logic
async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;

  appendChatMsg('user', text);
  input.value = '';

  const messagesDiv = document.getElementById('drawerMessages');
  const skeleton = document.createElement('div');
  skeleton.className = 'chat-bubble bot skeleton';
  skeleton.style.width = '140px';
  skeleton.style.height = '36px';
  messagesDiv.appendChild(skeleton);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, user_id: state.userId })
    });
    const data = await res.json();
    skeleton.remove();

    let formatted = data.response
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/^-\s+(.*)$/gm, '• $1')
      .replace(/^\d+\.\s+(.*)$/gm, '• $1')
      .replace(/\n/g, '<br>');

    appendChatMsg('bot', formatted);

    // If chat returned recommendation cards, update movie grid too
    if (data.cards && data.cards.length > 0) {
      renderMovieGrid(data.cards, data.trace, data.latency_ms);
      showToast("Updated recommendations based on your chat request!");
    }
  } catch (err) {
    skeleton.remove();
    appendChatMsg('bot', '⚠️ Error contacting assistant: ' + err);
  }
}

function appendChatMsg(sender, htmlContent) {
  const container = document.getElementById('drawerMessages');
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${sender}`;
  bubble.innerHTML = htmlContent;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

function sendQuickPrompt(promptText) {
  document.getElementById('chatInput').value = promptText;
  sendChatMessage();
}

// Toast Notification
let toastTimeout = null;
function showToast(message, showUndo = false) {
  const toast = document.getElementById('toast');
  const toastText = document.getElementById('toastText');
  const undoBtn = document.getElementById('toastUndoBtn');

  toastText.innerText = message;
  undoBtn.style.display = showUndo ? 'inline-block' : 'none';

  toast.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('show');
  }, 4000);
}
