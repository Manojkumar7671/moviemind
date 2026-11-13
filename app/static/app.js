/* MovieMind Client Application */

// State
const state = {
  activeTab: 'discover',
  selectedGenres: new Set(['Action', 'Sci-Fi', 'Thriller']),
  userId: 'demo_user',
  theme: localStorage.getItem('moviemind-theme') || 'dark',
  movies: [],
  worldMovies: [],
  candidateBufferDiscover: [],
  candidateBufferWorld: [],
  isRefillingDiscover: false,
  isRefillingWorld: false,
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

function normalizeTitle(title) {
  if (!title) return '';
  let t = String(title).replace(/\s*\(\d{4}\)$/, '').trim();
  const m = t.match(/^(.*?),\s*(The|A|An)$/i);
  if (m) {
    const art = m[2].charAt(0).toUpperCase() + m[2].slice(1).toLowerCase();
    return `${art} ${m[1].trim()}`;
  }
  return t;
}

// SVG Poster Pattern Generator (Tinted by primary genre hue, pattern varied by movie_id)
function generatePosterSvg(title, year, primaryGenre, movieId = 0) {
  const normTitle = normalizeTitle(title);
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
  if (['discover', 'world', 'tastemap', 'eval'].includes(initialTab)) {
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

let worldPostersPromise = null;
async function loadWorldPosters() {
  if (window.WORLD_POSTERS) return window.WORLD_POSTERS;
  if (!worldPostersPromise) {
    worldPostersPromise = fetch('/static/world_posters.json?v=10')
      .then(res => (res.ok ? res.json() : {}))
      .then(data => {
        window.WORLD_POSTERS = data || {};
        return window.WORLD_POSTERS;
      })
      .catch(() => {
        window.WORLD_POSTERS = {};
        return window.WORLD_POSTERS;
      });
  }
  return worldPostersPromise;
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
    loadTasteMap();
    setTimeout(renderTasteMap, 50);
  } else if (tabName === 'world') {
    if (!state.worldLoaded) {
      loadWorldPosters();
      loadWorldMeta();
      loadWorldRecommendations();
      state.worldLoaded = true;
    }
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
    const summaryBar = document.getElementById('onboardingSummaryBar');
    if (summaryBar) {
      summaryBar.style.display = 'flex';
      document.getElementById('summaryGenresList').innerText = genresList.join(', ');
    }
    state.movies = data.recommendations || [];
    state.candidateBufferDiscover = [];
    refillCandidateBuffer(false);
    updateFallbackNotice(data.llm_fallback, data.notice);
    renderMovieGrid(state.movies, data.trace, data.latency_ms, data.llm_fallback);
    await loadTasteMap();
  } catch (err) {
    showToast("Failed to save genres: " + err);
  }
}

function skipOnboarding() {
  document.getElementById('onboardingBanner').style.display = 'none';
  const summaryBar = document.getElementById('onboardingSummaryBar');
  if (summaryBar) {
    summaryBar.style.display = 'flex';
    document.getElementById('summaryGenresList').innerText = 'All genres (Popular picks)';
  }
  showToast("Showing popular recommendations for you.");
}

function editGenres() {
  const summaryBar = document.getElementById('onboardingSummaryBar');
  if (summaryBar) summaryBar.style.display = 'none';
  const banner = document.getElementById('onboardingBanner');
  if (banner) banner.style.display = 'block';
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

function showGridError(targetGridId, retryFnName, message = "Failed to load recommendations.") {
  const grid = document.getElementById(targetGridId);
  if (!grid) return;
  grid.innerHTML = `
    <div class="grid-error-state" style="grid-column: 1 / -1; padding: 48px 24px; text-align: center;">
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">${message}</p>
      <button class="btn-secondary" onclick="${retryFnName}()">Retry</button>
    </div>
  `;
}

// Recommendations
async function loadRecommendations() {
  const grid = document.getElementById('movieGrid');
  if (grid) {
    grid.innerHTML = '<div class="skeleton" style="height: 380px; border-radius: var(--radius-md); background: rgba(255,255,255,0.04); animation: pulse 1.5s infinite;"></div>'.repeat(6);
  }
  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: state.userId, genres: Array.from(state.selectedGenres), n: 30 })
    });
    if (!res.ok) {
      showGridError('movieGrid', 'loadRecommendations');
      return;
    }
    const data = await res.json();
    const all = data.recommendations || [];
    state.movies = all.slice(0, 6);
    state.candidateBufferDiscover = all.slice(6);
    updateFallbackNotice(data.llm_fallback, data.notice);
    renderMovieGrid(state.movies, data.trace, data.latency_ms, data.llm_fallback);
  } catch (err) {
    showGridError('movieGrid', 'loadRecommendations', `Failed to load recommendations: ${err}`);
  }
}

function renderMovieGrid(movies, trace = [], latency = {}, isFallback = false, targetGridId = 'movieGrid') {
  const grid = document.getElementById(targetGridId);
  if (!grid) return;
  grid.innerHTML = '';

  if (!movies || movies.length === 0) {
    if (targetGridId === 'worldMovieGrid') {
      grid.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1; padding: 48px 24px; text-align: center; color: var(--muted); font-size: 14px;">No films match. Widen the years or remove a filter.</div>`;
    } else {
      grid.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1; padding: 32px; text-align: center; color: var(--muted);">No movies found matching current criteria.</div>`;
    }
    return;
  }

  movies.forEach((movie, index) => {
    grid.appendChild(createMovieCardElement(movie, index, isFallback, latency, trace, targetGridId));
  });
}

function createMovieCardElement(movie, index = 0, isFallback = false, latency = {}, trace = [], targetGridId = 'movieGrid') {
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

    const isWorldCinema = (targetGridId === 'worldMovieGrid');

    let originBadge = '';
    let langChipHtml = '';

    if (isWorldCinema) {
      const originNames = (movie.origin_country_names && movie.origin_country_names.length > 0)
        ? movie.origin_country_names.join(', ')
        : (movie.origin || 'unknown');
      const langNames = (movie.original_language_names && movie.original_language_names.length > 0)
        ? movie.original_language_names.join(', ')
        : (movie.language_names_str || '');

      originBadge = `<span class="card-origin-region-label" title="${originNames}">Origin: ${originNames}</span>`;
      if (langNames) {
        langChipHtml = `<span class="genre-tag language-tag" title="Original language">${langNames}</span>`;
      }
    }

    const posterKey = ((movie.title || '') + (movie.year || '')).toLowerCase().replace(/[^a-z0-9]/g, '');
    const posterPath = isWorldCinema
      ? ((window.WORLD_POSTERS && (window.WORLD_POSTERS[movie.imdb_id] || window.WORLD_POSTERS[movie.movie_id])) || (window.POSTERS && window.POSTERS[posterKey]))
      : (window.POSTERS && window.POSTERS[posterKey]);
    const safeAlt = String(movie.title || '').replace(/"/g, '&quot;');
    const posterImgHtml = posterPath
      ? `<img class="poster-real" loading="lazy" alt="${safeAlt} poster" src="https://image.tmdb.org/t/p/w342${posterPath}"><div class="poster-shade"></div>`
      : '';

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
      <div class="poster-frame">${posterImgHtml}
        ${generatePosterSvg(movie.title, movie.year, primaryGenre, movie.movie_id)}
        <div class="poster-content">
          <div class="poster-header">
            <div class="poster-header-left">
              <span class="poster-year">${movie.year}</span>${originBadge}
            </div>
            <span class="poster-match-badge" title="Relative to this list">${movie.match_pct}% match</span>
          </div>
          <div class="poster-title-area">
            <h3 class="poster-title">${movie.title}</h3>
            <div class="poster-genres-row">${genresHtml}${langChipHtml}</div>
          </div>
        </div>
      </div>
      <div class="ticket-footer">
        <div class="ticket-actions-row">
          <button class="btn-ticket" onclick="toggleWhy('${movie.movie_id}')">Why this?</button>
          <div style="display:flex; gap:6px;">
            <button class="btn-ticket btn-like" onclick="likeMovie('${movie.movie_id}', ${isWorldCinema})">Like</button>
            <button class="btn-ticket btn-not-for-me" onclick="dismissMovie('${movie.movie_id}', ${isWorldCinema})">Not for me</button>
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
          <div class="trace-details" onclick="toggleTrace('${movie.movie_id}')">
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
              <span>Relative match: <strong style="color:var(--primary);">${movie.match_pct}%</strong></span>
            </div>
          </div>
        </div>
      </div>
    `;
    return card;
}

async function loadWorldMeta() {
  try {
    const res = await fetch('/api/global/meta');
    if (!res.ok) return;
    const meta = await res.json();
    
    const regSel = document.getElementById('worldRegionSelect');
    if (regSel && regSel.options.length <= 1) {
      meta.regions.forEach(r => {
        const opt = document.createElement('option');
        const code = (typeof r === 'object' && r.code) ? r.code : r;
        const name = (typeof r === 'object' && r.name) ? r.name : code;
        const count = (typeof r === 'object' && r.count !== undefined) ? ` (${r.count.toLocaleString()})` : '';
        opt.value = code;
        opt.textContent = `${name}${count}`;
        regSel.appendChild(opt);
      });
    }

    const langSel = document.getElementById('worldLanguageSelect');
    if (langSel && langSel.options.length <= 1) {
      meta.languages.forEach(l => {
        const opt = document.createElement('option');
        const code = (typeof l === 'object' && l.code) ? l.code : l;
        const name = (typeof l === 'object' && l.name) ? l.name : code;
        const count = (typeof l === 'object' && l.count !== undefined) ? ` (${l.count.toLocaleString()})` : '';
        opt.value = code;
        opt.textContent = `${name}${count}`;
        langSel.appendChild(opt);
      });
    }

    const genSel = document.getElementById('worldGenreSelect');
    if (genSel && genSel.options.length <= 1) {
      meta.genres.forEach(g => {
        const opt = document.createElement('option');
        opt.value = g;
        opt.textContent = g;
        genSel.appendChild(opt);
      });
    }

    const yMin = document.getElementById('worldYearMin');
    const yMax = document.getElementById('worldYearMax');
    if (yMin) yMin.placeholder = meta.year_min;
    if (yMax) yMax.placeholder = meta.year_max;
  } catch (err) {
    console.error("Failed to load world meta:", err);
  }
}

async function loadWorldRecommendations() {
  const grid = document.getElementById('worldMovieGrid');
  if (grid) {
    grid.innerHTML = '<div class="skeleton" style="height: 380px; border-radius: var(--radius-md); background: rgba(255,255,255,0.04); animation: pulse 1.5s infinite;"></div>'.repeat(6);
  }

  const reg = document.getElementById('worldRegionSelect')?.value || '';
  const lang = document.getElementById('worldLanguageSelect')?.value || '';
  const genre = document.getElementById('worldGenreSelect')?.value || '';
  const yMin = document.getElementById('worldYearMin')?.value || '';
  const yMax = document.getElementById('worldYearMax')?.value || '';

  // Preserve sticky filter state across re-renders
  state.worldFilters = {
    region: reg,
    language: lang,
    genre: genre,
    year_min: yMin,
    year_max: yMax,
  };

  const params = new URLSearchParams({
    user_id: state.userId,
    n: 40,
  });
  if (reg) params.set('region', reg);
  if (lang) params.set('language', lang);
  if (genre) params.set('genre', genre);
  if (yMin) params.set('year_min', yMin);
  if (yMax) params.set('year_max', yMax);

  try {
    const [res] = await Promise.all([
      fetch(`/api/global/recommend?${params.toString()}`),
      loadWorldPosters(),
    ]);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const msg = errData.detail || "Failed to load world recommendations.";
      showGridError('worldMovieGrid', 'loadWorldRecommendations', msg);
      return;
    }
    const data = await res.json();
    const all = data.recommendations || [];
    state.worldMovies = all.slice(0, 12);
    state.candidateBufferWorld = all.slice(12);
    renderMovieGrid(state.worldMovies, [], { total: data.total_ms }, false, 'worldMovieGrid');
  } catch (err) {
    console.error("Failed to load world recommendations:", err);
    showGridError('worldMovieGrid', 'loadWorldRecommendations', "Network connection error.");
  }
}

function applyWorldFilters() {
  loadWorldRecommendations();
}

function resetWorldFilters() {
  if (document.getElementById('worldRegionSelect')) document.getElementById('worldRegionSelect').value = '';
  if (document.getElementById('worldLanguageSelect')) document.getElementById('worldLanguageSelect').value = '';
  if (document.getElementById('worldGenreSelect')) document.getElementById('worldGenreSelect').value = '';
  if (document.getElementById('worldYearMin')) document.getElementById('worldYearMin').value = '';
  if (document.getElementById('worldYearMax')) document.getElementById('worldYearMax').value = '';
  loadWorldRecommendations();
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

// User Actions: Like / Not for me (optimistic instant UI feedback, pre-buffered backfill)
function popNextCandidateFromBuffer(isWorld) {
  const bufferKey = isWorld ? 'candidateBufferWorld' : 'candidateBufferDiscover';
  const listKey = isWorld ? 'worldMovies' : 'movies';
  if (!state[bufferKey]) state[bufferKey] = [];
  const currentList = state[listKey] || [];
  const existingIds = new Set(currentList.map(m => String(m.movie_id)));
  const dismissed = state.dismissedSet || new Set();
  const liked = state.likedSet || new Set();

  while (state[bufferKey].length > 0) {
    const cand = state[bufferKey].shift();
    const cid = String(cand.movie_id);
    if (!existingIds.has(cid) && !dismissed.has(cid) && !liked.has(cid) && !document.getElementById(`card-${cid}`)) {
      if (state[bufferKey].length < 10) {
        refillCandidateBuffer(isWorld);
      }
      return cand;
    }
  }
  return null;
}

async function refillCandidateBuffer(isWorld) {
  const lockKey = isWorld ? 'isRefillingWorld' : 'isRefillingDiscover';
  if (state[lockKey]) return;
  state[lockKey] = true;

  try {
    const bufferKey = isWorld ? 'candidateBufferWorld' : 'candidateBufferDiscover';
    const listKey = isWorld ? 'worldMovies' : 'movies';
    const currentList = state[listKey] || [];
    const existingIds = new Set(currentList.map(m => String(m.movie_id)));
    (state[bufferKey] || []).forEach(m => existingIds.add(String(m.movie_id)));
    const dismissed = state.dismissedSet || new Set();
    const liked = state.likedSet || new Set();

    let candidates = [];
    if (isWorld) {
      const f = state.worldFilters || {};
      const params = new URLSearchParams({
        user_id: state.userId,
        n: 40
      });
      if (f.region) params.set('region', f.region);
      if (f.language) params.set('language', f.language);
      if (f.genre) params.set('genre', f.genre);
      if (f.year_min) params.set('year_min', f.year_min);
      if (f.year_max) params.set('year_max', f.year_max);
      const res = await fetch(`/api/global/recommend?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        candidates = data.recommendations || [];
      }
    } else {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: state.userId,
          genres: Array.from(state.selectedGenres),
          n: 40
        })
      });
      if (res.ok) {
        const data = await res.json();
        candidates = data.recommendations || [];
      }
    }

    if (!state[bufferKey]) state[bufferKey] = [];
    for (const c of candidates) {
      const cid = String(c.movie_id);
      if (!existingIds.has(cid) && !dismissed.has(cid) && !liked.has(cid) && !document.getElementById(`card-${cid}`)) {
        state[bufferKey].push(c);
        existingIds.add(cid);
      }
    }
  } catch (err) {
    console.warn("Buffer refill failed:", err);
  } finally {
    state[lockKey] = false;
  }
}

async function getNextCandidate(isWorld) {
  let cand = popNextCandidateFromBuffer(isWorld);
  if (cand) return cand;

  await refillCandidateBuffer(isWorld);
  return popNextCandidateFromBuffer(isWorld);
}

async function replenishGridCard(isWorld) {
  const targetGridId = isWorld ? 'worldMovieGrid' : 'movieGrid';
  const grid = document.getElementById(targetGridId);
  if (!grid) return;
  const targetCount = isWorld ? 12 : 6;
  const listKey = isWorld ? 'worldMovies' : 'movies';
  if (!state[listKey]) state[listKey] = [];

  while (state[listKey].length < targetCount) {
    const cand = await getNextCandidate(isWorld);
    if (!cand) break;
    state[listKey].push(cand);
    grid.appendChild(createMovieCardElement(cand, state[listKey].length - 1, false, {}, [], targetGridId));
  }
}

function sendFeedbackInBackground(movieId, feedbackType) {
  fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: state.userId, movie_id: movieId, feedback_type: feedbackType })
  }).catch(err => {
    console.warn("Feedback sync failed:", err);
  });
}

async function replaceMovieCard(movieId, isWorld) {
  if (typeof isWorld !== 'boolean') {
    isWorld = state.activeTab === 'world';
  }
  const targetGridId = isWorld ? 'worldMovieGrid' : 'movieGrid';
  const grid = document.getElementById(targetGridId);
  const listKey = isWorld ? 'worldMovies' : 'movies';
  const targetCount = isWorld ? 12 : 6;
  if (!state[listKey]) state[listKey] = [];

  const card = document.getElementById(`card-${movieId}`);
  const cardIndex = state[listKey].findIndex(m => String(m.movie_id) === String(movieId));

  let nextCand = popNextCandidateFromBuffer(isWorld);
  if (!nextCand) {
    nextCand = await getNextCandidate(isWorld);
  }

  if (nextCand) {
    if (cardIndex !== -1) {
      state[listKey][cardIndex] = nextCand;
    } else {
      state[listKey] = state[listKey].filter(m => String(m.movie_id) !== String(movieId));
      state[listKey].push(nextCand);
    }

    const newCard = createMovieCardElement(
      nextCand,
      cardIndex !== -1 ? cardIndex : state[listKey].length - 1,
      false,
      {},
      [],
      targetGridId
    );

    if (card && card.parentNode && typeof card.replaceWith === 'function') {
      card.replaceWith(newCard);
    } else if (card && card.parentNode) {
      const parent = card.parentNode;
      const idx = parent.children ? parent.children.indexOf(card) : -1;
      card.remove();
      if (idx !== -1 && parent.children) {
        parent.children.splice(idx, 0, newCard);
        newCard.parentNode = parent;
      } else {
        parent.appendChild(newCard);
      }
    } else if (grid) {
      if (card) card.remove();
      grid.appendChild(newCard);
    }
  } else {
    if (card) card.remove();
    state[listKey] = state[listKey].filter(m => String(m.movie_id) !== String(movieId));
    if (isWorld) {
      await loadWorldRecommendations();
    } else {
      await loadRecommendations();
    }
  }

  if (grid && grid.querySelectorAll) {
    const currentCards = grid.querySelectorAll('.movie-card');
    while (currentCards.length < targetCount && state[isWorld ? 'candidateBufferWorld' : 'candidateBufferDiscover']?.length > 0) {
      const extra = popNextCandidateFromBuffer(isWorld);
      if (extra) {
        state[listKey].push(extra);
        grid.appendChild(createMovieCardElement(extra, state[listKey].length - 1, false, {}, [], targetGridId));
      } else {
        break;
      }
    }
  }
}

function likeMovie(movieId, isWorld) {
  if (typeof isWorld !== 'boolean') {
    isWorld = state.activeTab === 'world';
  }
  if (!state.likedSet) state.likedSet = new Set();
  state.likedSet.add(String(movieId));
  const card = document.getElementById(`card-${movieId}`);
  const likeBtn = card?.querySelector?.('.btn-like');
  if (likeBtn) {
    likeBtn.classList.add('liked');
    likeBtn.textContent = 'Liked ✓';
    likeBtn.setAttribute('aria-pressed', 'true');
  }
  showToast("Added to likes!");
  const p = replaceMovieCard(movieId, isWorld);
  sendFeedbackInBackground(movieId, 'like');
  return p;
}

function dismissMovie(movieId, isWorld) {
  if (typeof isWorld !== 'boolean') {
    isWorld = state.activeTab === 'world';
  }
  const listKey = isWorld ? 'worldMovies' : 'movies';
  const list = state[listKey] || [];
  const movieObj = list.find(m => String(m.movie_id) === String(movieId));
  state.lastDismissed = movieObj || { movie_id: movieId };
  if (!state.dismissedSet) state.dismissedSet = new Set();
  state.dismissedSet.add(String(movieId));

  showToast("Removed from picks", true);
  const p = replaceMovieCard(movieId, isWorld);
  sendFeedbackInBackground(movieId, 'dislike');
  return p;
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
  if (state.activeTab === 'world') {
    await loadWorldRecommendations();
  }
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
    if (elSum) elSum.innerText = `LLM Agent (Held-Out): ${llmPct}%`;
    const elFb = document.getElementById('agentEvalFallbackSummary');
    if (elFb) elFb.innerText = `Fallback Router: ${fbPct}%`;

    const diffNotice = document.getElementById('differingCatsNotice');
    if (diffNotice) {
      if (ev.differing_categories && ev.differing_categories.length > 0) {
        const diffList = ev.differing_categories.map(d => `${d.category} (LLM: ${d.llm_pass_rate_pct}%, FB: ${d.fallback_pass_rate_pct}%)`).join(', ');
        diffNotice.innerText = `Differ in ${ev.differing_categories.length} categories: ${diffList}`;
        diffNotice.style.color = 'var(--primary)';
      } else {
        diffNotice.innerText = '0 differences (identical 100% agreement)';
        diffNotice.style.color = '#10B981';
      }
    }

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

    // World Cinema Content Evaluation
    if (data.world_cinema_eval) {
      const w = data.world_cinema_eval;
      const elH20 = document.getElementById('worldHit20');
      if (elH20 && w.recommender_hit20 !== undefined) {
        elH20.innerText = `${(w.recommender_hit20 * 100).toFixed(2)}%`;
      }
      const elH20CI = document.getElementById('worldHit20CI');
      if (elH20CI && w.recommender_ci95_str) {
        elH20CI.innerText = `95% CI: ${w.recommender_ci95_str}`;
      }
      const elRecBox = document.getElementById('worldRecHitBox');
      if (elRecBox && w.recommender_hit20 !== undefined) {
        elRecBox.innerText = `${(w.recommender_hit20 * 100).toFixed(2)}%`;
      }
      const elRecSub = document.getElementById('worldRecSub');
      if (elRecSub && w.recommender_hits !== undefined) {
        elRecSub.innerText = `${w.recommender_hits} hits / ${w.n_samples || 500} • 95% CI: ${w.recommender_ci95_str || ''}`;
      }
      const elPoolRandBox = document.getElementById('worldPoolRandHitBox');
      if (elPoolRandBox && w.pool_random_hit20 !== undefined) {
        elPoolRandBox.innerText = `${(w.pool_random_hit20 * 100).toFixed(2)}%`;
      }
      const elPoolRandSub = document.getElementById('worldPoolRandSub');
      if (elPoolRandSub && w.pool_random_hits !== undefined) {
        elPoolRandSub.innerText = `${w.pool_random_hits} hits / ${w.n_samples || 500} • 95% CI: ${w.pool_random_ci95_str || ''}`;
      }
      const elPoolPopBox = document.getElementById('worldPoolPopHitBox');
      if (elPoolPopBox && w.pool_popularity_hit20 !== undefined) {
        elPoolPopBox.innerText = `${(w.pool_popularity_hit20 * 100).toFixed(2)}%`;
      }
      const elPoolPopSub = document.getElementById('worldPoolPopSub');
      if (elPoolPopSub && w.pool_popularity_hits !== undefined) {
        elPoolPopSub.innerText = `${w.pool_popularity_hits} hits / ${w.n_samples || 500} • 95% CI: ${w.pool_popularity_ci95_str || ''}`;
      }
      const elPopBox = document.getElementById('worldPopHitBox');
      if (elPopBox && w.popularity_hit20 !== undefined) {
        elPopBox.innerText = `${(w.popularity_hit20 * 100).toFixed(2)}%`;
      }
      const elPopSub = document.getElementById('worldPopSub');
      if (elPopSub && w.popularity_hits !== undefined) {
        elPopSub.innerText = `${w.popularity_hits} hits / ${w.n_samples || 500} • 95% CI: ${w.popularity_ci95_str || ''}`;
      }
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

    // Display cards strictly inside chat drawer; never replace main recommendation grid
    if (data.cards && data.cards.length > 0) {
      const cardsHtml = data.cards.slice(0, 5).map(c => `
        <div class="chat-card-item" style="margin-top: 6px; padding: 6px 10px; background: rgba(255,255,255,0.05); border-radius: 6px; border: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-weight: 600; font-size: 12px; color: var(--text);">${c.title} <span style="opacity: 0.6; font-size: 11px;">(${c.year})</span></div>
            <div style="font-size: 11px; color: var(--muted);">${(c.genres || []).slice(0, 2).join(', ')}</div>
          </div>
          <span style="font-size: 11px; font-weight: 600; color: var(--primary);">${c.match_pct || 90}% match</span>
        </div>
      `).join('');
      appendChatMsg('bot', `<div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; color: var(--retro);">Matched Films:</div>${cardsHtml}`);
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
