// ONCO-SEG Lite – interactive leaderboard (static, GitHub Pages)
// Reads docs/leaderboard.csv. On aiikram.github.io/onco-seg-lite-challenge/ it reads the raw file
// from the default branch so new scores appear without waiting for a Pages rebuild.

const COLUMNS = [
  { key: "rank", label: "Rank", num: true, on: true },
  { key: "team", label: "Participant", on: true },
  { key: "score", label: "Mean Dice", num: true, on: true },
  { key: "gap", label: "Δ Human Top", num: true, on: true },
  { key: "iou", label: "IoU", num: true, on: true },
  { key: "lesion_dice", label: "Lesion Dice", num: true, on: true },
  { key: "specificity", label: "Specificity", num: true, on: true },
  { key: "model_type", label: "Type", on: true },
  { key: "architecture", label: "Architecture", on: true },
  { key: "params", label: "Params", num: true, on: true },
  { key: "llm", label: "LLM", on: false },
  { key: "date", label: "Date", on: true },
  { key: "n_submissions", label: "#Subs", num: true, on: false },
  { key: "run_id", label: "Run ID", on: false },
];

let allRows = [];
let sortKey = "rank";
let sortAsc = true;

function csvUrl() {
  const host = window.location.hostname;
  const m = host.match(/^([^.]+)\.github\.io$/);
  const repo = window.location.pathname.split("/").filter(Boolean)[0];
  if (m && repo) {
    return `https://raw.githubusercontent.com/${m[1]}/${repo}/HEAD/docs/leaderboard.csv?t=${Date.now()}`;
  }
  return `leaderboard.csv?t=${Date.now()}`;
}

function parseCSV(text) {
  const lines = text.replace(/^﻿/, "").trim().split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
  return lines.slice(1).map(line => {
    const vals = line.split(",");
    const o = {};
    headers.forEach((h, i) => (o[h] = (vals[i] || "").trim()));
    return o;
  }).filter(r => r.team && !isNaN(parseFloat(r.score)));
}

function withRanks(rows) {
  // count submissions per team, find each team's best, rank teams (ties share rank)
  const counts = {}, best = {};
  rows.forEach(r => {
    counts[r.team] = (counts[r.team] || 0) + 1;
    const s = parseFloat(r.score);
    if (!(r.team in best) || s > best[r.team]) best[r.team] = s;
  });
  const bestScores = Object.values(best).sort((a, b) => b - a);
  const humanScores = rows.filter(r => r.model_type === "human").map(r => parseFloat(r.score));
  const humanTop = humanScores.length ? Math.max(...humanScores) : null;
  return rows.map(r => {
    const s = parseFloat(r.score);
    return {
      ...r,
      n_submissions: counts[r.team],
      gap: humanTop === null ? "" : (parseFloat(r.score) - humanTop).toFixed(4),
      isBest: s === best[r.team],
      rank: 1 + bestScores.filter(b => b > s + 1e-9).length,
    };
  });
}

async function load() {
  try {
    let res = await fetch(csvUrl());
    if (!res.ok) res = await fetch(`leaderboard.csv?t=${Date.now()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allRows = withRanks(parseCSV(await res.text()));
    renderStats();
    render();
  } catch (e) {
    document.getElementById("leaderboardTable").innerHTML =
      `<div class="loading" style="color:#b91c1c">❌ Could not load leaderboard.csv (${esc(e.message)})</div>`;
  }
}

function renderStats() {
  const teams = new Set(allRows.map(r => r.team));
  const bestOf = f => {
    const s = allRows.filter(f).map(r => parseFloat(r.score));
    return s.length ? Math.max(...s).toFixed(4) : "–";
  };
  set("statTeams", teams.size);
  set("statSubs", allRows.length);
  set("statBest", bestOf(() => true));
  set("statHuman", bestOf(r => r.model_type === "human"));
  set("statLLM", bestOf(r => r.model_type === "llm-only"));
}

function render() {
  const q = document.getElementById("searchBox").value.toLowerCase();
  const type = document.getElementById("typeFilter").value;
  const days = parseInt(document.getElementById("dateFilter").value || "0", 10);
  const bestOnly = document.getElementById("bestOnly").checked;
  const cutoff = days ? Date.now() - days * 864e5 : 0;

  let rows = allRows.filter(r =>
    (!bestOnly || r.isBest) &&
    (!type || r.model_type === type) &&
    (!cutoff || Date.parse(r.date) >= cutoff) &&
    (!q || Object.values(r).some(v => String(v).toLowerCase().includes(q)))
  );
  if (bestOnly) { // one row per team even if a team tied with itself
    const seen = new Set();
    rows = rows.filter(r => !seen.has(r.team) && seen.add(r.team));
  }

  const col = COLUMNS.find(c => c.key === sortKey);
  rows.sort((a, b) => {
    let x = a[sortKey], y = b[sortKey];
    if (col && col.num) { x = parseFloat(x) || 0; y = parseFloat(y) || 0; }
    if (x < y) return sortAsc ? -1 : 1;
    if (x > y) return sortAsc ? 1 : -1;
    return String(a.date).localeCompare(String(b.date));
  });

  const cols = COLUMNS.filter(c => c.on);
  if (!rows.length) {
    document.getElementById("leaderboardTable").innerHTML =
      `<div class="loading">${allRows.length ? "No entries match the filters." : "No submissions yet – be the first!"}</div>`;
    return;
  }
  const head = cols.map(c =>
    `<th data-key="${c.key}" class="${c.num ? "num " : ""}${c.key === sortKey ? "sorted" + (sortAsc ? " asc" : "") : ""}">${c.label}</th>`).join("");
  const body = rows.map(r => "<tr>" + cols.map(c => cell(c, r)).join("") + "</tr>").join("");
  const el = document.getElementById("leaderboardTable");
  el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  el.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.key;
    if (k === sortKey) sortAsc = !sortAsc;
    else { sortKey = k; sortAsc = ["rank", "team", "model_type", "architecture", "llm", "run_id"].includes(k); }
    render();
  }));
}

function cell(c, r) {
  const v = r[c.key];
  switch (c.key) {
    case "rank": {
      const medal = { 1: "🥇 ", 2: "🥈 ", 3: "🥉 " }[v] || "";
      return `<td class="num">${medal}${v}</td>`;
    }
    case "team": return `<td><strong>${esc(v)}</strong></td>`;
    case "score": return `<td class="num score">${parseFloat(v).toFixed(4)}</td>`;
    case "model_type": return `<td><span class="badge ${esc(v)}">${esc(v || "unspecified")}</span></td>`;
    case "gap": {
      if (v === "" || v == null) return `<td class="num">–</td>`;
      const g = parseFloat(v);
      const color = g >= 0 ? "#166534" : "#b91c1c";
      return `<td class="num" style="color:${color}">${g >= 0 ? "+" : ""}${g.toFixed(4)}</td>`;
    }
    case "params": return `<td class="num">${v && !isNaN(v) ? Number(v).toLocaleString() : "–"}</td>`;
    default: return `<td class="${c.num ? "num" : ""}">${esc(v === "" || v == null ? "–" : v)}</td>`;
  }
}

function buildColumnMenu() {
  const menu = document.getElementById("columnMenu");
  menu.innerHTML = COLUMNS.map((c, i) =>
    `<label><input type="checkbox" data-i="${i}" ${c.on ? "checked" : ""}> ${c.label}</label>`).join("");
  menu.querySelectorAll("input").forEach(cb => cb.addEventListener("change", () => {
    COLUMNS[cb.dataset.i].on = cb.checked;
    render();
  }));
  document.getElementById("toggleColumns").addEventListener("click", () => menu.classList.toggle("hidden"));
}

const set = (id, v) => (document.getElementById(id).textContent = v);
function esc(t) { const d = document.createElement("div"); d.textContent = String(t); return d.innerHTML; }

document.addEventListener("DOMContentLoaded", () => {
  buildColumnMenu();
  ["searchBox"].forEach(id => document.getElementById(id).addEventListener("input", render));
  ["typeFilter", "dateFilter", "bestOnly"].forEach(id => document.getElementById(id).addEventListener("change", render));
  load();
  setInterval(load, 5 * 60 * 1000);
});
