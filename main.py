import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="Apex Quant Engine API",
    version="26.0",
    description="Backend de prédictions quantitatives de football pour application mobile"
)

# Autoriser toutes les origines pour permettre la connexion depuis l'application mobile ou le web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. CONFIGURATION & CLÉ API FOOTBALL-DATA
# ==========================================
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "1e9518e7585349f9abe6d5a29ddb83b1")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "FL1": "Ligue 1",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "CL": "Ligue des Champions",
    "EL": "UEFA Europa League",
    "ELC": "Championship (Angleterre)"
}

# ==========================================
# 2. MOTEUR MATHÉMATIQUE APEX QUANT
# ==========================================
def dixon_coles_adjustment(x: int, y: int, h_xg: float, a_xg: float, rho: float = -0.06) -> float:
    if x == 0 and y == 0: return max(0.01, 1.0 - (h_xg * a_xg * rho))
    elif x == 0 and y == 1: return max(0.01, 1.0 + (h_xg * rho))
    elif x == 1 and y == 0: return max(0.01, 1.0 + (a_xg * rho))
    elif x == 1 and y == 1: return max(0.01, 1.0 - rho)
    return 1.0

def prob_to_odds(p: float) -> float:
    if p <= 0: return 99.00
    return round(100.0 / p, 2)

async def fetch_api(client: httpx.AsyncClient, endpoint: str) -> Optional[Dict[str, Any]]:
    try:
        response = await client.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=10.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

async def get_advanced_league_stats(client: httpx.AsyncClient, league_code: str):
    data = await fetch_api(client, f"competitions/{league_code}/standings")
    stats = {}
    avg_goals = 1.35
    
    if data and "standings" in data and len(data["standings"]) > 0:
        table_total = data["standings"][0].get("table", [])
        table_home = data["standings"][1].get("table", []) if len(data["standings"]) > 1 else table_total
        table_away = data["standings"][2].get("table", []) if len(data["standings"]) > 2 else table_total
        
        dict_home = {r["team"]["name"]: r for r in table_home}
        dict_away = {r["team"]["name"]: r for r in table_away}
        
        total_played, total_gf = 0, 0
        
        for row in table_total:
            name = row["team"]["name"]
            played = max(1, row.get("playedGames", 1))
            pts = row.get("points", 0)
            gf = row.get("goalsFor", 0)
            ga = row.get("goalsAgainst", 0)
            form = row.get("form", "D,D,D,D,D")
            
            h_row = dict_home.get(name, row)
            h_played = max(1, h_row.get("playedGames", 1))
            h_gf = h_row.get("goalsFor", gf / 2)
            h_ga = h_row.get("goalsAgainst", ga / 2)
            
            a_row = dict_away.get(name, row)
            a_played = max(1, a_row.get("playedGames", 1))
            a_gf = a_row.get("goalsFor", gf / 2)
            a_ga = a_row.get("goalsAgainst", ga / 2)
            
            pts_per_game = pts / played
            elo_rating = 1500 + (pts_per_game - 1.30) * 180 + ((gf - ga) / played) * 40
            
            form_pts = 0
            if form:
                clean_form = str(form).replace(",", "").upper()
                for char in clean_form[-5:]:
                    if char == 'W': form_pts += 3
                    elif char == 'D': form_pts += 1
            
            form_factor = float(np.clip(0.88 + (form_pts / 50.0), 0.88, 1.12))
            seed_val = sum(ord(c) for c in name)
            card_rate = float(np.clip(1.8 + (seed_val % 12) * 0.18 + (ga / played) * 0.25, 1.5, 4.2))
            corner_rate = float(np.clip(4.2 + (seed_val % 15) * 0.20 + (gf / played) * 0.45, 3.8, 7.5))
            midfield_control = float(np.clip(0.85 + (pts / (played * 3.0)) * 0.35, 0.75, 1.30))
            
            stats[name] = {
                "gf_pg": gf / played, "ga_pg": ga / played,
                "home_gf_pg": h_gf / h_played, "home_ga_pg": h_ga / h_played,
                "away_gf_pg": a_gf / a_played, "away_ga_pg": a_ga / a_played,
                "elo": elo_rating, "form_factor": form_factor,
                "card_rate": card_rate, "corner_rate": corner_rate,
                "midfield": midfield_control
            }
            total_played += played
            total_gf += gf
            
        if total_played > 0:
            avg_goals = max(0.9, (total_gf / total_played) / 2.0)
            
    return stats, avg_goals

def run_quant_prediction_v23(h_name: str, a_name: str, score_h: int = 0, score_a: int = 0, elapsed_min: int = 0, team_stats: dict = {}, avg_goals: float = 1.35, is_live: bool = False):
    default_stat = {
        "gf_pg": 1.35, "ga_pg": 1.25, "home_gf_pg": 1.45, "home_ga_pg": 1.10, 
        "away_gf_pg": 1.15, "away_ga_pg": 1.35, "elo": 1500, "form_factor": 1.0,
        "card_rate": 2.3, "corner_rate": 5.0, "midfield": 1.0
    }
    
    h_stat = team_stats.get(h_name, default_stat)
    a_stat = team_stats.get(a_name, default_stat)
    
    h_att = h_stat["home_gf_pg"] / max(0.1, avg_goals)
    h_def = h_stat["home_ga_pg"] / max(0.1, avg_goals)
    a_att = a_stat["away_gf_pg"] / max(0.1, avg_goals)
    a_def = a_stat["away_ga_pg"] / max(0.1, avg_goals)

    elo_diff = h_stat["elo"] - a_stat["elo"]
    elo_adj = np.clip(elo_diff / 800.0, -0.25, 0.25)

    full_h_xg = float(np.clip(avg_goals * h_att * a_def * (1.0 + elo_adj) * h_stat["form_factor"], 0.4, 3.2))
    full_a_xg = float(np.clip(avg_goals * a_att * h_def * (1.0 - elo_adj) * a_stat["form_factor"], 0.3, 2.8))

    if is_live:
        rem_factor = max(0.05, (90.0 - float(elapsed_min)) / 90.0) if elapsed_min > 0 else 0.50
        rem_h_xg = full_h_xg * rem_factor
        rem_a_xg = full_a_xg * rem_factor
    else:
        rem_factor = 1.0
        rem_h_xg = full_h_xg
        rem_a_xg = full_a_xg

    max_g = 8
    matrix = np.zeros((max_g, max_g))

    for dh in range(max_g - score_h):
        for da in range(max_g - score_a):
            p_h = poisson.pmf(dh, rem_h_xg)
            p_a = poisson.pmf(da, rem_a_xg)
            adj = dixon_coles_adjustment(dh, da, rem_h_xg, rem_a_xg)
            
            final_h = score_h + dh
            final_a = score_a + da
            if final_h < max_g and final_a < max_g:
                matrix[final_h, final_a] = p_h * p_a * adj

    tot_p = np.sum(matrix)
    if tot_p > 0: matrix /= tot_p

    flat_idx = np.argsort(matrix.ravel())[::-1]
    top_3_scores = []
    for idx in flat_idx[:3]:
        gh, ga = np.unravel_index(idx, matrix.shape)
        top_3_scores.append({"score": f"{gh}-{ga}", "prob": round(float(matrix[gh, ga] * 100), 1)})

    p_h = float(np.sum(np.tril(matrix, -1))) * 100
    p_n = float(np.sum(np.diag(matrix))) * 100
    p_a = float(np.sum(np.triu(matrix, 1))) * 100

    prob_o15 = round(float((1.0 - (matrix[0,0] + matrix[1,0] + matrix[0,1])) * 100), 1)
    prob_o25 = round(float((1.0 - np.sum([matrix[i,j] for i in range(3) for j in range(3) if i+j <= 2])) * 100), 1)
    prob_u35 = round(float(np.sum([matrix[i,j] for i in range(max_g) for j in range(max_g) if i+j <= 3]) * 100), 1)
    
    prob_h_goal = round(float((1.0 - poisson.pmf(0, rem_h_xg)) * 100), 1)
    prob_a_goal = round(float((1.0 - poisson.pmf(0, rem_a_xg)) * 100), 1)

    attack_drive = (full_h_xg + full_a_xg) / 2.5
    exp_c_tot = round(float(np.clip(((h_stat["corner_rate"] + a_stat["corner_rate"]) * attack_drive * 0.90) * rem_factor, 1.5, 14.0)), 1)
    
    prob_c_4_5 = round(float((1.0 - poisson.cdf(4, exp_c_tot)) * 100), 1) if exp_c_tot > 0 else 0
    prob_c_6_5 = round(float((1.0 - poisson.cdf(6, exp_c_tot)) * 100), 1) if exp_c_tot > 0 else 0
    prob_c_8_5 = round(float((1.0 - poisson.cdf(8, exp_c_tot)) * 100), 1) if exp_c_tot > 0 else 0
    prob_c_10_5 = round(float((1.0 - poisson.cdf(10, exp_c_tot)) * 100), 1) if exp_c_tot > 0 else 0

    referee_strictness = round(0.88 + ((sum(ord(c) for c in h_name + a_name) % 35) * 0.01), 2)
    midfield_clash = (h_stat["midfield"] + a_stat["midfield"]) / 2.0
    intensity_mult = 1.10 if abs(h_stat["elo"] - a_stat["elo"]) < 80 else 1.0
    
    base_cards = (h_stat["card_rate"] + a_stat["card_rate"]) / 2.0
    exp_k_tot = round(float(np.clip((base_cards * referee_strictness * midfield_clash * intensity_mult) * rem_factor, 0.8, 8.5)), 1)
    
    prob_k_1_5 = round(float((1.0 - poisson.cdf(1, exp_k_tot)) * 100), 1) if exp_k_tot > 0 else 0
    prob_k_2_5 = round(float((1.0 - poisson.cdf(2, exp_k_tot)) * 100), 1) if exp_k_tot > 0 else 0
    prob_k_3_5 = round(float((1.0 - poisson.cdf(3, exp_k_tot)) * 100), 1) if exp_k_tot > 0 else 0
    prob_k_4_5 = round(float((1.0 - poisson.cdf(4, exp_k_tot)) * 100), 1) if exp_k_tot > 0 else 0

    rem_xg_tot = rem_h_xg + rem_a_xg
    prob_more_goals = round(float((1.0 - poisson.pmf(0, rem_xg_tot)) * 100), 1)
    prob_more_corners_2plus = round(float((1.0 - poisson.cdf(1, exp_c_tot)) * 100), 1)

    best_pick = ""
    best_prob = 0.0
    pick_type = ""
    
    if is_live:
        if prob_more_goals >= 68.0:
            best_pick = "⚡ En Direct : Au moins 1 BUT supplémentaire"
            best_prob = prob_more_goals
            pick_type = "LIVE_GOAL"
        elif prob_more_corners_2plus >= 72.0:
            best_pick = "⛳ En Direct : Au moins 2 CORNERS supplémentaires"
            best_prob = prob_more_corners_2plus
            pick_type = "LIVE_CORNER"
        else:
            best_pick = f"🔒 En Direct : Score {score_h}-{score_a} conservé"
            best_prob = round(100.0 - prob_more_goals, 1)
            pick_type = "LIVE_STABLE"
    else:
        if p_h >= 62.0:
            best_pick = f"🔥 Victoire Directe : {h_name}"
            best_prob = p_h
            pick_type = "HOME_WIN"
        elif p_a >= 62.0:
            best_pick = f"🔥 Victoire Directe : {a_name}"
            best_prob = p_a
            pick_type = "AWAY_WIN"
        elif (p_h + p_n) >= 73.0:
            best_pick = f"🛡️ Double Chance : {h_name} ou Nul (1X)"
            best_prob = round(p_h + p_n, 1)
            pick_type = "1X"
        elif (p_a + p_n) >= 73.0:
            best_pick = f"🛡️ Double Chance : Nul ou {a_name} (X2)"
            best_prob = round(p_a + p_n, 1)
            pick_type = "X2"
        elif prob_o15 >= 75.0:
            best_pick = "⚽ Plus de 1.5 Buts au Total"
            best_prob = prob_o15
            pick_type = "O15"
        elif prob_u35 >= 75.0:
            best_pick = "🛡️ Moins de 3.5 Buts au Total"
            best_prob = prob_u35
            pick_type = "U35"
        elif prob_h_goal >= 78.0:
            best_pick = f"⚽ {h_name} marque au moins 1 but"
            best_prob = prob_h_goal
            pick_type = "HOME_GOAL"
        elif prob_a_goal >= 78.0:
            best_pick = f"⚽ {a_name} marque au moins 1 but"
            best_prob = prob_a_goal
            pick_type = "AWAY_GOAL"
        else:
            if (p_h + p_n) >= (p_a + p_n):
                best_pick = f"🛡️ Double Chance : {h_name} ou Nul (1X)"
                best_prob = round(p_h + p_n, 1)
                pick_type = "1X"
            else:
                best_pick = f"🛡️ Double Chance : Nul ou {a_name} (X2)"
                best_prob = round(p_a + p_n, 1)
                pick_type = "X2"

    return {
        "p_h": round(p_h, 1), "p_n": round(p_n, 1), "p_a": round(p_a, 1),
        "odds_h": prob_to_odds(p_h), "odds_n": prob_to_odds(p_n), "odds_a": prob_to_odds(p_a),
        "odds_o15": prob_to_odds(prob_o15), "odds_o25": prob_to_odds(prob_o25),
        "top_3_scores": top_3_scores,
        "prob_o15": prob_o15, "prob_o25": prob_o25, "prob_u35": prob_u35,
        "corners": {
            "tot": exp_c_tot, "p_4_5": prob_c_4_5, "p_6_5": prob_c_6_5, 
            "p_8_5": prob_c_8_5, "p_10_5": prob_c_10_5,
            "odds_4_5": prob_to_odds(prob_c_4_5), "odds_6_5": prob_to_odds(prob_c_6_5)
        },
        "cards": {
            "tot": exp_k_tot, "p_1_5": prob_k_1_5, "p_2_5": prob_k_2_5, 
            "p_3_5": prob_k_3_5, "p_4_5": prob_k_4_5,
            "odds_1_5": prob_to_odds(prob_k_1_5), "odds_2_5": prob_to_odds(prob_k_2_5)
        },
        "best_pick": best_pick,
        "best_prob": best_prob,
        "pick_type": pick_type,
        "selected_odds": prob_to_odds(best_prob)
    }

# ==========================================
# 3. ENDPOINTS API POUR APPLICATION MOBILE
# ==========================================
@app.get("/")
def home():
    return {"status": "Connecté", "system": "Apex Quant Engine v26.0", "engine": "FastAPI Mobile Core"}

@app.get("/api/v1/predictions/{league_code}")
async def get_league_predictions(league_code: str = "PL"):
    if league_code not in COMPETITIONS:
        raise HTTPException(status_code=400, detail="Code de compétition invalide.")
    
    async with httpx.AsyncClient() as client:
        stats, avg_goals = await get_advanced_league_stats(client, league_code)
        matches_data = await fetch_api(client, f"competitions/{league_code}/matches")
        
        matches = matches_data.get("matches", []) if matches_data else []
        predictions_list = []
        
        today_dt = datetime.utcnow()
        next_week_dt = today_dt + timedelta(days=8)
        
        for m in matches:
            if m.get('status') in ['SCHEDULED', 'TIMED']:
                utc_str = m.get('utcDate', '')
                if utc_str:
                    try:
                        m_dt = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S")
                        if today_dt - timedelta(hours=3) <= m_dt <= next_week_dt:
                            h_team = m['homeTeam']['name']
                            a_team = m['awayTeam']['name']
                            
                            pred = run_quant_prediction_v23(h_team, a_team, 0, 0, 0, stats, avg_goals, is_live=False)
                            
                            predictions_list.append({
                                "match_id": m.get("id"),
                                "date_utc": utc_str,
                                "home_team": h_team,
                                "away_team": a_team,
                                "predictions": pred
                            })
                    except Exception:
                        continue

        return {
            "league_code": league_code,
            "league_name": COMPETITIONS[league_code],
            "total_matches": len(predictions_list),
            "data": predictions_list
        }

@app.get("/api/v1/coupon")
async def get_multi_league_coupon():
    async with httpx.AsyncClient() as client:
        all_matches = []
        multi_stats = {}
        multi_avg = {}
        
        for comp_code in COMPETITIONS.keys():
            stats, avg_g = await get_advanced_league_stats(client, comp_code)
            multi_stats[comp_code] = stats
            multi_avg[comp_code] = avg_g
            
            raw = await fetch_api(client, f"competitions/{comp_code}/matches")
            matches = raw.get("matches", []) if raw else []
            
            today_dt = datetime.utcnow()
            next_week_dt = today_dt + timedelta(days=8)
            
            for m in matches:
                if m.get('status') in ['SCHEDULED', 'TIMED']:
                    utc_str = m.get('utcDate', '')
                    if utc_str:
                        try:
                            m_dt = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S")
                            if today_dt - timedelta(hours=3) <= m_dt <= next_week_dt:
                                m_entry = dict(m)
                                m_entry['league_code'] = comp_code
                                all_matches.append(m_entry)
                        except Exception:
                            continue

        analyzed_list = []
        for m in all_matches:
            c_code = m['league_code']
            h_team = m['homeTeam']['name']
            a_team = m['awayTeam']['name']
            
            pred = run_quant_prediction_v23(
                h_team, a_team, 0, 0, 0, 
                team_stats=multi_stats.get(c_code, {}), 
                avg_goals=multi_avg.get(c_code, 1.35), 
                is_live=False
            )
            
            if pred["best_prob"] >= 70.0:
                analyzed_list.append({
                    "league": COMPETITIONS[c_code],
                    "match": f"{h_team} vs {a_team}",
                    "date_utc": m['utcDate'],
                    "pick": pred["best_pick"],
                    "prob": pred["best_prob"],
                    "type": pred["pick_type"],
                    "odds": pred["selected_odds"],
                    "top_score": pred['top_3_scores'][0]['score']
                })
        
        analyzed_list.sort(key=lambda x: x["prob"], reverse=True)
        
        selected_coupon = []
        league_counts = {}
        type_counts = {}
        
        for item in analyzed_list:
            lg = item["league"]
            tp = item["type"]
            if league_counts.get(lg, 0) < 2 and type_counts.get(tp, 0) < 2:
                selected_coupon.append(item)
                league_counts[lg] = league_counts.get(lg, 0) + 1
                type_counts[tp] = type_counts.get(tp, 0) + 1
            if len(selected_coupon) == 5:
                break
                
        if len(selected_coupon) < 5:
            for item in analyzed_list:
                if item not in selected_coupon:
                    selected_coupon.append(item)
                if len(selected_coupon) == 5:
                    break

        total_odds = 1.0
        sum_prob = 0.0
        for leg in selected_coupon:
            total_odds *= leg["odds"]
            sum_prob += leg["prob"]
            
        return {
            "coupon_type": "Combiné 5 Matchs Haute Fiabilité",
            "total_combined_odds": round(total_odds, 2),
            "average_confidence": round(sum_prob / max(1, len(selected_coupon)), 1),
            "selections": selected_coupon
        }