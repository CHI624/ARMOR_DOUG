from flask import Flask, render_template, request, redirect, url_for, send_from_directory, current_app, jsonify, session
import sqlite3
import os
import time
import subprocess
import json
import datetime
import sys
import uuid
import requests
from pathlib import Path
from werkzeug.utils import secure_filename
from paths_config import Paths, ensure_directories
from dmd_inference_ensemble import run_ensemble_inference

app = Flask(
    __name__,
    static_folder=str(Paths.STATIC_ROOT),
    template_folder=str(Paths.TEMPLATES),
)

# Ensure expected runtime directories exist before use
ensure_directories()

# ---- Use existing DB if defined, else fallback ----
DB_FILE = Path(globals().get("DB_PATH", Paths.DB_FILE))

# ---- Ensure uploads & analysis directories exist ----
UPLOAD_FOLDER = Paths.UPLOADS
ANALYSIS_FOLDER = Paths.ANALYSIS
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ANALYSIS_FOLDER.mkdir(parents=True, exist_ok=True)
BASE_DIR = Paths.BASE
INFERENCE_SCRIPT = BASE_DIR / "dmd_inference_ensemble.py"
TEMP_DIR = Paths.TEMP
OUTPUT_DIR = Paths.SEVERITY_OUTPUTS
# Default to current interpreter unless overridden via PYTHON_PATH env var
PYTHON_PATH = os.environ.get("PYTHON_PATH", sys.executable)
ALLOWED_FILE_EXT = {"png", "jpg", "jpeg", "gif", "wav", "mp3", "m4a", "ogg", "webp", "mp4"}
TEXT_UPLOAD_FOLDER = Paths.STATIC_TEXT


# Ensure folders exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:5000/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
# Toggle between external/proxy LLM and local LLM server
LLM_MODE = os.environ.get("LLM_MODE", "external")  # "external" | "local"
LLM_LOCAL_BASE_URL = os.environ.get("LLM_LOCAL_BASE_URL", "http://127.0.0.1:53789")
LLM_LOCAL_MODEL = os.environ.get("LLM_LOCAL_MODEL", "Qwen2.5-72B.Q4_K_M.gguf")
LLM_LOCAL_TEMPERATURE = float(os.environ.get("LLM_LOCAL_TEMPERATURE", "0.7"))
LLM_LOCAL_MAX_TOKENS = int(os.environ.get("LLM_LOCAL_MAX_TOKENS", "256"))


def call_local_llm(prompt: str) -> str | None:
    """
    Send a chat-style request to a local Oobabooga/text-generation-webui server.
    Endpoint and generation parameters are controlled via env vars:
      - LLM_API_URL (default http://127.0.0.1:5000/v1/chat/completions)
      - LLM_MODEL (optional), LLM_TEMPERATURE, LLM_MAX_TOKENS
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if LLM_MODEL:
        payload["model"] = LLM_MODEL

    try:
        resp = requests.post(LLM_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"LLM request failed: {exc}")
        return None

    # Try OpenAI-compatible response shape first, then fall back
    try:
        if isinstance(data, dict):
            if data.get("choices"):
                choice = data["choices"][0]
                msg = choice.get("message", {}) if isinstance(choice, dict) else {}
                return msg.get("content") or choice.get("text")
            if data.get("results"):
                return data["results"][0].get("text")
    except Exception:
        pass
    return None


def call_local_chat_server(prompt: str) -> str | None:
    """
    Call the user-provided local LLM API (default http://127.0.0.1:53789).
    """
    url = f"{LLM_LOCAL_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": LLM_LOCAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": LLM_LOCAL_MAX_TOKENS,
        "temperature": LLM_LOCAL_TEMPERATURE,
        "top_p": 0.95,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        if not resp.ok:
            print("Local LLM error status:", resp.status_code)
            print("Local LLM body:", resp.text[:1000])
            resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"Local LLM request failed: {exc}")
        return None


def call_llm(prompt: str) -> str | None:
    """
    Dispatch between external/proxy LLM and local LLM based on LLM_MODE env.
    """
    if LLM_MODE.lower() == "local":
        return call_local_chat_server(prompt)
    return call_local_llm(prompt)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/ETI")
def ETI_HOME():
    return render_template("ETI_HOME.html")
@app.route("/ARMOR_Simu")
def ARMOR_Simu():
    return render_template("ARMOR_Simu.html")
@app.route("/CMMD_risk_choice")
def CMMD_risk_choice():
    return render_template("CMMD_risk_choice.html")
@app.route("/ARMOR_QUER")
def ARMOR_QUER():
    return render_template("ARMOR_QUER.html")
@app.route("/ARMOR_QUER_RESEARCH")
def ARMOR_QUER_RESEARCH():
    return render_template("ARMOR_QUER_RESEARCH.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/projects")
def projects():
    return render_template("projects.html")
@app.route("/CMMD_RISK_RESP")
def CMMD_RISK_RESP():
        # Try to pull last saved result from session (or fetch from DB)
    result = session.get("severity_result", None)

    if result:
        cmmd_score = f"{result['label']} (scores: {result['scores']})"
    else:
        cmmd_score = "N/A"
    return render_template("CMMD_RISK_RESP.html",
                                   CMMD_SCORE=cmmd_score,
        # you can also pass individual scores if you want:
        FINAL_LABEL=result['label'] if result else "N/A",
        SCORES=result['scores'] if result else {})
@app.route("/ARMOR_QUER_INPUT")
def ARMOR_QUER_INPUT():
    return render_template("ARMOR_QUER_INPUT.html")
@app.route("/ARMOR_ASSES")
def ARMOR_ASSES():
    return render_template("ARMOR_ASSES.html")
@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")
def init_db():
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS armor_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_oper TEXT,
            c2_per_op TEXT,
            c2_per_n TEXT,
            c2_per_ex TEXT,
            rank TEXT,
            c2_oper_ex TEXT,
            c2_per_com TEXT,
            loc_oper TEXT,
            c2_per_in TEXT,
            situations TEXT,
            conditions TEXT,
            facts TEXT,
            terr_wea TEXT,
            num_enem TEXT,
            troops_oper_ex TEXT,
            civil_con TEXT,
            time_days INTEGER,
            time_hours INTEGER,
            time_min INTEGER,
            c2_lessons TEXT,
            descr_haz TEXT,
            exposed_haz TEXT,
            experienced_hazard TEXT,
            past_haz TEXT,
            immediate_haz TEXT,
            complete_mission TEXT,
            loss_mission TEXT,
            death TEXT,
            perm_disability TEXT,
            loss_equipment TEXT,
            property_damage TEXT,
            facility_damage TEXT,
            collateral_damage TEXT,
            combined_hazards TEXT,
            severity_score TEXT,
            severity_score_label TEXT,
            single_risk_score TEXT,
            risk_score TEXT,
            risk_score_label TEXT,
            final_risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route("/submit_form", methods=["POST"])
def submit_form():
    """
    Handles researcher form submission:
    - saves uploaded image
    - saves hazard text to file
    - runs ensemble inference
    - stores results in DB
    """

    form_data = request.form.to_dict()

    # 1) Save uploaded image
    image_file = request.files.get("MISS_IMAGE")
    image_path = None
    if image_file and image_file.filename != "":
        filename = secure_filename(image_file.filename)
        image_path = UPLOAD_FOLDER / filename
        image_file.save(image_path)

    # 2) Save hazard description text into a .txt file
    descr_haz = form_data.get("DESCR_HAZ", "").strip()
    text_filename = f"hazard_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.txt"
    tmp_text_path = UPLOAD_FOLDER / text_filename
    with open(tmp_text_path, "w") as f:
        f.write(descr_haz)

    # 3) Run DMD Ensemble Inference (severity by default)
    try:
        ensemble_index, ensemble_result, label_mapping = run_ensemble_inference(
            text_file=tmp_text_path,
            image_file=image_path,
            task="severity"
        )
        final_label = label_mapping[ensemble_index]

        # ✅ Save inference results to a .txt file for frontend / researcher
        result_filename = f"inference_result_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.txt"
        result_path = UPLOAD_FOLDER / result_filename
        with open(result_path, "w") as f:
            f.write(f"Final Label: {final_label}\n")
            f.write("Scores:\n")
            for label, score in ensemble_result.items():
                f.write(f"  {label}: {score:.4f}\n")

    except Exception as e:
        print("❌ Inference failed:", e)
        final_label = "error"
        ensemble_result = {}
        result_path = None
        
    data = request.form.to_dict()

    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('''
        INSERT INTO armor_records (
            date_oper, c2_per_op, c2_per_n, c2_per_ex, rank, c2_oper_ex, 
            c2_per_com, loc_oper, c2_per_in, situations, conditions, facts, terr_wea, 
            num_enem, troops_oper_ex, civil_con, time_days, time_hours, time_min, 
            c2_lessons, descr_haz, exposed_haz, experienced_hazard, past_haz, 
            immediate_haz, complete_mission, loss_mission, death, perm_disability, 
            loss_equipment, property_damage, facility_damage, collateral_damage, 
            combined_hazards, severity_score, severity_score_label, single_risk_score, risk_score, 
            risk_score_label, final_risk_level
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        data.get("DATE_OPER"),
        data.get("C2_PER_OP"),
        data.get("C2_PER_N"),
        data.get("C2_PER_EX"),
        data.get("RANK"),
        data.get("C2_OPER_EX"),
        data.get("C2_PER_COM"),
        data.get("LOC_OPER"),
        data.get("C2_PER_IN"),
        data.get("SITUATIONS"),
        data.get("CONDITIONS"),
        data.get("FACTS"),
        data.get("TERR_WEA"),
        data.get("NUM_ENEM"),
        data.get("TROOPS_OPER_EX"),
        data.get("CIVIL_CON"),
        data.get("TIMEDAYS"),
        data.get("TIMEHOURS"),
        data.get("TIMEMIN"),
        data.get("C2_LESSONS"),
        data.get("DESCR_HAZ"),
        data.get("EXPOSED_HAZ"),
        data.get("experienced_hazard"),
        data.get("PAST_HAZ"),
        data.get("IMMEDIATE_HAZ"),
        data.get("COMPLETE_MISSION"),
        data.get("LOSS_MISSION"),
        data.get("DEATH"),
        data.get("PERM_DISABILITY"),
        data.get("LOSS_EQUIPMENT"),
        data.get("PROPERTY_DAMAGE"),
        data.get("FACILITY_DAMAGE"),
        data.get("COLLATERAL_DAMAGE"),
        data.get("COMBINED_HAZARDS"),
        data.get("SEVERITY_SCORE"),
        data.get("SEVERITY_SCORE_LABEL"),
        data.get("SINGLE_RISK_SCORE"),
        data.get("RISK_SCORE"),
        data.get("RISK_SCORE_LABEL"),
        data.get("FINAL_RISK_LEVEL")
    ))
    conn.commit()
    conn.close()

    return redirect(url_for("ARMOR_Simu_Output"))

def query_db(query):
    print("Using DB file:", DB_FILE.resolve())
    con = sqlite3.connect(str(DB_FILE))
    con.row_factory = sqlite3.Row  # makes rows behave like dicts
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    con.close()
    return rows

@app.route("/view/<table>")
def view_table(table):
    rows = query_db(f"SELECT * FROM {table}")
    if rows:
        columns = rows[0].keys()
        return render_template("view.html", columns=columns, rows=rows)
    else:
        return render_template("view.html", columns=[], rows=[])
@app.route("/ARMOR_Simu_Output")
def ARMOR_Simu_Output():
    db_path = DB_FILE
    if not db_path.exists():
        # No DB -> nothing to show
        return render_template(
            "ARMOR_Simu_Output.html",
            SEVERITY_SCORE="N/A",
            RISK_SCORE="N/A",
            FINAL_RISK_LEVEL="N/A",
            SINGLE_RISK_SCORE="N/A",
            isMissImageAvailable=False
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # fixed query: include FROM <table>
    cursor.execute("""
        SELECT severity_score, single_risk_score, risk_score, final_risk_level
        FROM armor_records
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row:
        # access by column names (Row acts like a dict)
        severity_score = row["severity_score"] if row["severity_score"] is not None else "N/A"
        risk_score = row["risk_score"] if row["risk_score"] is not None else "N/A"
        final_risk_level = row["final_risk_level"] if row["final_risk_level"] is not None else "N/A"
        single_risk_score = row["single_risk_score"] if row["single_risk_score"] is not None else "N/A"

        # example of a check for an image/other column existence:
        is_miss_image_available = bool(row["severity_score"])  # or some other column that indicates image
    else:
        severity_score = "N/A"
        risk_score = "N/A"
        final_risk_level = "N/A"
        single_risk_score = "N/A"
        is_miss_image_available = False

    return render_template(
        "ARMOR_Simu_Output.html",
        SEVERITY_SCORE=severity_score,
        RISK_SCORE=risk_score,
        FINAL_RISK_LEVEL=final_risk_level,
        SINGLE_RISK_SCORE=single_risk_score,  # set this if you have the column
        isMissImageAvailable=is_miss_image_available
    )
# ---- Helper: create the quer_submissions table if needed ----
def ensure_quer_table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS quer_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher TEXT,
            job TEXT,
            job_descr TEXT,
            hazards_json TEXT,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# call this once on app start (ensure you call later in __main__)
# ensure_quer_table()

# ---- Helper: safe file save (returns relative static path or None) ----
def save_quer_image(file_storage):
    if not file_storage:
        return None
    filename = secure_filename(file_storage.filename)
    if filename == "":
        return None
    unique = f"{uuid.uuid4().hex}_{filename}"
    dest = UPLOAD_FOLDER / unique
    file_storage.save(dest)
    # return path relative to static (so templates can use url_for('static', filename=...))
    return f"uploads/{unique}"

# ---- Route to handle form submission (quer_submit) ----
@app.route("/quer_submit", methods=["POST"])
def quer_submit():
    # Ensure table exists
    ensure_quer_table()

    # Basic form fields
    researcher = request.form.get("RESEARCHER_N", "").strip()
    job = request.form.get("JOB", "").strip()
    job_descr = request.form.get("JOB_DESCR", "").strip()

    # hazardsList is a hidden input that contains JSON from the client
    hazards_text = request.form.get("hazardsList", "").strip()
    # Validate JSON (store as text anyway)
    hazards_json_str = ""
    if hazards_text:
        try:
            parsed = json.loads(hazards_text)
            # Optionally sanitize or normalize parsed here
            hazards_json_str = json.dumps(parsed)  # store canonical JSON string
        except Exception:
            # If not valid JSON, store raw text but record that it failed to parse
            hazards_json_str = json.dumps({"invalid_json_raw": hazards_text})

    # file upload
    file_obj = request.files.get("QUER_IMAGE")
    image_relpath = save_quer_image(file_obj)  # e.g. "uploads/uuid_name.jpg" or None

    # Insert into DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO quer_submissions (researcher, job, job_descr, hazards_json, image_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        researcher,
        job,
        job_descr,
        hazards_json_str,
        image_relpath,
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()
    inserted_id = c.lastrowid
    conn.close()

    # Optionally: immediately run analysis on inserted row or queue background processing.
    # e.g.: process_event_by_id(inserted_id)  # if you implemented generic processing that accepts this submission
    # But for now we just redirect to a result page showing what was saved.

    #return redirect(url_for("quer_result", submission_id=inserted_id))
    return redirect(url_for("ARMOR_QUER_INPUT"))

# ---- Result route: show the saved submission ----
@app.route("/quer_result/<int:submission_id>")
def quer_result(submission_id):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM quer_submissions WHERE id = ?", (submission_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return f"No submission with id {submission_id}", 404

    # hazards_json -> try to pretty-print for template
    hazards_pretty = ""
    try:
        hazards_pretty = json.dumps(json.loads(row["hazards_json"]), indent=2)
    except Exception:
        hazards_pretty = row["hazards_json"] or ""

    # Pass values to template (you need to create quer_result.html or adapt an existing template)
    return render_template(
        "quer_result.html",
        submission=dict(row),
        HAZARDS_JSON=hazards_pretty,
        IMAGE_URL=(url_for("static", filename=row["image_path"]) if row["image_path"] else None)
    )

# ---- Debug/listing route: view all queries ----
@app.route("/view_queries")
def view_queries():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM quer_submissions ORDER BY id DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()
    return render_template("view_queries.html", rows=rows)
def ensure_scores_table():
    """Create a compact table for score-only records if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS armor_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            num_haz INTEGER,
            combined_hazards TEXT,
            single_risk_score TEXT,
            severity_score TEXT,
            severity_score_label TEXT,
            risk_score TEXT,
            risk_score_label TEXT,
            final_risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Ensure table exists on import/startup
ensure_scores_table()


@app.route("/Analysis_asess_submit", methods=["POST"])
def Analysis_asess_submit():
    """
    Accepts the Stage-2 assessment form and writes only the computed score fields
    into the new armor_scores table.
    """
    ensure_scores_table()
    form = request.form

    # Read the hidden/posted fields generated by your front-end JS
    try:
        num_haz = form.get("NUM_HAZ")
        num_haz = int(num_haz) if num_haz not in (None, "") else None
    except Exception:
        num_haz = None

    combined_hazards = form.get("COMBINED_HAZARDS", "").strip()
    single_risk_score = form.get("SINGLE_RISK_SCORE", "").strip()
    severity_score = form.get("SEVERITY_SCORE", "").strip()
    severity_score_label = form.get("SEVERITY_SCORE_LABEL", "").strip()
    risk_score = form.get("RISK_SCORE", "").strip()
    risk_score_label = form.get("RISK_SCORE_LABEL", "").strip()
    final_risk_level = form.get("FINAL_RISK_LEVEL", "").strip()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO armor_scores (
            num_haz,
            combined_hazards,
            single_risk_score,
            severity_score,
            severity_score_label,
            risk_score,
            risk_score_label,
            final_risk_level,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        num_haz,
        combined_hazards,
        single_risk_score,
        severity_score,
        severity_score_label,
        risk_score,
        risk_score_label,
        final_risk_level,
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()
    inserted_id = c.lastrowid
    conn.close()

    # Redirect to the results page (it will read from armor_scores)
    return redirect(url_for("ARMOR_REPORT"))
# Note: create two templates:
# - quer_result.html to display the saved submission (show fields, link to the saved image using IMAGE_URL, and the hazards JSON)
# - view_queries.html to present a simple table of recent submissions for debugging
def ensure_researcher_table():
    """Create a compact table for score-only records if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ANALYSIS_RESEARCHER_TABLE (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            DESCR_HAZ INTEGER,
            RISK_SCORE TEXT,
            SINGLE_RISK_SCORE TEXT,
            NUM_HAZ TEXT,
            SEVERITY_SCORE TEXT,
            COMBINED_HAZARDS TEXT,
            FINAL_RISK_LEVEL TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Ensure table exists on import/startup
ensure_researcher_table()
@app.route("/Analysis_asess_researcher_submit", methods=["POST"])
def Analysis_asess_researcher_submit():
    """
    Accepts the Stage-2 assessment form and writes only the computed score fields
    into the new armor_scores table.
    """
    ensure_scores_table()
    form = request.form

    # Read the hidden/posted fields generated by your front-end JS
    try:
        num_haz = form.get("NUM_HAZ")
        num_haz = int(num_haz) if num_haz not in (None, "") else None
    except Exception:
        num_haz = None

    descr_haz = form.get("DESCR_HAZ", "").strip()    
    risk_score = form.get("RISK_SCORE", "").strip()
    single_risk_score = form.get("SINGLE_RISK_SCORE", "").strip()
    num_haz = form.get("NUM_HAZ", "").strip()
    severity_score = form.get("SEVERITY_SCORE", "").strip()    
    combined_hazards = form.get("COMBINED_HAZARDS", "").strip()    
    final_risk_level = form.get("FINAL_RISK_LEVEL", "").strip()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO ANALYSIS_RESEARCHER_TABLE (
            DESCR_HAZ, RISK_SCORE, SINGLE_RISK_SCORE, NUM_HAZ, SEVERITY_SCORE, COMBINED_HAZARDS, FINAL_RISK_LEVEL, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        descr_haz,
        num_haz,
        combined_hazards,
        single_risk_score,
        severity_score,
        risk_score,
        final_risk_level,
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()
    inserted_id = c.lastrowid
    conn.close()

    # Redirect to the results page (it will read from armor_scores)
    return redirect(url_for("ARMOR_REPORT_RESEARCHER"))
# Note: create two templates:
# - quer_result.html to display the saved submission (show fields, link to the saved image using IMAGE_URL, and the hazards JSON)
# - view_queries.html to present a simple table of recent submissions for debugging
@app.route("/ARMOR_REPORT")
def ARMOR_REPORT():
    db_path = DB_FILE
    if not db_path.exists():
        # No DB -> nothing to show
        return render_template(
            "ARMOR_REPORT.html",
            SEVERITY_SCORE="N/A",
            RISK_SCORE="N/A",
            FINAL_RISK_LEVEL="N/A",
            SINGLE_RISK_SCORE="N/A",
            isMissImageAvailable=False
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # fixed query: include FROM <table>
    cursor.execute("""
        SELECT severity_score, single_risk_score, risk_score, final_risk_level
        FROM armor_scores
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row:
        # access by column names (Row acts like a dict)
        severity_score = row["severity_score"] if row["severity_score"] is not None else "N/A"
        risk_score = row["risk_score"] if row["risk_score"] is not None else "N/A"
        final_risk_level = row["final_risk_level"] if row["final_risk_level"] is not None else "N/A"
        single_risk_score = row["single_risk_score"] if row["single_risk_score"] is not None else "N/A"

        # example of a check for an image/other column existence:
        is_miss_image_available = bool(row["severity_score"])  # or some other column that indicates image
    else:
        severity_score = "N/A"
        risk_score = "N/A"
        final_risk_level = "N/A"
        single_risk_score = "N/A"
        is_miss_image_available = False


# ---- Helper: create the quer_submissions table if needed ----
    return render_template("ARMOR_REPORT.html",
        SEVERITY_SCORE=severity_score,
        RISK_SCORE=risk_score,
        FINAL_RISK_LEVEL=final_risk_level,
        SINGLE_RISK_SCORE=single_risk_score,  # set this if you have the column
        isMissImageAvailable=is_miss_image_available
    )
def allowed_file_ext(filename):
    if not filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_FILE_EXT

def save_uploaded_file(file_storage, subfolder=""):
    """
    Save uploaded FileStorage object into static/uploads (optionally in a subfolder).
    Returns relative path (relative to static/) or None on failure.
    """
    if not file_storage or file_storage.filename == "":
        return None
    if not allowed_file_ext(file_storage.filename):
        return None
    filename = secure_filename(file_storage.filename)
    dest_dir = UPLOAD_FOLDER / subfolder if subfolder else UPLOAD_FOLDER
    dest_dir.mkdir(parents=True, exist_ok=True)
    unique = f"{uuid.uuid4().hex}_{filename}"
    abs_path = dest_dir / unique
    file_storage.save(abs_path)
    # store path relative to static: "uploads/..." so templates can access via url_for('static', filename=rel)
    rel = Path("uploads") / subfolder / unique if subfolder else Path("uploads") / unique
    return str(rel).replace("\\", "/")

def init_quer_researcher_table():
    """Create a new minimal table to store researcher query submissions (score/table-free)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS quer_researcher (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_name TEXT,
            job TEXT,
            job_descr TEXT,
            hazards_json TEXT,
            image_path TEXT,
            audio_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# call this at startup (or include in your existing init_db)
init_quer_researcher_table()

@app.route("/quer_submit_researcher", methods=["POST"])
def quer_submit_researcher():
    """
    Handler for the Researcher 'Identify Hazards' form.
    Saves uploaded image/audio (if present), stores the core data in quer_researcher table,
    and redirects to a result/thank-you page.
    """
    # Use form.get(...) so missing fields won't raise KeyError
    researcher_name = request.form.get("RESEARCHER_N", "").strip()
    # the JS attempted to put the final selected job into a hidden input 'final-job'.
    # If you didn't add that hidden input to the HTML, this will attempt to read the visible selects/radios:
    job = request.form.get("final-job") or request.form.get("JOB") or ""
    job_descr = request.form.get("JOB_DESCR", "").strip()
    hazards_json = request.form.get("hazardsList", "").strip()  # JSON encoded string from form

    # files (form uses name="QUER_IMAGE" and "QUER_AUDIO")
    image_rel = None
    audio_rel = None
    if "QUER_IMAGE" in request.files:
        fimg = request.files["QUER_IMAGE"]
        image_rel = save_uploaded_file(fimg, subfolder="quer_researcher")
    if "QUER_AUDIO" in request.files:
        faud = request.files["QUER_AUDIO"]
        audio_rel = save_uploaded_file(faud, subfolder="quer_researcher")

    # Insert into DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO quer_researcher (
            researcher_name, job, job_descr, hazards_json, image_path, audio_path
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        researcher_name,
        job,
        job_descr,
        hazards_json,
        image_rel,
        audio_rel
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()

    # Optionally: flash message, redirect to a result page that shows the saved record
    # Make sure you have a template route e.g. /quer_researcher_result/<int:id>
    return redirect(url_for("ARMOR_QUER_INPUT_RESEARCHER"))
@app.route("/ARMOR_QUER_INPUT_RESEARCHER")
def ARMOR_QUER_INPUT_RESEARCHER():
    return render_template("ARMOR_QUER_INPUT_RESEARCHER.html")
app.secret_key = os.urandom(24)  # needed for sessions

@app.route("/store_model_choice", methods=["POST"])
def store_model_choice():
    data = request.get_json()
    model = data.get("model")

    # Initialize session storage if not set
    if "selected_models" not in session:
        session["selected_models"] = []

    # Add model if not already stored
    if model not in session["selected_models"]:
        session["selected_models"].append(model)

    return jsonify({"status": "ok", "selected_models": session["selected_models"]})

@app.route("/get_selected_models", methods=["GET"])
def get_selected_models():
    return jsonify(session.get("selected_models", []))


@app.route("/api/llm", methods=["POST"])
def llm_proxy():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    reply = call_llm(prompt)
    if not reply:
        return jsonify({"error": "LLM returned no content"}), 502
    return jsonify({"reply": reply})
@app.route("/ARMOR_PHOTO_MODEL_RESEARCHER")
def ARMOR_PHOTO_MODEL_RESEARCHER():
    return render_template("ARMOR_PHOTO_MODEL_RESEARCHER.html")
@app.route("/CMMD_CHOICE_PAGE")
def CMMD_CHOICE_PAGE():
    return render_template("CMMD_CHOICE_PAGE.html")
@app.route("/run_dmd_inference", methods=["POST"])
def run_dmd_inference():
    """
    Run DMD inference using the uploaded hazard text + image
    and save results to static/text/severity_result.txt
    """

    # === 1. Retrieve inputs from session ===
    form = session.get("form_res1", {})
    hazard_text = form.get("dl")
    img_path = form.get("quer_image")

    if not hazard_text or not img_path:
        return jsonify({"error": "Missing session input data"}), 400
    # === 2. Write hazard text to temp file (outside static) ===
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Clear old files in TEMP_DIR
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(TEMP_DIR / f)
        except OSError:
            pass
    # === 2. Write hazard text to temp file (outside static) ===
    hazard_text_file = TEMP_DIR / "hazard_input.txt"
    with open(hazard_text_file, "w") as f:
        f.write(hazard_text)

    # === 3. Define output paths in static/text/ ===
    text_output_dir = Paths.STATIC_TEXT
    text_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = text_output_dir / "severity_result.txt"

    # === 4. Run the Python model ===
    cmd = [
        PYTHON_PATH,
        INFERENCE_SCRIPT,
        hazard_text_file,
        img_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Python model timed out"}), 500

    if result.returncode != 0:
        return jsonify({
            "error": "Python model execution failed",
            "log": result.stderr
        }), 500

    # === 5. Save model output to file ===
    with open(output_file, "w") as f:
        f.write(result.stdout)

    # === 6. Wait until file is ready ===
    start = time.time()
    min_size = 10
    while (
        (not output_file.exists() or output_file.stat().st_size < min_size)
        and time.time() - start < 60
    ):
        time.sleep(0.2)

    if not output_file.exists() or output_file.stat().st_size < min_size:
        return jsonify({
            "error": "Output file not ready in time",
            "log": result.stdout
        }), 500

    # === 7. Store in session for later use ===
    session["severity_result_file"] = output_file

    # === 8. Return JSON response with proper static URL ===
    rel_path = os.path.relpath(output_file, start=Paths.STATIC_ROOT)
    static_url = url_for("static", filename=rel_path.replace("\\", "/"))

    return jsonify({
        "success": True,
        "predicted_url": static_url,   # URL the frontend can fetch
        "local_path": output_file      # Absolute local path (for debugging)
    })
@app.route("/ARMOR_ASESS_RESEARCHER")
def ARMOR_ASESS_RESEARCHER():
    return render_template("ARMOR_ASESS_RESEARCHER.html")
@app.route("/ARMOR_REPORT_RESEARCHER")
def ARMOR_REPORT_RESEARCHER():
    return render_template("ARMOR_REPORT_RESEARCHER.html")
@app.route("/ARMOR_CHOICE_PAGE")
def ARMOR_CHOICE_PAGE():
    return render_template("ARMOR_CHOICE_PAGE.html")
@app.route("/ETI_INFO_PAGE")
def ETI_INFO_PAGE():
    return render_template("ETI_INFO_PAGE.html")
@app.route("/ARMOR_HOME")
def ARMOR_HOME():
    return render_template("ARMOR_HOME.html")
if __name__ == "__main__":
    init_db()
    app.run(debug=True)  # Debug mode for development
    port=8080  # Default Flask port
