"""
FastAPI Server for Resume Analyzer & Improver
Endpoints pour l'analyse et l'amélioration de CV
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv

from resume_analyzer import ResumeAnalyzerAPI
from resume_improver import ResumeImprover

# Configuration
app = FastAPI(
    title="UtopiaHire Resume API",
    description="API pour l'analyse et l'amélioration de CV avec NLP + LLM",
    version="1.0.0"
)

# CORS - Ajuster selon vos besoins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le fichier .env
load_dotenv()
# Instances globales
analyzer_api = ResumeAnalyzerAPI()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # À définir dans les variables d'environnement

# Dossiers temporaires
TEMP_DIR = "temp_files"
OUTPUT_DIR = "generated_resumes"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# MODELS DE RÉPONSE
# ============================================

class AnalysisResponse(BaseModel):
    success: bool
    score: int
    level: str
    sections_detected: Dict[str, bool]
    sections_missing: List[str]
    contact_info: Dict
    high_priority_issues: List[Dict]
    medium_priority_issues: List[Dict]
    statistics: Dict
    timestamp: str


class ImprovementResponse(BaseModel):
    success: bool
    original_score: int
    improvements: Dict
    latex_generated: bool
    pdf_generated: bool
    files: Dict[str, str]
    timestamp: str


# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Endpoint de santé de l'API"""
    return {
        "service": "UtopiaHire Resume API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "improve": "/api/v1/improve",
            "full_process": "/api/v1/full-process",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Vérification de l'état de santé du service"""
    return {
        "status": "healthy",
        "analyzer": "ready",
        "improver": "ready" if GROQ_API_KEY else "missing_api_key",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_resume(file: UploadFile = File(...)):
    """
    Analyse un CV et retourne le score + recommandations
    
    Args:
        file: Fichier PDF du CV
        
    Returns:
        Analyse complète du CV avec score, sections, issues, etc.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    
    # Sauvegarder le fichier temporairement
    temp_path = os.path.join(TEMP_DIR, f"analyze_{datetime.now().timestamp()}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyser le CV
        result = analyzer_api.analyze(temp_path)
        
        # Formater la réponse
        response = AnalysisResponse(
            success=True,
            score=result['score'],
            level=result['level'],
            sections_detected=result['full_analysis']['analysis']['sections'],
            sections_missing=result['sections_missing'],
            contact_info=result['contact_info'],
            high_priority_issues=result['issues_to_fix']['high_priority'],
            medium_priority_issues=result['issues_to_fix']['medium_priority'],
            statistics={
                'word_count': result['full_analysis']['analysis']['format']['word_count'],
                'estimated_pages': result['full_analysis']['analysis']['format']['estimated_pages'],
                'experience_years': result['full_analysis']['analysis']['experience_duration']['total_experience_years'],
                'strong_verbs_count': result['full_analysis']['analysis']['verb_analysis']['strong_count'],
                'weak_verbs_count': result['full_analysis']['analysis']['verb_analysis']['weak_count'],
                'metrics_count': result['full_analysis']['analysis']['metrics']['metrics_count'],
                'bullet_points': result['full_analysis']['analysis']['bullets']['bullet_count']
            },
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")
    
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/improve", response_model=ImprovementResponse)
async def improve_resume(
    file: UploadFile = File(...),
    language: str = Form("en")
):
    """
    Améliore un CV avec l'IA (analyse + génération LaTeX + PDF)
    
    Args:
        file: Fichier PDF du CV
        language: Langue des suggestions ('en' ou 'fr')
        
    Returns:
        CV amélioré avec fichiers LaTeX et PDF générés
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503, 
            detail="Service d'amélioration indisponible: clé API Groq manquante"
        )
    
    if language not in ['en', 'fr']:
        raise HTTPException(status_code=400, detail="Langue doit être 'en' ou 'fr'")
    
    # Sauvegarder le fichier temporairement
    temp_path = os.path.join(TEMP_DIR, f"improve_{datetime.now().timestamp()}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Créer l'improver
        improver = ResumeImprover(GROQ_API_KEY)
        
        # Analyser et améliorer
        result = improver.analyze_and_improve(temp_path, language)
        
        # Créer un dossier de sortie unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(OUTPUT_DIR, f"improved_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Sauvegarder les résultats
        improver.save_improvements(result, output_folder)
        
        # Préparer les chemins des fichiers générés
        files = {
            'report': f"{output_folder}/improvement_report.txt",
            'latex': f"{output_folder}/improved_resume.tex",
            'pdf': f"{output_folder}/improved_resume.pdf",
            'json': f"{output_folder}/improvements.json"
        }
        
        # Vérifier si le PDF a été généré
        pdf_generated = os.path.exists(files['pdf'])
        
        response = ImprovementResponse(
            success=True,
            original_score=result['original_analysis']['score'],
            improvements={
                'professional_summary': result['improvements'].get('professional_summary', {}).get('generated_summary'),
                'experience_improved': 'experience' in result['improvements'],
                'skills_improved': 'skills' in result['improvements'],
                'bullet_suggestions': result['improvements'].get('bullet_suggestions', [])
            },
            latex_generated=True,
            pdf_generated=pdf_generated,
            files={
                'folder': output_folder,
                'report_url': f"/api/v1/download/{timestamp}/report",
                'latex_url': f"/api/v1/download/{timestamp}/latex",
                'pdf_url': f"/api/v1/download/{timestamp}/pdf" if pdf_generated else None,
                'json_url': f"/api/v1/download/{timestamp}/json"
            },
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'amélioration: {str(e)}")
    
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/full-process")
async def full_process(
    file: UploadFile = File(...),
    language: str = Form("en")
):
    """
    Processus complet: Analyse + Amélioration en une seule requête
    
    Args:
        file: Fichier PDF du CV
        language: Langue des suggestions ('en' ou 'fr')
        
    Returns:
        Analyse complète + CV amélioré
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503, 
            detail="Service indisponible: clé API Groq manquante"
        )
    
    # Sauvegarder le fichier temporairement
    temp_path = os.path.join(TEMP_DIR, f"full_{datetime.now().timestamp()}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Analyse
        analysis_result = analyzer_api.analyze(temp_path)
        
        # 2. Amélioration
        improver = ResumeImprover(GROQ_API_KEY)
        improvement_result = improver.analyze_and_improve(temp_path, language)
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(OUTPUT_DIR, f"full_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        improver.save_improvements(improvement_result, output_folder)
        
        pdf_generated = os.path.exists(f"{output_folder}/improved_resume.pdf")
        
        return {
            "success": True,
            "analysis": {
                "score": analysis_result['score'],
                "level": analysis_result['level'],
                "sections_detected": analysis_result['full_analysis']['analysis']['sections'],
                "high_priority_issues": analysis_result['issues_to_fix']['high_priority'],
                "statistics": {
                    'word_count': analysis_result['full_analysis']['analysis']['format']['word_count'],
                    'strong_verbs': analysis_result['full_analysis']['analysis']['verb_analysis']['strong_count'],
                    'metrics': analysis_result['full_analysis']['analysis']['metrics']['metrics_count']
                }
            },
            "improvement": {
                "improvements_generated": list(improvement_result['improvements'].keys()),
                "latex_generated": True,
                "pdf_generated": pdf_generated
            },
            "files": {
                'folder': output_folder,
                'report_url': f"/api/v1/download/{timestamp}/report",
                'latex_url': f"/api/v1/download/{timestamp}/latex",
                'pdf_url': f"/api/v1/download/{timestamp}/pdf" if pdf_generated else None,
                'json_url': f"/api/v1/download/{timestamp}/json"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/v1/download/{timestamp}/{file_type}")
async def download_file(timestamp: str, file_type: str):
    """
    Télécharge un fichier généré
    
    Args:
        timestamp: Timestamp du dossier généré
        file_type: Type de fichier (report, latex, pdf, json)
    """
    # Chercher le dossier correspondant
    possible_folders = [
        os.path.join(OUTPUT_DIR, f"improved_{timestamp}"),
        os.path.join(OUTPUT_DIR, f"full_{timestamp}")
    ]
    
    folder = None
    for f in possible_folders:
        if os.path.exists(f):
            folder = f
            break
    
    if not folder:
        raise HTTPException(status_code=404, detail="Fichiers non trouvés")
    
    # Mapper le type de fichier
    file_mapping = {
        'report': ('improvement_report.txt', 'text/plain'),
        'latex': ('improved_resume.tex', 'application/x-latex'),
        'pdf': ('improved_resume.pdf', 'application/pdf'),
        'json': ('improvements.json', 'application/json')
    }
    
    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail="Type de fichier invalide")
    
    filename, media_type = file_mapping[file_type]
    file_path = os.path.join(folder, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Fichier {filename} non trouvé")
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/api/v1/recommendations/{score}")
async def get_recommendations_by_score(score: int):
    """
    Retourne des recommandations génériques basées sur un score
    
    Args:
        score: Score du CV (0-100)
    """
    if score < 0 or score > 100:
        raise HTTPException(status_code=400, detail="Score doit être entre 0 et 100")
    
    if score >= 80:
        level = "Excellent"
        tips = [
            "Votre CV est de très haute qualité",
            "Assurez-vous de le maintenir à jour",
            "Personnalisez-le pour chaque candidature"
        ]
    elif score >= 60:
        level = "Bon"
        tips = [
            "Ajoutez plus de résultats quantifiables",
            "Renforcez vos verbes d'action",
            "Vérifiez la cohérence du formatage"
        ]
    elif score >= 40:
        level = "Moyen"
        tips = [
            "Restructurez vos sections principales",
            "Ajoutez des métriques concrètes",
            "Utilisez des verbes d'action forts",
            "Vérifiez les informations de contact"
        ]
    else:
        level = "À améliorer"
        tips = [
            "Ajoutez toutes les sections essentielles",
            "Complétez vos informations de contact",
            "Utilisez des bullet points pour les réalisations",
            "Quantifiez vos accomplissements",
            "Évitez les verbes passifs"
        ]
    
    return {
        "score": score,
        "level": level,
        "recommendations": tips
    }


# ============================================
# GESTION DES ERREURS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Erreur interne du serveur",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================
# DÉMARRAGE
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)