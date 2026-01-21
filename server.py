from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import jwt
import bcrypt
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'easywerkstatt')]

# JWT Settings
JWT_SECRET = os.environ.get('JWT_SECRET', 'easywerkstatt-secret-key-2024')
JWT_ALGORITHM = "HS256"

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Create the main app
app = FastAPI(title="EasyWerkstatt API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== MODELS =====================

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: str = "mechanic"

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    role: str
    created_at: str

class CustomerCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None

class CustomerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    created_at: str
    updated_at: str

class VehicleCreate(BaseModel):
    customer_id: str
    license_plate: Optional[str] = None
    vin: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    color: Optional[str] = None
    mileage: Optional[int] = None
    tire_size: Optional[str] = None
    notes: Optional[str] = None

class VehiclePhoto(BaseModel):
    id: str
    photo_type: str  # front, back, left, right, roof, windshield, vin, registration, tire, underside
    image_base64: str
    uploaded_at: str

class VehicleResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    customer_id: str
    license_plate: Optional[str] = None
    vin: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    color: Optional[str] = None
    mileage: Optional[int] = None
    tire_size: Optional[str] = None
    notes: Optional[str] = None
    photos: List[dict] = []
    ai_analysis: Optional[dict] = None
    created_at: str
    updated_at: str

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    vehicle_id: str
    analysis_type: str
    result: dict
    created_at: str

class SupplierLink(BaseModel):
    name: str
    url: str
    logo_color: str
    description: str

# ===================== HELPER FUNCTIONS =====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc).timestamp() + 86400 * 7  # 7 days
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({"id": payload["user_id"]}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ===================== AUTH ROUTES =====================

@api_router.post("/auth/register", response_model=dict)
async def register(user: UserCreate):
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    user_doc = {
        "id": user_id,
        "email": user.email,
        "password": hash_password(user.password),
        "name": user.name,
        "role": user.role,
        "created_at": now
    }
    
    await db.users.insert_one(user_doc)
    token = create_token(user_id, user.email, user.role)
    
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "created_at": now
        }
    }

@api_router.post("/auth/login", response_model=dict)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["id"], user["email"], user["role"])
    
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "created_at": user["created_at"]
        }
    }

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)

# ===================== CUSTOMER ROUTES =====================

@api_router.post("/customers", response_model=CustomerResponse)
async def create_customer(customer: CustomerCreate, current_user: dict = Depends(get_current_user)):
    customer_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    customer_doc = {
        "id": customer_id,
        **customer.model_dump(),
        "created_by": current_user["id"],
        "created_at": now,
        "updated_at": now
    }
    
    await db.customers.insert_one(customer_doc)
    return CustomerResponse(**customer_doc)

@api_router.get("/customers", response_model=List[CustomerResponse])
async def list_customers(current_user: dict = Depends(get_current_user)):
    customers = await db.customers.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [CustomerResponse(**c) for c in customers]

@api_router.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: str, current_user: dict = Depends(get_current_user)):
    customer = await db.customers.find_one({"id": customer_id}, {"_id": 0})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CustomerResponse(**customer)

@api_router.put("/customers/{customer_id}", response_model=CustomerResponse)
async def update_customer(customer_id: str, customer: CustomerCreate, current_user: dict = Depends(get_current_user)):
    existing = await db.customers.find_one({"id": customer_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    now = datetime.now(timezone.utc).isoformat()
    update_data = {**customer.model_dump(), "updated_at": now}
    
    await db.customers.update_one({"id": customer_id}, {"$set": update_data})
    updated = await db.customers.find_one({"id": customer_id}, {"_id": 0})
    return CustomerResponse(**updated)

@api_router.delete("/customers/{customer_id}")
async def delete_customer(customer_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.customers.delete_one({"id": customer_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    # Also delete associated vehicles
    await db.vehicles.delete_many({"customer_id": customer_id})
    return {"message": "Customer deleted"}

# ===================== VEHICLE ROUTES =====================

@api_router.post("/vehicles", response_model=VehicleResponse)
async def create_vehicle(vehicle: VehicleCreate, current_user: dict = Depends(get_current_user)):
    # Verify customer exists
    customer = await db.customers.find_one({"id": vehicle.customer_id})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    vehicle_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    vehicle_doc = {
        "id": vehicle_id,
        **vehicle.model_dump(),
        "photos": [],
        "ai_analysis": None,
        "created_by": current_user["id"],
        "created_at": now,
        "updated_at": now
    }
    
    await db.vehicles.insert_one(vehicle_doc)
    return VehicleResponse(**vehicle_doc)

@api_router.get("/vehicles", response_model=List[VehicleResponse])
async def list_vehicles(customer_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    query = {"customer_id": customer_id} if customer_id else {}
    vehicles = await db.vehicles.find(query, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [VehicleResponse(**v) for v in vehicles]

@api_router.get("/vehicles/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: str, current_user: dict = Depends(get_current_user)):
    vehicle = await db.vehicles.find_one({"id": vehicle_id}, {"_id": 0})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return VehicleResponse(**vehicle)

@api_router.put("/vehicles/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(vehicle_id: str, vehicle: VehicleCreate, current_user: dict = Depends(get_current_user)):
    existing = await db.vehicles.find_one({"id": vehicle_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    now = datetime.now(timezone.utc).isoformat()
    update_data = {**vehicle.model_dump(), "updated_at": now}
    
    await db.vehicles.update_one({"id": vehicle_id}, {"$set": update_data})
    updated = await db.vehicles.find_one({"id": vehicle_id}, {"_id": 0})
    return VehicleResponse(**updated)

@api_router.delete("/vehicles/{vehicle_id}")
async def delete_vehicle(vehicle_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.vehicles.delete_one({"id": vehicle_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"message": "Vehicle deleted"}

# ===================== PHOTO UPLOAD ROUTES =====================

@api_router.post("/vehicles/{vehicle_id}/photos")
async def upload_photo(
    vehicle_id: str,
    photo_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    vehicle = await db.vehicles.find_one({"id": vehicle_id})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Read and encode file
    content = await file.read()
    base64_image = base64.b64encode(content).decode('utf-8')
    
    photo_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    photo_doc = {
        "id": photo_id,
        "photo_type": photo_type,
        "image_base64": base64_image,
        "filename": file.filename,
        "content_type": file.content_type,
        "uploaded_at": now
    }
    
    await db.vehicles.update_one(
        {"id": vehicle_id},
        {"$push": {"photos": photo_doc}, "$set": {"updated_at": now}}
    )
    
    return {"message": "Photo uploaded", "photo_id": photo_id}

@api_router.delete("/vehicles/{vehicle_id}/photos/{photo_id}")
async def delete_photo(vehicle_id: str, photo_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.vehicles.update_one(
        {"id": vehicle_id},
        {"$pull": {"photos": {"id": photo_id}}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Photo not found")
    return {"message": "Photo deleted"}

# ===================== AI ANALYSIS ROUTES =====================

@api_router.post("/vehicles/{vehicle_id}/analyze")
async def analyze_vehicle(vehicle_id: str, current_user: dict = Depends(get_current_user)):
    vehicle = await db.vehicles.find_one({"id": vehicle_id}, {"_id": 0})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    if not vehicle.get("photos"):
        raise HTTPException(status_code=400, detail="No photos to analyze")
    
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")
    
    try:
        # Create chat instance
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"vehicle-analysis-{vehicle_id}",
            system_message="""Du bist ein Experte für Fahrzeugidentifikation und KFZ-Technik. 
            Analysiere die Fahrzeugbilder und extrahiere alle relevanten Informationen:
            - Marke und Modell
            - Bauserie/Generation
            - Baujahr (geschätzt)
            - Farbe
            - Besondere Merkmale
            - Reifengröße (falls sichtbar)
            - Fahrgestellnummer (falls sichtbar)
            - Zustand (Sichtbare Schäden, Rost, etc.)
            
            Antworte auf Deutsch und strukturiert in JSON-Format."""
        ).with_model("openai", "gpt-5.1")
        
        # Prepare images for analysis
        file_contents = []
        for photo in vehicle["photos"][:5]:  # Max 5 photos to avoid token limits
            file_contents.append(ImageContent(image_base64=photo["image_base64"]))
        
        # Create message with images
        user_message = UserMessage(
            text=f"""Analysiere diese Fahrzeugbilder und identifiziere das Fahrzeug.
            Bekannte Informationen:
            - Kennzeichen: {vehicle.get('license_plate', 'Unbekannt')}
            - VIN: {vehicle.get('vin', 'Unbekannt')}
            
            Bitte gib eine detaillierte Analyse in folgendem JSON-Format zurück:
            {{
                "make": "Marke",
                "model": "Modell",
                "series": "Bauserie",
                "year_estimated": "Baujahr",
                "color": "Farbe",
                "body_type": "Karosserieform",
                "tire_size": "Reifengröße falls erkennbar",
                "vin_detected": "VIN falls erkennbar",
                "condition": "Zustandsbeschreibung",
                "special_features": ["Besondere Merkmale"],
                "damage_notes": ["Sichtbare Schäden"],
                "confidence": "Konfidenz der Erkennung (hoch/mittel/niedrig)"
            }}""",
            file_contents=file_contents
        )
        
        response = await chat.send_message(user_message)
        
        # Parse response
        import json
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                analysis_result = json.loads(response[json_start:json_end])
            else:
                analysis_result = {"raw_response": response}
        except json.JSONDecodeError:
            analysis_result = {"raw_response": response}
        
        # Save analysis
        now = datetime.now(timezone.utc).isoformat()
        analysis_doc = {
            "id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "result": analysis_result,
            "analyzed_at": now
        }
        
        await db.vehicles.update_one(
            {"id": vehicle_id},
            {"$set": {"ai_analysis": analysis_doc, "updated_at": now}}
        )
        
        # Update vehicle with detected info if confidence is high
        if analysis_result.get("confidence") in ["hoch", "high"]:
            update_fields = {}
            if analysis_result.get("make") and not vehicle.get("make"):
                update_fields["make"] = analysis_result["make"]
            if analysis_result.get("model") and not vehicle.get("model"):
                update_fields["model"] = analysis_result["model"]
            if analysis_result.get("year_estimated") and not vehicle.get("year"):
                try:
                    update_fields["year"] = int(analysis_result["year_estimated"])
                except (ValueError, TypeError):
                    pass
            if analysis_result.get("color") and not vehicle.get("color"):
                update_fields["color"] = analysis_result["color"]
            if analysis_result.get("tire_size") and not vehicle.get("tire_size"):
                update_fields["tire_size"] = analysis_result["tire_size"]
            
            if update_fields:
                await db.vehicles.update_one({"id": vehicle_id}, {"$set": update_fields})
        
        return {"message": "Analysis complete", "analysis": analysis_doc}
        
    except Exception as e:
        logger.error(f"AI Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ===================== SERVICE BOOK (DIGITALES SCHECKHEFT) =====================

class ServiceBookEntry(BaseModel):
    entry_type: str  # tuv, repair, maintenance, oilChange, tireChange, inspection, other
    date: str
    mileage: Optional[int] = None
    description: str
    cost: Optional[float] = None
    workshop: Optional[str] = None
    next_due_date: Optional[str] = None
    next_due_mileage: Optional[int] = None

@api_router.post("/vehicles/{vehicle_id}/servicebook")
async def add_servicebook_entry(vehicle_id: str, entry: ServiceBookEntry, current_user: dict = Depends(get_current_user)):
    vehicle = await db.vehicles.find_one({"id": vehicle_id})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    entry_doc = {
        "id": entry_id,
        **entry.model_dump(),
        "created_by": current_user["id"],
        "created_at": now
    }
    
    await db.servicebook.insert_one({**entry_doc, "vehicle_id": vehicle_id})
    
    # Create reminder if next_due_date is set
    if entry.next_due_date:
        reminder_doc = {
            "id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "customer_id": vehicle.get("customer_id"),
            "reminder_type": entry.entry_type,
            "due_date": entry.next_due_date,
            "due_mileage": entry.next_due_mileage,
            "description": f"Fällig: {entry.entry_type}",
            "completed": False,
            "created_at": now
        }
        await db.reminders.insert_one(reminder_doc)
    
    return {"message": "Entry added", "id": entry_id}

@api_router.get("/vehicles/{vehicle_id}/servicebook")
async def get_servicebook(vehicle_id: str, current_user: dict = Depends(get_current_user)):
    entries = await db.servicebook.find(
        {"vehicle_id": vehicle_id}, 
        {"_id": 0}
    ).sort("date", -1).to_list(1000)
    return entries

@api_router.delete("/vehicles/{vehicle_id}/servicebook/{entry_id}")
async def delete_servicebook_entry(vehicle_id: str, entry_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.servicebook.delete_one({"id": entry_id, "vehicle_id": vehicle_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"message": "Entry deleted"}

# ===================== TÜV CHECKLIST =====================

class TuvChecklistData(BaseModel):
    checks: dict  # checkpoint_id: "passed" | "failed" | "pending"
    notes: dict   # checkpoint_id: "note text"

@api_router.post("/vehicles/{vehicle_id}/tuv-checklist")
async def save_tuv_checklist(vehicle_id: str, data: TuvChecklistData, current_user: dict = Depends(get_current_user)):
    vehicle = await db.vehicles.find_one({"id": vehicle_id})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    now = datetime.now(timezone.utc).isoformat()
    
    checklist_doc = {
        "vehicle_id": vehicle_id,
        "checks": data.checks,
        "notes": data.notes,
        "updated_at": now,
        "updated_by": current_user["id"]
    }
    
    await db.tuv_checklists.update_one(
        {"vehicle_id": vehicle_id},
        {"$set": checklist_doc},
        upsert=True
    )
    
    return {"message": "Checklist saved"}

@api_router.get("/vehicles/{vehicle_id}/tuv-checklist")
async def get_tuv_checklist(vehicle_id: str, current_user: dict = Depends(get_current_user)):
    checklist = await db.tuv_checklists.find_one({"vehicle_id": vehicle_id}, {"_id": 0})
    return checklist

# ===================== REMINDERS =====================

class ReminderCreate(BaseModel):
    vehicle_id: str
    reminder_type: str  # oilChange, tireChange, tuv, inspection, custom
    due_date: Optional[str] = None
    due_mileage: Optional[int] = None
    description: str

@api_router.post("/reminders")
async def create_reminder(reminder: ReminderCreate, current_user: dict = Depends(get_current_user)):
    vehicle = await db.vehicles.find_one({"id": reminder.vehicle_id})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    reminder_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    reminder_doc = {
        "id": reminder_id,
        **reminder.model_dump(),
        "customer_id": vehicle.get("customer_id"),
        "completed": False,
        "notified": False,
        "created_by": current_user["id"],
        "created_at": now
    }
    
    await db.reminders.insert_one(reminder_doc)
    return {"message": "Reminder created", "id": reminder_id}

@api_router.get("/reminders")
async def list_reminders(vehicle_id: Optional[str] = None, customer_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    query = {"completed": False}
    if vehicle_id:
        query["vehicle_id"] = vehicle_id
    if customer_id:
        query["customer_id"] = customer_id
    
    reminders = await db.reminders.find(query, {"_id": 0}).sort("due_date", 1).to_list(1000)
    return reminders

@api_router.put("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.reminders.update_one(
        {"id": reminder_id},
        {"$set": {"completed": True, "completed_at": datetime.now(timezone.utc).isoformat()}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"message": "Reminder completed"}

@api_router.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.reminders.delete_one({"id": reminder_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"message": "Reminder deleted"}

# ===================== LISTING GENERATOR =====================

class ListingRequest(BaseModel):
    condition: str
    known_defects: Optional[str] = None
    features: Optional[str] = None

@api_router.post("/vehicles/{vehicle_id}/generate-listing")
async def generate_listing(vehicle_id: str, request: ListingRequest, current_user: dict = Depends(get_current_user)):
    vehicle = await db.vehicles.find_one({"id": vehicle_id}, {"_id": 0})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")
    
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"listing-{vehicle_id}",
            system_message="""Du bist ein Experte für Fahrzeug-Inserate. 
            Erstelle professionelle, ehrliche und ansprechende Verkaufsanzeigen auf Deutsch.
            Betone positive Eigenschaften, verschweige aber keine bekannten Mängel."""
        ).with_model("openai", "gpt-5.1")
        
        condition_labels = {
            "new": "Neufahrzeug",
            "used": "Gebraucht",
            "certified": "Geprüft/Zertifiziert",
            "accident": "Unfallfahrzeug",
            "export": "Export",
            "forParts": "Für Bastler / Teileträger"
        }
        
        prompt = f"""Erstelle eine professionelle Verkaufsanzeige für folgendes Fahrzeug:

Fahrzeugdaten:
- Marke: {vehicle.get('make', 'Unbekannt')}
- Modell: {vehicle.get('model', 'Unbekannt')}
- Baujahr: {vehicle.get('year', 'Unbekannt')}
- Kilometerstand: {vehicle.get('mileage', 'Unbekannt')} km
- Farbe: {vehicle.get('color', 'Unbekannt')}
- Zustand: {condition_labels.get(request.condition, request.condition)}

Bekannte Mängel: {request.known_defects or 'Keine bekannt'}

Ausstattung: {request.features or 'Standardausstattung'}

Erstelle eine ansprechende, ehrliche Beschreibung für Online-Plattformen wie Mobile.de oder eBay Kleinanzeigen.
Die Beschreibung sollte 150-250 Wörter lang sein."""
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        return {"description": response}
        
    except Exception as e:
        logger.error(f"Listing generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ===================== SUPPLIER LINKS =====================

SUPPLIERS = [
    {
        "name": "Dolphin",
        "url": "https://www.dolphin-group.de",
        "logo_color": "#00539B",
        "description": "Ersatzteile & Werkstattbedarf"
    },
    {
        "name": "Gamma Reifenhandel",
        "url": "https://www.gamma-reifen.de",
        "logo_color": "#FF6B00",
        "description": "Reifen & Felgen"
    },
    {
        "name": "Reifen.com",
        "url": "https://www.reifen.com",
        "logo_color": "#DC2626",
        "description": "Online Reifenshop"
    },
    {
        "name": "Schmettau und Fuchs",
        "url": "https://www.schmettau-fuchs.de",
        "logo_color": "#1E3A8A",
        "description": "Kfz-Ersatzteile"
    },
    {
        "name": "Heise und Klatte",
        "url": "https://www.heise-klatte.de",
        "logo_color": "#16A34A",
        "description": "Autoteile Großhandel"
    },
    {
        "name": "Heil und Sohn",
        "url": "https://www.heil-und-sohn.de",
        "logo_color": "#7C3AED",
        "description": "Kfz-Teile & Zubehör"
    }
]

@api_router.get("/suppliers", response_model=List[dict])
async def list_suppliers():
    return SUPPLIERS

# ===================== AR SCANNER ROUTES =====================

class ARScanRequest(BaseModel):
    image_base64: str
    scan_mode: str  # tire, parts, damage

SUPPLIERS_MAP = {
    "tire": [
        {"name": "Reifen.com", "url": "https://www.reifen.com", "color": "#DC2626"},
        {"name": "Gamma Reifenhandel", "url": "https://www.gamma-reifen.de", "color": "#FF6B00"},
    ],
    "parts": [
        {"name": "Dolphin", "url": "https://www.dolphin-group.de", "color": "#00539B"},
        {"name": "Heil und Sohn", "url": "https://www.heil-und-sohn.de", "color": "#7C3AED"},
        {"name": "Schmettau und Fuchs", "url": "https://www.schmettau-fuchs.de", "color": "#1E3A8A"},
    ],
    "damage": [
        {"name": "Heise und Klatte", "url": "https://www.heise-klatte.de", "color": "#16A34A"},
        {"name": "Dolphin", "url": "https://www.dolphin-group.de", "color": "#00539B"},
    ]
}

@api_router.post("/ar/analyze")
async def ar_analyze(request: ARScanRequest, current_user: dict = Depends(get_current_user)):
    """
    AR-Light: Analyze camera frame and return detected items with supplier links
    """
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")
    
    scan_mode = request.scan_mode
    
    # Build prompt based on scan mode
    prompts = {
        "tire": """Analysiere dieses Bild eines Reifens. Erkenne:
            - Reifengröße (z.B. 205/55 R16)
            - Reifenmarke
            - Profiltiefe (geschätzt)
            - DOT-Nummer falls sichtbar
            - Zustand (gut/mittel/schlecht)
            
            Antworte in JSON:
            {
                "tire_size": "205/55 R16",
                "brand": "Continental",
                "tread_depth": "5mm",
                "dot_number": "2521",
                "condition": "gut",
                "needs_replacement": false,
                "confidence": "hoch"
            }""",
        
        "wheel": """Analysiere dieses Bild einer Felge/eines Rades. Erkenne:
            - Lochkreis/PCD (z.B. 5x112)
            - Anzahl der Schrauben
            - Felgendurchmesser (z.B. 16 Zoll)
            - Felgenbreite (z.B. 7J)
            - Einpresstiefe/ET (geschätzt)
            - Nabendurchmesser falls erkennbar
            - J-Nummer falls sichtbar
            
            Antworte in JSON:
            {
                "bolt_pattern": "5x112",
                "bolt_count": 5,
                "diameter": "16",
                "width": "7J",
                "offset_et": "45",
                "center_bore": "57.1",
                "j_number": "7Jx16 H2 ET45",
                "confidence": "hoch"
            }""",
        
        "dashboard": """Analysiere dieses Bild eines Armaturenbretts/Displays. Erkenne:
            - Fehlermeldungen/Warnleuchten
            - Service-Anzeigen
            - Kilometerstand
            - Tankfüllstand
            - Andere relevante Anzeigen
            
            Für jede Fehlermeldung gib auch mögliche Ursachen und Lösungen an.
            
            Antworte in JSON:
            {
                "error_codes": [
                    {"code": "P0300", "description": "Zündaussetzer", "severity": "mittel", "solution": "Zündkerzen prüfen"}
                ],
                "warning_lights": ["Motorcheck", "Reifendruck"],
                "service_due": true,
                "mileage_shown": "85432",
                "recommendations": ["OBD-Diagnose durchführen"],
                "confidence": "hoch"
            }""",
        
        "obd": """Analysiere dieses Bild eines OBD-Testers/Diagnosetools. Erkenne:
            - Angezeigte Fehlercodes
            - Beschreibung der Fehler
            - Schweregrad
            - Empfohlene Maßnahmen
            
            Für jeden Fehlercode suche nach der Bedeutung und Lösung.
            
            Antworte in JSON:
            {
                "dtc_codes": [
                    {"code": "P0420", "description": "Katalysator Wirkungsgrad unter Schwellwert", "severity": "mittel", "solution": "Katalysator und Lambdasonde prüfen"}
                ],
                "pending_codes": [],
                "freeze_frame_data": {},
                "recommendations": ["Professionelle Diagnose empfohlen"],
                "reset_possible": true,
                "confidence": "hoch"
            }""",
        
        "parts": """Analysiere dieses Bild aus dem Motorraum/Unterboden eines Fahrzeugs. Identifiziere:
            - Sichtbare Teile (Ölfilter, Luftfilter, Bremsen, etc.)
            - Position der Teile
            - Zustand der Teile
            - Fahrzeugtyp falls erkennbar
            
            Antworte in JSON:
            {
                "identified_parts": [
                    {"name": "Ölfilter", "location": "rechts vorne", "condition": "gut"},
                    {"name": "Luftfilter", "location": "mitte oben", "condition": "verschmutzt"}
                ],
                "vehicle_type": "PKW",
                "recommendations": ["Luftfilter wechseln"],
                "confidence": "mittel"
            }""",
        
        "damage": """Analysiere dieses Bild auf Fahrzeugschäden. Erkenne:
            - Art des Schadens (Rost, Delle, Kratzer, Riss, etc.)
            - Schweregrad (leicht/mittel/schwer)
            - Betroffene Stelle
            - Reparaturempfehlung
            
            Antworte in JSON:
            {
                "damages": [
                    {"type": "Rost", "severity": "mittel", "location": "Schweller links", "repair": "Schleifen und Lackieren"}
                ],
                "overall_condition": "reparaturbedürftig",
                "urgent": false,
                "confidence": "hoch"
            }"""
    }
    
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"ar-scan-{current_user['id']}",
            system_message="Du bist ein KFZ-Experte mit Spezialisierung auf Fahrzeugteile und Schäden. Analysiere Bilder präzise und gib strukturierte JSON-Antworten auf Deutsch."
        ).with_model("openai", "gpt-5.1")
        
        image_content = ImageContent(image_base64=request.image_base64)
        
        user_message = UserMessage(
            text=prompts.get(scan_mode, prompts["parts"]),
            file_contents=[image_content]
        )
        
        response = await chat.send_message(user_message)
        
        # Parse response
        import json
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                analysis = json.loads(response[json_start:json_end])
            else:
                analysis = {"raw_response": response}
        except json.JSONDecodeError:
            analysis = {"raw_response": response}
        
        # Build response based on scan mode
        detected_items = []
        description = ""
        confidence = analysis.get("confidence", "mittel")
        
        if scan_mode == "tire":
            if analysis.get("tire_size"):
                detected_items.append({"label": "Reifengröße", "value": analysis["tire_size"]})
            if analysis.get("brand"):
                detected_items.append({"label": "Marke", "value": analysis["brand"]})
            if analysis.get("tread_depth"):
                detected_items.append({"label": "Profiltiefe", "value": analysis["tread_depth"]})
            if analysis.get("condition"):
                detected_items.append({"label": "Zustand", "value": analysis["condition"]})
            
            if analysis.get("needs_replacement"):
                description = "⚠️ Reifen sollte ersetzt werden!"
            else:
                description = "Reifen in gutem Zustand."
                
        elif scan_mode == "parts":
            for part in analysis.get("identified_parts", []):
                detected_items.append({
                    "label": part.get("name", "Teil"),
                    "value": f"{part.get('location', '')} - {part.get('condition', '')}"
                })
            
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                description = "Empfehlungen: " + ", ".join(recommendations)
                
        elif scan_mode == "damage":
            for damage in analysis.get("damages", []):
                detected_items.append({
                    "label": damage.get("type", "Schaden"),
                    "value": f"{damage.get('location', '')} ({damage.get('severity', '')})"
                })
            
            if analysis.get("urgent"):
                description = "⚠️ Dringende Reparatur empfohlen!"
            else:
                description = f"Gesamtzustand: {analysis.get('overall_condition', 'unbekannt')}"
        
        elif scan_mode == "wheel":
            if analysis.get("bolt_pattern"):
                detected_items.append({"label": "Lochkreis", "value": analysis["bolt_pattern"]})
            if analysis.get("bolt_count"):
                detected_items.append({"label": "Schrauben", "value": str(analysis["bolt_count"])})
            if analysis.get("diameter"):
                detected_items.append({"label": "Durchmesser", "value": f'{analysis["diameter"]} Zoll'})
            if analysis.get("width"):
                detected_items.append({"label": "Breite", "value": analysis["width"]})
            if analysis.get("offset_et"):
                detected_items.append({"label": "Einpresstiefe", "value": f'ET{analysis["offset_et"]}'})
            if analysis.get("center_bore"):
                detected_items.append({"label": "Nabendurchmesser", "value": f'{analysis["center_bore"]}mm'})
            if analysis.get("j_number"):
                detected_items.append({"label": "J-Nummer", "value": analysis["j_number"]})
            
            description = "Felgendaten erfolgreich erkannt."
        
        elif scan_mode == "dashboard":
            for warning in analysis.get("warning_lights", []):
                detected_items.append({"label": "Warnleuchte", "value": warning})
            for error in analysis.get("error_codes", []):
                detected_items.append({
                    "label": error.get("code", "Fehler"),
                    "value": error.get("description", "")
                })
            
            if analysis.get("service_due"):
                description = "⚠️ Service fällig!"
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                description += " " + ", ".join(recommendations)
        
        elif scan_mode == "obd":
            for dtc in analysis.get("dtc_codes", []):
                detected_items.append({
                    "label": dtc.get("code", "DTC"),
                    "value": dtc.get("description", "")
                })
            
            if analysis.get("reset_possible"):
                description = "Fehlerspeicher kann zurückgesetzt werden."
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                description = ", ".join(recommendations)
            
            # Add search links for error codes
            if analysis.get("dtc_codes"):
                for dtc in analysis["dtc_codes"]:
                    code = dtc.get("code", "")
                    if code:
                        detected_items.append({
                            "label": f"🔍 {code} Lösung suchen",
                            "value": f"https://www.google.com/search?q={code}+lösung+reparatur"
                        })
        
        # Get relevant suppliers
        suppliers = SUPPLIERS_MAP.get(scan_mode, [])
        
        # Add wheel-specific suppliers
        if scan_mode == "wheel":
            suppliers = [
                {"name": "Reifen.com", "url": "https://www.reifen.com/felgen", "color": "#DC2626"},
                {"name": "Gamma Reifenhandel", "url": "https://www.gamma-reifen.de", "color": "#FF6B00"},
            ]
        
        return {
            "scan_mode": scan_mode,
            "detected_items": detected_items,
            "description": description,
            "confidence": confidence,
            "suppliers": suppliers,
            "raw_analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"AR Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analyse fehlgeschlagen: {str(e)}")

# ===================== AI SEARCH =====================

class AISearchRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []

@api_router.post("/ai/search")
async def ai_search(request: AISearchRequest, current_user: dict = Depends(get_current_user)):
    """
    Intelligente KI-Suche für Kunden, Fahrzeuge und technische Fragen
    """
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")
    
    query = request.query.lower()
    results_data = []
    sources = []
    
    # First, search in database for relevant data
    # Search customers
    customer_keywords = ["kunde", "customer", "angelegt", "registriert", "kontakt"]
    if any(kw in query for kw in customer_keywords):
        # Extract potential name from query
        customers = await db.customers.find({}, {"_id": 0}).to_list(100)
        for customer in customers:
            if customer["name"].lower() in query or any(word in customer["name"].lower() for word in query.split() if len(word) > 2):
                results_data.append({
                    "type": "customer",
                    "id": customer["id"],
                    "name": customer["name"],
                    "details": customer.get("phone") or customer.get("email") or ""
                })
                sources.append("Kundendatenbank")
    
    # Search vehicles
    vehicle_keywords = ["fahrzeug", "auto", "wagen", "vehicle", "car", "vin", "fahrgestellnummer", "kennzeichen"]
    if any(kw in query for kw in vehicle_keywords):
        vehicles = await db.vehicles.find({}, {"_id": 0}).to_list(100)
        for vehicle in vehicles:
            vehicle_name = f"{vehicle.get('make', '')} {vehicle.get('model', '')}".strip()
            license_plate = vehicle.get("license_plate", "")
            vin = vehicle.get("vin", "")
            
            # Check if any vehicle info matches query
            if (vehicle_name.lower() in query or 
                license_plate.lower() in query or 
                any(word in vehicle_name.lower() for word in query.split() if len(word) > 2)):
                results_data.append({
                    "type": "vehicle",
                    "id": vehicle["id"],
                    "name": f"{vehicle_name} ({license_plate})" if license_plate else vehicle_name or "Fahrzeug",
                    "details": f"VIN: {vin}" if vin else f"Baujahr: {vehicle.get('year', 'k.A.')}"
                })
                sources.append("Fahrzeugdatenbank")
    
    # Build context for AI
    db_context = ""
    if results_data:
        db_context = "\n\nGefundene Daten in der Datenbank:\n"
        for item in results_data[:5]:  # Limit to 5 results
            db_context += f"- {item['type'].capitalize()}: {item['name']}"
            if item.get('details'):
                db_context += f" ({item['details']})"
            db_context += "\n"
    
    # Chat history context
    history_context = ""
    if request.chat_history:
        history_context = "\n\nVorherige Konversation:\n"
        for msg in request.chat_history[-4:]:  # Last 4 messages
            role = "Benutzer" if msg.get("role") == "user" else "Assistent"
            history_context += f"{role}: {msg.get('content', '')}\n"
    
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"ai-search-{current_user['id']}",
            system_message="""Du bist ein KFZ-Experte und Werkstatt-Assistent für die EasyWerkstatt-Software.
            
Du kannst:
1. Technische Fragen zu Fahrzeugen beantworten (Position von Teilen, Reparaturtipps, Fehlercodes)
2. Informationen aus der Werkstatt-Datenbank suchen (Kunden, Fahrzeuge)
3. Allgemeine KFZ-Fragen beantworten

Antworte immer auf Deutsch, präzise und hilfreich.
Wenn Datenbankeinträge gefunden wurden, beziehe dich darauf.
Bei technischen Fragen gib praktische, umsetzbare Tipps.
Wenn du dir unsicher bist, sage es ehrlich."""
        ).with_model("openai", "gpt-5.1")
        
        prompt = f"""Benutzeranfrage: {request.query}
{db_context}
{history_context}

Bitte beantworte die Anfrage hilfreich und präzise."""
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Add AI knowledge as source if no DB results
        if not sources:
            sources.append("KI-Wissen")
        
        return {
            "answer": response,
            "data": results_data[:5],  # Limit results
            "sources": list(set(sources))  # Unique sources
        }
        
    except Exception as e:
        logger.error(f"AI Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suche fehlgeschlagen: {str(e)}")

# ===================== DASHBOARD STATS =====================

@api_router.get("/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    customer_count = await db.customers.count_documents({})
    vehicle_count = await db.vehicles.count_documents({})
    vehicles_with_photos = await db.vehicles.count_documents({"photos.0": {"$exists": True}})
    analyzed_vehicles = await db.vehicles.count_documents({"ai_analysis": {"$ne": None}})
    
    return {
        "total_customers": customer_count,
        "total_vehicles": vehicle_count,
        "vehicles_with_photos": vehicles_with_photos,
        "analyzed_vehicles": analyzed_vehicles
    }

# ===================== ROOT & HEALTH =====================

@api_router.get("/")
async def root():
    return {"message": "EasyWerkstatt API", "version": "1.0.0"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create indexes for better performance
@app.on_event("startup")
async def create_indexes():
    """Create MongoDB indexes for better query performance"""
    try:
        # Customer indexes
        await db.customers.create_index("id", unique=True)
        await db.customers.create_index("name")
        await db.customers.create_index("email")
        await db.customers.create_index("created_at")
        
        # Vehicle indexes
        await db.vehicles.create_index("id", unique=True)
        await db.vehicles.create_index("customer_id")
        await db.vehicles.create_index("license_plate")
        await db.vehicles.create_index("vin")
        await db.vehicles.create_index([("make", 1), ("model", 1)])
        await db.vehicles.create_index("created_at")
        
        # User indexes
        await db.users.create_index("id", unique=True)
        await db.users.create_index("email", unique=True)
        
        # Service book indexes
        await db.servicebook.create_index("vehicle_id")
        await db.servicebook.create_index("date")
        await db.servicebook.create_index("entry_type")
        
        # Reminders indexes
        await db.reminders.create_index("vehicle_id")
        await db.reminders.create_index("customer_id")
        await db.reminders.create_index("due_date")
        await db.reminders.create_index("completed")
        
        # TÜV checklist index
        await db.tuv_checklists.create_index("vehicle_id", unique=True)
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
