from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone
import os
import uuid
import jwt
import bcrypt
import logging

# ===================== ENV =====================

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME", "easywerkstatt")

JWT_SECRET = os.environ.get("JWT_SECRET", "easywerkstatt-secret")
JWT_ALGORITHM = "HS256"

FRONTEND_ORIGIN = "https://werkstatt-plus-frontend.vercel.app"

# ===================== APP =====================

app = FastAPI(title="EasyWerkstatt API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DB =====================

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# ===================== SECURITY =====================

security = HTTPBearer()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc).timestamp() + 60 * 60 * 24 * 7
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        user = await db.users.find_one({"id": payload["user_id"]}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ===================== ROUTER =====================

api = APIRouter(prefix="/api")

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
    notes: Optional[str] = None

class CustomerResponse(CustomerCreate):
    id: str
    created_at: str
    updated_at: str

# ===================== AUTH =====================

@api.post("/auth/register")
async def register(user: UserCreate):
    if await db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already exists")

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    doc = {
        "id": user_id,
        "email": user.email,
        "password": hash_password(user.password),
        "name": user.name,
        "role": user.role,
        "created_at": now
    }

    await db.users.insert_one(doc)

    return {
        "token": create_token(user_id, user.email, user.role),
        "user": {k: doc[k] for k in doc if k != "password"}
    }

@api.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "token": create_token(user["id"], user["email"], user["role"]),
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "created_at": user["created_at"]
        }
    }

@api.get("/auth/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    return user

# ===================== CUSTOMERS =====================

@api.post("/customers", response_model=CustomerResponse)
async def create_customer(
    customer: CustomerCreate,
    user: dict = Depends(get_current_user)
):
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "id": str(uuid.uuid4()),
        **customer.model_dump(),
        "created_at": now,
        "updated_at": now,
        "created_by": user["id"]
    }
    await db.customers.insert_one(doc)
    return doc

@api.get("/customers", response_model=List[CustomerResponse])
async def list_customers(user: dict = Depends(get_current_user)):
    return await db.customers.find({}, {"_id": 0}).to_list(1000)

# ===================== SYSTEM =====================

@api.get("/")
async def root():
    return {"message": "EasyWerkstatt API online"}

@api.get("/health")
async def health():
    return {"status": "ok"}

# ===================== INIT =====================

app.include_router(api)

@app.on_event("startup")
async def startup():
    await db.users.create_index("email", unique=True)
    await db.customers.create_index("id", unique=True)
    logging.info("Indexes ready")

@app.on_event("shutdown")
async def shutdown():
    client.close()
