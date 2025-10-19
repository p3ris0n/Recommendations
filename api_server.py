"""
FastAPI Server for UKFoodSaver Recommendation System - FIXED VERSION
Replace your api_server.py with this
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Literal
from contextlib import asynccontextmanager
from datetime import datetime
import pandas as pd
import uvicorn
import os

# Import your recommender system
from interaction_based_recommender import (
    UKFoodSaverRecommender,
    load_interaction_data,
    INTERACTION_WEIGHTS
)

# ============================================================================
# LIFESPAN CONTEXT MANAGER (Replaces deprecated startup/shutdown events)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # STARTUP
    print("=" * 70)
    print("UKFoodSaver Recommendations API Starting...")
    print("=" * 70)
    
    try:
        # Check for data files
        data_files = [
            'data/interactions.csv',
            'data/UKFS_testdata.csv',
            './interactions.csv',
            './UKFS_testdata.csv'
        ]
        
        data_file = None
        for file_path in data_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                data_file = file_path
                break
        
        if data_file:
            print(f"Loading data from: {data_file}")
            
            if 'UKFS_testdata' in data_file:
                df = pd.read_csv(data_file)
                if 'rating' in df.columns and 'interaction_type' not in df.columns:
                    df['interaction_type'] = df['rating'].apply(
                        lambda x: 'purchase' if x >= 1.5 else 'view'
                    )
                    df['timestamp'] = pd.Timestamp.now()
                interactions_df = load_interaction_data(df=df)
            else:
                interactions_df = load_interaction_data(data_file)
            
            recommender.train(interactions_df)
            print(f"✓ Model trained on {len(interactions_df)} interactions")
        else:
            print("⚠️  No valid data files found. Model starting untrained.")
            print("   Use POST /train to initialize the model")
            
    except Exception as e:
        print(f"⚠️  Startup training failed: {e}")
        print("   Model starting untrained. Use /train endpoint.")
    
    print("=" * 70)
    print("✓ API Ready!")
    print("=" * 70)
    
    yield  # App runs here
    
    # SHUTDOWN
    print("Shutting down...")

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="UKFoodSaver Recommendations API",
    description="Recommendation system for food marketplace platform",
    version="1.0.0",
    lifespan=lifespan  # Use lifespan instead of events
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = UKFoodSaverRecommender()

# ============================================================================
# PYDANTIC MODELS (Fixed protected namespace warning)
# ============================================================================

class InteractionLog(BaseModel):
    user_id: str
    item_id: str
    interaction_type: Literal['view', 'add_to_cart', 'purchase']
    timestamp: Optional[datetime] = None

class ItemMetadataInput(BaseModel):
    item_id: str
    postal_code: str
    keywords: List[str]
    store_id: str
    created_at: Optional[datetime] = None

class RecommendationResponse(BaseModel):
    type: str
    user_id: Optional[str] = None
    recommendations: List[dict]
    postal_code: Optional[str] = None
    keyword: Optional[str] = None
    count: int

class ComplementaryResponse(BaseModel):
    item_id: str
    complementary_items: List[dict]
    count: int

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # Fix warning
    
    status: str
    model_trained: bool
    last_train_time: Optional[str]
    total_interactions: int
    total_users: int
    total_items: int

# ============================================================================
# API ENDPOINTS (Keep all your existing endpoints exactly as they are)
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API welcome endpoint"""
    return {
        "message": "UKFoodSaver Recommendations API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ... (Keep ALL your other endpoints exactly as they are in your original file)
# Just copy them from your original api_server.py

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )