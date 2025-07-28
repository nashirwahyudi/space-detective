# standalone_api.py
"""
Standalone FastAPI for Space Detective Chat API
Self-contained API that includes chat logic and endpoints
"""

import asyncio
import logging
import os
import pickle
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Optional imports
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    print("h3-py not available. Install with: pip install h3-py")

try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADER CLASS
# =============================================================================

class FileBasedDataLoader:
    """Simple file-based data loader"""
    
    def __init__(self, csv_path: str, model_path: str, explainer_path: str):
        self.csv_path = csv_path
        self.model_path = model_path
        self.explainer_path = explainer_path
        
        # Loaded data
        self.training_data = None
        self.model = None
        self.explainer = None
        self.loaded = False
    
    def load_all(self) -> bool:
        """Load all data files"""
        try:
            # Check if files exist
            for file_path, name in [
                (self.csv_path, "CSV"),
                (self.model_path, "Model"),
                (self.explainer_path, "Explainer")
            ]:
                if not os.path.exists(file_path):
                    logger.error(f"{name} file not found: {file_path}")
                    return False
            
            # Load training data
            logger.info(f"Loading training data from {self.csv_path}")
            self.training_data = pd.read_csv(self.csv_path)
            logger.info(f"✅ Training data loaded: {self.training_data.shape}")
            
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load explainer
            logger.info(f"Loading explainer from {self.explainer_path}")
            with open(self.explainer_path, 'rb') as f:
                self.explainer = pickle.load(f)
            logger.info(f"✅ Explainer loaded: {type(self.explainer).__name__}")
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly statistics from training data"""
        if not self.loaded or self.training_data is None:
            return {}
        
        df = self.training_data
        total_records = len(df)
        anomaly_records = len(df[df['anomaly_score'] == -1]) if 'anomaly_score' in df.columns else 0
        
        # Province statistics
        province_stats = []
        if 'nmprov' in df.columns:
            for province in df['nmprov'].unique():
                province_data = df[df['nmprov'] == province]
                province_anomalies = len(province_data[province_data['anomaly_score'] == -1])
                province_total = len(province_data)
                
                province_stats.append({
                    "province": province,
                    "total_locations": province_total,
                    "anomaly_count": province_anomalies,
                    "anomaly_percentage": (province_anomalies / province_total * 100) if province_total > 0 else 0
                })
        
        # Top anomalous areas
        top_anomalies = []
        if anomaly_records > 0 and 'built_growth_pct_22_24' in df.columns:
            anomaly_data = df[df['anomaly_score'] == -1]
            if not anomaly_data.empty:
                top_areas = anomaly_data.nlargest(5, 'built_growth_pct_22_24')
                
                for _, row in top_areas.iterrows():
                    top_anomalies.append({
                        "h3_index": row['h3_index'],
                        "location": f"{row.get('nmdesa', 'Unknown')}, {row.get('nmkab', 'Unknown')}",
                        "province": row.get('nmprov', 'Unknown'),
                        "built_growth": float(row.get('built_growth_pct_22_24', 0)),
                        "wealth_index": float(row.get('RWI', 0))
                    })
        
        return {
            "total_locations": total_records,
            "anomaly_count": anomaly_records,
            "anomaly_percentage": (anomaly_records / total_records * 100) if total_records > 0 else 0,
            "province_statistics": sorted(province_stats, key=lambda x: x['anomaly_percentage'], reverse=True)[:5],
            "top_anomalies": top_anomalies
        }
    
    def get_location_info(self, h3_index: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific H3 location"""
        if not self.loaded or self.training_data is None:
            return None
        
        try:
            location_data = self.training_data[self.training_data['h3_index'] == h3_index]
            if location_data.empty:
                return None
            
            row = location_data.iloc[0]
            
            # Get coordinates
            lat, lng = None, None
            if H3_AVAILABLE:
                try:
                    lat, lng = h3.h3_to_geo(h3_index)
                except:
                    lat, lng = row.get('centroid_lat'), row.get('centroid_long')
            else:
                lat, lng = row.get('centroid_lat'), row.get('centroid_long')
            
            return {
                'h3_index': h3_index,
                'coordinates': {'latitude': lat, 'longitude': lng},
                'administrative': {
                    'province': row.get('nmprov', 'Unknown'),
                    'regency': row.get('nmkab', 'Unknown'),
                    'district': row.get('nmkec', 'Unknown'),
                    'village': row.get('nmdesa', 'Unknown')
                },
                'anomaly_info': {
                    'is_anomaly': row.get('anomaly_score', 1) == -1,
                    'anomaly_score': float(row.get('anomaly_score', 0)),
                    'anomaly_label': row.get('anomaly_label', 'Normal')
                },
                'indicators': {
                    'built_growth': float(row.get('built_growth_pct_22_24', 0)),
                    'wealth_index': float(row.get('RWI', 0)),
                    'night_lights': float(row.get('ntl_per_capita', 0)),
                    'built_per_rwi': float(row.get('built_per_rwi', 0)),
                    'urban_growth': float(row.get('urban_growth', 0)),
                    'veg_to_built': float(row.get('veg_to_built', 0))
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting location info for {h3_index}: {e}")
            return None


# =============================================================================
# MESSAGE ANALYZER
# =============================================================================

class MessageAnalyzer:
    """Analyzes user messages to determine intent"""
    
    def __init__(self):
        self.intent_patterns = {
            'location_analysis': [
                r'\b87[0-9a-f]{10,13}\b',  # H3 index pattern
                r'analiz.*lokasi',
                r'analyze.*location',
                r'location.*analysis'
            ],
            'anomaly_inquiry': [
                r'anomal[iy]',
                r'suspicious',
                r'mencurigakan',
                r'red flag',
                r'investigation'
            ],
            'statistics_request': [
                r'statist[ik]',
                r'summary',
                r'overview',
                r'ringkasan',
                r'top.*anomal',
                r'worst.*area',
                r'most.*suspicious'
            ],
            'help_request': [
                r'help',
                r'bantuan',
                r'how.*to',
                r'cara.*',
                r'panduan',
                r'guide'
            ],
            'shap_analysis': [
                r'shap',
                r'explain.*why',
                r'interpretasi',
                r'jelaskan.*mengapa',
                r'feature.*importance'
            ]
        }
    
    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze message and return intent and extracted information"""
        message_lower = message.lower()
        
        analysis = {
            'primary_intent': 'general_inquiry',
            'confidence': 0.0,
            'extracted_entities': {}
        }
        
        # Check for intents
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            intent_scores[intent] = score
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[primary_intent] > 0:
                analysis['primary_intent'] = primary_intent
                analysis['confidence'] = min(intent_scores[primary_intent] / 3.0, 1.0)
        
        # Extract H3 indexes
        h3_pattern = r'\b87[0-9a-f]{10,13}\b'
        h3_matches = re.findall(h3_pattern, message, re.IGNORECASE)
        if h3_matches:
            analysis['extracted_entities']['h3_indexes'] = h3_matches
        
        return analysis


# =============================================================================
# CHAT API CLASS
# =============================================================================

class ChatAPI:
    """Main Chat API class"""
    
    def __init__(self, csv_path: str, model_path: str, explainer_path: str, alibaba_api_key: str = None):
        self.data_loader = FileBasedDataLoader(csv_path, model_path, explainer_path)
        self.message_analyzer = MessageAnalyzer()
        self.conversation_history = []
        self.alibaba_api_key = alibaba_api_key
        
        # Setup Alibaba API if key provided
        if self.alibaba_api_key and DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.alibaba_api_key
    
    def initialize(self) -> bool:
        """Initialize the chat API by loading data"""
        logger.info("Initializing Space Detective Chat API...")
        success = self.data_loader.load_all()
        
        if success:
            logger.info("✅ Chat API initialized successfully!")
        else:
            logger.error("❌ Chat API initialization failed!")
        
        return success
    
    async def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Process user message and return response"""
        try:
            # Analyze message
            analysis = self.message_analyzer.analyze_message(message)
            
            # Generate response based on intent
            response_text = self._generate_response(message, analysis)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            })
            
            self.conversation_history.append({
                "role": "assistant", 
                "message": response_text,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "analysis": analysis
            })
            
            # Keep only recent history
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                "response": response_text,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "data_loaded": self.data_loader.loaded
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "response": f"🚨 Error processing your message: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Generate response based on intent"""
        intent = analysis['primary_intent']
        entities = analysis['extracted_entities']
        
        if not self.data_loader.loaded:
            return """🛰️ **SPACE DETECTIVE AI ASSISTANT**

⚠️ **Data Not Loaded**: Training data, model, or explainer files could not be loaded.

🔧 **Please Check:**
- File paths are correct
- Files exist and are accessible
- CSV format is valid
- PKL files are not corrupted

📋 **Current File Paths:**
- CSV: """ + self.data_loader.csv_path + """
- Model: """ + self.data_loader.model_path + """
- Explainer: """ + self.data_loader.explainer_path + """

**Once files are loaded properly, I'll be ready for analysis! 🚀**"""
        
        if intent == 'location_analysis':
            return self._generate_location_response(entities)
        elif intent == 'anomaly_inquiry':
            return self._generate_anomaly_response()
        elif intent == 'statistics_request':
            return self._generate_statistics_response()
        elif intent == 'shap_analysis':
            return self._generate_shap_response(entities)
        elif intent == 'help_request':
            return self._generate_help_response()
        else:
            return self._generate_general_response()
    
    def _generate_location_response(self, entities: Dict[str, Any]) -> str:
        """Generate location analysis response"""
        if 'h3_indexes' not in entities:
            return "❌ Please provide a valid H3 index for location analysis (e.g., 876529d53ffffff)."
        
        responses = []
        for h3_index in entities['h3_indexes']:
            location_info = self.data_loader.get_location_info(h3_index)
            
            if not location_info:
                responses.append(f"❌ Location {h3_index} not found in dataset.")
                continue
            
            if location_info['anomaly_info']['is_anomaly']:
                response = f"""🚨 **ANOMALY DETECTED: {location_info['administrative']['village']}, {location_info['administrative']['regency']}**

📍 **Location Details:**
- H3 Index: {h3_index}
- Province: {location_info['administrative']['province']}
- Status: **ANOMALY** ⚠️

📊 **Key Indicators:**
- Built Growth: {location_info['indicators']['built_growth']:.2f}%
- Wealth Index: {location_info['indicators']['wealth_index']:.3f}
- Night Lights per Capita: {location_info['indicators']['night_lights']:.3f}

🔍 **Why Suspicious:**
{'- Extremely high building growth' if location_info['indicators']['built_growth'] > 50 else ''}
{'- Economic mismatch detected' if location_info['indicators']['wealth_index'] < 0 and location_info['indicators']['built_growth'] > 30 else ''}
{'- Rapid urban development' if location_info['indicators']['urban_growth'] > 0 else ''}

🎯 **Recommended Actions:**
1. Field investigation and permit verification
2. Financial source analysis
3. Ownership pattern investigation
4. Continuous monitoring

💡 Ask for SHAP analysis: "Explain why {h3_index} is anomalous"
"""
            else:
                response = f"""✅ **NORMAL AREA: {location_info['administrative']['village']}, {location_info['administrative']['regency']}**

📍 **Location Details:**
- H3 Index: {h3_index}
- Province: {location_info['administrative']['province']}
- Status: **NORMAL** ✅

📊 **Characteristics:**
- Built Growth: {location_info['indicators']['built_growth']:.2f}% (Normal)
- Wealth Index: {location_info['indicators']['wealth_index']:.3f} (Consistent)
- Development: Balanced pattern

This area shows expected development patterns for the region.
"""
            
            responses.append(response)
        
        return "\n\n---\n\n".join(responses)
    
    def _generate_anomaly_response(self) -> str:
        """Generate anomaly overview"""
        stats = self.data_loader.get_anomaly_statistics()
        
        if not stats:
            return "❌ No anomaly data available."
        
        response = f"""🕵️ **ANOMALY DETECTION OVERVIEW**

📊 **Detection Status:**
- Total Locations: {stats['total_locations']:,}
- Anomalies Found: {stats['anomaly_count']:,}
- Detection Rate: {stats['anomaly_percentage']:.2f}%

🔍 **What Makes Areas Suspicious:**
1. Rapid building growth (>50%)
2. Economic mismatches (low wealth, high development)
3. Unusual development patterns
4. Land use changes

🏆 **High-Risk Provinces:**"""

        for i, prov in enumerate(stats['province_statistics'][:3], 1):
            risk = "🔴 HIGH" if prov['anomaly_percentage'] > 15 else "🟡 MEDIUM" if prov['anomaly_percentage'] > 5 else "🟢 LOW"
            response += f"\n{i}. {prov['province']}: {prov['anomaly_percentage']:.1f}% ({prov['anomaly_count']} locations) {risk}"

        if stats['top_anomalies']:
            response += "\n\n🚨 **Most Suspicious Areas:**"
            for i, area in enumerate(stats['top_anomalies'][:3], 1):
                urgency = "🔴 URGENT" if area['built_growth'] > 100 else "🟡 HIGH"
                response += f"\n{i}. {area['location']} - Growth: {area['built_growth']:.1f}% ({urgency})"

        return response
    
    def _generate_statistics_response(self) -> str:
        """Generate statistics response"""
        stats = self.data_loader.get_anomaly_statistics()
        
        if not stats:
            return "❌ No statistical data available."
        
        response = f"""📊 **COMPREHENSIVE STATISTICS**

🎯 **Overall Summary:**
- Total Locations: {stats['total_locations']:,}
- Anomalies: {stats['anomaly_count']:,}
- Normal Areas: {stats['total_locations'] - stats['anomaly_count']:,}
- Anomaly Rate: {stats['anomaly_percentage']:.2f}%

🏆 **Provincial Ranking:**"""

        for i, prov in enumerate(stats['province_statistics'], 1):
            risk = "🔴 HIGH" if prov['anomaly_percentage'] > 15 else "🟡 MEDIUM" if prov['anomaly_percentage'] > 5 else "🟢 LOW"
            response += f"\n{i}. {prov['province']}: {prov['anomaly_percentage']:.1f}% {risk}"

        if stats['top_anomalies']:
            response += "\n\n🚨 **Priority Targets:**"
            for i, area in enumerate(stats['top_anomalies'], 1):
                response += f"\n{i}. {area['location']} ({area['province']})"
                response += f"\n   H3: {area['h3_index']} | Growth: {area['built_growth']:.1f}%"

        return response
    
    def _generate_shap_response(self, entities: Dict[str, Any]) -> str:
        """Generate SHAP analysis response"""
        if 'h3_indexes' not in entities:
            return """🧠 **SHAP ANALYSIS SYSTEM**

SHAP explains WHY the AI flagged areas as suspicious.

📊 **How to Read SHAP:**
- Positive values → Push toward ANOMALY
- Negative values → Push toward NORMAL
- Magnitude → Strength of influence

🔍 **Key Features:**
- built_growth_pct_22_24: Building expansion
- RWI: Wealth index
- ntl_per_capita: Night lights
- built_per_rwi: Development vs wealth

💡 **Usage**: Provide H3 index for SHAP analysis
"""
        
        h3_index = entities['h3_indexes'][0]
        return self._get_shap_analysis(h3_index)
    
    def _generate_help_response(self) -> str:
        """Generate help response"""
        return f"""🤖 **SPACE DETECTIVE AI ASSISTANT - HELP**

**🔍 Available Commands:**

**1. Location Analysis**
- Paste H3 index: "876529d53ffffff"
- Get anomaly status and indicators

**2. Statistics & Overview**
- "What are the anomaly statistics?"
- "Show me the most suspicious areas"
- "Which provinces are high risk?"

**3. SHAP Explanations**
- "Explain why [H3 index] is anomalous"
- "Show feature importance"

**4. Investigation Support**
- Get recommended actions
- Understand risk patterns

**📊 System Status:**
- Data Loaded: {'✅ Yes' if self.data_loader.loaded else '❌ No'}
- Records: {self.data_loader.get_anomaly_statistics().get('total_locations', 0):,}
- Model: {'✅ Ready' if self.data_loader.model else '❌ Not Ready'}

**Ready to help with money laundering detection! 🚀**
"""
    
    def _generate_general_response(self) -> str:
        """Generate general response"""
        stats = self.data_loader.get_anomaly_statistics()
        
        return f"""🛰️ **SPACE DETECTIVE AI ASSISTANT**

I analyze satellite data to detect potential money laundering activities!

📊 **Current Status:**
- {stats.get('total_locations', 0):,} locations analyzed
- {stats.get('anomaly_count', 0):,} anomalies detected
- {stats.get('anomaly_percentage', 0):.2f}% anomaly rate

🔍 **I Can Help With:**
- Location analysis (provide H3 index)
- Anomaly statistics and insights
- SHAP explanations for AI decisions
- Investigation guidance

💬 **Try Asking:**
- "What are the most suspicious areas?"
- "Analyze [H3 index]"
- "Show me statistics"

**How can I assist your investigation?**
"""
    
    def _get_shap_analysis(self, h3_index: str) -> str:
        """Get SHAP analysis for location"""
        if not self.data_loader.explainer:
            return "❌ SHAP explainer not available."
        
        try:
            location_data = self.data_loader.training_data[
                self.data_loader.training_data['h3_index'] == h3_index
            ]
            
            if location_data.empty:
                return f"❌ Location {h3_index} not found for SHAP analysis."
            
            row = location_data.iloc[0]
            feature_columns = [col for col in self.data_loader.training_data.columns 
                             if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            
            if not feature_columns:
                return "❌ No feature columns found for SHAP analysis."
            
            # Prepare features
            features = row[feature_columns].fillna(0).values.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.data_loader.explainer.shap_values(features)
            
            # Format response
            response = f"""🧠 **SHAP ANALYSIS: {row.get('nmdesa', 'Unknown')}, {row.get('nmkab', 'Unknown')}**

📍 H3: {h3_index}
🎯 Status: {row.get('anomaly_label', 'Unknown')}

📊 **Top Feature Contributions:**
"""
            
            # Sort features by importance
            feature_importance = [(feature_columns[i], shap_values[0][i]) for i in range(len(feature_columns))]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (feature, shap_val) in enumerate(feature_importance[:8], 1):
                direction = "→ ANOMALY" if shap_val > 0 else "→ NORMAL"
                feature_value = row[feature] if not pd.isna(row[feature]) else 0
                response += f"\n{i}. {feature}: {feature_value:.3f}"
                response += f"\n   SHAP: {shap_val:+.3f} {direction}\n"
            
            return response
            
        except Exception as e:
            return f"❌ SHAP analysis failed: {str(e)}"
    
    def analyze_location(self, h3_index: str) -> Dict[str, Any]:
        """Direct location analysis"""
        location_info = self.data_loader.get_location_info(h3_index)
        
        if not location_info:
            return {"error": True, "message": f"Location {h3_index} not found"}
        
        return {
            "success": True,
            "location_info": location_info,
            "analysis": {
                "is_anomaly": location_info['anomaly_info']['is_anomaly'],
                "risk_level": "HIGH" if location_info['anomaly_info']['is_anomaly'] else "NORMAL",
                "key_indicators": location_info['indicators']
            }
        }
    
    def get_shap_analysis(self, h3_index: str) -> Dict[str, Any]:
        """Get SHAP analysis for API"""
        if not self.data_loader.explainer:
            return {"error": True, "message": "SHAP explainer not available"}
        
        try:
            location_data = self.data_loader.training_data[
                self.data_loader.training_data['h3_index'] == h3_index
            ]
            
            if location_data.empty:
                return {"error": True, "message": f"Location {h3_index} not found"}
            
            row = location_data.iloc[0]
            feature_columns = [col for col in self.data_loader.training_data.columns 
                             if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            
            features = row[feature_columns].fillna(0).values.reshape(1, -1)
            shap_values = self.data_loader.explainer.shap_values(features)
            
            shap_data = []
            for i, feature in enumerate(feature_columns):
                shap_data.append({
                    "feature": feature,
                    "value": float(row[feature]) if not pd.isna(row[feature]) else 0,
                    "shap_value": float(shap_values[0][i]),
                    "abs_importance": abs(float(shap_values[0][i]))
                })
            
            shap_data.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            return {
                "success": True,
                "h3_index": h3_index,
                "shap_analysis": shap_data,
                "top_factors": shap_data[:5]
            }
            
        except Exception as e:
            return {"error": True, "message": str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        stats = self.data_loader.get_anomaly_statistics()
        
        return {
            "system_status": {
                "data_loaded": self.data_loader.loaded,
                "model_available": self.data_loader.model is not None,
                "explainer_available": self.data_loader.explainer is not None
            },
            "data_info": {
                "total_records": stats.get('total_locations', 0),
                "anomaly_count": stats.get('anomaly_count', 0),
                "anomaly_percentage": stats.get('anomaly_percentage', 0)
            },
            "conversation_stats": {
                "total_messages": len(self.conversation_history)
            },
            "capabilities": [
                "Location Analysis", "Anomaly Statistics", 
                "SHAP Explanations", "Investigation Guidance"
            ]
        }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(default="default")
    include_analysis: bool = Field(default=False)

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    session_id: str
    data_loaded: bool
    analysis: Optional[Dict[str, Any]] = None
    error: bool = Field(default=False)

class LocationAnalysisRequest(BaseModel):
    h3_index: str = Field(..., min_length=15, max_length=15)
    include_shap: bool = Field(default=False)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, bool]


# =============================================================================
# FASTAPI APP
# =============================================================================

# Global chat API instance
chat_api_instance: Optional[ChatAPI] = None
startup_time = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chat_api_instance
    
    # Startup
    logger.info("🚀 Starting Space Detective API...")
    
    # File paths - EDIT THESE FOR YOUR FILES
    csv_path = os.getenv("CSV_PATH", "sample_data.csv")
    model_path = os.getenv("MODEL_PATH", "model.pkl")
    explainer_path = os.getenv("EXPLAINER_PATH", "explainer.pkl")
    alibaba_api_key = os.getenv("ALIBABA_API_KEY", 'sk-0d3d6181c41a40559b9e014269aa15c3')
    
    logger.info(f"📁 Loading data files:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Explainer: {explainer_path}")
    
    # Initialize chat API
    chat_api_instance = ChatAPI(csv_path, model_path, explainer_path, alibaba_api_key)
    
    if chat_api_instance.initialize():
        logger.info("✅ Space Detective API ready!")
    else:
        logger.error("❌ Failed to initialize API. Check file paths above.")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Space Detective API...")


# Create FastAPI app
app = FastAPI(
    title="Space Detective API",
    description="AI-powered money laundering detection via satellite intelligence",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility function
def validate_h3_index(h3_index: str) -> bool:
    """Validate H3 index format"""
    if not isinstance(h3_index, str) or len(h3_index) != 15:
        return False
    if not h3_index.startswith('8'):
        return False
    try:
        int(h3_index[1:], 16)
        return True
    except ValueError:
        return False


def get_chat_api() -> ChatAPI:
    """Get chat API instance"""
    if chat_api_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat API not initialized"
        )
    
    if not chat_api_instance.data_loader.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training data not loaded. Check file paths and restart."
        )
    
    return chat_api_instance


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🛰️ Space Detective API v2.0",
        "description": "AI-powered money laundering detection",
        "status": "operational",
        "endpoints": {
            "chat": "POST /api/chat",
            "location_analysis": "POST /api/analyze-location", 
            "shap_analysis": "POST /api/shap-analysis",
            "system_info": "GET /api/system-info",
            "health": "GET /api/health",
            "docs": "GET /docs"
        },
        "version": "2.0.0"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        chat_api = get_chat_api()
        
        response_data = await chat_api.chat(request.message, request.session_id)
        
        chat_response = ChatResponse(
            response=response_data["response"],
            timestamp=response_data["timestamp"],
            session_id=request.session_id,
            data_loaded=response_data.get("data_loaded", False),
            error=response_data.get("error", False)
        )
        
        if request.include_analysis and "analysis" in response_data:
            chat_response.analysis = response_data["analysis"]
        
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"🚨 Error: {str(e)}",
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            data_loaded=False,
            error=True
        )


@app.post("/api/analyze-location")
async def analyze_location_endpoint(request: LocationAnalysisRequest):
    """Analyze specific location"""
    try:
        if not validate_h3_index(request.h3_index):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid H3 index format"
            )
        
        chat_api = get_chat_api()
        
        analysis_result = chat_api.analyze_location(request.h3_index)
        
        if analysis_result.get("error"):
            return {
                "success": False,
                "h3_index": request.h3_index,
                "error_message": analysis_result["message"]
            }
        
        response = {
            "success": True,
            "h3_index": request.h3_index,
            "location_info": analysis_result["location_info"],
            "analysis": analysis_result["analysis"]
        }
        
        if request.include_shap:
            shap_result = chat_api.get_shap_analysis(request.h3_index)
            if shap_result.get("success"):
                response["shap_analysis"] = shap_result
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location analysis error: {e}")
        return {
            "success": False,
            "h3_index": request.h3_index,
            "error_message": str(e)
        }


@app.post("/api/shap-analysis")
async def shap_analysis_endpoint(h3_index: str, top_features: int = 10):
    """Get SHAP analysis"""
    try:
        if not validate_h3_index(h3_index):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid H3 index format"
            )
        
        chat_api = get_chat_api()
        
        shap_result = chat_api.get_shap_analysis(h3_index)
        
        if shap_result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=shap_result["message"]
            )
        
        # Limit features
        if "shap_analysis" in shap_result:
            shap_result["shap_analysis"] = shap_result["shap_analysis"][:top_features]
        
        return shap_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/system-info")
async def system_info_endpoint():
    """Get system information"""
    try:
        if chat_api_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat API not initialized"
            )
        
        system_info = chat_api_instance.get_system_info()
        uptime = str(datetime.now() - startup_time)
        
        return {
            "system_status": system_info["system_status"],
            "data_info": system_info["data_info"],
            "conversation_stats": system_info["conversation_stats"],
            "capabilities": system_info["capabilities"],
            "uptime": uptime,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/anomaly-statistics")
async def anomaly_statistics_endpoint():
    """Get anomaly statistics"""
    try:
        chat_api = get_chat_api()
        
        stats = chat_api.data_loader.get_anomaly_statistics()
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No statistics available"
            )
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "api": True,
        "chat_api": chat_api_instance is not None,
        "data_loaded": chat_api_instance.data_loader.loaded if chat_api_instance else False,
        "model_available": chat_api_instance.data_loader.model is not None if chat_api_instance else False,
        "explainer_available": chat_api_instance.data_loader.explainer is not None if chat_api_instance else False
    }
    
    overall_health = all(components.values())
    
    return HealthResponse(
        status="healthy" if overall_health else "degraded",
        timestamp=datetime.now().isoformat(),
        components=components
    )


@app.get("/api/sample-data")
async def sample_data_endpoint(limit: int = 10):
    """Get sample data for testing"""
    try:
        chat_api = get_chat_api()
        
        df = chat_api.data_loader.training_data
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data available"
            )
        
        # Get sample with essential columns
        essential_cols = ['h3_index', 'nmprov', 'nmkab', 'nmdesa', 'anomaly_score', 'built_growth_pct_22_24']
        available_cols = [col for col in essential_cols if col in df.columns]
        
        if not available_cols:
            available_cols = list(df.columns)[:6]  # First 6 columns as fallback
        
        sample_data = df.head(limit)[available_cols].to_dict('records')
        
        return {
            "success": True,
            "sample_data": sample_data,
            "total_records": len(df),
            "columns_shown": available_cols,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "Endpoint not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Space Detective Standalone API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--csv", help="Path to CSV file")
    parser.add_argument("--model", help="Path to model PKL file")
    parser.add_argument("--explainer", help="Path to explainer PKL file")
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.csv:
        os.environ["CSV_PATH"] = args.csv
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    if args.explainer:
        os.environ["EXPLAINER_PATH"] = args.explainer
    
    print("🛰️ Space Detective Standalone API")
    print("=" * 50)
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔧 Health: http://{args.host}:{args.port}/api/health")
    print(f"💬 Chat: POST http://{args.host}:{args.port}/api/chat")
    print("🛑 Stop with Ctrl+C")
    print()
    
    # Show file paths
    csv_path = os.getenv("CSV_PATH", "sample_data.csv")
    model_path = os.getenv("MODEL_PATH", "model.pkl")
    explainer_path = os.getenv("EXPLAINER_PATH", "explainer.pkl")
    
    print("📁 Data Files:")
    print(f"  CSV: {csv_path}")
    print(f"  Model: {model_path}")
    print(f"  Explainer: {explainer_path}")
    print()
    
    # Check if files exist
    missing_files = []
    for path, name in [(csv_path, "CSV"), (model_path, "Model"), (explainer_path, "Explainer")]:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("📝 Update file paths using:")
        print("   --csv /path/to/your/data.csv")
        print("   --model /path/to/your/model.pkl")
        print("   --explainer /path/to/your/explainer.pkl")
        print("\n🚀 Starting anyway (will show errors in API)...")
    
    print("\n🚀 Starting server...")
    
    # Run server
    uvicorn.run(
        "standalone_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


# =============================================================================
# QUICK TEST SCRIPT
# =============================================================================

def create_test_script():
    """Create a quick test script"""
    test_script = '''# test_api.py
"""
Quick test script for Space Detective API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/api/health")
    print("🔧 Health Check:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_chat():
    """Test chat endpoint"""
    response = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "What are the anomaly statistics?",
        "session_id": "test_session"
    })
    print("\\n💬 Chat Test:")
    print(f"Response: {response.json()['response'][:200]}...")
    return response.status_code == 200

def test_system_info():
    """Test system info"""
    response = requests.get(f"{BASE_URL}/api/system-info")
    print("\\n📊 System Info:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_sample_data():
    """Test sample data"""
    response = requests.get(f"{BASE_URL}/api/sample-data?limit=3")
    print("\\n📋 Sample Data:")
    if response.status_code == 200:
        data = response.json()
        print(f"Records: {len(data['sample_data'])}")
        if data['sample_data']:
            print(f"Sample H3: {data['sample_data'][0].get('h3_index', 'N/A')}")
    else:
        print(f"Error: {response.status_code}")
    return response.status_code == 200

if __name__ == "__main__":
    print("🧪 Testing Space Detective API...")
    print("=" * 40)
    
    tests = [
        ("Health Check", test_health),
        ("System Info", test_system_info), 
        ("Sample Data", test_sample_data),
        ("Chat", test_chat)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {e}"))
    
    print("\\n📋 Test Results:")
    for name, result in results:
        print(f"  {name}: {result}")
    
    success_count = len([r for r in results if "PASS" in r[1]])
    print(f"\\n🎯 {success_count}/{len(tests)} tests passed")
'''
    
    with open("test_api.py", "w") as f:
        f.write(test_script)
    
    print("📄 Created test_api.py - Run: python test_api.py")


# Create test script when module loads
if __name__ == "__main__":
    try:
        create_test_script()
    except:
        pass

## EXAMPLE USAGE
# // Chat with API
# fetch('http://localhost:8000/api/chat', {
#   method: 'POST',
#   headers: { 'Content-Type': 'application/json' },
#   body: JSON.stringify({
#     message: 'What are the most suspicious areas?',
#     session_id: 'user123',
#     include_analysis: true
#   })
# })
# .then(response => response.json())
# .then(data => console.log(data.response));

# // Analyze location
# fetch('http://localhost:8000/api/analyze-location', {
#   method: 'POST', 
#   headers: { 'Content-Type': 'application/json' },
#   body: JSON.stringify({
#     h3_index: '876529d53ffffff',
#     include_shap: true
#   })
# })
# .then(response => response.json())
# .then(data => console.log(data));