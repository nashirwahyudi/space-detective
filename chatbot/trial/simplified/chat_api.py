# chat_api.py
"""
Chat API for Space Detective v2.0 - File-Based Version
Simple AI chat interface using file-based data loading
"""

import asyncio
import logging
import json
import re
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import h3

# External AI API imports (optional)
try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("Dashscope not available. Install with: pip install dashscope")

logger = logging.getLogger(__name__)


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
            try:
                lat, lng = h3.h3_to_geo(h3_index)
            except:
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
            'extracted_entities': {},
            'suggested_actions': []
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
        
        # Extract province names
        province_keywords = [
            'sumatera utara', 'dki jakarta', 'jawa barat', 'jawa tengah', 'jawa timur',
            'kalimantan', 'sulawesi', 'papua', 'bali', 'lombok'
        ]
        found_provinces = [prov for prov in province_keywords if prov in message_lower]
        if found_provinces:
            analysis['extracted_entities']['provinces'] = found_provinces
        
        return analysis


class ResponseGenerator:
    """Generates responses based on intent and data"""
    
    def __init__(self, data_loader: FileBasedDataLoader, alibaba_api_key: str = None):
        self.data_loader = data_loader
        self.alibaba_api_key = alibaba_api_key
        self.message_analyzer = MessageAnalyzer()
        
        # Setup Alibaba API if key provided
        if self.alibaba_api_key and DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.alibaba_api_key
    
    async def generate_response(self, message: str) -> Dict[str, Any]:
        """Generate response to user message"""
        try:
            # Analyze message
            analysis = self.message_analyzer.analyze_message(message)
            
            # Generate response based on intent
            response_text = await self._generate_intent_response(message, analysis)
            
            # Enhance with external AI if available
            if self.alibaba_api_key and DASHSCOPE_AVAILABLE:
                enhanced_response = await self._enhance_with_external_ai(message, response_text)
                if enhanced_response:
                    response_text = enhanced_response
            
            return {
                "response": response_text,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "data_loaded": self.data_loader.loaded
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"🚨 I encountered an error: {str(e)}. Please try rephrasing your question.",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_intent_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Generate response based on detected intent"""
        
        intent = analysis['primary_intent']
        entities = analysis['extracted_entities']
        
        if not self.data_loader.loaded:
            return self._generate_no_data_response()
        
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
        """Generate location-specific analysis response"""
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
- Status: **{location_info['anomaly_info']['anomaly_label'].upper()}** ⚠️

📊 **Key Indicators:**
- Built Growth: {location_info['indicators']['built_growth']:.2f}%
- Wealth Index (RWI): {location_info['indicators']['wealth_index']:.3f}
- Night Lights per Capita: {location_info['indicators']['night_lights']:.3f}
- Built per RWI: {location_info['indicators']['built_per_rwi']:.3f}

🔍 **Suspicious Patterns:**
{'- Extremely high building growth rate' if location_info['indicators']['built_growth'] > 50 else ''}
{'- Low wealth vs high development mismatch' if location_info['indicators']['wealth_index'] < 0 and location_info['indicators']['built_growth'] > 30 else ''}
{'- Rapid urban expansion' if location_info['indicators']['urban_growth'] > 0 else ''}
{'- Vegetation to built conversion' if location_info['indicators']['veg_to_built'] > 0 else ''}

🎯 **Recommended Actions:**
1. **Field Investigation**: Verify actual development vs permits
2. **Financial Analysis**: Trace funding sources for construction
3. **Ownership Check**: Investigate property ownership patterns
4. **Activity Monitoring**: Track ongoing construction activities

💡 Ask for SHAP analysis to understand feature contributions: "Explain why {h3_index} is anomalous"
"""
            else:
                response = f"""✅ **NORMAL AREA: {location_info['administrative']['village']}, {location_info['administrative']['regency']}**

📍 **Location Details:**
- H3 Index: {h3_index}
- Province: {location_info['administrative']['province']}
- Status: **{location_info['anomaly_info']['anomaly_label'].upper()}** ✅

📊 **Assessment:**
This area shows normal development patterns consistent with economic indicators.

📈 **Characteristics:**
- Built Growth: {location_info['indicators']['built_growth']:.2f}% (Normal range)
- Wealth Index: {location_info['indicators']['wealth_index']:.3f} (Consistent)
- Development Pattern: Balanced and expected

This area can serve as a baseline for comparison with suspicious locations in the region.
"""
            
            responses.append(response)
        
        return "\n\n---\n\n".join(responses)
    
    def _generate_anomaly_response(self) -> str:
        """Generate anomaly overview response"""
        stats = self.data_loader.get_anomaly_statistics()
        
        if not stats:
            return "❌ No anomaly data available."
        
        response = f"""🕵️ **ANOMALY DETECTION OVERVIEW**

📊 **Current Detection Status:**
- Total Locations Analyzed: {stats['total_locations']:,}
- Anomalies Detected: {stats['anomaly_count']:,}
- Detection Rate: {stats['anomaly_percentage']:.2f}%

🔍 **What Makes an Area Suspicious:**
1. **Rapid Building Growth** (>50% in recent years)
2. **Economic Mismatch** (High development, low wealth)
3. **Unusual Development Patterns** (Sudden urban expansion)
4. **Land Use Changes** (Vegetation to built conversion)

🏆 **Top Risk Provinces:**"""

        for i, prov in enumerate(stats['province_statistics'][:3], 1):
            risk_level = "🔴 HIGH" if prov['anomaly_percentage'] > 15 else "🟡 MEDIUM" if prov['anomaly_percentage'] > 5 else "🟢 LOW"
            response += f"\n{i}. **{prov['province']}**: {prov['anomaly_percentage']:.1f}% anomaly rate ({prov['anomaly_count']} locations) {risk_level}"

        if stats['top_anomalies']:
            response += "\n\n🚨 **Most Suspicious Areas:**"
            for i, area in enumerate(stats['top_anomalies'][:3], 1):
                urgency = "🔴 URGENT" if area['built_growth'] > 100 else "🟡 HIGH PRIORITY"
                response += f"\n{i}. **{area['location']}** - Growth: {area['built_growth']:.1f}% ({urgency})"
                response += f"\n   H3: {area['h3_index']}"

        response += "\n\n💡 **For Investigation**: Provide specific H3 indexes for detailed analysis."
        
        return response
    
    def _generate_statistics_response(self) -> str:
        """Generate comprehensive statistics response"""
        stats = self.data_loader.get_anomaly_statistics()
        
        if not stats:
            return "❌ No statistical data available."
        
        response = f"""📊 **COMPREHENSIVE ANOMALY STATISTICS**

🎯 **Overall Summary:**
- **Total Locations**: {stats['total_locations']:,}
- **Anomalies Found**: {stats['anomaly_count']:,}
- **Normal Areas**: {stats['total_locations'] - stats['anomaly_count']:,}
- **Anomaly Rate**: {stats['anomaly_percentage']:.2f}%

🏆 **Provincial Risk Ranking:**"""

        for i, prov in enumerate(stats['province_statistics'], 1):
            risk_level = "🔴 HIGH" if prov['anomaly_percentage'] > 15 else "🟡 MEDIUM" if prov['anomaly_percentage'] > 5 else "🟢 LOW"
            response += f"\n{i}. **{prov['province']}**: {prov['anomaly_percentage']:.1f}% ({prov['anomaly_count']}/{prov['total_locations']}) {risk_level}"

        if stats['top_anomalies']:
            response += "\n\n🚨 **Priority Investigation Targets:**"
            for i, area in enumerate(stats['top_anomalies'], 1):
                urgency = "🔴 URGENT" if area['built_growth'] > 100 else "🟡 HIGH"
                response += f"\n{i}. **{area['location']}** ({area['province']})"
                response += f"\n   - H3: {area['h3_index']}"
                response += f"\n   - Growth: {area['built_growth']:.1f}% | Priority: {urgency}"

        response += f"\n\n📈 **System Info:**"
        response += f"\n- AI Model: Isolation Forest (File-based)"
        response += f"\n- Data Source: Local CSV file"
        response += f"\n- Features: Geospatial + Economic indicators"
        
        return response
    
    def _generate_shap_response(self, entities: Dict[str, Any]) -> str:
        """Generate SHAP analysis response"""
        if 'h3_indexes' not in entities:
            return """🧠 **SHAP ANALYSIS SYSTEM**

**🤖 How SHAP Explains AI Decisions:**

SHAP (SHapley Additive exPlanations) reveals WHY the AI flagged an area as suspicious by showing each feature's contribution.

📊 **How to Read SHAP Values:**
- **Positive Values**: Push toward "anomaly" prediction
- **Negative Values**: Push toward "normal" prediction
- **Value Magnitude**: Strength of influence
- **Final Score**: Combined result of all features

🔍 **Key Features in Our Model:**
- `built_growth_pct_22_24`: Building expansion rate
- `RWI`: Relative Wealth Index
- `ntl_per_capita`: Night lights per capita
- `built_per_rwi`: Development vs wealth ratio
- `urban_growth`: Urban expansion indicator
- `veg_to_built`: Land use change patterns

💡 **To Get SHAP Analysis**: Provide an H3 index (e.g., "Explain why 876529d53ffffff is anomalous")
"""
        
        # Get SHAP analysis for the first H3 index
        h3_index = entities['h3_indexes'][0]
        location_info = self.data_loader.get_location_info(h3_index)
        
        if not location_info:
            return f"❌ Location {h3_index} not found for SHAP analysis."
        
        try:
            # Get feature data for SHAP
            row = self.data_loader.training_data[self.data_loader.training_data['h3_index'] == h3_index].iloc[0]
            feature_columns = [col for col in self.data_loader.training_data.columns 
                             if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            
            if not feature_columns:
                return "❌ No feature columns found for SHAP analysis."
            
            # Prepare features for SHAP
            features = row[feature_columns].fillna(0).values.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.data_loader.explainer.shap_values(features)
            
            # Prepare SHAP analysis
            shap_data = []
            for i, feature in enumerate(feature_columns):
                shap_data.append({
                    "feature": feature,
                    "value": float(row[feature]) if not pd.isna(row[feature]) else 0,
                    "shap_value": float(shap_values[0][i]),
                    "impact": "increases anomaly likelihood" if shap_values[0][i] > 0 else "decreases anomaly likelihood",
                    "abs_importance": abs(float(shap_values[0][i]))
                })
            
            # Sort by absolute importance
            shap_data.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            response = f"""🧠 **SHAP ANALYSIS for {location_info['administrative']['village']}, {location_info['administrative']['regency']}**

📍 **Location**: {h3_index}
🎯 **Status**: {location_info['anomaly_info']['anomaly_label']}
🔢 **Base Value**: {self.data_loader.explainer.expected_value:.3f}

📊 **Feature Contributions (Top 10):**
"""
            
            for i, item in enumerate(shap_data[:10], 1):
                direction = "→ ANOMALY" if item['shap_value'] > 0 else "→ NORMAL"
                response += f"\n{i}. **{item['feature']}**: {item['value']:.3f}"
                response += f"\n   SHAP: {item['shap_value']:+.3f} {direction}"
                response += f"\n   Impact: {item['impact']}\n"
            
            # Interpretation
            positive_features = [item for item in shap_data if item['shap_value'] > 0]
            negative_features = [item for item in shap_data if item['shap_value'] < 0]
            
            response += f"\n🔍 **Interpretation:**"
            response += f"\n- **{len(positive_features)} features** push toward ANOMALY"
            response += f"\n- **{len(negative_features)} features** push toward NORMAL"
            
            if positive_features:
                top_positive = positive_features[0]
                response += f"\n- **Strongest anomaly indicator**: {top_positive['feature']} ({top_positive['shap_value']:+.3f})"
            
            return response
            
        except Exception as e:
            logger.error(f"SHAP analysis error: {e}")
            return f"❌ SHAP analysis failed for {h3_index}: {str(e)}"
    
    def _generate_help_response(self) -> str:
        """Generate help response"""
        return """🤖 **SPACE DETECTIVE AI ASSISTANT - HELP**

**🔍 What I Can Help With:**

**1. Location Analysis**
- Paste any H3 index (e.g., "876529d53ffffff") for detailed analysis
- Get anomaly status and key indicators
- Understand why an area is flagged

**2. Statistical Insights**
- "What are the anomaly statistics?"
- "Show me the most suspicious provinces"
- "Which areas need investigation?"

**3. SHAP Explanations**
- "Explain why [H3 index] is anomalous"
- "Show feature importance for [location]"
- Understand AI decision factors

**4. Investigation Support**
- Get recommended actions for suspicious areas
- Understand red flags and patterns
- Compare normal vs anomalous areas

**💬 Example Questions:**
- "Analyze 876529d53ffffff"
- "What are the most suspicious areas?"
- "Show me anomaly statistics"
- "Explain why this area is flagged"
- "Which provinces have highest anomaly rates?"

**🎯 Pro Tips:**
- Be specific with H3 indexes for detailed analysis
- Ask for SHAP analysis to understand AI decisions
- Use "statistics" or "overview" for general insights

**📊 Current Data Status:**
- Training Data: {'✅ Loaded' if self.data_loader.loaded else '❌ Not Loaded'}
- ML Model: {'✅ Ready' if self.data_loader.model else '❌ Not Ready'}
- SHAP Explainer: {'✅ Available' if self.data_loader.explainer else '❌ Not Available'}

**Ready to help with your money laundering detection analysis! 🚀**
"""
    
    def _generate_general_response(self) -> str:
        """Generate general response"""
        stats = self.data_loader.get_anomaly_statistics()
        
        response = f"""🛰️ **SPACE DETECTIVE AI ASSISTANT**

I'm here to help you analyze satellite data for money laundering detection!

📊 **Current Analysis Status:**
- {stats.get('total_locations', 0):,} locations analyzed
- {stats.get('anomaly_count', 0):,} anomalies detected
- {stats.get('anomaly_percentage', 0):.2f}% anomaly rate
- File-based data loading active

🔍 **I Can Help You With:**
- **Location Analysis**: Provide H3 indexes for detailed investigation
- **Anomaly Investigation**: Understand why areas are flagged
- **Statistical Insights**: Get comprehensive analysis overview
- **SHAP Explanations**: Understand AI model decisions

💬 **Popular Queries:**
- "What are the most suspicious areas?"
- "Analyze [H3 index]"
- "Show me anomaly statistics"
- "Explain why [location] is anomalous"

**How can I assist with your investigation today?**
"""
        return response
    
    def _generate_no_data_response(self) -> str:
        """Generate response when no data is loaded"""
        return """🛰️ **SPACE DETECTIVE AI ASSISTANT**

⚠️ **Data Not Loaded**: Please ensure your data files are properly loaded.

🔧 **Required Files:**
- Training data CSV file
- Trained ML model (PKL file)
- SHAP explainer (PKL file)

📋 **To Load Data:**
1. Check file paths in configuration
2. Ensure files exist and are accessible
3. Restart the application

💡 **I Can Still Help With:**
- General information about money laundering detection
- Explanation of how the system works
- Guidance on data requirements

**Once data is loaded, I'll be ready for full analysis! 🚀**
"""
    
    async def _enhance_with_external_ai(self, original_message: str, current_response: str) -> Optional[str]:
        """Enhance response using Alibaba AI (optional)"""
        if not self.alibaba_api_key or not DASHSCOPE_AVAILABLE:
            return None
        
        try:
            system_prompt = """You are an AI assistant for Space Detective, helping investigators analyze satellite data for money laundering detection. Enhance the response with additional insights while keeping it professional and actionable."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original question: {original_message}\n\nCurrent analysis: {current_response}\n\nProvide additional insights:"}
            ]
            
            response = Generation.call(
                model='qwen-turbo',
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            if response.status_code == 200:
                enhanced_text = response.output.text
                return f"{current_response}\n\n**🤖 Enhanced AI Insights:**\n{enhanced_text}"
            
        except Exception as e:
            logger.warning(f"External AI enhancement failed: {e}")
        
        return None


class ChatAPI:
    """Main Chat API class"""
    
    def __init__(self, csv_path: str, model_path: str, explainer_path: str, alibaba_api_key: str = None):
        self.data_loader = FileBasedDataLoader(csv_path, model_path, explainer_path)
        self.response_generator = ResponseGenerator(self.data_loader, alibaba_api_key)
        self.conversation_history = []
    
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
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            })
            
            # Generate response
            response_data = await self.response_generator.generate_response(message)
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "message": response_data["response"],
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "analysis": response_data.get("analysis")
            })
            
            # Keep only recent history (last 20 messages)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response_data
            
        except Exception as e:
            error_response = {
                "response": f"🚨 Error processing your message: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
            return error_response
    
    def get_conversation_history(self, session_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if session_id:
            history = [msg for msg in self.conversation_history if msg.get("session_id") == session_id]
        else:
            history = self.conversation_history
        
        return history[-limit:] if limit else history
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        stats = self.data_loader.get_anomaly_statistics() if self.data_loader.loaded else {}
        
        return {
            "system_status": {
                "data_loaded": self.data_loader.loaded,
                "model_available": self.data_loader.model is not None,
                "explainer_available": self.data_loader.explainer is not None
            },
            "data_info": {
                "total_records": stats.get('total_locations', 0),
                "anomaly_count": stats.get('anomaly_count', 0),
                "anomaly_percentage": stats.get('anomaly_percentage', 0),
                "provinces": len(stats.get('province_statistics', []))
            },
            "conversation_stats": {
                "total_messages": len(self.conversation_history),
                "active_sessions": len(set(msg.get("session_id") for msg in self.conversation_history))
            },
            "capabilities": [
                "Location Analysis (H3 index)",
                "Anomaly Statistics",
                "SHAP Explanations", 
                "Investigation Guidance",
                "Provincial Risk Assessment"
            ]
        }
    
    def analyze_location(self, h3_index: str) -> Dict[str, Any]:
        """Direct location analysis method"""
        location_info = self.data_loader.get_location_info(h3_index)
        
        if not location_info:
            return {
                "error": True,
                "message": f"Location {h3_index} not found in dataset"
            }
        
        return {
            "success": True,
            "location_info": location_info,
            "analysis": {
                "is_anomaly": location_info['anomaly_info']['is_anomaly'],
                "risk_level": "HIGH" if location_info['anomaly_info']['is_anomaly'] else "NORMAL",
                "key_indicators": location_info['indicators'],
                "administrative_info": location_info['administrative']
            }
        }
    
    def get_shap_analysis(self, h3_index: str) -> Dict[str, Any]:
        """Get SHAP analysis for specific location"""
        if not self.data_loader.loaded or not self.data_loader.explainer:
            return {
                "error": True,
                "message": "SHAP explainer not available"
            }
        
        try:
            # Get location data
            location_data = self.data_loader.training_data[
                self.data_loader.training_data['h3_index'] == h3_index
            ]
            
            if location_data.empty:
                return {
                    "error": True,
                    "message": f"Location {h3_index} not found"
                }
            
            row = location_data.iloc[0]
            
            # Get feature columns
            feature_columns = [col for col in self.data_loader.training_data.columns 
                             if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            
            if not feature_columns:
                return {
                    "error": True,
                    "message": "No feature columns found for SHAP analysis"
                }
            
            # Prepare features for SHAP
            features = row[feature_columns].fillna(0).values.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.data_loader.explainer.shap_values(features)
            
            # Prepare SHAP data
            shap_data = []
            for i, feature in enumerate(feature_columns):
                shap_data.append({
                    "feature": feature,
                    "value": float(row[feature]) if not pd.isna(row[feature]) else 0,
                    "shap_value": float(shap_values[0][i]),
                    "impact": "increases anomaly likelihood" if shap_values[0][i] > 0 else "decreases anomaly likelihood",
                    "abs_importance": abs(float(shap_values[0][i]))
                })
            
            # Sort by absolute importance
            shap_data.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            return {
                "success": True,
                "h3_index": h3_index,
                "location": f"{row.get('nmdesa', 'Unknown')}, {row.get('nmkab', 'Unknown')}, {row.get('nmprov', 'Unknown')}",
                "is_anomaly": row.get('anomaly_score', 1) == -1,
                "base_value": float(self.data_loader.explainer.expected_value),
                "shap_analysis": shap_data,
                "top_factors": shap_data[:5],
                "explanation": "SHAP values show how each feature contributed to the anomaly prediction."
            }
            
        except Exception as e:
            return {
                "error": True,
                "message": f"SHAP analysis failed: {str(e)}"
            }


# Utility functions
def validate_h3_index(h3_index: str) -> bool:
    """Validate H3 index format"""
    if not isinstance(h3_index, str):
        return False
    
    # H3 index should be 15 characters long and start with '8'
    if len(h3_index) != 15 or not h3_index.startswith('8'):
        return False
    
    # Should contain only hexadecimal characters
    try:
        int(h3_index[1:], 16)
        return True
    except ValueError:
        return False


def extract_h3_indexes(text: str) -> List[str]:
    """Extract all H3 indexes from text"""
    h3_pattern = r'\b87[0-9a-f]{10,13}\b'
    matches = re.findall(h3_pattern, text, re.IGNORECASE)
    return [match for match in matches if validate_h3_index(match)]


# Example usage and testing
async def main():
    """Example usage of the Chat API"""
    
    # File paths - adjust these to your actual file locations
    csv_path = "sample_data.csv"
    model_path = "model.pkl"
    explainer_path = "explainer.pkl"
    alibaba_api_key = None  # Optional: add your Alibaba API key here
    
    print("🛰️ Space Detective Chat API - File-Based Version")
    print("=" * 50)
    
    # Initialize Chat API
    chat_api = ChatAPI(csv_path, model_path, explainer_path, alibaba_api_key)
    
    if not chat_api.initialize():
        print("❌ Failed to initialize Chat API. Check your file paths:")
        print(f"  - CSV: {csv_path}")
        print(f"  - Model: {model_path}")
        print(f"  - Explainer: {explainer_path}")
        return
    
    # Get system info
    system_info = chat_api.get_system_info()
    print(f"✅ System initialized successfully!")
    print(f"📊 Data: {system_info['data_info']['total_records']} records")
    print(f"🚨 Anomalies: {system_info['data_info']['anomaly_count']} ({system_info['data_info']['anomaly_percentage']:.2f}%)")
    
    # Test conversation
    test_messages = [
        "Hello, what can you help me with?",
        "What are the anomaly statistics?",
        "Show me the most suspicious areas",
        "What makes an area suspicious?",
        "How does SHAP analysis work?"
    ]
    
    print(f"\n🧪 Testing conversation:")
    print("-" * 30)
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        
        response = await chat_api.chat(message, session_id="test_session")
        
        print(f"🤖 Assistant: {response['response'][:200]}...")
        
        if 'analysis' in response:
            intent = response['analysis'].get('primary_intent', 'unknown')
            confidence = response['analysis'].get('confidence', 0)
            print(f"   Intent: {intent} (confidence: {confidence:.2f})")
    
    # Test direct location analysis (if you have real H3 indexes in your data)
    print(f"\n🔍 Testing location analysis:")
    print("-" * 30)
    
    # Get a sample H3 index from the data
    if chat_api.data_loader.loaded and not chat_api.data_loader.training_data.empty:
        sample_h3 = chat_api.data_loader.training_data['h3_index'].iloc[0]
        print(f"Analyzing sample location: {sample_h3}")
        
        location_analysis = chat_api.analyze_location(sample_h3)
        if location_analysis.get('success'):
            print(f"✅ Location: {location_analysis['location_info']['administrative']['village']}")
            print(f"🎯 Status: {location_analysis['analysis']['risk_level']}")
            print(f"📊 Growth: {location_analysis['analysis']['key_indicators']['built_growth']:.2f}%")
        
        # Test SHAP analysis
        shap_analysis = chat_api.get_shap_analysis(sample_h3)
        if shap_analysis.get('success'):
            print(f"🧠 SHAP: Top factor - {shap_analysis['top_factors'][0]['feature']}")
    
    print(f"\n🎉 Chat API testing completed!")
    print(f"📈 Total messages: {len(chat_api.conversation_history)}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())


# =============================================================================
# SIMPLE SETUP SCRIPT
# =============================================================================

def create_simple_setup():
    """Create a simple setup script for easy usage"""
    
    setup_script = '''# simple_chat_setup.py
"""
Simple setup script for Space Detective Chat API
Edit the file paths below and run this script
"""

import asyncio
from chat_api import ChatAPI

async def run_chat():
    """Run the chat API with your data"""
    
    # 🔧 EDIT THESE PATHS TO YOUR ACTUAL FILES
    CSV_PATH = "sample_data.csv"           # Your training data CSV
    MODEL_PATH = "model.pkl"               # Your trained model PKL
    EXPLAINER_PATH = "explainer.pkl"       # Your SHAP explainer PKL
    ALIBABA_API_KEY = None                 # Optional: Your Alibaba API key
    
    print("🛰️ Starting Space Detective Chat API...")
    
    # Initialize
    chat_api = ChatAPI(CSV_PATH, MODEL_PATH, EXPLAINER_PATH, ALIBABA_API_KEY)
    
    if not chat_api.initialize():
        print("❌ Failed to load data files. Please check your paths:")
        print(f"  CSV: {CSV_PATH}")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Explainer: {EXPLAINER_PATH}")
        return
    
    print("✅ Chat API ready!")
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\\n💬 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            response = await chat_api.chat(user_input)
            print(f"🤖 Assistant: {response['response']}")
            
        except KeyboardInterrupt:
            print("\\n👋 Chat ended by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_chat())
'''
    
    with open("simple_chat_setup.py", "w") as f:
        f.write(setup_script)
    
    print("📄 Created simple_chat_setup.py")
    print("✏️  Edit the file paths in the script and run: python simple_chat_setup.py")


# Create the simple setup script when this module is imported
if __name__ == "__main__":
    create_simple_setup()