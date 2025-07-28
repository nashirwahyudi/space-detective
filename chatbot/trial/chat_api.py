# chat_api.py
"""
Chat API for Space Detective v2.0
Handles AI-powered conversations with RAG integration and context-aware responses
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import h3

# External AI API imports
try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("Dashscope not available. Install with: pip install dashscope")

from database_manager import DatabaseManager
from rag_system import RAGSystem, SearchResult
from config import AppConfig

logger = logging.getLogger(__name__)


class ChatContext:
    """Manages conversation context and history"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[Dict[str, str]] = []
        self.context_data: Dict[str, Any] = {}
        self.max_history = max_history
        self.session_start = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Keep only recent messages
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_messages = self.conversation_history[-6:]  # Last 3 exchanges
        context_parts = []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                context_parts.append(f"User asked: {msg['content'][:100]}...")
            elif msg["role"] == "assistant":
                context_parts.append(f"Assistant responded about: {msg['content'][:100]}...")
        
        return "\n".join(context_parts)
    
    def update_context_data(self, key: str, value: Any):
        """Update context data"""
        self.context_data[key] = value
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """Get context data"""
        return self.context_data.get(key, default)


class MessageAnalyzer:
    """Analyzes user messages to determine intent and extract information"""
    
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
            'model_explanation': [
                r'shap',
                r'explain.*model',
                r'why.*detect',
                r'interpretasi',
                r'jelaskan.*model'
            ],
            'statistics_request': [
                r'statist[ik]',
                r'summary',
                r'overview',
                r'ringkasan',
                r'top.*anomal',
                r'worst.*area'
            ],
            'help_request': [
                r'help',
                r'bantuan',
                r'how.*to',
                r'cara.*',
                r'panduan',
                r'guide'
            ],
            'procedure_inquiry': [
                r'procedure',
                r'prosedur',
                r'investigation.*step',
                r'checklist',
                r'guidelines'
            ]
        }
    
    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze message and return intent and extracted information"""
        message_lower = message.lower()
        
        analysis = {
            'primary_intent': 'general_inquiry',
            'confidence': 0.0,
            'extracted_entities': {},
            'suggested_actions': [],
            'requires_rag': False,
            'requires_database': False
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
        
        # Extract entities
        analysis['extracted_entities'] = self._extract_entities(message)
        
        # Determine requirements
        analysis['requires_rag'] = self._requires_rag(analysis['primary_intent'])
        analysis['requires_database'] = self._requires_database(analysis['primary_intent'])
        
        # Suggest actions
        analysis['suggested_actions'] = self._suggest_actions(analysis)
        
        return analysis
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message"""
        entities = {}
        
        # Extract H3 indexes
        h3_pattern = r'\b87[0-9a-f]{10,13}\b'
        h3_matches = re.findall(h3_pattern, message, re.IGNORECASE)
        if h3_matches:
            entities['h3_indexes'] = h3_matches
        
        # Extract province names (Indonesian provinces)
        province_keywords = [
            'sumatera utara', 'dki jakarta', 'jawa barat', 'jawa tengah', 'jawa timur',
            'kalimantan', 'sulawesi', 'papua', 'bali', 'lombok'
        ]
        message_lower = message.lower()
        found_provinces = [prov for prov in province_keywords if prov in message_lower]
        if found_provinces:
            entities['provinces'] = found_provinces
        
        # Extract numbers that might be thresholds or limits
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(number_pattern, message)
        if numbers:
            entities['numbers'] = [float(n) if '.' in n else int(n) for n in numbers]
        
        return entities
    
    def _requires_rag(self, intent: str) -> bool:
        """Determine if intent requires RAG search"""
        rag_intents = {'help_request', 'procedure_inquiry', 'model_explanation'}
        return intent in rag_intents
    
    def _requires_database(self, intent: str) -> bool:
        """Determine if intent requires database access"""
        db_intents = {'location_analysis', 'anomaly_inquiry', 'statistics_request'}
        return intent in db_intents
    
    def _suggest_actions(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest actions based on analysis"""
        actions = []
        
        intent = analysis['primary_intent']
        entities = analysis['extracted_entities']
        
        if intent == 'location_analysis' and 'h3_indexes' in entities:
            actions.append('perform_location_analysis')
            actions.append('get_shap_explanation')
        
        if intent == 'anomaly_inquiry':
            actions.append('get_anomaly_statistics')
            actions.append('search_procedures')
        
        if intent == 'statistics_request':
            actions.append('generate_summary_statistics')
        
        if intent == 'procedure_inquiry':
            actions.append('search_investigation_procedures')
        
        return actions


class ResponseGenerator:
    """Generates context-aware responses"""
    
    def __init__(self, db_manager: DatabaseManager, rag_system: RAGSystem, config: AppConfig):
        self.db_manager = db_manager
        self.rag_system = rag_system
        self.config = config
        self.message_analyzer = MessageAnalyzer()
    
    async def generate_response(self, message: str, chat_context: ChatContext) -> Dict[str, Any]:
        """Generate comprehensive response to user message"""
        try:
            # Analyze the message
            analysis = self.message_analyzer.analyze_message(message)
            
            # Get context-specific data
            context_data = {}
            rag_results = []
            
            # RAG search if needed
            if analysis['requires_rag']:
                rag_results = await self._perform_rag_search(message, analysis)
                context_data['rag_results'] = rag_results
            
            # Database queries if needed
            if analysis['requires_database']:
                db_context = await self._get_database_context(analysis)
                context_data.update(db_context)
            
            # Generate response based on intent
            response_text = await self._generate_intent_response(
                message, analysis, context_data, chat_context
            )
            
            # Enhance with external AI if available
            if self.config.external_apis.alibaba_api_key and DASHSCOPE_AVAILABLE:
                enhanced_response = await self._enhance_with_external_ai(
                    message, response_text, context_data, chat_context
                )
                if enhanced_response:
                    response_text = enhanced_response
            
            # Add to conversation history
            chat_context.add_message("user", message, {"analysis": analysis})
            chat_context.add_message("assistant", response_text, {
                "intent": analysis['primary_intent'],
                "rag_used": len(rag_results) > 0,
                "db_accessed": analysis['requires_database']
            })
            
            return {
                "response": response_text,
                "analysis": analysis,
                "context_data": context_data,
                "rag_results_count": len(rag_results),
                "suggestions": self._generate_suggestions(analysis, context_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"🚨 I encountered an error processing your request: {str(e)}. Please try rephrasing your question.",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _perform_rag_search(self, message: str, analysis: Dict[str, Any]) -> List[SearchResult]:
        """Perform RAG search based on message analysis"""
        try:
            # Determine search strategy based on intent
            search_query = message
            document_types = None
            
            if analysis['primary_intent'] == 'procedure_inquiry':
                document_types = ['investigation_guide', 'procedure']
                search_query = f"investigation procedures {message}"
            elif analysis['primary_intent'] == 'model_explanation':
                document_types = ['technical_guide']
                search_query = f"model explanation SHAP {message}"
            elif analysis['primary_intent'] == 'help_request':
                document_types = ['investigation_guide', 'technical_guide']
            
            # Perform search
            results = await self.rag_system.search_documents(
                query=search_query,
                max_results=self.config.rag.max_search_results,
                similarity_threshold=self.config.rag.similarity_threshold,
                document_types=document_types
            )
            
            return results
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    async def _get_database_context(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant database context based on analysis"""
        context = {}
        
        try:
            entities = analysis['extracted_entities']
            
            # Load training data if needed
            if analysis['primary_intent'] in ['location_analysis', 'anomaly_inquiry', 'statistics_request']:
                try:
                    training_data, dataset_info = await self.db_manager.load_training_data("main_dataset")
                    context['training_data'] = training_data
                    context['dataset_info'] = dataset_info
                except Exception as e:
                    logger.warning(f"Could not load training data: {e}")
            
            # Get specific location data
            if 'h3_indexes' in entities and 'training_data' in context:
                location_data = {}
                for h3_index in entities['h3_indexes']:
                    location_info = await self._get_location_info(h3_index, context['training_data'])
                    if location_info:
                        location_data[h3_index] = location_info
                
                if location_data:
                    context['location_data'] = location_data
            
            # Get anomaly statistics
            if analysis['primary_intent'] == 'statistics_request' and 'training_data' in context:
                stats = await self._calculate_anomaly_statistics(context['training_data'])
                context['anomaly_statistics'] = stats
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting database context: {e}")
            return {}
    
    async def _get_location_info(self, h3_index: str, training_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific H3 location"""
        try:
            location_data = training_data[training_data['h3_index'] == h3_index]
            if location_data.empty:
                return None
            
            row = location_data.iloc[0]
            
            # Get coordinates
            try:
                lat, lng = h3.h3_to_geo(h3_index)
            except:
                lat, lng = None, None
            
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
                    'confidence': row.get('confidence_score', 0)
                },
                'indicators': {
                    'built_growth': float(row.get('built_growth_pct_22_24', 0)),
                    'wealth_index': float(row.get('RWI', 0)),
                    'night_lights': float(row.get('ntl_sumut_monthly_mean', 0)),
                    'population': float(row.get('WPOP2020_sum', 0)) if 'WPOP2020_sum' in row else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting location info for {h3_index}: {e}")
            return None
    
    async def _calculate_anomaly_statistics(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive anomaly statistics"""
        try:
            total_records = len(training_data)
            anomaly_records = len(training_data[training_data['anomaly_score'] == -1]) if 'anomaly_score' in training_data.columns else 0
            
            # Province statistics
            province_stats = []
            if 'nmprov' in training_data.columns:
                for province in training_data['nmprov'].unique():
                    province_data = training_data[training_data['nmprov'] == province]
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
            if anomaly_records > 0 and 'built_growth_pct_22_24' in training_data.columns:
                anomaly_data = training_data[training_data['anomaly_score'] == -1]
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
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    async def _generate_intent_response(self, message: str, analysis: Dict[str, Any], 
                                      context_data: Dict[str, Any], chat_context: ChatContext) -> str:
        """Generate response based on detected intent"""
        
        intent = analysis['primary_intent']
        entities = analysis['extracted_entities']
        
        if intent == 'location_analysis':
            return await self._generate_location_response(entities, context_data)
        
        elif intent == 'anomaly_inquiry':
            return await self._generate_anomaly_response(context_data)
        
        elif intent == 'statistics_request':
            return await self._generate_statistics_response(context_data)
        
        elif intent == 'model_explanation':
            return await self._generate_explanation_response(context_data)
        
        elif intent == 'procedure_inquiry':
            return await self._generate_procedure_response(context_data)
        
        elif intent == 'help_request':
            return await self._generate_help_response(context_data)
        
        else:
            return await self._generate_general_response(message, context_data, chat_context)
    
    async def _generate_location_response(self, entities: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """Generate location-specific analysis response"""
        if 'h3_indexes' in entities and 'location_data' in context_data:
            responses = []
            
            for h3_index in entities['h3_indexes']:
                if h3_index in context_data['location_data']:
                    location_info = context_data['location_data'][h3_index]
                    
                    if location_info['anomaly_info']['is_anomaly']:
                        response = f"""🚨 **ANOMALY ANALYSIS FOR {location_info['administrative']['village']}, {location_info['administrative']['regency']}**

📍 **Location Details:**
- H3 Index: {h3_index}
- Province: {location_info['administrative']['province']}
- Status: **ANOMALY DETECTED** ⚠️

📊 **Key Indicators:**
- Built Growth: {location_info['indicators']['built_growth']:.2f}%
- Wealth Index: {location_info['indicators']['wealth_index']:.3f}
- Night Lights: {location_info['indicators']['night_lights']:.3f}

🔍 **Why This Area is Suspicious:**
{'- Extremely high building growth rate' if location_info['indicators']['built_growth'] > 50 else ''}
{'- Economic mismatch (low wealth, high development)' if location_info['indicators']['wealth_index'] < 0 and location_info['indicators']['built_growth'] > 30 else ''}
{'- Unusual night-time activity patterns' if location_info['indicators']['night_lights'] > 5 else ''}

🎯 **Recommended Actions:**
1. **Administrative Check**: Verify building permits and ownership records
2. **Financial Investigation**: Trace funding sources for construction
3. **Field Surveillance**: Monitor ongoing activities
4. **Cross-reference**: Check against transaction databases

💡 **Next Steps**: Request SHAP analysis for detailed feature explanation."""
                    else:
                        response = f"""✅ **NORMAL AREA ANALYSIS FOR {location_info['administrative']['village']}, {location_info['administrative']['regency']}**

📍 **Location Details:**
- H3 Index: {h3_index}
- Province: {location_info['administrative']['province']}
- Status: **NORMAL** ✅

📊 **Assessment:**
This area shows normal development patterns consistent with economic indicators.

📈 **Characteristics:**
- Built Growth: {location_info['indicators']['built_growth']:.2f}% (Normal range)
- Wealth Index: {location_info['indicators']['wealth_index']:.3f} (Consistent)
- Night Activity: {location_info['indicators']['night_lights']:.3f} (Expected level)

This area can serve as a baseline for comparison with suspicious locations."""
                    
                    responses.append(response)
                else:
                    responses.append(f"❌ Location {h3_index} not found in current dataset.")
            
            return "\n\n---\n\n".join(responses)
        
        else:
            return "❌ No valid H3 indexes found or location data unavailable. Please provide a valid H3 index (e.g., 876529d53ffffff)."
    
    async def _generate_anomaly_response(self, context_data: Dict[str, Any]) -> str:
        """Generate anomaly inquiry response"""
        if 'anomaly_statistics' in context_data:
            stats = context_data['anomaly_statistics']
            
            response = f"""🕵️ **ANOMALY DETECTION OVERVIEW (Database Enhanced)**

📊 **Current Detection Status:**
- Total Locations Analyzed: {stats['total_locations']:,}
- Anomalies Detected: {stats['anomaly_count']:,}
- Detection Rate: {stats['anomaly_percentage']:.2f}%

🔍 **What Makes an Area Suspicious:**
1. **Rapid Building Growth** (>50% in 2-3 years)
2. **Economic Mismatch** (High development, low wealth index)
3. **Unusual Night Activity** (High lights, low population)
4. **Land Use Anomalies** (Sudden vegetation conversion)

🏆 **Top Risk Provinces:**"""

            for i, prov in enumerate(stats['province_statistics'][:3], 1):
                response += f"\n{i}. **{prov['province']}**: {prov['anomaly_percentage']:.1f}% anomaly rate ({prov['anomaly_count']} locations)"

            if stats['top_anomalies']:
                response += "\n\n🚨 **Most Suspicious Areas:**"
                for i, area in enumerate(stats['top_anomalies'][:3], 1):
                    response += f"\n{i}. {area['location']} - Growth: {area['built_growth']:.1f}%"

            response += "\n\n💡 **For Investigation**: Provide specific H3 indexes for detailed analysis."
            
            return response
        
        else:
            return """🕵️ **ANOMALY DETECTION SYSTEM**

Our AI system identifies suspicious patterns that may indicate money laundering:

🔍 **Detection Methods:**
- **Satellite Imagery Analysis**: Rapid construction patterns
- **Economic Indicators**: Wealth vs development mismatches  
- **Night-time Activity**: Unusual light patterns
- **Land Use Changes**: Sudden vegetation to built conversion

🎯 **Investigation Process:**
1. **Automated Detection**: AI flags anomalous locations
2. **Context Analysis**: Cross-reference with economic data
3. **Field Verification**: On-site investigation recommendations
4. **Evidence Collection**: Document findings for authorities

💡 **To Get Started**: Ask for anomaly statistics or provide an H3 index for analysis."""
    
    async def _generate_statistics_response(self, context_data: Dict[str, Any]) -> str:
        """Generate statistics summary response"""
        if 'anomaly_statistics' in context_data:
            stats = context_data['anomaly_statistics']
            
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
                    response += f"\n   - Growth: {area['built_growth']:.1f}% | Wealth: {area['wealth_index']:.3f} | Priority: {urgency}"

            response += f"\n\n📈 **System Performance:**"
            response += f"\n- Database: PostgreSQL with real-time queries"
            response += f"\n- AI Model: Isolation Forest anomaly detection"
            response += f"\n- Analysis Coverage: Multi-provincial satellite intelligence"
            
            return response
        
        else:
            return "📊 **STATISTICS UNAVAILABLE**\n\nNo statistical data is currently loaded. Please ensure the training dataset is properly loaded from the database."
    
    async def _generate_explanation_response(self, context_data: Dict[str, Any]) -> str:
        """Generate model explanation response"""
        rag_context = ""
        if 'rag_results' in context_data and context_data['rag_results']:
            rag_context = "\n\n**📚 From Knowledge Base:**\n"
            for result in context_data['rag_results'][:2]:
                rag_context += f"- {result.title}: {result.content[:150]}...\n"
        
        base_response = """🧠 **AI MODEL EXPLANATION SYSTEM**

**🤖 How Our AI Detects Money Laundering:**

**1. Isolation Forest Algorithm:**
- Identifies data points that deviate significantly from normal patterns
- Trained on geospatial and economic indicators
- Flags locations with unusual development characteristics

**2. Key Features Analyzed:**
- `built_growth_pct_22_24`: Building expansion rate (2022-2024)
- `RWI`: Relative Wealth Index (economic prosperity)
- `ntl_per_capita`: Night-time lights per capita (activity levels)
- `veg_to_built`: Land use conversion patterns
- `built_per_rwi`: Development vs wealth ratio

**3. SHAP Explainability:**
- **Green Values**: Push toward "anomaly" prediction
- **Red Values**: Push toward "normal" prediction  
- **Bar Length**: Feature influence strength
- **Final Score**: Combined prediction confidence

**🔍 Anomaly Indicators:**
- Rapid building growth (>50%) in low-wealth areas
- High night activity with low population density
- Sudden land conversion from vegetation to built areas
- Economic indicators inconsistent with development level

💡 **For Detailed Analysis**: Provide an H3 index to get SHAP values for that specific location."""
        
        return base_response + rag_context
    
    async def _generate_procedure_response(self, context_data: Dict[str, Any]) -> str:
        """Generate investigation procedure response"""
        rag_context = ""
        if 'rag_results' in context_data and context_data['rag_results']:
            rag_context = "\n\n**📋 Investigation Procedures from Knowledge Base:**\n"
            for result in context_data['rag_results'][:3]:
                rag_context += f"\n**{result.title}:**\n{result.content[:200]}...\n"
        
        base_response = """📋 **MONEY LAUNDERING INVESTIGATION PROCEDURES**

**🔍 Phase 1: Initial Assessment**
1. **Satellite Analysis**: Review rapid construction patterns
2. **Data Validation**: Verify AI anomaly detection results
3. **Risk Scoring**: Assess priority level for investigation
4. **Resource Planning**: Allocate investigation team and timeline

**📊 Phase 2: Data Collection**
1. **Administrative Records**: Building permits, ownership documents
2. **Financial Intelligence**: Transaction patterns, funding sources  
3. **Economic Analysis**: Local wealth indicators vs development
4. **Field Intelligence**: Local interviews and observations

**🎯 Phase 3: Analysis & Verification**
1. **Pattern Recognition**: Compare with known money laundering cases
2. **SHAP Interpretation**: Understand AI decision factors
3. **Cross-referencing**: Check against financial databases
4. **Evidence Correlation**: Build comprehensive case profile

**⚖️ Phase 4: Reporting & Action**
1. **Documentation**: Prepare detailed investigation report
2. **Legal Review**: Ensure evidence meets legal standards
3. **Authority Notification**: Submit to relevant agencies
4. **Follow-up Monitoring**: Continuous surveillance recommendations

**🚨 Red Flags to Investigate:**
- Building growth >100% in 2-3 years
- Night-time activity inconsistent with population
- Complex ownership structures
- Cash-intensive construction projects
- Proximity to known high-risk areas"""
        
        return base_response + rag_context
    
    async def _generate_help_response(self, context_data: Dict[str, Any]) -> str:
        """Generate help and guidance response"""
        return """🤖 **SPACE DETECTIVE v2.0 AI ASSISTANT - COMPREHENSIVE HELP**

**🔍 What I Can Help With:**

**1. Location Analysis**
- Paste any H3 index (e.g., "876529d53ffffff") for detailed anomaly analysis
- Get SHAP explanations for AI decisions
- Compare multiple locations

**2. Investigation Support**
- "Show me investigation procedures for suspicious areas"
- "What are the red flags for money laundering?"
- "How do I interpret anomaly results?"

**3. Data Insights**
- "What are the anomaly statistics?"
- "Show me the most suspicious provinces"
- "Which areas need immediate investigation?"

**4. Technical Guidance** 
- "How does the AI model work?"
- "Explain SHAP values"
- "What features indicate money laundering?"

**5. Database Operations**
- Real-time PostgreSQL queries
- RAG-enhanced knowledge search
- Performance monitoring

**💬 Example Questions:**
- "Analyze H3 index 876529d53ffffff"
- "What makes an area suspicious?"
- "Show me the top 5 anomalous locations"
- "Search for investigation procedures"
- "How is Sumatera Utara performing?"

**🎯 Pro Tips:**
- Be specific with H3 indexes for detailed analysis
- Ask follow-up questions for deeper insights  
- Use natural language - I understand context
- Request comparisons between locations

**🔗 Advanced Features:**
- RAG-powered knowledge base search
- Real-time database integration
- Context-aware conversations
- Multi-language support (EN/ID)

**Ready to assist with your investigation needs! 🚀**"""
    
    async def _generate_general_response(self, message: str, context_data: Dict[str, Any], 
                                       chat_context: ChatContext) -> str:
        """Generate general response for unclassified messages"""
        # Check conversation context for better response
        context_summary = chat_context.get_context_summary()
        
        if 'training_data' in context_data:
            dataset_info = context_data.get('dataset_info', {})
            total_records = len(context_data['training_data'])
            
            response = f"""🛰️ **SPACE DETECTIVE v2.0 AI ASSISTANT**

I'm here to help you with AI-powered money laundering detection and investigation!

📊 **Current System Status:**
- {total_records:,} locations analyzed from PostgreSQL database
- Real-time anomaly detection active
- RAG knowledge base enabled
- Advanced SHAP explainability available

🔍 **I Can Help You With:**
- **Location Analysis**: Provide H3 indexes for detailed investigation guidance
- **Anomaly Investigation**: Understand why areas are flagged as suspicious  
- **Procedure Guidance**: Access investigation protocols and best practices
- **Statistical Insights**: Get comprehensive anomaly statistics and trends

💬 **Popular Queries:**
- "What are the most suspicious areas?"
- "Analyze [H3 index]" 
- "Show me investigation procedures"
- "Explain why this area is anomalous"
- "What patterns indicate money laundering?"

🎯 **Enhanced with v2.0:**
- PostgreSQL integration for real-time data
- RAG system for contextual knowledge access
- Advanced conversation understanding
- Multi-modal analysis capabilities

**How can I assist with your investigation today?**"""
        else:
            response = """🛰️ **SPACE DETECTIVE v2.0 AI ASSISTANT**

Hello! I'm your AI-powered assistant for geospatial money laundering detection.

⚠️ **Current Status**: Database connection needed for full functionality.

🔧 **Available Features:**
- Investigation procedure guidance
- Model explanation and SHAP interpretation  
- General money laundering detection advice
- Technical support for system setup

📋 **To Unlock Full Capabilities:**
1. Ensure PostgreSQL database is connected
2. Load training data and models
3. Initialize RAG knowledge base

💡 **I Can Still Help With:**
- "How does anomaly detection work?"
- "What are money laundering indicators?"
- "Explain investigation procedures" 
- "How to interpret AI results?"

**Ready to assist once your data is loaded! 🚀**"""
        
        return response
    
    async def _enhance_with_external_ai(self, original_message: str, current_response: str,
                                      context_data: Dict[str, Any], chat_context: ChatContext) -> Optional[str]:
        """Enhance response using external AI services"""
        if not DASHSCOPE_AVAILABLE:
            return None
        
        try:
            # Setup Alibaba API
            dashscope.api_key = self.config.external_apis.alibaba_api_key
            
            # Prepare enhanced context
            system_context = """You are an AI assistant for Space Detective, an advanced geospatial anomaly detection system for combating money laundering. You help investigators understand AI model decisions and provide actionable intelligence.

Key capabilities:
- Analyze satellite imagery patterns for suspicious construction
- Detect economic mismatches indicating potential money laundering
- Provide SHAP explanations for AI decisions
- Guide investigation procedures

Respond professionally but accessibly, focusing on actionable insights for investigators."""
            
            # Add context from RAG and database
            context_info = ""
            if 'rag_results' in context_data and context_data['rag_results']:
                context_info += "\nRelevant knowledge base information is available about investigation procedures."
            
            if 'training_data' in context_data:
                context_info += f"\nCurrent dataset contains {len(context_data['training_data'])} locations with real-time anomaly detection."
            
            # Prepare conversation history
            recent_history = chat_context.conversation_history[-4:] if len(chat_context.conversation_history) > 4 else chat_context.conversation_history
            
            messages = [
                {"role": "system", "content": system_context + context_info}
            ]
            
            # Add recent conversation context
            for msg in recent_history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"][:500]  # Truncate for API limits
                    })
            
            # Add current exchange
            messages.append({"role": "user", "content": original_message})
            
            # Call external AI
            response = Generation.call(
                model=self.config.external_apis.alibaba_model,
                messages=messages,
                temperature=self.config.external_apis.alibaba_temperature,
                max_tokens=self.config.external_apis.alibaba_max_tokens
            )
            
            if response.status_code == 200:
                enhanced_text = response.output.text
                
                # Combine with original response
                final_response = f"{current_response}\n\n**🤖 Enhanced AI Analysis:**\n{enhanced_text}"
                return final_response
            
        except Exception as e:
            logger.warning(f"External AI enhancement failed: {e}")
        
        return None
    
    def _generate_suggestions(self, analysis: Dict[str, Any], context_data: Dict[str, Any]) -> List[str]:
        """Generate helpful suggestions for the user"""
        suggestions = []
        
        intent = analysis['primary_intent']
        entities = analysis['extracted_entities']
        
        if intent == 'location_analysis':
            if 'h3_indexes' in entities:
                suggestions.append("Request SHAP analysis for detailed feature explanation")
                suggestions.append("Compare with neighboring areas")
                suggestions.append("Check investigation history for this location")
            else:
                suggestions.append("Provide an H3 index for location analysis")
        
        elif intent == 'anomaly_inquiry':
            suggestions.append("Ask for specific province statistics")
            suggestions.append("Request top suspicious areas list")
            suggestions.append("Search for investigation procedures")
        
        elif intent == 'statistics_request':
            suggestions.append("Request provincial risk ranking")
            suggestions.append("Ask for top anomalous areas")
            suggestions.append("Compare statistics across regions")
        
        else:
            suggestions.extend([
                "Try: 'Show me the most suspicious areas'",
                "Try: 'Analyze [H3 index]'",
                "Try: 'What are investigation procedures?'",
                "Try: 'Explain anomaly detection'"
            ])
        
        return suggestions[:4]  # Limit to 4 suggestions


# Main Chat API class
class ChatAPI:
    """Main Chat API orchestrator"""
    
    def __init__(self, db_manager: DatabaseManager, rag_system: RAGSystem, config: AppConfig):
        self.db_manager = db_manager
        self.rag_system = rag_system
        self.config = config
        self.response_generator = ResponseGenerator(db_manager, rag_system, config)
        self.active_sessions: Dict[str, ChatContext] = {}
        self.session_timeout = timedelta(hours=2)
    
    def _cleanup_expired_sessions(self):
        """Remove expired chat sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, context in self.active_sessions.items()
            if current_time - context.session_start > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
    
    def get_or_create_session(self, session_id: str) -> ChatContext:
        """Get existing session or create new one"""
        self._cleanup_expired_sessions()
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ChatContext()
        
        return self.active_sessions[session_id]
    
    async def process_message(self, message: str, session_id: str = "default",
                            use_rag: bool = True, max_context_docs: int = 3) -> Dict[str, Any]:
        """Process user message and generate response"""
        
        # Get or create chat session
        chat_context = self.get_or_create_session(session_id)
        
        try:
            # Generate response
            result = await self.response_generator.generate_response(message, chat_context)
            
            # Add session information
            result.update({
                "session_id": session_id,
                "message_count": len(chat_context.conversation_history),
                "session_duration": str(datetime.now() - chat_context.session_start),
                "system_info": {
                    "database_connected": self.db_manager.is_connected,
                    "rag_available": self.rag_system.initialized,
                    "external_ai_available": bool(self.config.external_apis.alibaba_api_key and DASHSCOPE_AVAILABLE)
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": f"🚨 I encountered an error processing your request: {str(e)}. Please try again.",
                "error": True,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_location(self, h3_index: str, session_id: str = "default") -> Dict[str, Any]:
        """Specialized endpoint for location analysis"""
        message = f"Analyze location {h3_index}"
        return await self.process_message(message, session_id, use_rag=False)
    
    async def compare_locations(self, h3_indexes: List[str], session_id: str = "default") -> Dict[str, Any]:
        """Compare multiple locations"""
        if len(h3_indexes) > 5:
            h3_indexes = h3_indexes[:5]  # Limit to 5 locations
        
        message = f"Compare these locations: {', '.join(h3_indexes)}"
        return await self.process_message(message, session_id, use_rag=False)
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about chat session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        context = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "session_start": context.session_start.isoformat(),
            "message_count": len(context.conversation_history),
            "duration": str(datetime.now() - context.session_start),
            "context_data_keys": list(context.context_data.keys()),
            "last_activity": context.conversation_history[-1]["timestamp"] if context.conversation_history else None
        }
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear specific chat session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    async def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversation history for session"""
        if session_id not in self.active_sessions:
            return []
        
        context = self.active_sessions[session_id]
        history = context.conversation_history[-limit:] if limit else context.conversation_history
        
        return history
    
    async def export_conversation(self, session_id: str) -> Optional[str]:
        """Export conversation as formatted text"""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        export_text = f"""# Space Detective Conversation Export
Session ID: {session_id}
Start Time: {context.session_start.isoformat()}
Duration: {datetime.now() - context.session_start}
Message Count: {len(context.conversation_history)}

---

"""
        
        for msg in context.conversation_history:
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            timestamp = msg["timestamp"]
            content = msg["content"]
            
            export_text += f"""## {role} - {timestamp}

{content}

---

"""
        
        return export_text
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get chat system statistics"""
        active_sessions = len(self.active_sessions)
        total_messages = sum(len(ctx.conversation_history) for ctx in self.active_sessions.values())
        
        return {
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": total_messages / active_sessions if active_sessions > 0 else 0,
            "system_uptime": str(datetime.now() - self.config.app_start_time) if hasattr(self.config, 'app_start_time') else "Unknown",
            "features_enabled": {
                "rag_system": self.rag_system.initialized,
                "database": self.db_manager.is_connected,
                "external_ai": bool(self.config.external_apis.alibaba_api_key and DASHSCOPE_AVAILABLE)
            }
        }


# Utility functions for chat API
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


def extract_h3_indexes(text: str) -> List[str]:
    """Extract all H3 indexes from text"""
    h3_pattern = r'\b87[0-9a-f]{10,13}\b'
    matches = re.findall(h3_pattern, text, re.IGNORECASE)
    return [match for match in matches if validate_h3_index(match)]


def sanitize_message(message: str, max_length: int = 5000) -> str:
    """Sanitize user message"""
    if not message:
        return ""
    
    # Remove excessive whitespace
    message = re.sub(r'\s+', ' ', message.strip())
    
    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "..."
    
    # Remove potentially harmful content
    message = re.sub(r'<script.*?</script>', '', message, flags=re.IGNORECASE | re.DOTALL)
    message = re.sub(r'javascript:', '', message, flags=re.IGNORECASE)
    
    return message


async def create_chat_api(db_manager: DatabaseManager, rag_system: RAGSystem, config: AppConfig) -> ChatAPI:
    """Factory function to create Chat API"""
    return ChatAPI(db_manager, rag_system, config)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        from config import AppConfig
        from database_manager import create_database_manager
        from rag_system import create_rag_system
        
        # Create configuration
        config = AppConfig()
        
        # Initialize components
        db_manager = await create_database_manager(config.database)
        rag_system = await create_rag_system(config.rag, config.database)
        
        # Create chat API
        chat_api = await create_chat_api(db_manager, rag_system, config)
        
        # Test conversation
        test_messages = [
            "Hello, what can you help me with?",
            "What are the most suspicious areas?",
            "Analyze H3 index 876529d53ffffff",
            "How does the anomaly detection work?",
            "Show me investigation procedures"
        ]
        
        print("=== Chat API Test ===")
        
        for message in test_messages:
            print(f"\n👤 User: {message}")
            
            response = await chat_api.process_message(message, session_id="test_session")
            
            print(f"🤖 Assistant: {response['response'][:200]}...")
            print(f"Intent: {response.get('analysis', {}).get('primary_intent', 'unknown')}")
            print(f"RAG Results: {response.get('rag_results_count', 0)}")
        
        # Get session info
        session_info = await chat_api.get_session_info("test_session")
        print(f"\n📊 Session Info: {session_info}")
        
        # Export conversation
        export = await chat_api.export_conversation("test_session")
        if export:
            print(f"\n📄 Conversation exported ({len(export)} characters)")
        
        # Cleanup
        await db_manager.close()
        await rag_system.close()
    
    # Run test
    import asyncio
    asyncio.run(main())