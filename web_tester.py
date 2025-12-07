#!/usr/bin/env python3
"""
NL2ATL Web-Based Interactive Tester
====================================

Flask-based web interface for testing and comparing NL→ATL translation models.

Features:
- Test fine-tuned model against standard LLMs (OpenAI GPT-4o-mini, Claude)
- Real-time response time tracking
- Side-by-side comparison of translations
- Syntax validation with error highlighting
- Save/export test results to JSON
- Clean, modern web interface

Requirements:
- flask>=3.0.0
- openai>=1.0.0
- anthropic>=0.18.0

Usage:
    python web_tester.py
    
    Then open http://localhost:5000 in your browser
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_file
from dotenv import load_dotenv

from atl_syntax import validate_atl_string

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_tester.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'nl2atl-tester-secret-key')

# =============================================================================
# Model Configuration
# =============================================================================

MODELS = {
    "finetuned": {
        "name": "Fine-tuned GPT-4o-mini",
        "provider": "openai",
        "model_id": "ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC",
        "color": "#4CAF50"
    },
    "openai": {
        "name": "GPT-4o-mini (Base)",
        "provider": "openai",
        "model_id": "gpt-4o-mini-2024-07-18",
        "color": "#2196F3"
    },
    "claude": {
        "name": "Claude 3 Haiku",
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "color": "#9C27B0"
    }
}

SYSTEM_PROMPT = (
    "You are an expert in translating natural language requirements into "
    "Alternating-time Temporal Logic (ATL) formulas. Given a natural language "
    "statement describing agent capabilities and temporal properties, generate "
    "ONLY the corresponding ATL formula using proper syntax with coalition operators "
    "⟨⟨...⟩⟩, temporal operators (G, F, X, U), and logical operators (∧, ∨, →, ¬). "
    "\n\nIMPORTANT: Return ONLY the ATL formula, nothing else. No explanations, "
    "no labels, no extra text. Just the formula itself."
)

# Test history storage
test_history: List[Dict[str, Any]] = []

# Example NL statements
EXAMPLES = [
    "Agents 1 and 2 can ensure the system never crashes",
    "The robot can eventually reach the goal state",
    "The controller can maintain temperature stability until the alarm is triggered",
    "The server team can ensure that every request is eventually followed by a response",
    "Agent 3 can guarantee that the system remains safe in the next state"
]


# =============================================================================
# Translation Functions
# =============================================================================

def translate_with_openai(nl_text: str, model_id: str) -> Dict[str, Any]:
    """
    Translate using OpenAI API with response time tracking.
    
    Returns:
        {
            "atl_formula": str,
            "response_time": float (seconds),
            "tokens_used": int,
            "error": Optional[str]
        }
    """
    logger.info(f"Starting OpenAI translation with model: {model_id}")
    logger.debug(f"Input NL text: {nl_text}")
    
    try:
        import openai
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        client = openai.OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_text}
        ]
        logger.debug(f"Sending request to OpenAI API...")
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        raw_content = response.choices[0].message.content.strip()
        logger.info(f"Received response in {response_time:.3f}s")
        logger.debug(f"Raw response: {raw_content}")
        
        # Clean up the response - remove common wrapper patterns
        atl_formula = raw_content
        
        # Remove markdown code blocks
        if "```" in atl_formula:
            lines = atl_formula.split("\n")
            atl_formula = "\n".join([l for l in lines if not l.strip().startswith("```")])
            atl_formula = atl_formula.strip()
            logger.debug(f"Removed code blocks: {atl_formula}")
        
        # Remove common prefixes like "ATL:", "Formula:", etc.
        for prefix in ["ATL:", "Formula:", "ATL Formula:", "Output:", "Result:"]:
            if atl_formula.startswith(prefix):
                atl_formula = atl_formula[len(prefix):].strip()
                logger.debug(f"Removed prefix '{prefix}': {atl_formula}")
        
        logger.info(f"Final ATL formula: {atl_formula}")
        
        return {
            "atl_formula": atl_formula,
            "response_time": response_time,
            "tokens_used": response.usage.total_tokens,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"OpenAI translation failed: {str(e)}", exc_info=True)
        return {
            "atl_formula": "",
            "response_time": 0.0,
            "tokens_used": 0,
            "error": str(e)
        }


def translate_with_anthropic(nl_text: str, model_id: str) -> Dict[str, Any]:
    """
    Translate using Anthropic API with response time tracking.
    
    Returns:
        {
            "atl_formula": str,
            "response_time": float (seconds),
            "tokens_used": int,
            "error": Optional[str]
        }
    """
    logger.info(f"Starting Anthropic translation with model: {model_id}")
    logger.debug(f"Input NL text: {nl_text}")
    
    try:
        import anthropic
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        
        client = anthropic.Anthropic(api_key=api_key)
        
        logger.debug(f"Sending request to Anthropic API...")
        
        start_time = time.time()
        
        message = client.messages.create(
            model=model_id,
            max_tokens=200,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": nl_text}
            ]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        raw_content = message.content[0].text.strip()
        logger.info(f"Received response in {response_time:.3f}s")
        logger.debug(f"Raw response: {raw_content}")
        
        # Clean up the response - remove common wrapper patterns
        atl_formula = raw_content
        
        # Remove markdown code blocks
        if "```" in atl_formula:
            lines = atl_formula.split("\n")
            atl_formula = "\n".join([l for l in lines if not l.strip().startswith("```")])
            atl_formula = atl_formula.strip()
            logger.debug(f"Removed code blocks: {atl_formula}")
        
        # Remove common prefixes like "ATL:", "Formula:", etc.
        for prefix in ["ATL:", "Formula:", "ATL Formula:", "Output:", "Result:"]:
            if atl_formula.startswith(prefix):
                atl_formula = atl_formula[len(prefix):].strip()
                logger.debug(f"Removed prefix '{prefix}': {atl_formula}")
        
        logger.info(f"Final ATL formula: {atl_formula}")
        
        return {
            "atl_formula": atl_formula,
            "response_time": response_time,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Anthropic translation failed: {str(e)}", exc_info=True)
        return {
            "atl_formula": "",
            "response_time": 0.0,
            "tokens_used": 0,
            "error": str(e)
        }


def translate_with_model(nl_text: str, model_key: str) -> Dict[str, Any]:
    """
    Translate using the specified model.
    
    Args:
        nl_text: Natural language text to translate
        model_key: Key from MODELS dict ("finetuned", "openai", "claude")
        
    Returns:
        Translation result with metrics
    """
    logger.info(f"translate_with_model called for: {model_key}")
    model_info = MODELS[model_key]
    logger.debug(f"Model info: {model_info}")
    
    if model_info["provider"] == "openai":
        return translate_with_openai(nl_text, model_info["model_id"])
    elif model_info["provider"] == "anthropic":
        return translate_with_anthropic(nl_text, model_info["model_id"])
    else:
        error_msg = f"Unknown provider: {model_info['provider']}"
        logger.error(error_msg)
        return {
            "atl_formula": "",
            "response_time": 0.0,
            "tokens_used": 0,
            "error": error_msg
        }


# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', models=MODELS, examples=EXAMPLES)


@app.route('/api/translate', methods=['POST'])
def api_translate():
    """
    Translate NL text using selected models.
    
    Request JSON:
        {
            "nl_text": str,
            "models": List[str]  # List of model keys to use
        }
    
    Response JSON:
        {
            "success": bool,
            "timestamp": str,
            "nl_text": str,
            "translations": {
                "model_key": {
                    "atl_formula": str,
                    "response_time": float,
                    "tokens_used": int,
                    "is_valid": bool,
                    "validation_errors": List[str],
                    "error": Optional[str]
                }
            }
        }
    """
    try:
        data = request.get_json()
        nl_text = data.get('nl_text', '').strip()
        selected_models = data.get('models', [])
        
        if not nl_text:
            return jsonify({
                'success': False,
                'error': 'No natural language text provided'
            }), 400
        
        if not selected_models:
            return jsonify({
                'success': False,
                'error': 'No models selected'
            }), 400
        
        logger.info(f"Translation request for models: {selected_models}")
        logger.info(f"NL text: {nl_text}")
        
        # Translate with each selected model
        translations = {}
        for model_key in selected_models:
            if model_key not in MODELS:
                logger.warning(f"Unknown model key: {model_key}")
                continue
            
            result = translate_with_model(nl_text, model_key)
            
            # Validate syntax
            is_valid_syntax = False
            validation_errors = []
            if result["atl_formula"] and not result["error"]:
                is_valid_syntax, validation_errors = validate_atl_string(result["atl_formula"])
                logger.info(f"Validation for {model_key}: valid={is_valid_syntax}, errors={validation_errors}")
            
            translations[model_key] = {
                **result,
                "is_valid": is_valid_syntax,
                "validation_errors": validation_errors
            }
        
        # Build response
        test_result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "nl_text": nl_text,
            "translations": translations
        }
        
        # Add to history
        test_history.append(test_result)
        
        logger.info("Translation completed successfully")
        return jsonify(test_result)
        
    except Exception as e:
        logger.error(f"Translation API error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    """
    Get test history.
    
    Response JSON:
        {
            "success": bool,
            "history": List[Dict]
        }
    """
    try:
        # Return last 20 tests
        return jsonify({
            'success': True,
            'history': test_history[-20:]
        })
    except Exception as e:
        logger.error(f"History API error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export', methods=['GET'])
def api_export():
    """
    Export test history as JSON file.
    
    Response:
        JSON file download
    """
    try:
        if not test_history:
            return jsonify({
                'success': False,
                'error': 'No test history to export'
            }), 400
        
        # Create temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"web_test_results_{timestamp}.json"
        filepath = Path("/tmp") / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_history, f, indent=2, ensure_ascii=False)
        
        return send_file(
            filepath,
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Export API error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    """
    Clear test history.
    
    Response JSON:
        {
            "success": bool
        }
    """
    try:
        global test_history
        test_history = []
        logger.info("Test history cleared")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Clear history API error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """
    Check API key status.
    
    Response JSON:
        {
            "success": bool,
            "openai_available": bool,
            "anthropic_available": bool
        }
    """
    return jsonify({
        'success': True,
        'openai_available': bool(os.environ.get("OPENAI_API_KEY")),
        'anthropic_available': bool(os.environ.get("ANTHROPIC_API_KEY"))
    })


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Launch the Flask web application."""
    logger.info("Starting NL2ATL Web Tester")
    
    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. OpenAI models will not work.")
        print("Warning: OPENAI_API_KEY not set. OpenAI models will not work.")
    else:
        logger.info("OPENAI_API_KEY found")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set. Claude will not work.")
        print("Warning: ANTHROPIC_API_KEY not set. Claude will not work.")
    else:
        logger.info("ANTHROPIC_API_KEY found")
    
    # Run Flask app
    print("\n" + "="*60)
    print("NL2ATL Web Tester Starting...")
    print("="*60)
    print("\nOpen your browser and navigate to:")
    print("    http://localhost:5030")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5030, debug=True)


if __name__ == "__main__":
    main()
