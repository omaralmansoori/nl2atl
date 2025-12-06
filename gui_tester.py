#!/usr/bin/env python3
"""
NL2ATL Interactive GUI Tester
==============================

Interactive graphical interface for testing and comparing NL→ATL translation models.

Features:
- Test fine-tuned model against standard LLMs (OpenAI GPT-4o-mini, Claude)
- Real-time response time tracking
- Side-by-side comparison of translations
- Syntax validation with error highlighting
- Save/export test results
- Manual quality assessment

Requirements:
- tkinter (requires system installation: brew install python-tk@3.13)
- openai>=1.0.0
- anthropic>=0.18.0

Usage:
    python gui_tester.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from atl_syntax import is_valid, validate_atl_string
from nl2atl import get_llm_client

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gui_tester.log')
    ]
)
logger = logging.getLogger(__name__)


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
# GUI Application
# =============================================================================

class NL2ATLTesterGUI:
    """Interactive GUI for testing NL2ATL translation models."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NL2ATL Interactive Tester")
        self.root.geometry("1400x900")
        
        # Test history
        self.test_history: List[Dict[str, Any]] = []
        
        # Model selection (all enabled by default)
        self.model_enabled = {key: tk.BooleanVar(value=True) for key in MODELS.keys()}
        
        # Translation state
        self.is_translating = False
        
        # Configure style
        self.setup_styles()
        
        # Build UI
        self.build_ui()
        
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for model results
        for model_key, model_info in MODELS.items():
            style.configure(
                f"{model_key}.TFrame",
                background="#f5f5f5",
                borderwidth=2,
                relief="solid"
            )
    
    def build_ui(self):
        """Build the main UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="NL2ATL Interactive Model Tester",
            font=("Helvetica", 18, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Input section
        self.build_input_section(main_frame)
        
        # Results section
        self.build_results_section(main_frame)
        
        # History section
        self.build_history_section(main_frame)
        
    def build_input_section(self, parent: ttk.Frame):
        """Build input section for NL text."""
        input_frame = ttk.LabelFrame(parent, text="Input", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # NL input
        nl_label = ttk.Label(input_frame, text="Natural Language Requirement:")
        nl_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.nl_text = scrolledtext.ScrolledText(
            input_frame,
            height=4,
            wrap=tk.WORD,
            font=("Courier", 11)
        )
        self.nl_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model selection
        model_select_label = ttk.Label(input_frame, text="Select Models to Use:")
        model_select_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 5))
        
        model_select_frame = ttk.Frame(input_frame)
        model_select_frame.grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        
        for idx, (model_key, model_info) in enumerate(MODELS.items()):
            cb = ttk.Checkbutton(
                model_select_frame,
                text=model_info["name"],
                variable=self.model_enabled[model_key]
            )
            cb.grid(row=0, column=idx, padx=(0, 15), sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=4, column=0, sticky=tk.W)
        
        self.translate_btn = ttk.Button(
            button_frame,
            text="Translate with Selected Models",
            command=self.translate_all
        )
        self.translate_btn.grid(row=0, column=0, padx=(0, 5))
        
        clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_input
        )
        clear_btn.grid(row=0, column=1, padx=(0, 5))
        
        load_btn = ttk.Button(
            button_frame,
            text="Load Example",
            command=self.load_example
        )
        load_btn.grid(row=0, column=2)
        
        # Progress indicator
        self.progress_label = ttk.Label(input_frame, text="", foreground="blue")
        self.progress_label.grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        
    def build_results_section(self, parent: ttk.Frame):
        """Build results section showing translations from all models."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        parent.rowconfigure(2, weight=1)
        
        # Create a result panel for each model
        self.result_panels = {}
        for idx, (model_key, model_info) in enumerate(MODELS.items()):
            panel = self.create_result_panel(results_frame, model_key, model_info)
            panel.grid(row=0, column=idx, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
            self.result_panels[model_key] = panel
            results_frame.columnconfigure(idx, weight=1)
    
    def create_result_panel(
        self,
        parent: ttk.Frame,
        model_key: str,
        model_info: Dict[str, Any]
    ) -> ttk.Frame:
        """Create a result panel for a specific model."""
        panel = ttk.Frame(parent, style=f"{model_key}.TFrame", padding="10")
        
        # Model name
        name_label = ttk.Label(
            panel,
            text=model_info["name"],
            font=("Helvetica", 12, "bold")
        )
        name_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # ATL formula
        atl_label = ttk.Label(panel, text="ATL Formula:")
        atl_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        atl_text = scrolledtext.ScrolledText(
            panel,
            height=3,
            wrap=tk.WORD,
            font=("Courier", 11),
            state='disabled'
        )
        atl_text.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Metrics frame
        metrics_frame = ttk.Frame(panel)
        metrics_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Response time
        time_label = ttk.Label(metrics_frame, text="Response Time:")
        time_label.grid(row=0, column=0, sticky=tk.W)
        time_value = ttk.Label(metrics_frame, text="--", font=("Courier", 10))
        time_value.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Tokens
        tokens_label = ttk.Label(metrics_frame, text="Tokens:")
        tokens_label.grid(row=1, column=0, sticky=tk.W)
        tokens_value = ttk.Label(metrics_frame, text="--", font=("Courier", 10))
        tokens_value.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Validation
        valid_label = ttk.Label(metrics_frame, text="Syntax Valid:")
        valid_label.grid(row=2, column=0, sticky=tk.W)
        valid_value = ttk.Label(metrics_frame, text="--", font=("Courier", 10))
        valid_value.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        # Error message
        error_text = scrolledtext.ScrolledText(
            panel,
            height=2,
            wrap=tk.WORD,
            font=("Courier", 9),
            state='disabled',
            background="#ffebee"
        )
        error_text.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        error_text.grid_remove()  # Hidden by default
        
        # Store widgets for later access
        setattr(panel, 'atl_text', atl_text)
        setattr(panel, 'time_value', time_value)
        setattr(panel, 'tokens_value', tokens_value)
        setattr(panel, 'valid_value', valid_value)
        setattr(panel, 'error_text', error_text)
        
        panel.columnconfigure(0, weight=1)
        
        return panel
    
    def build_history_section(self, parent: ttk.Frame):
        """Build history section showing past tests."""
        history_frame = ttk.LabelFrame(parent, text="Test History", padding="10")
        history_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        history_frame.columnconfigure(0, weight=1)
        
        # History controls
        controls_frame = ttk.Frame(history_frame)
        controls_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        export_btn = ttk.Button(
            controls_frame,
            text="Export Results",
            command=self.export_results
        )
        export_btn.grid(row=0, column=0, padx=(0, 5))
        
        clear_history_btn = ttk.Button(
            controls_frame,
            text="Clear History",
            command=self.clear_history
        )
        clear_history_btn.grid(row=0, column=1)
        
        # History list
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            height=6,
            wrap=tk.WORD,
            font=("Courier", 9),
            state='disabled'
        )
        self.history_text.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def translate_all(self):
        """Translate using selected models and display results."""
        nl_text = self.nl_text.get("1.0", tk.END).strip()
        
        if not nl_text:
            messagebox.showwarning("No Input", "Please enter a natural language requirement.")
            return
        
        # Check if at least one model is selected
        selected_models = [key for key, var in self.model_enabled.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("No Model Selected", "Please select at least one model to use.")
            return
        
        # Prevent concurrent translations
        if self.is_translating:
            return
        
        # Run translation in separate thread to prevent UI freezing
        thread = threading.Thread(target=self._translate_thread, args=(nl_text, selected_models))
        thread.daemon = True
        thread.start()
    
    def _translate_thread(self, nl_text: str, selected_models: List[str]):
        """Background thread for translation."""
        logger.info(f"Translation thread started for {len(selected_models)} models")
        logger.info(f"Selected models: {selected_models}")
        logger.info(f"NL text: {nl_text}")
        
        self.is_translating = True
        
        # Update UI on main thread
        self.root.after(0, self._set_translating_state, True)
        
        test_result = {
            "timestamp": datetime.now().isoformat(),
            "nl_text": nl_text,
            "translations": {}
        }
        
        try:
            # Translate with each selected model
            for idx, model_key in enumerate(selected_models, 1):
                logger.info(f"Processing model {idx}/{len(selected_models)}: {model_key}")
                
                # Update progress
                progress_text = f"Translating with {MODELS[model_key]['name']} ({idx}/{len(selected_models)})..."
                self.root.after(0, self.progress_label.config, {"text": progress_text})
                
                result = translate_with_model(nl_text, model_key)
                logger.info(f"Model {model_key} returned: {result}")
                
                # Validate syntax
                is_valid_syntax = False
                validation_errors = []
                if result["atl_formula"] and not result["error"]:
                    is_valid_syntax, validation_errors = validate_atl_string(result["atl_formula"])
                    logger.info(f"Validation for {model_key}: valid={is_valid_syntax}, errors={validation_errors}")
                
                # Store result
                test_result["translations"][model_key] = {
                    **result,
                    "is_valid": is_valid_syntax,
                    "validation_errors": validation_errors
                }
                
                # Update UI on main thread
                panel = self.result_panels[model_key]
                self.root.after(0, self.update_result_panel, panel, result)
                logger.info(f"Completed processing for {model_key}")
            
            # Add to history
            self.test_history.append(test_result)
            self.root.after(0, self.update_history_display)
            logger.info("Translation thread completed successfully")
            
        except Exception as e:
            logger.error(f"Translation thread error: {str(e)}", exc_info=True)
            self.root.after(0, messagebox.showerror, "Translation Error", f"Error during translation: {e}")
        
        finally:
            # Re-enable button
            self.is_translating = False
            self.root.after(0, self._set_translating_state, False)
            logger.info("Translation thread finished")
    
    def _set_translating_state(self, is_translating: bool):
        """Update UI state during translation."""
        if is_translating:
            self.translate_btn.config(state='disabled', text="Translating...")
            self.progress_label.config(text="Starting translation...")
        else:
            self.translate_btn.config(state='normal', text="Translate with Selected Models")
            self.progress_label.config(text="Translation complete!")
            # Clear progress after 3 seconds
            self.root.after(3000, lambda: self.progress_label.config(text=""))
    
    def update_result_panel(self, panel: ttk.Frame, result: Dict[str, Any]):
        """Update a result panel with translation results."""
        # Update ATL formula
        atl_text = panel.atl_text
        atl_text.config(state='normal')
        atl_text.delete("1.0", tk.END)
        atl_text.insert("1.0", result["atl_formula"])
        atl_text.config(state='disabled')
        
        # Update metrics
        panel.time_value.config(text=f"{result['response_time']:.3f}s")
        panel.tokens_value.config(text=str(result["tokens_used"]))
        
        # Validate syntax
        if result["atl_formula"] and not result["error"]:
            is_valid_syntax, errors = validate_atl_string(result["atl_formula"])
            panel.valid_value.config(
                text="✓ Yes" if is_valid_syntax else "✗ No",
                foreground="green" if is_valid_syntax else "red"
            )
            
            # Show errors if any
            if not is_valid_syntax:
                error_text = panel.error_text
                error_text.grid()
                error_text.config(state='normal')
                error_text.delete("1.0", tk.END)
                error_text.insert("1.0", "\n".join(errors))
                error_text.config(state='disabled')
            else:
                panel.error_text.grid_remove()
        else:
            panel.valid_value.config(text="--")
        
        # Show API errors
        if result["error"]:
            error_text = panel.error_text
            error_text.grid()
            error_text.config(state='normal')
            error_text.delete("1.0", tk.END)
            error_text.insert("1.0", f"Error: {result['error']}")
            error_text.config(state='disabled')
    
    def update_history_display(self):
        """Update the history display with recent tests."""
        self.history_text.config(state='normal')
        self.history_text.delete("1.0", tk.END)
        
        for idx, test in enumerate(reversed(self.test_history[-10:]), 1):
            self.history_text.insert(tk.END, f"[{test['timestamp']}] {test['nl_text'][:60]}...\n")
            
            for model_key, trans in test['translations'].items():
                model_name = MODELS[model_key]["name"]
                valid_str = "✓" if trans.get("is_valid", False) else "✗"
                time_str = f"{trans['response_time']:.3f}s"
                self.history_text.insert(
                    tk.END,
                    f"  {model_name}: {valid_str} {time_str} | {trans['atl_formula'][:40]}\n"
                )
            
            self.history_text.insert(tk.END, "\n")
        
        self.history_text.config(state='disabled')
    
    def clear_input(self):
        """Clear input text."""
        self.nl_text.delete("1.0", tk.END)
    
    def load_example(self):
        """Load an example NL statement."""
        examples = [
            "Agents 1 and 2 can ensure the system never crashes",
            "The robot can eventually reach the goal state",
            "The controller can maintain temperature stability until the alarm is triggered",
            "The server team can ensure that every request is eventually followed by a response",
            "Agent 3 can guarantee that the system remains safe in the next state"
        ]
        
        import random
        example = random.choice(examples)
        self.nl_text.delete("1.0", tk.END)
        self.nl_text.insert("1.0", example)
    
    def export_results(self):
        """Export test history to JSON file."""
        if not self.test_history:
            messagebox.showinfo("No Data", "No test results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"gui_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.test_history, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def clear_history(self):
        """Clear test history."""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all test history?"):
            self.test_history = []
            self.history_text.config(state='normal')
            self.history_text.delete("1.0", tk.END)
            self.history_text.config(state='disabled')


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Launch the GUI application."""
    logger.info("Starting NL2ATL GUI Tester")
    
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
    
    # Create and run GUI
    logger.info("Initializing GUI")
    root = tk.Tk()
    app = NL2ATLTesterGUI(root)
    logger.info("GUI started, entering main loop")
    root.mainloop()
    logger.info("GUI closed")


if __name__ == "__main__":
    main()
