#!/usr/bin/env python3
"""
Quantum ML Models Universal Launcher
====================================

Universal launcher for all quantum machine learning models.
Provides interactive interface to run, demo, and benchmark quantum models.

Usage:
    python quantum_launcher.py                    # Interactive mode
    python quantum_launcher.py --model quantum_gpt --demo
    python quantum_launcher.py --list
    python quantum_launcher.py --benchmark --all

Features:
- Interactive model selection
- Automated dependency checking
- Multiple execution modes
- Performance benchmarking
- Error handling and recovery
"""

import sys
import os
import argparse
import importlib
import time
from typing import Dict, List, Optional, Callable
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class QuantumMLLauncher:
    """Universal launcher for quantum ML models"""
    
    def __init__(self):
        self.models = {
            'quantum_compression': {
                'name': 'Quantum ACPP Compressor',
                'description': 'Revolutionary data compression using quantum superposition',
                'file': 'quantum_compression_model.py',
                'main_function': 'main_quantum_compression_demo',
                'class': 'QuantumCompressionAutoencoder',
                'features': [
                    '60-95% compression ratios',
                    'Quantum pattern recognition',
                    'Real-time compression',
                    'Production ready APIs'
                ],
                'requirements': ['pennylane', 'torch', 'numpy', 'matplotlib', 'scikit-learn']
            },
            'quantum_gpt': {
                'name': 'Quantum GPT',
                'description': "World's first quantum language model",
                'file': 'GPT Model.py',
                'main_function': 'demo_quantum_gpt',
                'class': 'QuantumGPT',
                'features': [
                    'Exponential memory efficiency: 24x fewer parameters',
                    'Parallel token processing via quantum superposition',
                    'Quantum attention mechanisms',
                    'Real text generation'
                ],
                'requirements': ['pennylane', 'torch', 'numpy', 'matplotlib']
            }
        }
        
        self.dependency_status = {}
        self.check_dependencies()
    
    def print_header(self):
        """Print the launcher header"""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("🚀 QUANTUM ML MODELS LAUNCHER")
        print("=" * 50)
        print(f"Universal interface for quantum machine learning{Colors.END}")
        print()
    
    def check_dependencies(self):
        """Check availability of required dependencies"""
        all_requirements = set()
        for model_info in self.models.values():
            all_requirements.update(model_info['requirements'])
        
        for package in all_requirements:
            try:
                importlib.import_module(package)
                self.dependency_status[package] = True
            except ImportError:
                self.dependency_status[package] = False
    
    def print_dependency_status(self):
        """Print status of dependencies"""
        print(f"{Colors.BOLD}📦 Dependency Status:{Colors.END}")
        
        for package, available in self.dependency_status.items():
            status = f"{Colors.GREEN}✅ Available{Colors.END}" if available else f"{Colors.RED}❌ Missing{Colors.END}"
            print(f"   {package:15} {status}")
        
        missing = [pkg for pkg, available in self.dependency_status.items() if not available]
        if missing:
            print(f"\n{Colors.YELLOW}💡 To install missing packages:{Colors.END}")
            print(f"   pip install {' '.join(missing)}")
        print()
    
    def list_models(self):
        """List all available models"""
        print(f"{Colors.BOLD}🔬 Available Quantum Models:{Colors.END}")
        print()
        
        for i, (key, model_info) in enumerate(self.models.items(), 1):
            # Check if model dependencies are available
            deps_available = all(self.dependency_status.get(dep, False) for dep in model_info['requirements'])
            status = f"{Colors.GREEN}Ready{Colors.END}" if deps_available else f"{Colors.YELLOW}Needs deps{Colors.END}"
            
            print(f"{Colors.BOLD}[{i}] {model_info['name']}{Colors.END} ({status})")
            print(f"    {Colors.CYAN}{model_info['description']}{Colors.END}")
            print(f"    File: {model_info['file']}")
            
            print(f"    {Colors.BOLD}Features:{Colors.END}")
            for feature in model_info['features']:
                print(f"      • {feature}")
            print()
    
    def interactive_menu(self):
        """Show interactive model selection menu"""
        while True:
            self.print_header()
            self.list_models()
            
            print(f"{Colors.BOLD}🎯 Options:{Colors.END}")
            print("[1] Quantum ACPP Compressor - Data compression demo")
            print("[2] Quantum GPT - Language model demo")
            print("[3] Benchmark all models")
            print("[4] Check dependencies")
            print("[5] Show model details")
            print("[0] Exit")
            print()
            
            try:
                choice = input(f"{Colors.BOLD}Select option [0-5]: {Colors.END}").strip()
                
                if choice == '0':
                    print(f"\n{Colors.GREEN}👋 Thanks for using Quantum ML Models!{Colors.END}")
                    break
                elif choice == '1':
                    self.run_model('quantum_compression', mode='demo')
                elif choice == '2':
                    self.run_model('quantum_gpt', mode='demo')
                elif choice == '3':
                    self.benchmark_all_models()
                elif choice == '4':
                    self.print_dependency_status()
                    input("Press Enter to continue...")
                elif choice == '5':
                    self.show_model_details()
                else:
                    print(f"{Colors.RED}Invalid choice. Please select 0-5.{Colors.END}")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
                time.sleep(2)
    
    def show_model_details(self):
        """Show detailed information about models"""
        print(f"\n{Colors.BOLD}📊 Detailed Model Information:{Colors.END}")
        print()
        
        for key, model_info in self.models.items():
            deps_available = all(self.dependency_status.get(dep, False) for dep in model_info['requirements'])
            
            print(f"{Colors.BOLD}{Colors.BLUE}{model_info['name']}{Colors.END}")
            print(f"Status: {'✅ Ready' if deps_available else '⚠️ Missing dependencies'}")
            print(f"File: {model_info['file']}")
            print(f"Main class: {model_info['class']}")
            print(f"Demo function: {model_info['main_function']}")
            
            print(f"\nRequirements:")
            for req in model_info['requirements']:
                status = "✅" if self.dependency_status.get(req, False) else "❌"
                print(f"  {status} {req}")
            
            print(f"\nKey Features:")
            for feature in model_info['features']:
                print(f"  • {feature}")
            print("-" * 50)
        
        input("Press Enter to continue...")
    
    def run_model(self, model_key: str, mode: str = 'demo'):
        """Run a specific model"""
        if model_key not in self.models:
            print(f"{Colors.RED}❌ Unknown model: {model_key}{Colors.END}")
            return
        
        model_info = self.models[model_key]
        
        # Check dependencies
        missing_deps = [dep for dep in model_info['requirements'] 
                       if not self.dependency_status.get(dep, False)]
        
        if missing_deps:
            print(f"{Colors.RED}❌ Missing dependencies: {', '.join(missing_deps)}{Colors.END}")
            print(f"{Colors.YELLOW}Install with: pip install {' '.join(missing_deps)}{Colors.END}")
            return
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}🚀 Launching {model_info['name']}...{Colors.END}")
        print(f"{Colors.CYAN}{model_info['description']}{Colors.END}")
        print()
        
        try:
            # Import and run the model
            module_name = model_info['file'].replace('.py', '').replace(' ', '_').lower()
            
            if model_key == 'quantum_compression':
                import quantum_compression_model
                print(f"{Colors.BOLD}Running Quantum Compression Demo...{Colors.END}")
                quantum_compression_model.main_quantum_compression_demo()
                
            elif model_key == 'quantum_gpt':
                # Import with proper module name handling
                spec = importlib.util.spec_from_file_location("quantum_gpt", "GPT Model.py")
                quantum_gpt_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(quantum_gpt_module)
                
                print(f"{Colors.BOLD}Running Quantum GPT Demo...{Colors.END}")
                quantum_gpt_module.demo_quantum_gpt()
            
            print(f"\n{Colors.GREEN}✅ {model_info['name']} demo completed successfully!{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error running {model_info['name']}:{Colors.END}")
            print(f"{Colors.RED}{str(e)}{Colors.END}")
            
            # Print traceback in debug mode
            if '--debug' in sys.argv:
                print(f"\n{Colors.YELLOW}Debug traceback:{Colors.END}")
                traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def benchmark_all_models(self):
        """Run benchmarks for all available models"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}🏁 Benchmarking All Quantum Models{Colors.END}")
        print("=" * 50)
        
        results = {}
        
        for model_key, model_info in self.models.items():
            missing_deps = [dep for dep in model_info['requirements'] 
                           if not self.dependency_status.get(dep, False)]
            
            if missing_deps:
                print(f"{Colors.YELLOW}⚠️ Skipping {model_info['name']} - missing dependencies{Colors.END}")
                continue
            
            print(f"\n{Colors.BOLD}🔬 Benchmarking {model_info['name']}...{Colors.END}")
            
            try:
                start_time = time.time()
                
                if model_key == 'quantum_compression':
                    import quantum_compression_model
                    # Run a quick compression test
                    model, result = quantum_compression_model.main_quantum_compression_demo()
                    results[model_key] = {
                        'status': 'success',
                        'time': time.time() - start_time,
                        'metrics': result
                    }
                    
                elif model_key == 'quantum_gpt':
                    spec = importlib.util.spec_from_file_location("quantum_gpt", "GPT Model.py")
                    quantum_gpt_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(quantum_gpt_module)
                    
                    model, tokenizer = quantum_gpt_module.demo_quantum_gpt()
                    results[model_key] = {
                        'status': 'success',
                        'time': time.time() - start_time,
                        'model': model
                    }
                
                print(f"{Colors.GREEN}✅ {model_info['name']} benchmark completed{Colors.END}")
                
            except Exception as e:
                print(f"{Colors.RED}❌ {model_info['name']} benchmark failed: {e}{Colors.END}")
                results[model_key] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Print benchmark summary
        print(f"\n{Colors.BOLD}{Colors.BLUE}📊 BENCHMARK SUMMARY{Colors.END}")
        print("=" * 50)
        
        for model_key, result in results.items():
            model_name = self.models[model_key]['name']
            
            if result['status'] == 'success':
                time_taken = result['time']
                print(f"{Colors.GREEN}✅ {model_name}: {time_taken:.2f}s{Colors.END}")
            else:
                print(f"{Colors.RED}❌ {model_name}: {result['error']}{Colors.END}")
        
        input("\nPress Enter to continue...")
    
    def run_specific_model(self, model_name: str, mode: str):
        """Run a specific model from command line"""
        # Handle different name formats
        model_key = None
        for key, info in self.models.items():
            if (key == model_name or 
                info['name'].lower().replace(' ', '_') == model_name.lower() or
                model_name.lower() in key.lower()):
                model_key = key
                break
        
        if not model_key:
            print(f"{Colors.RED}❌ Model '{model_name}' not found{Colors.END}")
            self.list_models()
            return
        
        self.run_model(model_key, mode)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Universal Quantum ML Models Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantum_launcher.py                          # Interactive mode
  python quantum_launcher.py --list                   # List all models
  python quantum_launcher.py --model quantum_gpt --demo
  python quantum_launcher.py --model compression --benchmark
  python quantum_launcher.py --benchmark --all
        """
    )
    
    parser.add_argument('--model', '-m', help='Model to run (quantum_gpt, quantum_compression)')
    parser.add_argument('--mode', choices=['demo', 'benchmark', 'interactive'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run benchmarks')
    parser.add_argument('--all', '-a', action='store_true', help='Run all models')
    parser.add_argument('--deps', action='store_true', help='Check dependencies')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    launcher = QuantumMLLauncher()
    
    try:
        if args.list:
            launcher.print_header()
            launcher.list_models()
            
        elif args.deps:
            launcher.print_header()
            launcher.print_dependency_status()
            
        elif args.benchmark and args.all:
            launcher.print_header()
            launcher.benchmark_all_models()
            
        elif args.model:
            launcher.print_header()
            mode = 'benchmark' if args.benchmark else args.mode
            launcher.run_specific_model(args.model, mode)
            
        else:
            # Interactive mode
            launcher.interactive_menu()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        if args.debug:
            traceback.print_exc()

if __name__ == "__main__":
    main()
