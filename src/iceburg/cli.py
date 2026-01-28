import json
import os
import sys
from pathlib import Path
from typing import Optional

import click

# Import existing modules only
try:
    from .blockchain.smart_contracts import BlockchainVerificationSystem
except ImportError:
    BlockchainVerificationSystem = None

from .config import load_config

# COREME Dual-Layer System imports
try:
    from .dual_layer_demo import demonstrate_dual_layer_system, test_radical_discovery
except ImportError:
    demonstrate_dual_layer_system = None
    test_radical_discovery = None

from .graph_store import KnowledgeGraph

try:
    from .interface import ANALYSIS_MODES, smart_interface_main
except ImportError:
    ANALYSIS_MODES = {}
    smart_interface_main = None

from .protocol import iceberg_protocol

try:
    from .suppression_resistant_storage import SuppressionResistantStorageSystem
except ImportError:
    SuppressionResistantStorageSystem = None

from .vectorstore import VectorStore

# Autonomous system imports
try:
    from .autonomous.research_orchestrator import AutonomousResearchOrchestrator
    from .evolution.evolution_pipeline import EvolutionPipeline
    from .monitoring.unified_performance_tracker import get_global_tracker
except ImportError:
    AutonomousResearchOrchestrator = None
    EvolutionPipeline = None
    get_global_tracker = None


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("query", type=str)
@click.option(
    "--out",
    "-o",
    type=click.Path(),
    help="Output file path for results",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(list(ANALYSIS_MODES.keys()) if ANALYSIS_MODES else ["default"]),
    default="default",
    help="Analysis mode to use",
)
def analyze(query: str, out: Optional[str], verbose: bool, mode: str) -> None:
    """Run ICEBURG analysis on a query."""
    if smart_interface_main:
        smart_interface_main(query, out, verbose, mode)
    else:
        click.echo("Smart interface not available")


@main.command()
@click.argument("query", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def research(query: str, out: Optional[str], verbose: bool) -> None:
    """Run ICEBURG research protocol on a query."""
    try:
        result = iceberg_protocol(query, verbose=verbose)
        if out:
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument("query", type=str)
@click.option("--depth", "-d", type=click.Choice(["quick", "standard", "deep"]), default="standard", help="Investigation depth")
@click.option("--out", "-o", type=click.Path(), help="Output file path for dossier markdown")
@click.option("--vault", is_flag=True, default=True, help="Save to investigation vault")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output with progress")
def investigate(query: str, depth: str, out: Optional[str], vault: bool, verbose: bool) -> None:
    """
    Run full ICEBURG investigation on a subject.
    
    This runs the complete Dossier Protocol:
    1. Gatherer - Multi-source intelligence collection
    2. Decoder - Symbol/pattern analysis  
    3. Mapper - Network relationship mapping
    4. Synthesizer - Final dossier compilation
    5. Indexer - Auto-ingest into PEGASUS graph
    
    Examples:
        iceburg investigate "Erik Strombeck asset hiding"
        iceburg investigate "Vladimir Putin" --depth deep --verbose
        iceburg investigate "Humboldt County property transfers" -o report.md
    """
    try:
        from .config import load_config
        from .protocols.dossier import DossierSynthesizer
        from .investigations import Investigation, get_investigation_store
        
        cfg = load_config()
        
        # Progress callback
        def progress(msg: str):
            if verbose:
                click.echo(f"  {msg}")
        
        if verbose:
            click.echo("🧊 ICEBURG Investigation Protocol")
            click.echo("=" * 50)
            click.echo(f"📋 Query: {query}")
            click.echo(f"🔍 Depth: {depth}")
            click.echo()
        
        # Run full Dossier Protocol
        synthesizer = DossierSynthesizer(cfg)
        dossier = synthesizer.generate_dossier(
            query=query, 
            depth=depth,
            thinking_callback=progress if verbose else None
        )
        
        if verbose:
            click.echo()
            click.echo("📊 Results:")
            click.echo(f"   Sources gathered: {dossier.metadata.get('total_sources', 0)}")
            click.echo(f"   Entities found: {dossier.metadata.get('entities_found', 0)}")
            click.echo(f"   Symbols detected: {dossier.metadata.get('symbols_detected', 0)}")
            click.echo(f"   Relationships mapped: {dossier.metadata.get('relationships_mapped', 0)}")
            click.echo(f"   Generation time: {dossier.metadata.get('generation_time_seconds', 0):.1f}s")
        
        # Save to vault if requested
        if vault:
            try:
                investigation = Investigation.from_dossier(dossier, query, depth)
                store = get_investigation_store()
                inv_id = store.save(investigation)
                if verbose:
                    click.echo(f"   Saved to vault: {inv_id}")
            except Exception as e:
                if verbose:
                    click.echo(f"   ⚠️ Vault save failed: {e}")
        
        # Output
        markdown = dossier.to_markdown()
        
        if out:
            with open(out, "w") as f:
                f.write(markdown)
            click.echo(f"\n✅ Dossier saved to: {out}")
        else:
            click.echo()
            click.echo(markdown)
        
        if verbose:
            click.echo()
            click.echo("✅ Investigation complete!")
            click.echo("   View in PEGASUS: http://localhost:8000/pegasus.html")
        
    except Exception as e:
        click.echo(f"❌ Investigation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
@click.argument("query", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def chat(query: str, out: Optional[str], verbose: bool) -> None:
    """Fast chat responses for simple questions (30s target)."""
    try:
        from .unified_interface import UnifiedICEBURG
        import asyncio
        
        iceburg = UnifiedICEBURG()
        result = asyncio.run(iceburg.process(query))
        
        if out:
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument("description", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--platform", default="macos", help="Target platform (macos, ios, android, web)")
@click.option("--framework", default="swiftui", help="Framework to use (swiftui, react, flutter)")
@click.option("--features", multiple=True, help="Features to include in the application")
def build(description: str, out: Optional[str], verbose: bool, platform: str, framework: str, features: tuple) -> None:
    """Build applications using ICEBURG's software lab."""
    try:
        from .unified_interface import UnifiedICEBURG
        import asyncio
        
        iceburg = UnifiedICEBURG()
        context = {
            "platform": platform,
            "framework": framework,
            "features": list(features) if features else []
        }
        result = asyncio.run(iceburg.process(description, context))
        
        if out:
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument("spec", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--steps", default=1000, help="Number of simulation steps")
def simulate(spec: str, out: Optional[str], verbose: bool, steps: int) -> None:
    """Simulate AGI civilization with world models and multi-agent systems."""
    try:
        from .unified_interface import UnifiedICEBURG
        import asyncio
        
        iceburg = UnifiedICEBURG()
        context = {"steps": steps}
        result = asyncio.run(iceburg.process(spec, context))
        
        if out:
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument("query", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def dual_layer(query: str, out: Optional[str], verbose: bool) -> None:
    """Run COREME dual-layer system analysis."""
    if demonstrate_dual_layer_system:
        result = demonstrate_dual_layer_system(query, verbose=verbose)
        if out:
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Dual-layer system not available")


@main.command()
@click.argument("query", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def radical_discovery(query: str, out: Optional[str], verbose: bool) -> None:
    """Run radical discovery analysis."""
    if test_radical_discovery:
        result = test_radical_discovery(query, verbose=verbose)
        if out:
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Radical discovery system not available")


@main.command()
@click.argument("query", type=str)
@click.option("--out", "-o", type=click.Path(), help="Output file path for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def package(query: str, out: Optional[str], verbose: bool) -> None:
    """Package ICEBURG analysis results."""
    try:
        result = iceberg_protocol(query, verbose=verbose)
        
        # Create package structure
        package_data = {
            "query": query,
            "result": result,
            "timestamp": json.dumps({"timestamp": "2024-01-01T00:00:00Z"}),
            "version": "1.0.0"
        }
        
        if out:
            with open(out, "w") as f:
                json.dump(package_data, f, indent=2)
        else:
            click.echo(json.dumps(package_data, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument('description')
@click.option('--out-dir', default='dist', help='Output directory for generated application')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--platform', default='macos', help='Target platform (macos, ios, android, web)')
@click.option('--framework', default='swiftui', help='Framework to use (swiftui, react, flutter)')
@click.option('--features', multiple=True, help='Features to include in the application')
@click.option('--sign', is_flag=True, help='Code sign the generated app (requires Developer ID)')
@click.option('--notarize', is_flag=True, help='Notarize the app for distribution (requires Apple Developer account)')
@click.option('--dmg', is_flag=True, help='Create DMG for distribution')
def build_app(description, out_dir, verbose, platform, framework, features, sign, notarize, dmg):
    """Build applications using ICEBURG's one-shot generation system."""
    try:
        from .agents.architect import Architect
        from .config import load_config
        from pathlib import Path
        
        config = load_config()
        architect = Architect(config)
        
        if verbose:
            click.echo("🧠 ICEBURG One-Shot App Builder")
            click.echo("=" * 50)
            click.echo(f"📝 Description: {description}")
            click.echo(f"🔧 Framework: {framework}")
            click.echo(f"⚙️ Features: {list(features) if features else 'default'}")
            click.echo(f"🖥️ Platform: {platform}")
            click.echo(f"📁 Output: {out_dir}")
            if sign:
                click.echo("🔐 Code signing: enabled")
            if notarize:
                click.echo("📋 Notarization: enabled")
            if dmg:
                click.echo("💿 DMG creation: enabled")
            click.echo()
        
        # Generate the application
        app_request = {
            'description': description,
            'app_type': 'desktop' if platform in ['macos'] else 'mobile' if platform in ['ios', 'android'] else 'web',
            'framework': framework,
            'features': list(features) if features else [],
            'platform': platform,
            'output_dir': str(Path(out_dir))
        }
        
        import asyncio
        result = asyncio.run(architect.generate_application(app_request, verbose=verbose))
        
        if verbose:
            click.echo(f"✅ Generated application: {result}")
        
        # Post-processing: signing, notarization, DMG creation
        if sign or notarize or dmg:
            from .packaging.macos_packager import MacOSPackager
            packager = MacOSPackager()
            
            if sign:
                if verbose:
                    click.echo("🔐 Code signing application...")
                packager.sign_app(result, verbose=verbose)
            
            if notarize:
                if verbose:
                    click.echo("📋 Notarizing application...")
                packager.notarize_app(result, verbose=verbose)
            
            if dmg:
                if verbose:
                    click.echo("💿 Creating DMG...")
                dmg_path = packager.create_dmg(result, verbose=verbose)
                if verbose:
                    click.echo(f"📦 DMG created: {dmg_path}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

@main.command()
@click.argument('description')
@click.option('--out-dir', default='dist', help='Output directory for generated application')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--platform', default='macos', help='Target platform (macos, ios, android, web)')
@click.option('--framework', default='swiftui', help='Framework to use (swiftui, react, flutter)')
@click.option('--features', multiple=True, help='Features to include in the application')
def build_intelligent(description, out_dir, verbose, platform, framework, features):
    """Generate applications using ICEBURG's autonomous development pipeline (legacy)"""
    try:
        from .agents.architect import Architect
        from .security.redteam import RedTeamAnalyzer
        from .security.review_gate import ReviewGate
        
        if verbose:
            click.echo("🚀 Starting ICEBURG autonomous app generation...")
        
        # Initialize ICEBURG components
        architect = Architect()
        
        # Prepare app specification
        app_spec = {
            'description': description,
            'app_type': 'desktop' if platform in ['macos'] else 'mobile' if platform in ['ios', 'android'] else 'web',
            'framework': framework,
            'features': list(features) if features else [],
            'platform': platform,
            'output_dir': out_dir
        }
        
        if verbose:
            click.echo(f"📋 App specification: {app_spec}")
        
        # Generate application
        success = architect.generate_application(app_spec, verbose=verbose)
        
        if success:
            click.echo("✅ Application generated successfully!")
            click.echo(f"📁 Output directory: {out_dir}")
        else:
            click.echo("❌ Application generation failed")
            return 1
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
@click.argument('problem')
@click.option('--departments', multiple=True, help='Departments to include in brainstorming')
@click.option('--scale', default=1, help='Scaling factor for think tank departments')
@click.option('--goals', multiple=True, help='Session goals for brainstorming')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def mega_brainstorm(problem, departments, scale, goals, verbose):
    """Start a mega brainstorming session across multiple think tank departments"""
    try:
        from .departments.mega_brainstorm import MegaBrainstormSystem
        
        if verbose:
            click.echo("🧠 Starting ICEBURG Mega Brainstorm Session...")
        
        # Initialize mega brainstorm system
        mega_system = MegaBrainstormSystem()
        
        # Prepare session context
        context = {
            'scale_factor': scale,
            'session_goals': list(goals) if goals else [],
            'verbose': verbose
        }
        
        if verbose:
            click.echo(f"📋 Problem: {problem}")
            click.echo(f"🏢 Departments: {list(departments) if departments else 'All available'}")
            click.echo(f"📈 Scale factor: {scale}")
        
        # Start mega brainstorm session
        session = mega_system.start_mega_brainstorm(
            problem=problem,
            context=context,
            participating_departments=list(departments) if departments else None,
            session_goals=list(goals) if goals else [],
            auto_scale=True
        )
        
        if verbose:
            click.echo(f"🎯 Session started: {session.session_id}")
            click.echo(f"👥 Participating departments: {len(session.participating_departments)}")
            click.echo(f"🤖 Total agents: {session.total_agents}")
        
        # Wait for session to complete (in a real implementation, this would be asynchronous)
        import time
        time.sleep(5)  # Simulate processing time
        
        # Generate synthesis
        synthesis = mega_system.synthesize_mega_brainstorm(session.session_id)
        
        if verbose:
            click.echo("📊 Mega Brainstorm Results:")
            click.echo(f"   Total ideas: {synthesis.get('total_ideas', 0)}")
            click.echo(f"   Collaborations: {synthesis.get('collaborations', 0)}")
            click.echo(f"   Synthesis quality: {synthesis.get('synthesis_quality', 0):.2f}")
        
        click.echo("✅ Mega brainstorm session completed!")
        return synthesis
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def think_tank_status(verbose):
    """Get status of think tank departments and brainstorming capabilities"""
    try:
        from .departments.mega_brainstorm import MegaBrainstormSystem
        
        if verbose:
            click.echo("📊 Getting ICEBURG Think Tank Status...")
        
        # Initialize mega brainstorm system
        mega_system = MegaBrainstormSystem()
        
        # Get status
        status = mega_system.get_mega_brainstorm_status()
        
        click.echo("🏢 ICEBURG Think Tank Status:")
        click.echo(f"   Active sessions: {status['active_sessions']}")
        click.echo(f"   Total sessions: {status['total_sessions']}")
        click.echo(f"   Total departments: {status['total_departments']}")
        click.echo(f"   Total agents: {status['total_agents']}")
        
        if verbose:
            click.echo("\n📈 Scaling Status:")
            scaling_status = status['scaling_status']
            click.echo(f"   CPU usage: {scaling_status['system_metrics']['cpu_usage']:.2%}")
            click.echo(f"   Memory usage: {scaling_status['system_metrics']['memory_usage']:.2%}")
            click.echo(f"   Active tasks: {scaling_status['system_metrics']['active_tasks']}")
            click.echo(f"   Monitoring active: {scaling_status['monitoring_active']}")
        
        return status
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
@click.argument('description')
@click.option('--out-dir', default='dist', help='Output directory for generated application')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--sign', is_flag=True, help='Code sign the generated app (requires Developer ID)')
@click.option('--notarize', is_flag=True, help='Notarize the app for distribution (requires Apple Developer account)')
@click.option('--dmg', is_flag=True, help='Create DMG for distribution')
@click.option('--bundle-id', help='Custom bundle identifier (e.g., com.company.appname)')
@click.option('--team-id', help='Apple Developer Team ID for signing')
@click.option('--app-name', help='Custom app name (defaults to generated name)')
def one_shot(description, out_dir, verbose, sign, notarize, dmg, bundle_id, team_id, app_name):
    """One-shot macOS app generation with full build/sign/notarize/DMG pipeline."""
    try:
        from .agents.architect import Architect
        from .packaging.macos_packager import MacOSPackager
        from .packaging.info_plist_generator import InfoPlistGenerator
        from .packaging.entitlements_generator import EntitlementsGenerator
        from pathlib import Path
        import asyncio
        
        if verbose:
            click.echo("🚀 ICEBURG One-Shot macOS App Builder")
            click.echo("=" * 50)
            click.echo(f"📝 Description: {description}")
            click.echo(f"📁 Output: {out_dir}")
            click.echo(f"🔐 Code signing: {'enabled' if sign else 'disabled'}")
            click.echo(f"📋 Notarization: {'enabled' if notarize else 'disabled'}")
            click.echo(f"💿 DMG creation: {'enabled' if dmg else 'disabled'}")
            if bundle_id:
                click.echo(f"🆔 Bundle ID: {bundle_id}")
            if team_id:
                click.echo(f"👥 Team ID: {team_id}")
            if app_name:
                click.echo(f"📱 App Name: {app_name}")
            click.echo()
        
        # Initialize components
        config = load_config()
        architect = Architect(config)
        packager = MacOSPackager()
        info_plist_gen = InfoPlistGenerator()
        entitlements_gen = EntitlementsGenerator()
        
        # Generate the application
        app_request = {
            'description': description,
            'app_type': 'desktop',
            'framework': 'swiftui',
            'features': ['native_macos', 'code_signing', 'notarization'],
            'platform': 'macos',
            'output_dir': str(Path(out_dir)),
            'bundle_id': bundle_id,
            'app_name': app_name
        }
        
        if verbose:
            click.echo("🧠 Generating application with ICEBURG Architect...")
        
        result = asyncio.run(architect.generate_application(app_request, verbose=verbose))
        
        if verbose:
            click.echo(f"✅ Generated application: {result}")
        
        # Generate Info.plist
        if verbose:
            click.echo("📋 Generating Info.plist...")
        
        info_plist_path = Path(result) / "Contents" / "Info.plist"
        info_plist_gen.generate_info_plist(
            bundle_id=bundle_id or "com.iceburg.generated",
            app_name=app_name or "ICEBURG App",
            version="1.0.0",
            output_path=str(info_plist_path),
            verbose=verbose
        )
        
        # Generate entitlements
        if verbose:
            click.echo("🔐 Generating entitlements...")
        
        entitlements_path = Path(result) / "Contents" / "entitlements.plist"
        entitlements_gen.generate_entitlements(
            bundle_id=bundle_id or "com.iceburg.generated",
            output_path=str(entitlements_path),
            verbose=verbose
        )
        
        # Code signing
        if sign:
            if verbose:
                click.echo("🔐 Code signing application...")
            
            success = packager.sign_app(
                app_path=result,
                team_id=team_id,
                verbose=verbose
            )
            
            if not success:
                click.echo("❌ Code signing failed")
                return 1
        
        # Notarization
        if notarize:
            if verbose:
                click.echo("📋 Notarizing application...")
            
            success = packager.notarize_app(
                app_path=result,
                team_id=team_id,
                verbose=verbose
            )
            
            if not success:
                click.echo("❌ Notarization failed")
                return 1
        
        # DMG creation
        if dmg:
            if verbose:
                click.echo("💿 Creating DMG...")
            
            dmg_path = packager.create_dmg(
                app_path=result,
                output_dir=out_dir,
                verbose=verbose
            )
            
            if dmg_path:
                click.echo(f"📦 DMG created: {dmg_path}")
            else:
                click.echo("❌ DMG creation failed")
                return 1
        
        # Final status
        click.echo("✅ One-shot macOS app generation completed!")
        click.echo(f"📁 App location: {result}")
        
        if dmg:
            click.echo(f"💿 DMG location: {dmg_path}")
        
        if sign:
            click.echo("🔐 App is code signed")
        
        if notarize:
            click.echo("📋 App is notarized and ready for distribution")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@main.group()
def autonomous():
    """Autonomous ICEBURG operations."""
    pass


@autonomous.command()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--max-queries", default=5, help="Maximum concurrent queries")
@click.option("--cycle-interval", default=300, help="Research cycle interval in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def start_research(config, max_queries, cycle_interval, verbose):
    """Start autonomous research mode."""
    if not AutonomousResearchOrchestrator:
        click.echo("❌ Autonomous research not available - missing dependencies")
        return 1
    
    try:
        import asyncio
        
        # Load configuration
        config_data = {}
        if config:
            with open(config, 'r') as f:
                config_data = json.load(f)
        
        # Add CLI options to config
        config_data.update({
            "max_concurrent_queries": max_queries,
            "research_cycle_interval": cycle_interval
        })
        
        # Create orchestrator
        orchestrator = AutonomousResearchOrchestrator(config_data)
        
        async def run_research():
            await orchestrator.start_autonomous_research()
            
            if verbose:
                click.echo("🔬 Autonomous research started")
                click.echo(f"   Max concurrent queries: {max_queries}")
                click.echo(f"   Cycle interval: {cycle_interval}s")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(10)
                    status = orchestrator.get_research_status()
                    if verbose:
                        click.echo(f"   Status: {status['cycle_count']} cycles, {status['current_queries']} active queries")
            except KeyboardInterrupt:
                click.echo("\n🛑 Stopping autonomous research...")
                await orchestrator.stop_autonomous_research()
                click.echo("✅ Autonomous research stopped")
        
        # Run the research loop
        asyncio.run(run_research())
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error starting autonomous research: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@autonomous.command()
@click.option("--trigger", default="manual", help="Trigger reason for evolution")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def evolve(trigger, config, verbose):
    """Trigger system evolution."""
    if not EvolutionPipeline:
        click.echo("❌ Evolution pipeline not available - missing dependencies")
        return 1
    
    try:
        import asyncio
        
        # Load configuration
        config_data = {}
        if config:
            with open(config, 'r') as f:
                config_data = json.load(f)
        
        # Create pipeline
        pipeline = EvolutionPipeline(config_data)
        
        async def run_evolution():
            if verbose:
                click.echo(f"🧬 Starting system evolution (trigger: {trigger})")
            
            job_id = await pipeline.evolve_system(trigger)
            
            if verbose:
                click.echo(f"   Job ID: {job_id}")
            
            # Monitor progress
            while True:
                status = pipeline.get_job_status(job_id)
                if not status:
                    click.echo("❌ Job not found")
                    return 1
                
                if verbose:
                    click.echo(f"   Stage: {status['stage']} - {status['status']}")
                
                if status['status'] in ['completed', 'failed']:
                    break
                
                await asyncio.sleep(5)
            
            # Show final results
            if status['status'] == 'completed':
                click.echo("✅ Evolution completed successfully")
            else:
                click.echo(f"❌ Evolution failed: {status.get('error_message', 'Unknown error')}")
                return 1
            
            return 0
        
        # Run evolution
        return asyncio.run(run_evolution())
        
    except Exception as e:
        click.echo(f"❌ Error during evolution: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@autonomous.command()
@click.option("--hours", default=24, help="Hours of data to analyze")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def status(hours, verbose):
    """Show autonomous system status."""
    try:
        # Performance tracker status
        if get_global_tracker:
            tracker = get_global_tracker()
            summary = tracker.get_performance_summary(hours)
            
            click.echo("📊 Performance Status:")
            click.echo(f"   Time period: {hours} hours")
            click.echo(f"   Total queries: {summary.get('total_queries', 0)}")
            click.echo(f"   Success rate: {summary.get('success_rate', 0):.1%}")
            
            averages = summary.get('averages', {})
            if averages:
                click.echo(f"   Avg response time: {averages.get('response_time', 0):.2f}s")
                click.echo(f"   Avg accuracy: {averages.get('accuracy', 0):.2f}")
                click.echo(f"   Memory usage: {averages.get('memory_usage_mb', 0):.0f}MB")
                click.echo(f"   CPU usage: {averages.get('cpu_usage_percent', 0):.1f}%")
            
            regressions = summary.get('recent_regressions', [])
            if regressions:
                click.echo(f"   Recent regressions: {len(regressions)}")
                for regression in regressions:
                    click.echo(f"     - {regression.metric_name}: {regression.regression_percent:.1f}%")
        else:
            click.echo("❌ Performance tracker not available")
        
        # Evolution pipeline status
        if EvolutionPipeline:
            pipeline = EvolutionPipeline()
            pipeline_status = pipeline.get_pipeline_status()
            
            click.echo("\n🧬 Evolution Pipeline Status:")
            click.echo(f"   Total jobs: {pipeline_status['total_jobs']}")
            click.echo(f"   Active jobs: {pipeline_status['active_jobs']}")
            click.echo(f"   Completed jobs: {pipeline_status['completed_jobs']}")
            click.echo(f"   Failed jobs: {pipeline_status['failed_jobs']}")
            click.echo(f"   Success rate: {pipeline_status['success_rate']:.1%}")
        else:
            click.echo("\n❌ Evolution pipeline not available")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@autonomous.command()
@click.option("--job-id", help="Specific job ID to show")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def jobs(job_id, verbose):
    """Show evolution jobs."""
    if not EvolutionPipeline:
        click.echo("❌ Evolution pipeline not available - missing dependencies")
        return 1
    
    try:
        pipeline = EvolutionPipeline()
        
        if job_id:
            # Show specific job
            status = pipeline.get_job_status(job_id)
            if not status:
                click.echo(f"❌ Job {job_id} not found")
                return 1
            
            click.echo(f"Job {job_id}:")
            click.echo(f"   Stage: {status['stage']}")
            click.echo(f"   Status: {status['status']}")
            click.echo(f"   Created: {status['created_at']}")
            if status['started_at']:
                click.echo(f"   Started: {status['started_at']}")
            if status['completed_at']:
                click.echo(f"   Completed: {status['completed_at']}")
            if status['error_message']:
                click.echo(f"   Error: {status['error_message']}")
        else:
            # Show all jobs
            all_jobs = pipeline.get_all_jobs()
            if not all_jobs:
                click.echo("No jobs found")
                return 0
            
            click.echo("Evolution Jobs:")
            for job in all_jobs:
                status_emoji = "✅" if job['status'] == 'completed' else "❌" if job['status'] == 'failed' else "🔄"
                click.echo(f"   {status_emoji} {job['job_id'][:8]} - {job['stage']} - {job['status']}")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error getting jobs: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()