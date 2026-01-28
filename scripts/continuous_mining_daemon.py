#!/usr/bin/env python3
"""
Continuous Mining Daemon
Keeps mining data from multiple sources continuously
"""

import time
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'mining' / 'daemon.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config():
    """Load mining configuration."""
    config_file = Path(__file__).parent.parent / 'config' / 'mining_config.json'
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {
        'enabled': True,
        'interval_seconds': 3600,
        'sources': {
            'web_intelligence': True,
            'wayback_machine': True,
            'county_portals': True,
            'business_records': True
        }
    }

def run_miner(script_name, wait_time=300):
    """Run a miner script and wait before next run."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logger.warning(f"Script not found: {script_name}")
        return
    
    while True:
        try:
            logger.info(f"Running: {script_name}")
            print(f"\n{'='*80}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running: {script_name}")
            print(f"{'='*80}\n")
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per run
            )
            
            if result.returncode == 0:
                print(result.stdout)
                logger.info(f"{script_name} completed successfully")
            else:
                print(f"Error: {result.stderr}")
                logger.error(f"{script_name} failed: {result.stderr}")
            
            print(f"\n⏳ Waiting {wait_time} seconds before next run...")
            logger.info(f"Waiting {wait_time} seconds before next {script_name} run")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            logger.info("Mining daemon stopped by user")
            print("\n🛑 Mining daemon stopped")
            break
        except subprocess.TimeoutExpired:
            logger.warning(f"{script_name} timed out after 30 minutes")
            print(f"⚠️  {script_name} timed out")
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error running {script_name}: {e}", exc_info=True)
            print(f"⚠️  Error: {e}")
            time.sleep(60)  # Wait 1 minute on error

def run_all_miners(config):
    """Run all enabled miners in sequence."""
    miners = []
    
    if config['sources'].get('web_intelligence', True):
        miners.append(('mine_strombeck_intelligence.py', 300))
    
    if config['sources'].get('wayback_machine', True):
        miners.append(('wayback_miner.py', 600))  # Longer wait for Wayback
    
    if config['sources'].get('county_portals', True):
        miners.append(('selenium_county_scraper.py', 900))  # Even longer for Selenium
    
    if config['sources'].get('business_records', True):
        miners.append(('deep_property_miner.py', 300))
    
    interval = config.get('interval_seconds', 3600)
    
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"Starting mining cycle #{cycle}")
        print(f"\n{'='*80}")
        print(f"MINING CYCLE #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        for script_name, wait_time in miners:
            if config.get('enabled', True):
                run_miner(script_name, wait_time)
            else:
                logger.info("Mining disabled in config")
                time.sleep(60)
        
        logger.info(f"Cycle #{cycle} complete. Waiting {interval} seconds until next cycle")
        print(f"\n✅ Cycle #{cycle} complete. Waiting {interval} seconds until next cycle...")
        time.sleep(interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous mining daemon')
    parser.add_argument('--script', type=str, help='Script to run continuously')
    parser.add_argument('--interval', type=int, help='Interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    config = load_config()
    
    if args.interval:
        config['interval_seconds'] = args.interval
    
    if args.script:
        # Run single script continuously
        run_miner(args.script, config.get('interval_seconds', 300))
    elif args.once:
        # Run all miners once
        if config['sources'].get('web_intelligence', True):
            run_miner('mine_strombeck_intelligence.py', 0)
        if config['sources'].get('wayback_machine', True):
            run_miner('wayback_miner.py', 0)
        if config['sources'].get('county_portals', True):
            run_miner('selenium_county_scraper.py', 0)
        if config['sources'].get('business_records', True):
            run_miner('deep_property_miner.py', 0)
    else:
        # Run all miners continuously
        run_all_miners(config)
