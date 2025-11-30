#!/usr/bin/env python3
"""
Index Physiology-Celestial-Chemistry data into VectorStore for agent access.
This enables recursive analysis and deeper understanding generation.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from iceburg.config import load_config
from iceburg.vectorstore import VectorStore

def chunk_data_for_indexing(data: dict, prefix: str = "") -> list[tuple[str, dict]]:
    """
    Chunk large data structures into searchable text segments.
    Returns list of (text, metadata) tuples.
    """
    chunks = []
    
    for key, value in data.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            chunks.extend(chunk_data_for_indexing(value, current_path))
        elif isinstance(value, list):
            # Process list items
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    chunks.extend(chunk_data_for_indexing(item, f"{current_path}[{i}]"))
                else:
                    text = f"{current_path}[{i}]: {item}"
                    chunks.append((text, {
                        "source": "physiology_celestial_chemistry",
                        "data_path": f"{current_path}[{i}]",
                        "data_type": type(item).__name__
                    }))
        else:
            # Leaf value - create searchable text
            text = f"{current_path}: {value}"
            chunks.append((text, {
                "source": "physiology_celestial_chemistry",
                "data_path": current_path,
                "data_type": type(value).__name__
            }))
    
    return chunks

def create_summary_chunks(data: dict) -> list[tuple[str, dict]]:
    """Create high-level summary chunks for better semantic search."""
    chunks = []
    
    # Celestial bodies summary
    if "celestial_bodies" in data:
        for body, info in data["celestial_bodies"].items():
            summary = f"Celestial body {body} correlates with organ system {info.get('organ_system', 'unknown')}. "
            if "voltage_gates" in info:
                summary += f"Primary voltage gates: {', '.join(info['voltage_gates'].get('primary', []))}. "
            if "neurotransmitters" in info:
                for nt, nt_info in info["neurotransmitters"].items():
                    summary += f"Neurotransmitter {nt} (correlation: {nt_info.get('correlation_strength', 'unknown')}). "
            if "hormones" in info:
                for h, h_info in info["hormones"].items():
                    summary += f"Hormone {h} (correlation: {h_info.get('correlation_strength', 'unknown')}). "
            
            chunks.append((summary, {
                "source": "physiology_celestial_chemistry",
                "type": "celestial_body_summary",
                "celestial_body": body,
                "organ_system": info.get("organ_system", "")
            }))
    
    # Voltage gates summary
    if "voltage_gated_ion_channels" in data:
        for channel, info in data["voltage_gated_ion_channels"].items():
            summary = f"Voltage-gated {channel} channel: resting potential {info.get('resting_potential_mv', 'unknown')} mV, "
            summary += f"activation threshold {info.get('activation_threshold_mv', 'unknown')} mV, "
            summary += f"celestial modulation {info.get('celestial_modulation', 'unknown')}, "
            summary += f"behavioral correlation: {info.get('behavioral_correlation', 'unknown')}. "
            if "associated_celestial_body" in info:
                summary += f"Associated with {info['associated_celestial_body']}. "
            if "associated_neurotransmitter" in info:
                summary += f"Interacts with {info['associated_neurotransmitter']} neurotransmitter."
            
            chunks.append((summary, {
                "source": "physiology_celestial_chemistry",
                "type": "voltage_gate_summary",
                "channel_type": channel,
                "celestial_body": info.get("associated_celestial_body", "")
            }))
    
    # Neurotransmitter summary
    if "neurotransmitters" in data:
        for nt, info in data["neurotransmitters"].items():
            summary = f"Neurotransmitter {nt}: {info.get('chemical_structure', 'unknown')} structure. "
            if "celestial_correlation" in info:
                corr = info["celestial_correlation"]
                summary += f"Correlates with {corr.get('planet', 'unknown')} (strength: {corr.get('correlation_strength', 'unknown')}). "
            summary += f"Functions: {', '.join(info.get('functions', []))}. "
            
            chunks.append((summary, {
                "source": "physiology_celestial_chemistry",
                "type": "neurotransmitter_summary",
                "neurotransmitter": nt,
                "celestial_body": info.get("celestial_correlation", {}).get("planet", "")
            }))
    
    # Hormone summary
    if "hormones" in data:
        for h, info in data["hormones"].items():
            summary = f"Hormone {h}: {info.get('chemical_structure', 'unknown')} structure. "
            if "celestial_correlation" in info:
                corr = info["celestial_correlation"]
                summary += f"Correlates with {corr.get('planet', 'unknown')} (strength: {corr.get('correlation_strength', 'unknown')}, "
                summary += f"confidence: {corr.get('confidence', 'unknown')}). "
            summary += f"Functions: {', '.join(info.get('functions', []))}. "
            
            chunks.append((summary, {
                "source": "physiology_celestial_chemistry",
                "type": "hormone_summary",
                "hormone": h,
                "celestial_body": info.get("celestial_correlation", {}).get("planet", "")
            }))
    
    return chunks

def index_physiology_celestial_data():
    """Index physiology-celestial-chemistry data into VectorStore."""
    cfg = load_config()
    vs = VectorStore(cfg)
    
    # Load JSON data
    data_file = Path('data/physiology_celestial_chemistry_data.json')
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading physiology-celestial-chemistry data...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract main data structure
    main_data = data.get("physiology_celestial_chemistry_mapping", {})
    
    # Create summary chunks for better semantic search
    print("📝 Creating summary chunks...")
    summary_chunks = create_summary_chunks(main_data)
    
    # Create detailed chunks
    print("🔍 Creating detailed data chunks...")
    detailed_chunks = chunk_data_for_indexing(main_data)
    
    # Combine summary and detailed chunks
    all_chunks = summary_chunks + detailed_chunks
    
    # Also index the markdown document
    md_file = Path('docs/PHYSIOLOGY_CELESTIAL_CHEMISTRY_MAPPING.md')
    if md_file.exists():
        print(f"📄 Indexing markdown documentation...")
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Split markdown into sections
        sections = md_content.split('\n## ')
        for i, section in enumerate(sections):
            if section.strip():
                section_title = section.split('\n')[0] if section else f"Section {i}"
                all_chunks.append((section, {
                    "source": "physiology_celestial_chemistry",
                    "type": "markdown_section",
                    "section_title": section_title,
                    "document": "PHYSIOLOGY_CELESTIAL_CHEMISTRY_MAPPING.md"
                }))
    
    print(f"📚 Indexing {len(all_chunks)} chunks into VectorStore...")
    
    # Index in batches
    batch_size = 50
    indexed_count = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [chunk[0] for chunk in batch]
        metadatas = [chunk[1] for chunk in batch]
        
        try:
            vs.add(texts=texts, metadatas=metadatas)
            indexed_count += len(batch)
            print(f"  ✅ Indexed batch {i//batch_size + 1} ({indexed_count}/{len(all_chunks)})")
        except Exception as e:
            print(f"  ❌ Error indexing batch {i//batch_size + 1}: {e}")
    
    print(f"\n✅ Successfully indexed {indexed_count} chunks")
    print(f"📊 Data is now accessible to agents via semantic search")
    print(f"🔬 Agents can now perform recursive analysis on this data")

if __name__ == "__main__":
    index_physiology_celestial_data()

