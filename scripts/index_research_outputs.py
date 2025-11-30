#!/usr/bin/env python3
"""Index ICEBURG's research outputs into VectorStore for agent access."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from iceburg.config import load_config
from iceburg.vectorstore import VectorStore

def index_research_outputs():
    """Index all research outputs into VectorStore."""
    cfg = load_config()
    vs = VectorStore(cfg)
    
    research_dir = Path('data/research_outputs')
    if not research_dir.exists():
        print(f"❌ Research outputs directory not found: {research_dir}")
        return
    
    md_files = list(research_dir.glob('*.md'))
    print(f"📚 Found {len(md_files)} research output files")
    print("")
    
    indexed_count = 0
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content.strip()) < 50:  # Skip very short files
                continue
            
            # Split into chunks if very long
            max_chunk_size = 5000
            chunks = []
            if len(content) > max_chunk_size:
                # Split by sections (## headers)
                sections = content.split('\n## ')
                current_chunk = sections[0] if sections else content
                for section in sections[1:]:
                    if len(current_chunk) + len(section) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = '## ' + section
                    else:
                        current_chunk += '\n## ' + section
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks = [content]
            
            # Index each chunk
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                
                chunk_id = vs.add(
                    texts=[chunk],
                    metadatas=[{
                        'source': 'research_outputs',
                        'file': md_file.name,
                        'chunk': i,
                        'type': 'research',
                        'iceburg_research': True
                    }]
                )
                indexed_count += 1
                print(f"✅ Indexed: {md_file.name} (chunk {i+1}/{len(chunks)})")
        
        except Exception as e:
            print(f"❌ Error indexing {md_file.name}: {e}")
    
    print("")
    print(f"✅ Successfully indexed {indexed_count} research chunks")
    print("")
    print("Agents can now access ICEBURG's research outputs!")

if __name__ == '__main__':
    index_research_outputs()

