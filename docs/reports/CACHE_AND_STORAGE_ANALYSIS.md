# Cache & Storage Analysis - ICEBURG Project

## 1. Cache Cleanup Safety for ICEBURG

### ✅ SAFE TO CLEAN (Won't affect ICEBURG data)

**System-Level Caches:**
- `~/Library/Caches/com.apple.python` - **615MB** (519MB actual)
  - System Python cache, not ICEBURG-specific
  - Safe to clean: `rm -rf ~/Library/Caches/com.apple.python/*`
  
- `~/Library/Caches/pip` - **368MB** (367MB actual)
  - pip package download cache
  - Safe to clean: `pip cache purge`
  - Will re-download packages when needed (slower installs temporarily)

- `~/Library/Caches/Homebrew` - **146MB**
  - Homebrew package cache
  - Safe to clean: `brew cleanup --prune=all`

**Total Safe to Clean: ~1.1GB**

### ⚠️ DO NOT CLEAN (ICEBURG Project Data)

**Active ICEBURG Data:**
- `~/Documents/iceburg_matrix/matrix.db` - **1.7GB** ✅ ACTIVE DATABASE
- `~/Documents/iceburg_matrix/opensanctions/` - **2.4GB** ✅ SOURCE DATA
- `/Users/jackdanger/Desktop/Projects/iceburg/data/chroma/` - **396KB** ✅ VECTOR DB
- `/Users/jackdanger/Desktop/Projects/iceburg/models/` - **3.9GB** ✅ ML MODELS

**ICEBURG Code References:**
The project actively uses:
- `~/Documents/iceburg_matrix/matrix.db` (referenced in `matrix_store.py`, `api.py`, `clean_matrix_relationships.py`)
- `~/Documents/iceburg_matrix/opensanctions/opensanctions_sanctions.json` (referenced in `api.py`, `populate_matrix_relationships.py`)

## 2. Other ICEBURG Projects Found

### Main Project (Active)
- **`/Users/jackdanger/Desktop/Projects/iceburg`** - **9.3GB**
  - Current working project
  - Contains all source code, models, frontend
  - **KEEP THIS**

### Duplicate/Backup Projects
- **`/Users/jackdanger/Desktop/ICEBERG2/iceburg`** - Size unknown
  - Appears to be a duplicate/backup copy
  - Contains similar structure (docs, examples, models)
  - **CAN BE ARCHIVED/DELETED** if confirmed duplicate

- **`/Users/jackdanger/Desktop/iceburg_backup`** - Size unknown
  - Backup directory
  - **CAN BE ARCHIVED/DELETED** if confirmed backup

### Documents Folder ICEBURG Data
- **`~/Documents/iceburg_matrix/`** - **4.1GB** ✅ ACTIVE
  - `matrix.db`: 1.7GB (active database)
  - `opensanctions/`: 2.4GB (source data)
  - **KEEP THIS** - actively used by project

- **`~/Documents/iceburg_backups/`** - **24MB**
  - Contains backup from Nov 7, 2025
  - **CAN BE ARCHIVED** if backup is no longer needed

- **`~/Documents/iceburg_data/`** - **2.8MB**
  - Investigation data storage
  - **KEEP THIS** - small, actively used

## 3. Documents Folder Analysis (4.2GB Total)

### Breakdown:
1. **`iceburg_matrix/`** - **4.1GB** (97% of Documents)
   - `matrix.db`: **1.7GB** - SQLite database with 1.5M entities, 1.4M relationships
   - `opensanctions/opensanctions_sanctions.json`: **2.4GB** - Source JSON data file
   - `fec/`, `icij/`: Empty directories

2. **`iceburg_backups/`** - **24MB**
   - Backup from November 2025
   - Multiple backup versions

3. **`iceburg_data/`** - **2.8MB**
   - Investigation storage
   - Small, actively used

### What's Taking Up Space:

**The 2.4GB `opensanctions_sanctions.json` file is the largest item in Documents.**

This is source data used to populate the matrix database. It's referenced in:
- `src/populate_matrix_relationships.py`
- `src/iceburg/colossus/api.py`

**Options:**
1. **Keep it** - Needed for re-populating database if needed
2. **Archive it** - Move to external storage if database is fully populated
3. **Compress it** - Could save ~50-70% space with gzip/bzip2

## Recommendations

### Immediate Actions (Free ~1.1GB safely):

```bash
# 1. Clean pip cache (368MB)
pip cache purge

# 2. Clean Homebrew cache (146MB)
brew cleanup --prune=all

# 3. Clean Python system cache (615MB) - optional, more aggressive
rm -rf ~/Library/Caches/com.apple.python/*
```

### Storage Optimization:

1. **Archive Duplicate Projects** (if confirmed duplicates):
   - `/Users/jackdanger/Desktop/ICEBERG2/iceburg`
   - `/Users/jackdanger/Desktop/iceburg_backup`
   - Could free several GB

2. **Compress opensanctions JSON** (save ~1-1.5GB):
   ```bash
   cd ~/Documents/iceburg_matrix/opensanctions
   gzip opensanctions_sanctions.json
   # Update code to handle .json.gz if needed
   ```

3. **Archive Old Backups** (24MB):
   - `~/Documents/iceburg_backups/` - move to external storage

### Summary

- **Cache cleanup**: ✅ Safe, will free ~1.1GB
- **ICEBURG data**: ⚠️ Keep all active data (matrix.db, opensanctions, models)
- **Documents space**: 4.1GB is mostly the opensanctions JSON (2.4GB) + matrix.db (1.7GB)
- **Duplicate projects**: Found ICEBERG2 and iceburg_backup - can archive if duplicates
