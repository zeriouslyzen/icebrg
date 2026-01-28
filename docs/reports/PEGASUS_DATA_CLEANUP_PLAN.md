# PEGASUS Data Cleanup & Full Fix Plan

## Problem Summary

**Critical Data Integrity Issue:**
- 1.4M relationships in database
- Only 30,162 (2.1%) are valid (both entities exist)
- 1,398,712 (97.9%) are invalid (missing entities)
- Result: Network queries return center node but no edges

**Root Cause:**
- Relationships populated from OpenSanctions JSON
- Entity IDs in relationships don't match entity IDs in entities table
- Likely ID format mismatch or incomplete entity import

## Solution Strategy

Two-phase approach:
1. **Data Cleanup**: Remove invalid relationships, keep only valid ones
2. **Code Optimization**: Improve network queries to work with cleaned data

## Implementation Plan

### Phase 1: Data Cleanup Script

**File**: `scripts/clean_matrix_relationships.py` (new file)

**Purpose**: Remove invalid relationships, keep only those where both entities exist

### Phase 2: Optimize Network Query

**File**: `src/iceburg/colossus/matrix_store.py`

**Changes**: Use JOIN in relationship query to only get valid relationships

### Phase 3: Add Data Quality Endpoint

**File**: `src/iceburg/colossus/api.py`

**New Endpoint**: `GET /api/colossus/data-quality`

### Phase 4: Add Cleanup API Endpoint

**File**: `src/iceburg/colossus/api.py`

**New Endpoint**: `POST /api/colossus/cleanup`

### Phase 5: Update Frontend

**File**: `frontend/pegasus.html`

**Changes**: Add data quality indicators and cleanup UI

### Phase 6: Test with Cleaned Data

Verify all pathways work with cleaned data

## Expected Results

**Before Cleanup:**
- 1,428,874 relationships (2.1% valid)
- Network queries return 0 edges

**After Cleanup:**
- ~30,162 relationships (100% valid)
- Network queries return edges for connected entities
- Graph visualization shows connected nodes
