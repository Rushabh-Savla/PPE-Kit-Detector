"""
Script to update existing face embeddings with worker display names from database.
Run this once to migrate from old format (just embeddings) to new format (embeddings + names).
"""
import asyncio
import pickle
from pathlib import Path
from database import connect_to_mongodb, get_database

async def update_face_names():
    """Update face embeddings with worker names from database."""
    await connect_to_mongodb()
    db = get_database()

    embeddings_file = Path("../known_faces/embeddings.pkl")

    if not embeddings_file.exists():
        print("No embeddings file found. Nothing to update.")
        return

    # Load existing embeddings
    with open(embeddings_file, 'rb') as f:
        known_faces = pickle.load(f)

    print(f"Found {len(known_faces)} registered faces")

    updated = 0
    for employee_id, face_data in known_faces.items():
        # Skip if already in new format
        if isinstance(face_data, dict):
            print(f"  {employee_id}: Already has display name - {face_data.get('display_name')}")
            continue

        # Look up worker in database
        worker = await db.workers.find_one({"employee_id": employee_id})

        if worker:
            display_name = worker["name"]
            print(f"  {employee_id}: Adding display name - {display_name}")

            # Convert to new format
            known_faces[employee_id] = {
                "embedding": face_data,
                "display_name": display_name
            }
            updated += 1
        else:
            print(f"  {employee_id}: Worker not found in database, keeping as-is")
            # Convert to new format with employee_id as display name
            known_faces[employee_id] = {
                "embedding": face_data,
                "display_name": employee_id
            }
            updated += 1

    # Save updated embeddings
    if updated > 0:
        with open(embeddings_file, 'wb') as f:
            pickle.dump(known_faces, f)
        print(f"\nUpdated {updated} face registrations")
    else:
        print("\nNo updates needed")

if __name__ == "__main__":
    asyncio.run(update_face_names())
