"""
Fix admin user in database
"""
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_ATLAS_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/sih_safety_system"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def fix_admin():
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.get_default_database()

    # Delete all existing users
    result = await db.users.delete_many({})
    print(f"Deleted {result.deleted_count} existing users")

    # Create new admin with proper password hash
    password_hash = pwd_context.hash("admin123")
    print(f"Generated password hash: {password_hash[:50]}...")

    await db.users.insert_one({
        "username": "superadmin",
        "password_hash": password_hash,
        "full_name": "System Administrator",
        "email": "admin@minesafety.com",
        "phone": "+91-9999999999",
        "role": "super_admin",
        "is_active": True,
        "created_at": datetime.utcnow()
    })
    print("Created new superadmin user")

    # Verify
    user = await db.users.find_one({"username": "superadmin"})
    if user:
        print(f"Verified user exists: {user['username']}")
        print(f"Password hash length: {len(user['password_hash'])}")

        # Test password verification
        is_valid = pwd_context.verify("admin123", user["password_hash"])
        print(f"Password verification test: {'PASS' if is_valid else 'FAIL'}")

    client.close()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(fix_admin())
