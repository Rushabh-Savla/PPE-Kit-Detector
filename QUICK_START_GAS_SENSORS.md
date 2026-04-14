# 🚀 Quick Start - Gas Sensor Integration

## Step-by-Step Setup (5 minutes)

### Step 1: Get Your Mine ID

1. Open Command Prompt and start MongoDB shell:
   ```bash
   mongo
   ```

2. Use your database:
   ```bash
   use sih_safety_system
   ```

3. Find your mines:
   ```bash
   db.mines.find().pretty()
   ```

4. Copy the `_id` value (looks like: `507f1f77bcf86cd799439011`)

### Step 2: Configure the Bridge Script

1. Open `backend/gas_bridge.py`

2. Find these lines (around line 17-30) and update:
   ```python
   # REQUIRED - Paste your mine ID here
   MINE_ID = "507f1f77bcf86cd799439011"  # ← CHANGE THIS!

   # OPTIONAL - Add zone/gate if needed
   ZONE_ID = None  # Set if sensor is in specific zone
   GATE_ID = None  # Set if sensor is at specific gate

   # Give your sensor a name
   SENSOR_ID = "GAS_SENSOR_001"  # You can change this

   # Login credentials (use any valid user)
   USERNAME = "shift_day1"
   PASSWORD = "shift123"
   ```

3. Save the file

### Step 3: Install Dependencies

Open Command Prompt in the backend directory:
```bash
cd c:\Users\Admin\Desktop\sih-final\backend
pip install aiohttp
```

### Step 4: Start Everything!

You need **3 terminal windows**:

#### Terminal 1 - Gas Sensor Script (Port 8000)
```bash
cd c:\Users\Admin\Desktop\sih-final\nextjs-frontend\gas-detection-files
python gas.py
```
Should show: `Running on http://0.0.0.0:8000`

#### Terminal 2 - Main Backend (Port 8001)
```bash
cd c:\Users\Admin\Desktop\sih-final\backend
python -m uvicorn main:app --reload --port 8001
```
Should show: `Uvicorn running on http://127.0.0.1:8001`

#### Terminal 3 - Gas Bridge
**Option A - Easy Way (Double-click):**
- Double-click `backend/START_GAS_BRIDGE.bat`

**Option B - Command Line:**
```bash
cd c:\Users\Admin\Desktop\sih-final\backend
python gas_bridge.py
```

### Step 5: Update Frontend API URL

Edit `nextjs-frontend/.env.local` (create if doesn't exist):
```
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Step 6: Start Frontend

```bash
cd c:\Users\Admin\Desktop\sih-final\nextjs-frontend
npm run dev
```

### Step 7: View Gas Monitoring!

1. Open browser: `http://localhost:3000`
2. Login: `superadmin` / `admin123`
3. Click **"Gas Monitoring"** in sidebar
4. Watch the magic happen! ✨

---

## 🎯 What You Should See

### Terminal 3 (Gas Bridge) Output:
```
======================================================================
  GAS SENSOR BRIDGE - Main Backend Integration
======================================================================

Configuration:
  Gas Sensor URL:  http://localhost:8000/data
  Main API URL:    http://localhost:8001/api/gas-sensors/readings
  Username:        shift_day1
  Mine ID:         507f1f77bcf86cd799439011
  Sensor ID:       GAS_SENSOR_001
  Poll Interval:   10 seconds

======================================================================

✓ [14:23:15] Logged in as shift_day1
ℹ [14:23:15] Polling every 10 seconds. Press Ctrl+C to stop.

✓ [14:23:15] CH₄:  245.3 PPM | CO:  12.2 PPM | Severity: NORMAL
✓ [14:23:25] CH₄:  267.1 PPM | CO:  13.4 PPM | Severity: NORMAL
⚠ [14:23:35] CH₄: 5420.0 PPM | CO:  27.1 PPM | Severity: MEDIUM   | ALERT CREATED
```

### Browser (Gas Monitoring Page):
- 🟢 **ALL CLEAR** banner (if safe)
- 🔴 **ALERT - ACTION REQUIRED** banner (if dangerous)
- Live sensor cards with current readings
- Real-time graphs and trends
- Safety guidelines

---

## 🔧 Troubleshooting

### "MINE_ID not configured!"
- You forgot to set `MINE_ID` in `gas_bridge.py`
- Follow Step 1 & 2 above

### "Timeout connecting to gas sensor"
- Your `gas.py` script isn't running
- Make sure Terminal 1 is running on port 8000
- Check: `http://localhost:8000/data` in browser

### "Login failed"
- Check username/password in `gas_bridge.py`
- Default: `shift_day1` / `shift123`
- Try: `superadmin` / `admin123`

### "Connection refused to main API"
- Your main backend isn't running
- Make sure Terminal 2 is running on port 8001
- Check: `http://localhost:8001/health` in browser

### No data showing in Gas Monitoring page
- Make sure all 3 terminals are running
- Check browser console for errors
- Verify `NEXT_PUBLIC_API_URL=http://localhost:8001` in frontend

---

## 📊 Port Configuration

If you want everything on different ports:

1. **Gas Sensor (gas.py)** - Default: 8000
   ```python
   # In gas.py, change:
   uvicorn.run(app, host="0.0.0.0", port=8000)  # Change this
   ```

2. **Main Backend** - Default: 8001
   ```bash
   uvicorn main:app --reload --port 8001  # Change this
   ```

3. **Update gas_bridge.py**:
   ```python
   GAS_SENSOR_URL = "http://localhost:8000/data"  # Match gas.py port
   MAIN_API_URL = "http://localhost:8001/api/gas-sensors/readings"  # Match backend port
   LOGIN_URL = "http://localhost:8001/auth/login"  # Match backend port
   ```

4. **Update frontend .env.local**:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8001  # Match backend port
   ```

---

## 🎊 Testing Without Hardware

Want to test without physical sensors? Add fake data:

```python
# In gas_bridge.py, add this function:
async def generate_fake_data():
    import random
    return {
        "status": "connected",
        "methane_ppm": random.uniform(200, 15000),  # Random methane
        "env": {
            "pressure_hpa": random.uniform(990, 1020),
            "altitude_m": random.uniform(200, 300),
        }
    }

# Replace this line in main_loop():
sensor_data = await fetch_sensor_data(session)

# With this:
sensor_data = await generate_fake_data()  # Use fake data
```

Run the bridge and watch alerts trigger when levels get high!

---

## ✅ Success Checklist

- [ ] MongoDB running
- [ ] Mine ID configured in `gas_bridge.py`
- [ ] Terminal 1: `gas.py` running on port 8000
- [ ] Terminal 2: Main backend running on port 8001
- [ ] Terminal 3: `gas_bridge.py` running and showing readings
- [ ] Frontend running on port 3000
- [ ] Can login and see "Gas Monitoring" in sidebar
- [ ] Gas Monitoring page shows live sensor data
- [ ] Alerts trigger when gas levels are high

---

## 🆘 Need Help?

Common issues:
1. **Port already in use**: Change ports as described above
2. **Module not found**: Run `pip install aiohttp`
3. **Can't connect to MongoDB**: Make sure MongoDB is running
4. **No data in UI**: Check all 3 terminals are running and no errors

Your system is ready! Happy monitoring! 🛡️
