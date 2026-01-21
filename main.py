"""
PilotView - Web app to view comma device recorded routes
"""

import os
import re
import bz2
import json
import hashlib
import shutil
import tempfile
import platform
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from flask import Flask, render_template, jsonify, send_file, request, abort, Response

app = Flask(__name__)

# Configuration - can be overridden via environment variable
DATA_DIR = os.environ.get('COMMA_DATA_DIR', '/data/media/0/realdata')

# Video cache directory for transcoded files (enables seeking)
CACHE_DIR = os.environ.get('COMMA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'comma-cam-cache'))

# Track transcoding jobs in progress
transcoding_jobs = {}
transcoding_lock = threading.Lock()

# Regex patterns for parsing route/segment names
# Format 1: Date-based format: 2024-01-21--16-30-45--0
ROUTE_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})--(\d+)$')
# Format 2: Dongle ID with date: a]f0-9]{16}|2024-01-21--16-30-45--0
DONGLE_ROUTE_PATTERN = re.compile(r'^([a-f0-9]{16})\|(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})--(\d+)$')
# Format 3: Hex-based format: 00000000--5692873cee--0 (dongle_id--route_id--segment)
HEX_ROUTE_PATTERN = re.compile(r'^([a-f0-9]{8})--([a-f0-9]{10})--(\d+)$')


def parse_segment_name(name: str) -> dict | None:
    """Parse a segment directory name into components."""
    # Try hex-based format first (most common for local data): 00000000--5692873cee--0
    match = HEX_ROUTE_PATTERN.match(name)
    if match:
        return {
            'dongle_id': match.group(1),
            'route_id': match.group(2),
            'route_time': None,  # No timestamp in this format
            'segment_num': int(match.group(3)),
            'route_name': f"{match.group(1)}--{match.group(2)}"
        }
    
    # Try with dongle ID and date
    match = DONGLE_ROUTE_PATTERN.match(name)
    if match:
        return {
            'dongle_id': match.group(1),
            'route_id': None,
            'route_time': match.group(2),
            'segment_num': int(match.group(3)),
            'route_name': f"{match.group(1)}|{match.group(2)}"
        }
    
    # Try date-only format
    match = ROUTE_PATTERN.match(name)
    if match:
        return {
            'dongle_id': None,
            'route_id': None,
            'route_time': match.group(1),
            'segment_num': int(match.group(2)),
            'route_name': match.group(1)
        }
    
    return None


def get_segment_files(segment_path: Path) -> dict:
    """Get files available in a segment directory."""
    files = {}
    file_types = {
        'rlog': ['rlog.bz2', 'rlog.zst', 'rlog'],
        'qlog': ['qlog.bz2', 'qlog.zst', 'qlog'],
        'fcamera': ['fcamera.hevc'],
        'dcamera': ['dcamera.hevc'],
        'ecamera': ['ecamera.hevc'],
        'qcamera': ['qcamera.ts'],
    }
    
    for file_type, filenames in file_types.items():
        for filename in filenames:
            file_path = segment_path / filename
            if file_path.exists():
                files[file_type] = {
                    'filename': filename,
                    'path': str(file_path),
                    'size': file_path.stat().st_size
                }
                break
    
    return files


def scan_routes(data_dir: str) -> dict:
    """Scan the data directory for routes and segments."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}
    
    routes = defaultdict(lambda: {'segments': [], 'start_time': None, 'end_time': None})
    
    for entry in data_path.iterdir():
        if not entry.is_dir():
            continue
        
        parsed = parse_segment_name(entry.name)
        if not parsed:
            continue
        
        route_name = parsed['route_name']
        segment_num = parsed['segment_num']
        files = get_segment_files(entry)
        
        segment_info = {
            'segment_num': segment_num,
            'path': str(entry),
            'name': entry.name,
            'files': files
        }
        
        routes[route_name]['segments'].append(segment_info)
        routes[route_name]['route_time'] = parsed.get('route_time')
        routes[route_name]['route_id'] = parsed.get('route_id')
        routes[route_name]['dongle_id'] = parsed.get('dongle_id')
    
    # Sort segments by number and calculate route info
    for route_name, route_data in routes.items():
        route_data['segments'].sort(key=lambda x: x['segment_num'])
        route_data['segment_count'] = len(route_data['segments'])
        
        # Parse datetime from route_time if available
        if route_data.get('route_time'):
            try:
                dt = datetime.strptime(route_data['route_time'], '%Y-%m-%d--%H-%M-%S')
                route_data['start_time'] = dt.isoformat()
                route_data['start_timestamp'] = dt.timestamp()
            except ValueError:
                pass
    
    return dict(routes)


def decompress_log_file(log_path: str) -> bytes:
    """Decompress a log file (handles .zst and .bz2 with streaming for zst)."""
    import zstandard as zstd
    
    with open(log_path, 'rb') as f:
        data = f.read()
    
    # Check magic bytes and decompress accordingly
    if data.startswith(b'\x28\xB5\x2F\xFD'):  # zstd magic
        # Use streaming decompression for zst files (handles missing content size)
        dctx = zstd.ZstdDecompressor()
        with open(log_path, 'rb') as f:
            reader = dctx.stream_reader(f)
            return reader.read()
    elif data.startswith(b'BZh'):  # bz2 magic
        return bz2.decompress(data)
    else:
        # Uncompressed
        return data


def read_log_messages(log_path: str, message_types: list[str] = None, limit: int = 1000) -> list[dict]:
    """Read messages from an rlog or qlog file."""
    try:
        from openpilot_logging.cereal import log as capnp_log
        
        # Decompress the log file
        dat = decompress_log_file(log_path)
        
        # Parse capnp messages
        ents = capnp_log.Event.read_multiple_bytes(dat)
        messages = []
        
        for msg in ents:
            if limit and len(messages) >= limit:
                break
            
            try:
                msg_type = msg.which()
                
                if message_types and msg_type not in message_types:
                    continue
                
                msg_data = {
                    'type': msg_type,
                    'logMonoTime': msg.logMonoTime,
                    'valid': getattr(msg, 'valid', True),
                }
                
                # Extract specific data based on message type
                if msg_type == 'initData':
                    init = msg.initData
                    msg_data['data'] = {
                        'kernelVersion': str(init.kernelVersion) if hasattr(init, 'kernelVersion') else None,
                        'dongleId': str(init.dongleId) if hasattr(init, 'dongleId') else None,
                        'deviceType': str(init.deviceType) if hasattr(init, 'deviceType') else None,
                        'version': str(init.version) if hasattr(init, 'version') else None,
                        'gitCommit': str(init.gitCommit) if hasattr(init, 'gitCommit') else None,
                        'gitBranch': str(init.gitBranch) if hasattr(init, 'gitBranch') else None,
                    }
                    # Try to get wall time for time correction
                    if hasattr(init, 'wallTimeNanos'):
                        msg_data['data']['wallTimeNanos'] = init.wallTimeNanos
                
                elif msg_type == 'gpsLocationExternal' or msg_type == 'gpsLocation':
                    gps = getattr(msg, msg_type)
                    msg_data['data'] = {
                        'latitude': gps.latitude if hasattr(gps, 'latitude') else None,
                        'longitude': gps.longitude if hasattr(gps, 'longitude') else None,
                        'altitude': gps.altitude if hasattr(gps, 'altitude') else None,
                        'speed': gps.speed if hasattr(gps, 'speed') else None,
                        'bearing': gps.bearingDeg if hasattr(gps, 'bearingDeg') else None,
                        'timestamp': gps.unixTimestampMillis if hasattr(gps, 'unixTimestampMillis') else None,
                    }
                
                elif msg_type == 'carState':
                    cs = msg.carState
                    msg_data['data'] = {
                        'vEgo': cs.vEgo if hasattr(cs, 'vEgo') else None,
                        'aEgo': cs.aEgo if hasattr(cs, 'aEgo') else None,
                        'steeringAngleDeg': cs.steeringAngleDeg if hasattr(cs, 'steeringAngleDeg') else None,
                        'gas': cs.gas if hasattr(cs, 'gas') else None,
                        'brake': cs.brakePressed if hasattr(cs, 'brakePressed') else None,
                        'cruiseState': {
                            'enabled': cs.cruiseState.enabled if hasattr(cs.cruiseState, 'enabled') else None,
                            'speed': cs.cruiseState.speed if hasattr(cs.cruiseState, 'speed') else None,
                        } if hasattr(cs, 'cruiseState') else None,
                    }
                
                elif msg_type == 'controlsState':
                    ctrl = msg.controlsState
                    msg_data['data'] = {
                        'enabled': ctrl.enabled if hasattr(ctrl, 'enabled') else None,
                        'active': ctrl.active if hasattr(ctrl, 'active') else None,
                        'alertText1': str(ctrl.alertText1) if hasattr(ctrl, 'alertText1') else None,
                        'alertText2': str(ctrl.alertText2) if hasattr(ctrl, 'alertText2') else None,
                        'alertType': str(ctrl.alertType) if hasattr(ctrl, 'alertType') else None,
                        'state': str(ctrl.state) if hasattr(ctrl, 'state') else None,
                    }
                
                elif msg_type == 'carControl':
                    cc = msg.carControl
                    actuators = cc.actuators if hasattr(cc, 'actuators') else None
                    msg_data['data'] = {
                        'enabled': cc.enabled if hasattr(cc, 'enabled') else None,
                        'steeringAngleDeg': actuators.steeringAngleDeg if actuators and hasattr(actuators, 'steeringAngleDeg') else None,
                        'steer': actuators.steer if actuators and hasattr(actuators, 'steer') else None,
                        'accel': actuators.accel if actuators and hasattr(actuators, 'accel') else None,
                    }
                
                elif msg_type == 'deviceState':
                    ds = msg.deviceState
                    msg_data['data'] = {
                        'cpuTempC': list(ds.cpuTempC) if hasattr(ds, 'cpuTempC') else None,
                        'memoryUsagePercent': ds.memoryUsagePercent if hasattr(ds, 'memoryUsagePercent') else None,
                        'freeSpacePercent': ds.freeSpacePercent if hasattr(ds, 'freeSpacePercent') else None,
                        'batteryPercent': ds.batteryPercent if hasattr(ds, 'batteryPercent') else None,
                        'networkType': str(ds.networkType) if hasattr(ds, 'networkType') else None,
                    }
                
                elif msg_type == 'clocks':
                    clocks = msg.clocks
                    msg_data['data'] = {
                        'wallTimeNanos': clocks.wallTimeNanos if hasattr(clocks, 'wallTimeNanos') else None,
                        'monoTime': clocks.monoTime if hasattr(clocks, 'monoTime') else None,
                    }
                
                else:
                    # For other message types, just include the type
                    msg_data['data'] = None
                
                messages.append(msg_data)
                
            except Exception as e:
                # Skip messages that fail to parse
                continue
        
        return messages
        
    except Exception as e:
        return [{'error': str(e)}]


def get_gps_time_from_segment(log_path: str) -> int | None:
    """Extract GPS time from liveLocationKalman in a segment. Returns unixTimestampMillis or None."""
    try:
        from openpilot_logging.cereal import log as capnp_log
        
        dat = decompress_log_file(log_path)
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        for msg in ents:
            try:
                msg_type = msg.which()
                if msg_type == 'liveLocationKalmanDEPRECATED':
                    loc = msg.liveLocationKalmanDEPRECATED
                    if hasattr(loc, 'unixTimestampMillis'):
                        ts = loc.unixTimestampMillis
                        # Valid timestamp should be after year 2020 (in ms)
                        if ts > 1577836800000:  # 2020-01-01 00:00:00 UTC
                            return ts
            except:
                continue
        return None
    except:
        return None


def get_route_start_time(segments: list) -> dict:
    """
    Get the route start time by searching segments for valid GPS time.
    Each segment is ~1 minute, so we calculate: start_time = gps_time - (segment_num * 60000ms)
    """
    # Try segments in order, checking a few at different points
    # Check first, then skip ahead to find GPS lock faster
    segments_to_check = []
    segment_nums = [s['segment_num'] for s in segments]
    
    # Add first few segments
    for i in range(min(3, len(segment_nums))):
        segments_to_check.append(segment_nums[i])
    
    # Add some middle segments (GPS might lock after a few minutes)
    mid = len(segment_nums) // 2
    for i in range(max(0, mid - 1), min(len(segment_nums), mid + 2)):
        if segment_nums[i] not in segments_to_check:
            segments_to_check.append(segment_nums[i])
    
    # Add some later segments
    for i in range(max(0, len(segment_nums) - 3), len(segment_nums)):
        if segment_nums[i] not in segments_to_check:
            segments_to_check.append(segment_nums[i])
    
    for seg in segments:
        if seg['segment_num'] not in segments_to_check:
            continue
            
        log_file = None
        for log_type in ['rlog', 'qlog']:
            if log_type in seg['files']:
                log_file = seg['files'][log_type]['path']
                break
        
        if not log_file:
            continue
        
        gps_time_ms = get_gps_time_from_segment(log_file)
        if gps_time_ms:
            # Calculate route start time by subtracting segment offset
            # Each segment is approximately 1 minute (60000 ms)
            segment_offset_ms = seg['segment_num'] * 60 * 1000
            route_start_ms = gps_time_ms - segment_offset_ms
            
            return {
                'baseTimeMs': route_start_ms,
                'timeSource': 'gps',
                'gpsSegment': seg['segment_num'],
                'gpsTimeMs': gps_time_ms,
            }
    
    return {'baseTimeMs': None, 'timeSource': None}


def get_time_correction(log_path: str) -> dict:
    """Get time correction data from log by reading initData and clocks messages."""
    try:
        from openpilot_logging.cereal import log as capnp_log
        
        # Decompress the log file
        dat = decompress_log_file(log_path)
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        init_data = None
        first_clock = None
        first_gps_time = None
        first_live_loc = None
        
        for msg in ents:
            try:
                msg_type = msg.which()
                
                if msg_type == 'initData' and init_data is None:
                    init = msg.initData
                    init_data = {
                        'logMonoTime': msg.logMonoTime,
                        'wallTimeNanos': init.wallTimeNanos if hasattr(init, 'wallTimeNanos') else None,
                    }
                
                elif msg_type == 'clocks' and first_clock is None:
                    clocks = msg.clocks
                    first_clock = {
                        'logMonoTime': msg.logMonoTime,
                        'wallTimeNanos': clocks.wallTimeNanos if hasattr(clocks, 'wallTimeNanos') else None,
                    }
                
                elif (msg_type == 'gpsLocationExternal' or msg_type == 'gpsLocation') and first_gps_time is None:
                    gps = getattr(msg, msg_type)
                    if hasattr(gps, 'unixTimestampMillis') and gps.unixTimestampMillis > 1577836800000:
                        first_gps_time = {
                            'logMonoTime': msg.logMonoTime,
                            'unixTimestampMillis': gps.unixTimestampMillis,
                        }
                
                elif msg_type == 'liveLocationKalmanDEPRECATED' and first_live_loc is None:
                    loc = msg.liveLocationKalmanDEPRECATED
                    if hasattr(loc, 'unixTimestampMillis') and loc.unixTimestampMillis > 1577836800000:
                        first_live_loc = {
                            'logMonoTime': msg.logMonoTime,
                            'unixTimestampMillis': loc.unixTimestampMillis,
                        }
                
                # Stop once we have all time sources
                if init_data and first_clock and (first_gps_time or first_live_loc):
                    break
                    
            except Exception:
                continue
        
        # Calculate the base wall time - prioritize GPS sources
        base_time = None
        time_source = None
        mono_offset = None
        
        if first_gps_time and first_gps_time['unixTimestampMillis'] > 0:
            base_time = first_gps_time['unixTimestampMillis']
            mono_offset = first_gps_time['logMonoTime']
            time_source = 'gps'
        elif first_live_loc and first_live_loc['unixTimestampMillis'] > 0:
            base_time = first_live_loc['unixTimestampMillis']
            mono_offset = first_live_loc['logMonoTime']
            time_source = 'gps_kalman'
        elif first_clock and first_clock['wallTimeNanos']:
            base_time = first_clock['wallTimeNanos'] // 1_000_000
            mono_offset = first_clock['logMonoTime']
            time_source = 'clocks'
        elif init_data and init_data['wallTimeNanos']:
            base_time = init_data['wallTimeNanos'] // 1_000_000
            mono_offset = init_data['logMonoTime']
            time_source = 'initData'
        
        return {
            'baseTimeMs': base_time,
            'monoOffset': mono_offset,
            'timeSource': time_source,
            'initData': init_data,
            'firstClock': first_clock,
            'firstGpsTime': first_gps_time,
            'firstLiveLoc': first_live_loc,
        }
        
    except Exception as e:
        return {'error': str(e)}


# Flask Routes

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html')


@app.route('/favicon.svg')
def favicon():
    """Serve the favicon."""
    return send_file('static/favicon.svg', mimetype='image/svg+xml')


@app.route('/api/config')
def get_config():
    """Get current configuration."""
    return jsonify({
        'dataDir': DATA_DIR,
        'exists': os.path.exists(DATA_DIR)
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    """Update configuration."""
    global DATA_DIR
    data = request.get_json()
    if 'dataDir' in data:
        DATA_DIR = data['dataDir']
    return jsonify({'dataDir': DATA_DIR, 'exists': os.path.exists(DATA_DIR)})


@app.route('/api/routes')
def list_routes():
    """List all routes in the data directory."""
    routes = scan_routes(DATA_DIR)
    
    # Check if we should fetch corrected timestamps from logs
    with_time_correction = request.args.get('with_time_correction', 'false').lower() == 'true'
    
    # Convert to list and sort by start time
    route_list = []
    for route_name, route_data in routes.items():
        route_entry = {
            'name': route_name,
            **route_data
        }
        
        # Optionally get corrected timestamp from GPS data across segments
        if with_time_correction and route_data['segments']:
            # Search multiple segments for valid GPS time
            time_data = get_route_start_time(route_data['segments'])
            if time_data.get('baseTimeMs'):
                route_entry['corrected_start_timestamp_ms'] = time_data['baseTimeMs']
                route_entry['time_source'] = time_data.get('timeSource')
                route_entry['gps_segment'] = time_data.get('gpsSegment')
        
        route_list.append(route_entry)
    
    # Sort by corrected timestamp if available, otherwise by folder name timestamp
    if with_time_correction:
        route_list.sort(
            key=lambda x: x.get('corrected_start_timestamp_ms') or x.get('start_timestamp', 0) * 1000,
            reverse=True
        )
    else:
        route_list.sort(key=lambda x: x.get('start_timestamp', 0), reverse=True)
    
    return jsonify({
        'dataDir': DATA_DIR,
        'routes': route_list,
        'count': len(route_list)
    })


@app.route('/api/routes/<path:route_name>')
def get_route(route_name):
    """Get details for a specific route."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    return jsonify({
        'name': route_name,
        **routes[route_name]
    })


@app.route('/api/routes/<path:route_name>/segments/<int:segment_num>')
def get_segment(route_name, segment_num):
    """Get details for a specific segment."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    segment = next((s for s in route['segments'] if s['segment_num'] == segment_num), None)
    
    if not segment:
        abort(404, description="Segment not found")
    
    return jsonify(segment)


@app.route('/api/routes/<path:route_name>/segments/<int:segment_num>/logs')
def get_segment_logs(route_name, segment_num):
    """Get log messages from a segment."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    segment = next((s for s in route['segments'] if s['segment_num'] == segment_num), None)
    
    if not segment:
        abort(404, description="Segment not found")
    
    # Get query parameters
    message_types = request.args.getlist('types')
    limit = request.args.get('limit', 1000, type=int)
    
    # Find log file
    log_file = None
    for log_type in ['rlog', 'qlog']:
        if log_type in segment['files']:
            log_file = segment['files'][log_type]['path']
            break
    
    if not log_file:
        abort(404, description="No log file found in segment")
    
    messages = read_log_messages(log_file, message_types or None, limit)
    
    return jsonify({
        'segment': segment_num,
        'logFile': log_file,
        'messageCount': len(messages),
        'messages': messages
    })


@app.route('/api/routes/<path:route_name>/segments/<int:segment_num>/time')
def get_segment_time(route_name, segment_num):
    """Get time correction data for a segment."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    segment = next((s for s in route['segments'] if s['segment_num'] == segment_num), None)
    
    if not segment:
        abort(404, description="Segment not found")
    
    # Find log file
    log_file = None
    for log_type in ['rlog', 'qlog']:
        if log_type in segment['files']:
            log_file = segment['files'][log_type]['path']
            break
    
    if not log_file:
        abort(404, description="No log file found in segment")
    
    time_data = get_time_correction(log_file)
    
    return jsonify({
        'segment': segment_num,
        'logFile': log_file,
        **time_data
    })


@app.route('/api/routes/<path:route_name>/time')
def get_route_time(route_name):
    """Get time correction data for a route by searching segments for GPS time."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    
    if not route['segments']:
        abort(404, description="No segments in route")
    
    # Search segments for valid GPS time
    time_data = get_route_start_time(route['segments'])
    
    return jsonify({
        'route': route_name,
        'routeTime': route.get('route_time'),
        'segmentCount': len(route['segments']),
        **time_data
    })


@app.route('/api/routes/<path:route_name>/segments/<int:segment_num>/steering')
def get_segment_steering(route_name, segment_num):
    """Get steering data (carState and carControl) for a segment, optimized for visualization."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    segment = next((s for s in route['segments'] if s['segment_num'] == segment_num), None)
    
    if not segment:
        abort(404, description="Segment not found")
    
    # Find log file
    log_file = None
    for log_type in ['rlog', 'qlog']:
        if log_type in segment['files']:
            log_file = segment['files'][log_type]['path']
            break
    
    if not log_file:
        abort(404, description="No log file found in segment")
    
    try:
        from openpilot_logging.cereal import log as capnp_log
        
        dat = decompress_log_file(log_file)
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        steering_data = []
        first_carstate_time = None  # Use first carState time as base
        actual_count = 0
        desired_count = 0
        msg_types_seen = set()
        sample_angles = []  # Collect sample angle values for debugging
        
        # First pass: find the first carState timestamp to use as base
        for msg in ents:
            try:
                if msg.which() == 'carState':
                    first_carstate_time = msg.logMonoTime
                    break
            except:
                continue
        
        # Reset the iterator by re-reading
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        for msg in ents:
            try:
                msg_type = msg.which()
                mono_time = msg.logMonoTime
                
                # Use first carState time as base, or fall back to this message's time
                if first_carstate_time is None:
                    first_carstate_time = mono_time
                
                # Calculate relative time in seconds from start of carState data
                rel_time = (mono_time - first_carstate_time) / 1e9
                
                msg_types_seen.add(msg_type)
                
                if msg_type == 'carState':
                    cs = msg.carState
                    # Try multiple attribute names for steering angle
                    angle = None
                    angle_attr_used = None
                    for attr in ['steeringAngleDeg', 'steeringAngle', 'steeringWheelAngle']:
                        if hasattr(cs, attr):
                            val = getattr(cs, attr)
                            if val is not None:
                                angle = float(val)
                                angle_attr_used = attr
                                break
                    
                    # Try multiple attribute names for speed
                    speed = None
                    for attr in ['vEgo', 'vEgoRaw', 'speed']:
                        if hasattr(cs, attr):
                            val = getattr(cs, attr)
                            if val is not None:
                                speed = float(val)
                                break
                    
                    steering_data.append({
                        't': round(rel_time, 3),
                        'type': 'actual',
                        'angle': angle,
                        'speed': speed,
                    })
                    actual_count += 1
                    
                    # Collect sample angle values for debugging - always first 10 samples
                    if len(sample_angles) < 10:
                        sample_angles.append({
                            't': round(rel_time, 3),
                            'angle': angle,
                            'attr': angle_attr_used,
                            'speed': speed
                        })
                
                elif msg_type == 'controlsState':
                    # controlsState contains the desired steering angle
                    ctrl = msg.controlsState
                    desired_angle = None
                    if hasattr(ctrl, 'steeringAngleDesiredDeg'):
                        desired_angle = ctrl.steeringAngleDesiredDeg
                    elif hasattr(ctrl, 'lateralControlState'):
                        lcs = ctrl.lateralControlState
                        # Check different lateral control types
                        lcs_type = lcs.which() if hasattr(lcs, 'which') else None
                        if lcs_type and hasattr(lcs, lcs_type):
                            lcs_state = getattr(lcs, lcs_type)
                            if hasattr(lcs_state, 'steeringAngleDesiredDeg'):
                                desired_angle = lcs_state.steeringAngleDesiredDeg
                    
                    if desired_angle is not None:
                        steering_data.append({
                            't': round(rel_time, 3),
                            'type': 'desired',
                            'angle': desired_angle,
                            'enabled': ctrl.enabled if hasattr(ctrl, 'enabled') else None,
                            'source': 'controlsState',
                        })
                        desired_count += 1
                
                elif msg_type == 'carControl':
                    cc = msg.carControl
                    actuators = cc.actuators if hasattr(cc, 'actuators') else None
                    if actuators:
                        # Try steeringAngleDeg first
                        desired_angle = None
                        if hasattr(actuators, 'steeringAngleDeg'):
                            val = actuators.steeringAngleDeg
                            if val is not None:
                                desired_angle = float(val)
                        
                        if desired_angle is not None:
                            steering_data.append({
                                't': round(rel_time, 3),
                                'type': 'desired',
                                'angle': desired_angle,
                                'enabled': bool(cc.enabled) if hasattr(cc, 'enabled') else None,
                                'source': 'carControl',
                            })
                            desired_count += 1
                        
            except Exception:
                continue
        
        # Sort by time
        steering_data.sort(key=lambda x: x['t'])
        
        # Filter relevant message types for debug
        relevant_types = ['carState', 'carControl', 'controlsState']
        seen_relevant = [t for t in relevant_types if t in msg_types_seen]
        
        # Count non-zero angles
        non_zero_actual = len([d for d in steering_data if d['type'] == 'actual' and d.get('angle') and d['angle'] != 0])
        non_zero_desired = len([d for d in steering_data if d['type'] == 'desired' and d.get('angle') and d['angle'] != 0])
        
        return jsonify({
            'segment': segment_num,
            'count': len(steering_data),
            'actualCount': actual_count,
            'desiredCount': desired_count,
            'nonZeroActual': non_zero_actual,
            'nonZeroDesired': non_zero_desired,
            'messageTypesSeen': seen_relevant,
            'sampleAngles': sample_angles[:10],  # Return sample values for debugging
            'data': steering_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/routes/<path:route_name>/segments/<int:segment_num>/steering-test')
def test_steering_data(route_name, segment_num):
    """Simple test endpoint to check steering data extraction."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        return jsonify({'error': 'Route not found'})
    
    route = routes[route_name]
    segment = next((s for s in route['segments'] if s['segment_num'] == segment_num), None)
    
    if not segment:
        return jsonify({'error': 'Segment not found'})
    
    log_file = None
    for log_type in ['rlog', 'qlog']:
        if log_type in segment['files']:
            log_file = segment['files'][log_type]['path']
            break
    
    if not log_file:
        return jsonify({'error': 'No log file'})
    
    try:
        from openpilot_logging.cereal import log as capnp_log
        
        dat = decompress_log_file(log_file)
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        results = {
            'carState_samples': [],
            'carControl_samples': [],
            'message_counts': {},
            'first_carstate_time': None,
        }
        
        # First pass: find first carState timestamp
        for msg in ents:
            try:
                if msg.which() == 'carState':
                    results['first_carstate_time'] = msg.logMonoTime
                    break
            except:
                continue
        
        # Reset iterator
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        count = 0
        for msg in ents:
            count += 1
            if count > 5000:  # Limit iterations
                break
                
            try:
                msg_type = msg.which()
                results['message_counts'][msg_type] = results['message_counts'].get(msg_type, 0) + 1
                
                base_time = results['first_carstate_time'] or msg.logMonoTime
                rel_time = (msg.logMonoTime - base_time) / 1e9
                
                # Get first 10 carState samples
                if msg_type == 'carState' and len(results['carState_samples']) < 10:
                    cs = msg.carState
                    results['carState_samples'].append({
                        't': round(rel_time, 3),
                        'steeringAngleDeg': float(cs.steeringAngleDeg) if hasattr(cs, 'steeringAngleDeg') else None,
                        'vEgo': float(cs.vEgo) if hasattr(cs, 'vEgo') else None,
                    })
                
                # Get first 10 carControl samples
                if msg_type == 'carControl' and len(results['carControl_samples']) < 10:
                    cc = msg.carControl
                    actuators = cc.actuators if hasattr(cc, 'actuators') else None
                    results['carControl_samples'].append({
                        't': round(rel_time, 3),
                        'enabled': bool(cc.enabled) if hasattr(cc, 'enabled') else None,
                        'steeringAngleDeg': float(actuators.steeringAngleDeg) if actuators and hasattr(actuators, 'steeringAngleDeg') else None,
                    })
                    
            except Exception as e:
                results['parse_error'] = str(e)
                continue
        
        results['total_messages_checked'] = count
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/routes/<path:route_name>/segments/<int:segment_num>/debug')
def debug_segment_messages(route_name, segment_num):
    """Debug endpoint to inspect message structure in a segment."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    segment = next((s for s in route['segments'] if s['segment_num'] == segment_num), None)
    
    if not segment:
        abort(404, description="Segment not found")
    
    log_file = None
    for log_type in ['rlog', 'qlog']:
        if log_type in segment['files']:
            log_file = segment['files'][log_type]['path']
            break
    
    if not log_file:
        abort(404, description="No log file found")
    
    try:
        from openpilot_logging.cereal import log as capnp_log
        
        dat = decompress_log_file(log_file)
        ents = capnp_log.Event.read_multiple_bytes(dat)
        
        result = {
            'segment': segment_num,
            'carState': None,
            'carControl': None,
            'controlsState': None,
        }
        
        for msg in ents:
            try:
                msg_type = msg.which()
                
                # Get first carState message and list all attributes
                if msg_type == 'carState' and result['carState'] is None:
                    cs = msg.carState
                    attrs = {}
                    for attr in dir(cs):
                        if not attr.startswith('_'):
                            try:
                                val = getattr(cs, attr)
                                if not callable(val):
                                    if isinstance(val, (int, float, str, bool, type(None))):
                                        attrs[attr] = val
                                    else:
                                        attrs[attr] = str(type(val).__name__)
                            except:
                                pass
                    result['carState'] = attrs
                
                # Get first carControl message
                if msg_type == 'carControl' and result['carControl'] is None:
                    cc = msg.carControl
                    attrs = {}
                    for attr in dir(cc):
                        if not attr.startswith('_'):
                            try:
                                val = getattr(cc, attr)
                                if not callable(val):
                                    if isinstance(val, (int, float, str, bool, type(None))):
                                        attrs[attr] = val
                                    elif attr == 'actuators' and hasattr(val, 'steeringAngleDeg'):
                                        attrs['actuators'] = {
                                            'steeringAngleDeg': val.steeringAngleDeg if hasattr(val, 'steeringAngleDeg') else None,
                                            'steer': val.steer if hasattr(val, 'steer') else None,
                                            'accel': val.accel if hasattr(val, 'accel') else None,
                                        }
                                    else:
                                        attrs[attr] = str(type(val).__name__)
                            except:
                                pass
                    result['carControl'] = attrs
                
                # Get first controlsState message
                if msg_type == 'controlsState' and result['controlsState'] is None:
                    ctrl = msg.controlsState
                    attrs = {}
                    for attr in dir(ctrl):
                        if not attr.startswith('_'):
                            try:
                                val = getattr(ctrl, attr)
                                if not callable(val):
                                    if isinstance(val, (int, float, str, bool, type(None))):
                                        attrs[attr] = val
                                    else:
                                        attrs[attr] = str(type(val).__name__)
                            except:
                                pass
                    result['controlsState'] = attrs
                
                # Stop once we have all
                if all(v is not None for v in result.values()):
                    break
                    
            except Exception as e:
                continue
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/routes/<path:route_name>/gps')
def get_route_gps(route_name):
    """Get GPS track data for a route (all segments)."""
    routes = scan_routes(DATA_DIR)
    
    if route_name not in routes:
        abort(404, description="Route not found")
    
    route = routes[route_name]
    gps_points = []
    segment_start_times = {}  # First carState mono time for each segment
    
    # Read GPS data from all segments
    for segment in route['segments']:
        log_file = None
        for log_type in ['rlog', 'qlog']:
            if log_type in segment['files']:
                log_file = segment['files'][log_type]['path']
                break
        
        if not log_file:
            continue
        
        try:
            from openpilot_logging.cereal import log as capnp_log
            dat = decompress_log_file(log_file)
            ents = capnp_log.Event.read_multiple_bytes(dat)
            
            segment_num = segment['segment_num']
            first_frame_time = None
            segment_gps = []
            
            for msg in ents:
                try:
                    msg_type = msg.which()
                    
                    # Find the first frame time (roadEncodeIdx or carState) for video sync
                    if first_frame_time is None:
                        if msg_type in ('roadEncodeIdx', 'carState'):
                            first_frame_time = msg.logMonoTime
                    
                    if msg_type in ('gpsLocationExternal', 'gpsLocation'):
                        gps = getattr(msg, msg_type)
                        if hasattr(gps, 'latitude') and hasattr(gps, 'longitude'):
                            lat = gps.latitude
                            lon = gps.longitude
                            # Filter out invalid GPS coordinates
                            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
                                point = {
                                    'lat': lat,
                                    'lon': lon,
                                    'speed': gps.speed if hasattr(gps, 'speed') else None,
                                    'bearing': gps.bearingDeg if hasattr(gps, 'bearingDeg') else None,
                                    'time': gps.unixTimestampMillis if hasattr(gps, 'unixTimestampMillis') else None,
                                    'segment': segment_num,
                                    'logMonoTime': msg.logMonoTime
                                }
                                segment_gps.append(point)
                except Exception:
                    continue
            
            # Store segment start time and add relative time to GPS points
            if first_frame_time is not None:
                segment_start_times[segment_num] = first_frame_time
                for point in segment_gps:
                    # Calculate relative time within segment (aligned with video time)
                    point['t'] = round((point['logMonoTime'] - first_frame_time) / 1e9, 3)
            
            gps_points.extend(segment_gps)
            
        except Exception as e:
            continue
    
    # Sort by time
    gps_points.sort(key=lambda x: (x.get('segment', 0), x.get('t', 0)))
    
    # Downsample if too many points (keep every 5th point if > 3000)
    # GPS comes at ~1Hz, so a 60-segment route has ~3600 points
    # We want enough points for smooth interpolation (~1 point per 5 seconds)
    if len(gps_points) > 3000:
        gps_points = gps_points[::5]
    elif len(gps_points) > 1500:
        gps_points = gps_points[::2]
    
    return jsonify({
        'route': route_name,
        'pointCount': len(gps_points),
        'segmentStartTimes': segment_start_times,
        'points': gps_points
    })


@app.route('/api/video/<path:file_path>')
def serve_video(file_path):
    """Serve video files with appropriate MIME types."""
    # Ensure the path is within the data directory for security
    full_path = Path(DATA_DIR) / file_path
    
    if not full_path.exists():
        abort(404, description="Video file not found")
    
    # Check if path is within data directory
    try:
        full_path.resolve().relative_to(Path(DATA_DIR).resolve())
    except ValueError:
        abort(403, description="Access denied")
    
    # Determine MIME type based on file extension
    filename = full_path.name.lower()
    if filename.endswith('.ts'):
        mimetype = 'video/mp2t'
    elif filename.endswith('.hevc'):
        mimetype = 'video/hevc'
    elif filename.endswith('.mp4'):
        mimetype = 'video/mp4'
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(full_path, mimetype=mimetype)


@app.route('/api/file/<path:file_path>')
def serve_file(file_path):
    """Serve any file from the data directory with appropriate MIME types."""
    full_path = Path(DATA_DIR) / file_path
    
    if not full_path.exists():
        abort(404, description="File not found")
    
    # Check if path is within data directory
    try:
        full_path.resolve().relative_to(Path(DATA_DIR).resolve())
    except ValueError:
        abort(403, description="Access denied")
    
    # Determine MIME type for video files
    filename = full_path.name.lower()
    mimetype = None
    if filename.endswith('.ts'):
        mimetype = 'video/mp2t'
    elif filename.endswith('.hevc'):
        mimetype = 'video/hevc'
    elif filename.endswith('.mp4'):
        mimetype = 'video/mp4'
    
    if mimetype:
        return send_file(full_path, mimetype=mimetype)
    return send_file(full_path)


def check_ffmpeg_available():
    """Check if FFmpeg is available on the system."""
    return shutil.which('ffmpeg') is not None


@lru_cache(maxsize=1)
def detect_hardware_acceleration():
    """
    Detect available hardware acceleration for video encoding.
    Returns dict with acceleration info.
    """
    result = {
        'available': False,
        'type': None,
        'encoder': 'libx264',  # Default software encoder
        'decoder': None,
        'platform': platform.system(),
        'machine': platform.machine(),
    }
    
    if not check_ffmpeg_available():
        return result
    
    # Check platform and architecture
    system = platform.system()
    machine = platform.machine()
    
    # Detect Apple Silicon (macOS with ARM64)
    is_apple_silicon = (system == 'Darwin' and machine == 'arm64')
    
    # Also check for Intel Mac with VideoToolbox support
    is_macos = (system == 'Darwin')
    
    if is_macos:
        # Check if VideoToolbox encoder is available in FFmpeg
        try:
            proc = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if 'h264_videotoolbox' in proc.stdout:
                result['available'] = True
                result['type'] = 'videotoolbox'
                result['encoder'] = 'h264_videotoolbox'
                
                if is_apple_silicon:
                    result['type'] = 'apple_silicon_videotoolbox'
                    # Apple Silicon has hardware HEVC decoder too
                    result['decoder'] = 'hevc_videotoolbox'
                
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"Error checking FFmpeg encoders: {e}")
    
    # Check for NVIDIA NVENC (Linux/Windows)
    elif system in ('Linux', 'Windows'):
        try:
            proc = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if 'h264_nvenc' in proc.stdout:
                result['available'] = True
                result['type'] = 'nvenc'
                result['encoder'] = 'h264_nvenc'
            elif 'h264_vaapi' in proc.stdout:
                # Intel/AMD VAAPI on Linux
                result['available'] = True
                result['type'] = 'vaapi'
                result['encoder'] = 'h264_vaapi'
            elif 'h264_qsv' in proc.stdout:
                # Intel Quick Sync
                result['available'] = True
                result['type'] = 'qsv'
                result['encoder'] = 'h264_qsv'
                
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"Error checking FFmpeg encoders: {e}")
    
    return result


@app.route('/api/capabilities')
def get_capabilities():
    """Get server capabilities (e.g., FFmpeg availability, hardware acceleration)."""
    ffmpeg_available = check_ffmpeg_available()
    hw_accel = detect_hardware_acceleration()
    
    # Calculate cache stats
    cache_dir = Path(CACHE_DIR)
    cache_files = list(cache_dir.glob('*.mp4')) if cache_dir.exists() else []
    cache_size = sum(f.stat().st_size for f in cache_files)
    
    return jsonify({
        'ffmpeg': ffmpeg_available,
        'streaming': ffmpeg_available,
        'hardware_acceleration': hw_accel,
        'cache': {
            'directory': str(cache_dir),
            'files': len(cache_files),
            'size_mb': round(cache_size / (1024 * 1024), 1),
        }
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cached video files."""
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.exists():
        return jsonify({'cleared': 0, 'message': 'Cache directory does not exist'})
    
    cleared = 0
    for f in cache_dir.glob('*.mp4'):
        try:
            f.unlink()
            cleared += 1
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
    
    return jsonify({'cleared': cleared, 'message': f'Cleared {cleared} cached files'})


def get_cache_path(source_path: str) -> Path:
    """Get the cache file path for a source video."""
    path_hash = hashlib.md5(source_path.encode()).hexdigest()[:16]
    source_name = Path(source_path).stem
    cache_filename = f"{source_name}_{path_hash}.mp4"
    return Path(CACHE_DIR) / cache_filename


def is_transcoding(source_path: str) -> bool:
    """Check if a video is currently being transcoded."""
    with transcoding_lock:
        return source_path in transcoding_jobs


def start_cache_transcode(source_path: str, is_hevc: bool):
    """Start background transcoding to cache file."""
    cache_path = get_cache_path(source_path)
    
    # Already cached
    if cache_path.exists():
        return
    
    # Check if already transcoding
    with transcoding_lock:
        if source_path in transcoding_jobs:
            return
        transcoding_jobs[source_path] = True
    
    # Ensure cache directory exists
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    def transcode():
        try:
            temp_path = cache_path.with_suffix('.tmp.mp4')
            hw_accel = detect_hardware_acceleration()
            
            cmd = ['ffmpeg', '-hide_banner', '-y']
            
            # Add hardware decoder if available
            if hw_accel.get('decoder') == 'hevc_videotoolbox':
                cmd.extend(['-hwaccel', 'videotoolbox'])
            
            cmd.extend(['-i', source_path])
            
            if is_hevc:
                # Transcode HEVC to H.264
                accel_type = hw_accel.get('type')
                if accel_type in ('videotoolbox', 'apple_silicon_videotoolbox'):
                    cmd.extend(['-c:v', 'h264_videotoolbox', '-q:v', '65', '-allow_sw', '1'])
                elif accel_type == 'nvenc':
                    cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23'])
                elif accel_type == 'vaapi':
                    cmd.extend(['-vaapi_device', '/dev/dri/renderD128', '-c:v', 'h264_vaapi', '-qp', '23'])
                elif accel_type == 'qsv':
                    cmd.extend(['-c:v', 'h264_qsv', '-preset', 'fast', '-global_quality', '23'])
                else:
                    cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '23'])
            else:
                # Just remux TS to MP4
                cmd.extend(['-c:v', 'copy'])
            
            cmd.extend(['-c:a', 'aac', '-movflags', '+faststart', str(temp_path)])
            
            print(f"Caching: {source_path} -> {cache_path}")
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            
            if result.returncode == 0 and temp_path.exists():
                temp_path.rename(cache_path)
                print(f"Cached: {cache_path}")
            else:
                print(f"Cache failed: {result.stderr.decode()[:200]}")
                if temp_path.exists():
                    temp_path.unlink()
        except Exception as e:
            print(f"Cache error: {e}")
        finally:
            with transcoding_lock:
                transcoding_jobs.pop(source_path, None)
    
    thread = threading.Thread(target=transcode, daemon=True)
    thread.start()


@app.route('/api/video-info/<path:file_path>')
def get_video_info(file_path):
    """Get video metadata and cache status."""
    full_path = Path(DATA_DIR) / file_path
    
    if not full_path.exists():
        abort(404, description="Video file not found")
    
    # Check if path is within data directory
    try:
        full_path.resolve().relative_to(Path(DATA_DIR).resolve())
    except ValueError:
        abort(403, description="Access denied")
    
    # Check cache status
    cache_path = get_cache_path(str(full_path))
    cached = cache_path.exists()
    transcoding = is_transcoding(str(full_path))
    
    return jsonify({
        'duration': 60,  # Comma segments are always 1 minute
        'size': full_path.stat().st_size,
        'filename': full_path.name,
        'cached': cached,
        'transcoding': transcoding,
    })


def build_ffmpeg_transcode_command(input_path: str) -> list:
    """
    Build FFmpeg command for HEVC transcoding with optimal hardware acceleration.
    """
    hw_accel = detect_hardware_acceleration()
    
    cmd = ['ffmpeg', '-hide_banner']
    
    # Add hardware decoder if available (Apple Silicon)
    if hw_accel.get('decoder') == 'hevc_videotoolbox':
        cmd.extend(['-hwaccel', 'videotoolbox'])
    
    # Input file
    cmd.extend(['-i', input_path])
    
    # Video encoding settings based on available hardware
    accel_type = hw_accel.get('type')
    
    if accel_type in ('videotoolbox', 'apple_silicon_videotoolbox'):
        # Apple VideoToolbox (macOS)
        cmd.extend([
            '-c:v', 'h264_videotoolbox',
            '-q:v', '65',  # Quality (0-100, higher is better, 65 is good balance)
            '-allow_sw', '1',  # Allow software fallback
        ])
    elif accel_type == 'nvenc':
        # NVIDIA NVENC
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',  # Medium preset (p1=fastest, p7=slowest)
            '-cq', '23',  # Constant quality
        ])
    elif accel_type == 'vaapi':
        # Intel/AMD VAAPI (Linux)
        cmd.extend([
            '-vaapi_device', '/dev/dri/renderD128',
            '-c:v', 'h264_vaapi',
            '-qp', '23',
        ])
    elif accel_type == 'qsv':
        # Intel Quick Sync
        cmd.extend([
            '-c:v', 'h264_qsv',
            '-preset', 'fast',
            '-global_quality', '23',
        ])
    else:
        # Software encoding (libx264)
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
        ])
    
    # Audio and output settings
    cmd.extend([
        '-c:a', 'aac',
        '-movflags', 'frag_keyframe+empty_moov+faststart',
        '-f', 'mp4',
        '-'
    ])
    
    return cmd


def build_ffmpeg_remux_command(input_path: str) -> list:
    """
    Build FFmpeg command for remuxing (no transcoding) - fast copy to MP4 container.
    Used for .ts files that are already H.264.
    """
    cmd = [
        'ffmpeg', '-hide_banner',
        '-i', input_path,
        '-c:v', 'copy',  # Copy video stream without re-encoding
        '-c:a', 'aac',   # Re-encode audio to AAC (in case of non-compatible audio)
        '-movflags', 'frag_keyframe+empty_moov+faststart',
        '-f', 'mp4',
        '-'
    ]
    return cmd


@app.route('/api/stream/<path:file_path>')
def stream_video(file_path):
    """Stream video file, using cache for proper seeking support."""
    full_path = Path(DATA_DIR) / file_path
    
    if not full_path.exists():
        abort(404, description="Video file not found")
    
    # Check if path is within data directory
    try:
        full_path.resolve().relative_to(Path(DATA_DIR).resolve())
    except ValueError:
        abort(403, description="Access denied")
    
    if not check_ffmpeg_available():
        abort(503, description="FFmpeg not available for streaming")
    
    filename = full_path.name.lower()
    is_hevc = filename.endswith('.hevc')
    is_ts = filename.endswith('.ts')
    
    if not (is_hevc or is_ts):
        # For other files, serve as-is
        return send_file(full_path)
    
    # Check if cached version exists
    cache_path = get_cache_path(str(full_path))
    if cache_path.exists():
        # Serve cached file with range request support (enables seeking)
        return send_file(cache_path, mimetype='video/mp4', conditional=True)
    
    # Start background caching
    start_cache_transcode(str(full_path), is_hevc)
    
    # Stream while caching in background
    if is_hevc:
        cmd = build_ffmpeg_transcode_command(str(full_path))
    else:
        cmd = build_ffmpeg_remux_command(str(full_path))
    
    def generate():
        """Generator that yields video chunks from FFmpeg."""
        process = None
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1024 * 64
            )
            
            while True:
                chunk = process.stdout.read(1024 * 64)
                if not chunk:
                    break
                yield chunk
            
            process.wait()
            
        except Exception as e:
            print(f"FFmpeg streaming error: {e}")
        finally:
            if process and process.poll() is None:
                process.terminate()
    
    return Response(
        generate(),
        mimetype='video/mp4',
        headers={
            'Content-Type': 'video/mp4',
            'Cache-Control': 'no-cache',
        }
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PilotView')
    parser.add_argument('--data-dir', '-d', type=str, default=DATA_DIR,
                        help='Path to comma data directory')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind to')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir
    print(f"Starting PilotView")
    print(f"Data directory: {DATA_DIR}")
    print(f"Server: http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
