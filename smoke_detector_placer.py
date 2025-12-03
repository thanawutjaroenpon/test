#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke Detector Auto-Placer for DXF
- Easy to use: Just provide input DXF file
- Automatically places smoke detectors according to international standards
- Exports DXF with detectors and PDF preview
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import ezdxf
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
try:
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    HAS_DRAWING = True
except ImportError:
    HAS_DRAWING = False
    print("Warning: ezdxf drawing add-on not available")


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-place smoke detectors in rooms from a DXF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Simple usage (automatic detection):
    python smoke_detector_placer.py input.dxf
  
  Custom output names:
    python smoke_detector_placer.py input.dxf --out output.dxf --pdf preview.pdf
  
  Specify room layer:
    python smoke_detector_placer.py input.dxf --rooms-layer ROOMS,WALLS
  
  Use EN54-14 standard:
    python smoke_detector_placer.py input.dxf --std EN54-14
        """
    )
    p.add_argument("in_path", help="Input DXF file path")
    p.add_argument("--out", dest="out_path", default=None, 
                   help="Output DXF path (default: input_with_detectors.dxf)")
    p.add_argument("--pdf", dest="pdf_path", default=None,
                   help="Output PDF preview path (default: input_preview.pdf)")
    p.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV export path")
    p.add_argument("--rooms-layer", dest="room_layers", default=None,
                   help="Comma-separated layer names containing rooms (default: auto-detect)")
    p.add_argument("--std", dest="standard", choices=["NFPA72", "EN54-14", "CUSTOM"], default="NFPA72",
                   help="Spacing standard (default: NFPA72)")
    p.add_argument("--spacing", dest="spacing", type=float, default=None,
                   help="Custom spacing in meters between detectors")
    p.add_argument("--margin", dest="margin", type=float, default=0.5,
                   help="Wall clearance margin in meters (default: 0.5)")
    p.add_argument("--grid", dest="grid", choices=["square", "hex"], default="square",
                   help="Grid style: square or hex (default: square)")
    p.add_argument("--min-room-area", dest="min_area", type=float, default=1.0,
                   help="Skip rooms smaller than this area in m² (default: 1.0)")
    p.add_argument("--units", dest="units", choices=["m", "mm", "ft", "in"], default=None,
                   help="Drawing units (default: auto-detect)")
    p.add_argument("--room-name-layer", dest="room_name_layer", default=None,
                   help="Layer containing room name text")
    p.add_argument("--inspect", action="store_true", 
                   help="Inspect DXF layers and polygons, then exit")
    p.add_argument("--no-pdf", action="store_true",
                   help="Skip PDF generation")
    p.add_argument("--offset-x", dest="offset_x", type=float, default=None,
                   help="X offset to shift detector positions (auto-detect if not specified)")
    p.add_argument("--offset-y", dest="offset_y", type=float, default=None,
                   help="Y offset to shift detector positions")
    p.add_argument("--coverage-circles", action="store_true",
                   help="Draw detector coverage radius circles in the output DXF")
    p.add_argument("--coverage-radius", dest="coverage_radius", type=float, default=None,
                   help="Custom coverage radius in meters (requires --coverage-circles)")
    return p.parse_args()


def auto_detect_units(doc) -> str:
    """Auto-detect the units used in the DXF file based on drawing size."""
    msp = doc.modelspace()
    try:
        box = msp.bbox()
        max_dim = max(box.size.x, box.size.y)
    except Exception:
        minx, miny, maxx, maxy = _bbox_fallback(msp)
        max_dim = max(maxx - minx, maxy - miny)
    
    if max_dim > 2000:
        return "mm"
    elif max_dim > 50:
        return "m"
    else:
        return "m"


def auto_detect_room_layers(doc) -> List[str]:
    """Auto-detect layers that contain closed polygons (likely rooms)."""
    from collections import defaultdict
    msp = doc.modelspace()
    counts = defaultdict(int)
    
    for e in msp.query("LWPOLYLINE"):
        try:
            if e.closed:
                pts = [(v[0], v[1]) for v in e.get_points()]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    counts[e.dxf.layer.upper()] += 1
        except Exception:
            pass
    
    for e in msp.query("POLYLINE"):
        try:
            if e.is_closed:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    counts[e.dxf.layer.upper()] += 1
        except Exception:
            pass
    
    # Return layers with at least 1 closed polygon
    detected = [layer for layer, count in counts.items() if count >= 1]
    return detected if detected else ["ROOMS"]


def auto_detect_offset(doc, rooms: List[Tuple[str, Polygon]], room_layers: List[str]) -> Tuple[float, float]:
    """Auto-detect offset to align rooms with main architectural drawing."""
    if not rooms:
        return 0.0, 0.0
    
    msp = doc.modelspace()
    
    # Get room bounds
    room_xs: List[float] = []
    room_ys: List[float] = []
    room_polys = [poly for _, poly in rooms]
    total_room_area = sum(poly.area for poly in room_polys)
    
    for poly in room_polys:
        bounds = poly.bounds
        room_xs.extend([bounds[0], bounds[2]])
        room_ys.extend([bounds[1], bounds[3]])
    
    room_minx = min(room_xs)
    room_maxx = max(room_xs)
    room_miny = min(room_ys)
    room_maxy = max(room_ys)
    
    # Use union of rooms to reduce influence of small outliers when computing bounds.
    try:
        merged = unary_union(room_polys)
        if isinstance(merged, Polygon):
            main_poly = merged
        elif hasattr(merged, "geoms"):
            polys = [g for g in merged.geoms if isinstance(g, Polygon)]
            main_poly = max(polys, key=lambda p: p.area) if polys else None
        else:
            main_poly = None
    except Exception:
        main_poly = None
    
    if main_poly:
        bounds = main_poly.bounds
        room_minx, room_miny, room_maxx, room_maxy = bounds
        total_room_area = max(main_poly.area, 1.0)
    else:
        total_room_area = max(total_room_area, 1.0)
    
    room_centerx = (room_minx + room_maxx) / 2
    room_centery = (room_miny + room_maxy) / 2
    room_width = max(room_maxx - room_minx, 1.0)
    room_height = max(room_maxy - room_miny, 1.0)

    # Priority list: architectural layers that typically define the building
    priority_keywords = ["WALL", "A-WALL", "I-WALL", "ARCH", "A-CLNG", "A-DOOR", "A-WIND"]
    
    # Convert room_layers to set for faster lookup
    room_layer_set = set(room_layers)
    
    # Try to find an outline layer that closely matches room dimensions.
    best_outline = None
    best_outline_score = float("inf")
    tolerance_ratio = 0.15  # allow 15% mismatch in width/height
    room_area = max(total_room_area, 1.0)
    
    def evaluate_outline(layer_name: str, bounds, area: float):
        nonlocal best_outline, best_outline_score
        minx, miny, maxx, maxy = bounds
        width = max(maxx - minx, 1.0)
        height = max(maxy - miny, 1.0)
        width_ratio = width / room_width
        height_ratio = height / room_height
        if not (1 - tolerance_ratio <= width_ratio <= 1 + tolerance_ratio):
            return
        if not (1 - tolerance_ratio <= height_ratio <= 1 + tolerance_ratio):
            return
        area_ratio = area / room_area if area > 0 else 1.0
        # Score based on closeness of width/height plus a small penalty for area mismatch.
        score = abs(1 - width_ratio) + abs(1 - height_ratio) + abs(1 - area_ratio) * 0.5
        # Reward layers that contain architectural keywords so they win ties naturally.
        for keyword in priority_keywords:
            if keyword in layer_name:
                score *= 0.8
                break
        if score < best_outline_score:
            best_outline_score = score
            best_outline = {
                "layer": layer_name,
                "bounds": bounds,
                "offset_x": minx - room_minx,
                "offset_y": miny - room_miny,
            }
    
    # Inspect closed polylines first—they often define usable outlines.
    for e in msp.query("LWPOLYLINE"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set or not e.closed:
                continue
            pts = [(v[0], v[1]) for v in e.get_points()]
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not (poly.is_valid and poly.area > 0):
                continue
            evaluate_outline(layer, poly.bounds, poly.area)
        except Exception:
            continue
    
    for e in msp.query("POLYLINE"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set or not e.is_closed:
                continue
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not (poly.is_valid and poly.area > 0):
                continue
            evaluate_outline(layer, poly.bounds, poly.area)
        except Exception:
            continue
    
    if best_outline:
        offset_x = best_outline["offset_x"]
        offset_y = best_outline["offset_y"]
        offset_magnitude = (offset_x ** 2 + offset_y ** 2) ** 0.5
        room_size = max(room_width, room_height)
        print("")
        print("🔍 Auto-detected drawing offset via outline match:")
        print(f"   Reference layer: {best_outline['layer']}")
        print(f"   Room bounds: X [{room_minx:.0f}, {room_maxx:.0f}]  Y [{room_miny:.0f}, {room_maxy:.0f}]")
        minx, miny, maxx, maxy = best_outline["bounds"]
        print(f"   Layer bounds: X [{minx:.0f}, {maxx:.0f}]  Y [{miny:.0f}, {maxy:.0f}]")
        print(f"   Applying offset: ({offset_x:.0f}, {offset_y:.0f})")
        if offset_magnitude < room_size * 0.01:
            print("   Detected offset is very small; rooms likely already aligned.")
            return 0.0, 0.0
        return offset_x, offset_y
    
    # First, try to find architectural layers
    from collections import defaultdict
    layer_coords = defaultdict(lambda: {"xs": [], "ys": [], "count": 0})
    
    for e in msp:
        try:
            layer = e.dxf.layer.upper()
            # Skip room layers
            if layer in room_layer_set:
                continue
            
            dxftype = e.dxftype()
            
            if dxftype == "LINE":
                layer_coords[layer]["xs"].extend([e.dxf.start.x, e.dxf.end.x])
                layer_coords[layer]["ys"].extend([e.dxf.start.y, e.dxf.end.y])
                layer_coords[layer]["count"] += 1
            elif dxftype == "LWPOLYLINE":
                for v in e.get_points():
                    layer_coords[layer]["xs"].append(v[0])
                    layer_coords[layer]["ys"].append(v[1])
                layer_coords[layer]["count"] += 1
        except Exception:
            pass
    
    # Find the best architectural layer
    priority_xs = []
    priority_ys = []
    priority_layer_found = None
    
    # Evaluate layer bounding boxes as potential outlines (helps when outline is made of lines)
    for layer, data in layer_coords.items():
        xs, ys = data["xs"], data["ys"]
        if layer in room_layer_set or data["count"] < 5 or len(xs) < 4 or len(ys) < 4:
            continue
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        width = max(maxx - minx, 1.0)
        height = max(maxy - miny, 1.0)
        approx_area = width * height
        evaluate_outline(layer, (minx, miny, maxx, maxy), approx_area)
    
    if best_outline:
        offset_x = best_outline["offset_x"]
        offset_y = best_outline["offset_y"]
        offset_magnitude = (offset_x ** 2 + offset_y ** 2) ** 0.5
        room_size = max(room_width, room_height)
        print("")
        print("🔍 Auto-detected drawing offset via layer extents:")
        print(f"   Reference layer: {best_outline['layer']}")
        print(f"   Room bounds: X [{room_minx:.0f}, {room_maxx:.0f}]  Y [{room_miny:.0f}, {room_maxy:.0f}]")
        minx, miny, maxx, maxy = best_outline["bounds"]
        print(f"   Layer bounds: X [{minx:.0f}, {maxx:.0f}]  Y [{miny:.0f}, {maxy:.0f}]")
        print(f"   Applying offset: ({offset_x:.0f}, {offset_y:.0f})")
        if offset_magnitude < room_size * 0.01:
            print("   Detected offset is very small; rooms likely already aligned.")
            return 0.0, 0.0
        return offset_x, offset_y
    
    # Try priority keywords first
    for keyword in priority_keywords:
        for layer, data in layer_coords.items():
            if keyword in layer and data["count"] > 10:  # Must have reasonable number of entities
                priority_xs.extend(data["xs"])
                priority_ys.extend(data["ys"])
                if not priority_layer_found:
                    priority_layer_found = layer
    
    # If no priority layers found, use the layer with most entities
    if not priority_xs:
        sorted_layers = sorted(layer_coords.items(), key=lambda x: -x[1]["count"])
        if sorted_layers:
            for layer, data in sorted_layers[:3]:  # Use top 3 layers
                priority_xs.extend(data["xs"])
                priority_ys.extend(data["ys"])
                if not priority_layer_found:
                    priority_layer_found = layer
    
    if not priority_xs:
        print("⚠️  Could not auto-detect offset (no suitable layers found)")
        return 0.0, 0.0
    
    main_minx, main_maxx = min(priority_xs), max(priority_xs)
    main_miny, main_maxy = min(priority_ys), max(priority_ys)
    main_centerx = (main_minx + main_maxx) / 2
    main_centery = (main_miny + main_maxy) / 2
    
    # Calculate offset to align centers
    offset_x = main_centerx - room_centerx
    offset_y = main_centery - room_centery
    
    # Check if offset is significant (> 1% of drawing size)
    room_size = max(room_maxx - room_minx, room_maxy - room_miny)
    offset_magnitude = (offset_x**2 + offset_y**2)**0.5
    
    if offset_magnitude < room_size * 0.01:
        # Offset is negligible, rooms are likely already aligned
        print(f"✅ Rooms are already aligned with main drawing (offset < 1%)")
        return 0.0, 0.0
    
    print(f"")
    print(f"🔍 Auto-detected drawing offset:")
    print(f"   Rooms center: ({room_centerx:.0f}, {room_centery:.0f})")
    print(f"   Main drawing center: ({main_centerx:.0f}, {main_centery:.0f})")
    if priority_layer_found:
        print(f"   Reference layer: {priority_layer_found}")
    print(f"   Applying offset: ({offset_x:.0f}, {offset_y:.0f})")
    
    return offset_x, offset_y


def unit_scale(units: str) -> float:
    if units == "m":
        return 1.0
    if units == "mm":
        return 1000.0
    if units == "ft":
        return 1.0 / 0.3048
    if units == "in":
        return 1.0 / 0.0254
    return 1.0


def create_building_bounds(doc) -> Polygon:
    """Create building bounds using union of all room polygons (most accurate)"""
    msp = doc.modelspace()
    room_polygons = []
    
    # Collect all closed polylines (rooms)
    for entity in msp:
        try:
            if entity.dxftype() == "LWPOLYLINE" and entity.closed:
                points = entity.get_points()
                if points and len(points) >= 3:
                    try:
                        poly = Polygon([(pt[0], pt[1]) for pt in points])
                        if poly.is_valid and poly.area > 0:
                            room_polygons.append(poly)
                    except Exception:
                        continue
            elif entity.dxftype() == "POLYLINE" and entity.is_closed:
                points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                if points and len(points) >= 3:
                    try:
                        poly = Polygon(points)
                        if poly.is_valid and poly.area > 0:
                            room_polygons.append(poly)
                    except Exception:
                        continue
        except Exception:
            continue
    
    if not room_polygons:
        return None
    
    # Create union of all room polygons to get the actual building shape
    try:
        from shapely.ops import unary_union
        building_union = unary_union(room_polygons)
        
        # If union is a single polygon, use it directly
        if isinstance(building_union, Polygon):
            return building_union
        # If union is a MultiPolygon, get the largest one
        elif hasattr(building_union, 'geoms'):
            largest_poly = max(building_union.geoms, key=lambda p: p.area)
            return largest_poly
        else:
            # Fallback to bounding box
            bounds = building_union.bounds
            return Polygon([
                (bounds[0], bounds[1]),
                (bounds[2], bounds[1]),
                (bounds[2], bounds[3]),
                (bounds[0], bounds[3])
            ])
    except Exception:
        # Fallback to simple bounding box
        all_xs = []
        all_ys = []
        for poly in room_polygons:
            bounds = poly.bounds
            all_xs.extend([bounds[0], bounds[2]])
            all_ys.extend([bounds[1], bounds[3]])
        
        minx, maxx = min(all_xs), max(all_xs)
        miny, maxy = min(all_ys), max(all_ys)
        
        return Polygon([
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy)
        ])


def dxf_to_room_polygons(doc, layers: List[str]) -> List[Tuple[str, Polygon]]:
    msp = doc.modelspace()
    polys: List[Tuple[str, Polygon]] = []

    def lwpoly_to_poly(e):
        pts = [(v[0], v[1]) for v in e.get_points()]
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return Polygon(pts)

    for e in msp.query("LWPOLYLINE"):
        try:
            if e.dxf.layer.upper() in layers and e.closed:
                poly = lwpoly_to_poly(e)
                if poly.is_valid and poly.area > 0:
                    polys.append((e.dxf.layer, poly))
        except Exception:
            continue

    for e in msp.query("POLYLINE"):
        try:
            if e.dxf.layer.upper() in layers and e.is_closed:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    polys.append((e.dxf.layer, poly))
        except Exception:
            continue

    return polys


def map_room_names(doc, polygons: List[Polygon], name_layer: str) -> Dict[int, str]:
    msp = doc.modelspace()
    texts = []
    for e in msp.query("TEXT MTEXT"):
        try:
            if e.dxf.layer.upper() == name_layer:
                # TEXT vs MTEXT text extraction
                try:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else e.text
                except Exception:
                    try:
                        txt = e.plain_text()
                    except Exception:
                        txt = ""
                if not txt:
                    continue
                # insertion point
                try:
                    x = float(e.dxf.insert.x)
                    y = float(e.dxf.insert.y)
                except Exception:
                    try:
                        x = float(e.dxf.location.x)
                        y = float(e.dxf.location.y)
                    except Exception:
                        continue
                texts.append((txt.strip(), (x, y)))
        except Exception:
            continue

    names: Dict[int, str] = {}
    centroids = [p.centroid for p in polygons]
    for i, c in enumerate(centroids):
        if not texts:
            break
        cx, cy = c.x, c.y
        nearest = min(texts, key=lambda t: (t[1][0] - cx) ** 2 + (t[1][1] - cy) ** 2)
        names[i] = nearest[0]
    return names


def hex_grid_points_in_poly(poly: Polygon, spacing: float):
    minx, miny, maxx, maxy = poly.bounds
    h = spacing * math.sin(math.radians(60))
    points = []
    y = miny
    row = 0
    while y <= maxy:
        offset = 0.0 if row % 2 == 0 else spacing * 0.5
        x = minx + offset
        while x <= maxx:
            pt = Point(x, y)
            if poly.contains(pt):
                points.append((x, y))
            x += spacing
        y += h
        row += 1
    return points


def square_grid_points_in_poly(poly: Polygon, spacing: float):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + spacing * 0.5, spacing)
    ys = np.arange(miny, maxy + spacing * 0.5, spacing)
    pts = []
    for y in ys:
        for x in xs:
            if poly.contains(Point(x, y)):
                pts.append((x, y))
    return pts


def compute_spacing(standard: str, custom_spacing: Optional[float]):
    # USER REQUEST: ล็อค spacing ที่ 9.1 เมตรเสมอ
    # Always return 9.1 meters spacing (locked)
    return 9.1, "Locked spacing: 9.1 meters (user request: ระยะห่างระหว่างจุดปักล็อคที่ 9.1เมตร)"


def compute_default_coverage_radius(standard: str, spacing_m: float) -> float:
    """Return default detector coverage radius in meters based on standard."""
    if spacing_m <= 0:
        return 0.0
    if standard == "NFPA72":
        # NFPA 72: 30 ft (9.1 m) spacing corresponds to ~21 ft (6.4 m) coverage radius.
        ratio = 6.4 / 9.1
        return spacing_m * ratio
    if standard == "EN54-14":
        # EN 54-14: Spacing 8.66 m (hex grid) ensures ≤7.5 m to nearest detector.
        ratio = 7.5 / 8.66
        return spacing_m * ratio
    return spacing_m * 0.5


def remove_excessive_overlaps(points: List[Tuple[float, float]], coverage_radius: float, max_overlap_ratio: float = 0.5) -> List[Tuple[float, float]]:
    """
    Remove detectors where coverage circles overlap too much.
    USER REQUEST: "ไม่จำเป็นต้องถี่ ถ้าดูแล้ววงมันซ้อนเยอะเกินจำเป็นก็ไม่ต้องปัก แต่ก็ยังต้องทำให้ครอบคลุม"
    
    Args:
        points: List of (x, y) detector positions
        coverage_radius: Radius of coverage circle in same units as points
        max_overlap_ratio: Maximum allowed overlap ratio (0.0-1.0). 
                          If overlap area > max_overlap_ratio * circle_area, remove the detector.
                          Default 0.5 means if >50% of circle overlaps, remove it.
    
    Returns:
        Filtered list of points with excessive overlaps removed
    """
    if not points or coverage_radius <= 0:
        return points
    
    if len(points) <= 1:
        return points
    
    circle_area = math.pi * (coverage_radius ** 2)
    max_overlap_area = circle_area * max_overlap_ratio
    
    # Create coverage circles for all points
    coverage_circles = [Point(x, y).buffer(coverage_radius, resolution=12) for (x, y) in points]
    
    # For each point, calculate how much its circle overlaps with others
    points_to_keep = []
    removed_count = 0
    
    for i, (x, y) in enumerate(points):
        current_circle = coverage_circles[i]
        total_overlap_area = 0.0
        
        # Check overlap with all other circles
        for j, other_circle in enumerate(coverage_circles):
            if i == j:
                continue
            
            try:
                # Calculate intersection area
                if current_circle.intersects(other_circle):
                    intersection = current_circle.intersection(other_circle)
                    if intersection.area > 0:
                        total_overlap_area += intersection.area
            except Exception:
                # If calculation fails, assume some overlap for safety
                total_overlap_area += circle_area * 0.1
        
        # If total overlap exceeds threshold, remove this detector
        if total_overlap_area > max_overlap_area:
            removed_count += 1
            overlap_percent = (total_overlap_area / circle_area) * 100
            print(f"   🗑️  Removing detector at ({x:.2f}, {y:.2f}): overlap {overlap_percent:.1f}% > {max_overlap_ratio*100:.0f}% threshold")
        else:
            points_to_keep.append((x, y))
    
    if removed_count > 0:
        print(f"   ✅ Removed {removed_count} detector(s) with excessive overlap (kept {len(points_to_keep)}/{len(points)})")
    
    return points_to_keep


def fill_coverage_gaps(points_by_room: List[Dict], coverage_radius: float, target_polygon: Polygon) -> List[Tuple[float, float]]:
    """Add detectors in uncovered zones to maintain coverage continuity."""
    if coverage_radius is None or coverage_radius <= 0:
        return []
    if target_polygon is None:
        return []
    
    all_points = [(x, y) for room in points_by_room for (x, y) in room["points"]]
    if not all_points:
        return []
    
    try:
        coverage_shapes = [Point(x, y).buffer(coverage_radius, resolution=12) for (x, y) in all_points]
        coverage_union = unary_union(coverage_shapes)
        uncovered = target_polygon.difference(coverage_union)
    except Exception:
        return []
    
    min_gap_area = math.pi * (coverage_radius ** 2) * 0.05
    added_points: List[Tuple[float, float]] = []
    
    while not uncovered.is_empty:
        if uncovered.geom_type == "Polygon":
            geometries = [uncovered]
        elif hasattr(uncovered, "geoms"):
            geometries = [g for g in uncovered.geoms if g.area > 0]
        else:
            break
        
        if not geometries:
            break
        
        largest = max(geometries, key=lambda g: g.area)
        if largest.area < min_gap_area:
            break
        
        candidate_point = largest.representative_point()
        if not target_polygon.contains(candidate_point):
            candidate_point = largest.centroid
        
        x, y = candidate_point.x, candidate_point.y
        min_dist = min(math.hypot(x - px, y - py) for (px, py) in all_points)
        if min_dist < coverage_radius * 0.6:
            coverage_union = coverage_union.union(largest)
            uncovered = target_polygon.difference(coverage_union)
            continue
        
        added_points.append((x, y))
        all_points.append((x, y))
        new_shape = Point(x, y).buffer(coverage_radius, resolution=12)
        coverage_union = coverage_union.union(new_shape)
        uncovered = target_polygon.difference(coverage_union)
    
    return added_points


def offset_interior(poly: Polygon, margin: float) -> Polygon:
    inset = poly.buffer(-margin, join_style=2)
    if inset.is_empty:
        return poly
    if inset.geom_type == "MultiPolygon":
        areas = [(g.area, g) for g in inset.geoms]
        return max(areas, key=lambda t: t[0])[1]
    return inset


def is_hollow_polygon(poly: Polygon, msp, room_layer_set: set, min_area_units: float, max_area_units: float) -> bool:
    """
    Check if a polygon is "hollow" (empty inside) - meaning it's likely a column.
    A hollow polygon has:
    1. Small rectangular/square shape
    2. No significant entities inside (no grid, no patterns, no other polygons)
    
    Returns True if polygon appears to be a hollow column.
    """
    try:
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if min(width, height) <= 0:
            return False
        
        # Must be roughly square/rectangular
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 6.0:
            return False
        
        # Must be in column size range
        if not (min_area_units <= poly.area <= max_area_units):
            return False
        
        # Check if there are entities inside the polygon
        # If there are many entities inside, it's probably not a hollow column
        entities_inside = 0
        entities_inside_area = 0.0
        
        # Sample points inside the polygon to check for entities
        centroid = poly.centroid
        sample_points = [
            centroid,
            Point(bounds[0] + width * 0.25, bounds[1] + height * 0.25),
            Point(bounds[0] + width * 0.75, bounds[1] + height * 0.25),
            Point(bounds[0] + width * 0.25, bounds[1] + height * 0.75),
            Point(bounds[0] + width * 0.75, bounds[1] + height * 0.75),
        ]
        
        # Check for entities that are inside or intersect the polygon
        for entity in msp:
            try:
                layer = entity.dxf.layer.upper()
                if layer in room_layer_set:
                    continue
                
                # Check if entity is inside the polygon
                entity_inside = False
                entity_area = 0.0
                
                if entity.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                    try:
                        if entity.dxftype() == "LWPOLYLINE":
                            if entity.closed:
                                pts = [(v[0], v[1]) for v in entity.get_points()]
                                if len(pts) >= 3:
                                    entity_poly = Polygon(pts)
                                    if entity_poly.is_valid:
                                        entity_area = entity_poly.area
                                        # Check if entity polygon is inside the column polygon
                                        if poly.contains(entity_poly.centroid):
                                            entity_inside = True
                        elif entity.dxftype() == "POLYLINE":
                            if entity.is_closed:
                                pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                                if len(pts) >= 3:
                                    entity_poly = Polygon(pts)
                                    if entity_poly.is_valid:
                                        entity_area = entity_poly.area
                                        if poly.contains(entity_poly.centroid):
                                            entity_inside = True
                    except Exception:
                        pass
                elif entity.dxftype() in ("LINE", "CIRCLE", "ARC"):
                    try:
                        # Check if entity's center/points are inside
                        if entity.dxftype() == "LINE":
                            mid_x = (entity.dxf.start.x + entity.dxf.end.x) / 2
                            mid_y = (entity.dxf.start.y + entity.dxf.end.y) / 2
                            if poly.contains(Point(mid_x, mid_y)):
                                entity_inside = True
                        elif entity.dxftype() == "CIRCLE":
                            if poly.contains(Point(entity.dxf.center.x, entity.dxf.center.y)):
                                entity_inside = True
                        elif entity.dxftype() == "ARC":
                            if poly.contains(Point(entity.dxf.center.x, entity.dxf.center.y)):
                                entity_inside = True
                    except Exception:
                        pass
                
                if entity_inside:
                    entities_inside += 1
                    entities_inside_area += entity_area
            except Exception:
                continue
        
        # If there are many entities inside (more than 2-3), it's probably not a hollow column
        # If total area of entities inside is significant (>20% of polygon area), it's not hollow
        if entities_inside > 3:
            return False
        if entities_inside_area > poly.area * 0.2:
            return False
        
        # If we get here, it's likely a hollow column
        return True
    except Exception:
        return False


def extract_columns(doc, room_layers: List[str], min_area_m2: float = 0.01, max_area_m2: float = 100.0) -> List[Polygon]:
    """
    Extract column/pillar polygons from DXF file.
    Columns are typically small closed polygons that are not rooms.
    NEW: Also detects "hollow" rectangles (empty inside, no grid/pattern) as columns.
    Can be POLYLINE, LWPOLYLINE, SOLID, HATCH, or INSERT/BLOCK entities.
    
    Args:
        doc: DXF document
        room_layers: List of layer names that contain rooms (to exclude)
        min_area_m2: Minimum area in m² to consider as column (default 0.05 m²)
        max_area_m2: Maximum area in m² to consider as column (default 20.0 m²)
    
    Returns:
        List of Polygon objects representing columns
    """
    msp = doc.modelspace()
    room_layer_set = set(layer.upper() for layer in room_layers)
    
    # Keywords that might indicate columns/pillars
    column_keywords = ["COLUMN", "PILLAR", "POST", "COL", "PIL", "BEAM", "STRUCT"]
    
    columns: List[Polygon] = []
    
    # Detect units to calculate area threshold
    try:
        bbox = msp.bbox()
        max_dim = max(bbox.size.x, bbox.size.y)
    except Exception:
        minx, miny, maxx, maxy = _bbox_fallback(msp)
        max_dim = max(maxx - minx, maxy - miny)
    
    # Guess units: if max dimension > 2000, likely mm, else meters
    if max_dim > 2000:
        scale = 1000.0  # mm to m
    else:
        scale = 1.0  # meters
    
    min_area_units = min_area_m2 * (scale ** 2)
    max_area_units = max_area_m2 * (scale ** 2)
    
    # Helper function to check if polygon is a column
    def is_column_polygon(poly: Polygon, layer: str, area: float) -> bool:
        if not (min_area_units <= area <= max_area_units):
            return False
        
        # Check if layer name suggests it's a column
        is_column_layer = any(keyword in layer for keyword in column_keywords)
        
        # Check if it's a small rectangle/square (typical column shape)
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if min(width, height) <= 0:
            return False
        aspect_ratio = max(width, height) / min(width, height)
        # More lenient: accept aspect ratio up to 6.0 for rectangular columns
        is_small_rect = aspect_ratio < 6.0 and area < max_area_units
        
        return is_column_layer or is_small_rect
    
    # Extract closed polylines that might be columns
    for e in msp.query("LWPOLYLINE"):
        try:
            layer = e.dxf.layer.upper()
            # Skip room layers
            if layer in room_layer_set:
                continue
            
            if not e.closed:
                continue
            
            pts = [(v[0], v[1]) for v in e.get_points()]
            if len(pts) < 3:
                continue
            
            poly = Polygon(pts)
            if not (poly.is_valid and poly.area > 0):
                continue
            
            if is_column_polygon(poly, layer, poly.area):
                columns.append(poly)
        except Exception:
            continue
    
    # Also check POLYLINE entities
    for e in msp.query("POLYLINE"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            if not e.is_closed:
                continue
            
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(pts) < 3:
                continue
            
            poly = Polygon(pts)
            if not (poly.is_valid and poly.area > 0):
                continue
            
            if is_column_polygon(poly, layer, poly.area):
                columns.append(poly)
        except Exception:
            continue
    
    # Check INSERT/BLOCK entities that might be columns
    # These are often used for structural elements like columns
    for e in msp.query("INSERT"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            block_name = e.dxf.name.upper()
            
            # Check if block name suggests it's a column
            is_column_block = any(keyword in block_name for keyword in column_keywords)
            is_column_layer = any(keyword in layer for keyword in column_keywords)
            
            if not (is_column_block or is_column_layer):
                continue
            
            # Get block definition to check its size
            try:
                block_def = doc.blocks.get(block_name)
                if block_def is None:
                    continue
                
                # Get bounding box of block
                try:
                    block_bbox = block_def.bbox()
                    if block_bbox is None:
                        continue
                    (x1, y1, _), (x2, y2, _) = block_bbox.extmin, block_bbox.extmax
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    area = width * height
                except Exception:
                    # Fallback: estimate from block entities
                    minx = miny = float("inf")
                    maxx = maxy = float("-inf")
                    for be in block_def:
                        try:
                            if be.dxftype() in ("LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC"):
                                be_bbox = be.bbox()
                                if be_bbox:
                                    (bx1, by1, _), (bx2, by2, _) = be_bbox.extmin, be_bbox.extmax
                                    minx = min(minx, bx1)
                                    miny = min(miny, by1)
                                    maxx = max(maxx, bx2)
                                    maxy = max(maxy, by2)
                        except Exception:
                            continue
                    if minx == float("inf"):
                        continue
                    width = maxx - minx
                    height = maxy - miny
                    area = width * height
                
                # Check if size is in column range
                if min_area_units <= area <= max_area_units:
                    # Create a polygon representing the column block
                    insert_x = e.dxf.insert.x
                    insert_y = e.dxf.insert.y
                    scale_x = e.dxf.xscale if hasattr(e.dxf, 'xscale') else 1.0
                    scale_y = e.dxf.yscale if hasattr(e.dxf, 'yscale') else 1.0
                    
                    # Create rectangle polygon at insertion point
                    half_w = (width * scale_x) / 2
                    half_h = (height * scale_y) / 2
                    col_poly = Polygon([
                        (insert_x - half_w, insert_y - half_h),
                        (insert_x + half_w, insert_y - half_h),
                        (insert_x + half_w, insert_y + half_h),
                        (insert_x - half_w, insert_y + half_h)
                    ])
                    if col_poly.is_valid:
                        columns.append(col_poly)
            except Exception:
                continue
        except Exception:
            continue
    
    # Also check for rectangular shapes made from LINE entities (common for columns)
    # Group lines by layer and check if they form small rectangles
    from collections import defaultdict
    line_groups = defaultdict(list)
    
    for e in msp.query("LINE"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            # Check if layer suggests it's a column
            is_column_layer = any(keyword in layer for keyword in column_keywords)
            if not is_column_layer:
                continue
            
            line_groups[layer].append(e)
        except Exception:
            continue
    
    # Try to form rectangles from lines on the same layer
    for layer, lines in line_groups.items():
        if len(lines) < 4:  # Need at least 4 lines for a rectangle
            continue
        
        # Collect all endpoints
        points = []
        for line in lines:
            try:
                points.append((line.dxf.start.x, line.dxf.start.y))
                points.append((line.dxf.end.x, line.dxf.end.y))
            except Exception:
                continue
        
        if len(points) < 4:
            continue
        
        # Find bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        width = maxx - minx
        height = maxy - miny
        area = width * height
        
        # Check if it's a small rectangle (likely a column)
        if min_area_units <= area <= max_area_units:
            # Create polygon from bounding box
            rect_poly = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ])
            if rect_poly.is_valid:
                columns.append(rect_poly)
    
    # Check SOLID entities (CRITICAL: often used for filled rectangles like orange/tan columns)
    # These are the filled squares shown in the user's image - MUST be detected!
    print(f"   🔍 Checking SOLID entities (filled rectangles like orange/tan columns)...")
    solid_count = 0
    for e in msp.query("SOLID"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            # Get SOLID vertices - these are filled rectangles, likely columns
            try:
                v1 = e.dxf.v1
                v2 = e.dxf.v2
                v3 = e.dxf.v3
                v4 = e.dxf.v4 if hasattr(e.dxf, 'v4') else v3
                
                # Create polygon from SOLID vertices
                solid_poly = Polygon([(v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y), (v4.x, v4.y)])
                if solid_poly.is_valid and solid_poly.area > 0:
                    # Check if it's a small rectangular shape (likely column)
                    # IMPORTANT: Don't filter by layer name - filled SOLIDs are often columns!
                    if min_area_units <= solid_poly.area <= max_area_units:
                        bounds = solid_poly.bounds
                        width = bounds[2] - bounds[0]
                        height = bounds[3] - bounds[1]
                        if min(width, height) > 0:
                            aspect_ratio = max(width, height) / min(width, height)
                            # Accept rectangular shapes (columns are often square or slightly rectangular)
                            if aspect_ratio < 6.0:
                                # Check for duplicates
                                already_added = False
                                for existing_col in columns:
                                    try:
                                        if existing_col.intersects(solid_poly):
                                            intersection = existing_col.intersection(solid_poly)
                                            if intersection.area > solid_poly.area * 0.5:
                                                already_added = True
                                                break
                                    except Exception:
                                        pass
                                if not already_added:
                                    columns.append(solid_poly)
                                    solid_count += 1
            except Exception:
                continue
        except Exception:
            continue
    if solid_count > 0:
        print(f"      Found {solid_count} column(s) from SOLID entities")
    
    # Check HATCH entities (CRITICAL: filled areas, often used for colored columns like orange/tan)
    # These are the filled squares shown in the user's image - MUST be detected!
    print(f"   🔍 Checking HATCH entities (filled areas like colored columns)...")
    hatch_count = 0
    for e in msp.query("HATCH"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            # Get HATCH boundary paths (check ALL hatches, not just column-named ones)
            # Filled columns are often represented as HATCH entities
            try:
                if hasattr(e, 'paths') and e.paths:
                    for path in e.paths:
                        if hasattr(path, 'vertices'):
                            vertices = [(v[0], v[1]) for v in path.vertices if len(v) >= 2]
                            if len(vertices) >= 3:
                                hatch_poly = Polygon(vertices)
                                if hatch_poly.is_valid and hatch_poly.area > 0:
                                    # Check if it's a small rectangular shape (likely column)
                                    # IMPORTANT: Don't filter by layer name - filled HATCHes are often columns!
                                    if min_area_units <= hatch_poly.area <= max_area_units:
                                        bounds = hatch_poly.bounds
                                        width = bounds[2] - bounds[0]
                                        height = bounds[3] - bounds[1]
                                        if min(width, height) > 0:
                                            aspect_ratio = max(width, height) / min(width, height)
                                            # Accept rectangular shapes (columns are often square or slightly rectangular)
                                            if aspect_ratio < 6.0:
                                                # Check for duplicates
                                                already_added = False
                                                for existing_col in columns:
                                                    try:
                                                        if existing_col.intersects(hatch_poly):
                                                            intersection = existing_col.intersection(hatch_poly)
                                                            if intersection.area > hatch_poly.area * 0.5:
                                                                already_added = True
                                                                break
                                                    except Exception:
                                                        pass
                                                if not already_added:
                                                    columns.append(hatch_poly)
                                                    hatch_count += 1
            except Exception:
                continue
        except Exception:
            continue
    if hatch_count > 0:
        print(f"      Found {hatch_count} column(s) from HATCH entities")
    
    # ULTRA-AGGRESSIVE detection: find ALL small closed polygons regardless of layer name
    # NEW: Also check if polygons are "hollow" (empty inside) - these are likely columns
    # This is critical to catch filled columns (like orange/tan squares in the image)
    # Exclude only room layers and very large/small shapes
    print(f"   🔍 Scanning all closed polygons for column detection (including hollow detection)...")
    
    hollow_count = 0
    for e in msp.query("LWPOLYLINE"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            if not e.closed:
                continue
            
            pts = [(v[0], v[1]) for v in e.get_points()]
            if len(pts) < 3:
                continue
            
            poly = Polygon(pts)
            if not (poly.is_valid and poly.area > 0):
                continue
            
            # Check if it's a small rectangle/square (likely a column)
            if min_area_units <= poly.area <= max_area_units:
                bounds = poly.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                if min(width, height) <= 0:
                    continue
                aspect_ratio = max(width, height) / min(width, height)
                
                # Small, roughly square/rectangular shapes are likely columns
                # Be more lenient: accept aspect ratio up to 6.0
                if aspect_ratio < 6.0:
                    # NEW: Check if it's a hollow polygon (empty inside, no grid/pattern)
                    is_hollow = is_hollow_polygon(poly, msp, room_layer_set, min_area_units, max_area_units)
                    
                    # Accept if it's hollow OR if it matches column criteria
                    if is_hollow or is_column_polygon(poly, layer, poly.area):
                        # Check if it's not already in the list (avoid duplicates)
                        already_added = False
                        for existing_col in columns:
                            try:
                                # Check if they overlap significantly
                                if existing_col.intersects(poly):
                                    intersection = existing_col.intersection(poly)
                                    if intersection.area > poly.area * 0.5:  # More than 50% overlap
                                        already_added = True
                                        break
                            except Exception:
                                pass
                        if not already_added:
                            columns.append(poly)
                            if is_hollow:
                                hollow_count += 1
        except Exception:
            continue
    
    if hollow_count > 0:
        print(f"      Found {hollow_count} hollow column(s) (empty inside, no grid/pattern)")
    
    # Also check POLYLINE entities with same aggressive approach
    for e in msp.query("POLYLINE"):
        try:
            layer = e.dxf.layer.upper()
            if layer in room_layer_set:
                continue
            
            if not e.is_closed:
                continue
            
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(pts) < 3:
                continue
            
            poly = Polygon(pts)
            if not (poly.is_valid and poly.area > 0):
                continue
            
            if min_area_units <= poly.area <= max_area_units:
                bounds = poly.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                if min(width, height) <= 0:
                    continue
                aspect_ratio = max(width, height) / min(width, height)
                
                if aspect_ratio < 6.0:
                    already_added = False
                    for existing_col in columns:
                        try:
                            if existing_col.intersects(poly):
                                intersection = existing_col.intersection(poly)
                                if intersection.area > poly.area * 0.5:
                                    already_added = True
                                    break
                        except Exception:
                            pass
                    if not already_added:
                        columns.append(poly)
        except Exception:
            continue
    
    return columns


def place_detectors_in_room(room_poly: Polygon, spacing_units: float, margin_units: float, grid_type: str, building_bounds: Polygon = None, columns: Optional[List[Polygon]] = None):
    # Default buffer distance: 6.4m radius
    buffer_distance_default = 6.4 * (1000.0 if spacing_units > 100 else 1.0)
    if room_poly.area <= 0:
        return [], buffer_distance_default
    inner = offset_interior(room_poly, margin_units)
    if inner.is_empty:
        return [], buffer_distance_default
    if grid_type == "hex":
        pts = hex_grid_points_in_poly(inner, spacing_units)
    else:
        pts = square_grid_points_in_poly(inner, spacing_units)
    if not pts:
        c = inner.centroid
        if inner.contains(c):
            pts = [(c.x, c.y)]
    
    # Default buffer distance: 6.4m radius
    buffer_distance_default = 6.4 * (1000.0 if spacing_units > 100 else 1.0)
    
    # Filter points that are outside building bounds if provided
    if building_bounds and pts:
        filtered_pts = []
        for (x, y) in pts:
            pt = Point(x, y)
            # Moderate filtering: keep points that are inside building bounds
            # Allow points close to boundary (within 0.5 meter tolerance)
            if building_bounds.contains(pt) or building_bounds.distance(pt) <= 0.5:
                filtered_pts.append((x, y))
        pts = filtered_pts
        # If no points after building bounds filtering, return early
        if not pts:
            return [], buffer_distance_default
    
    # Filter points that overlap with columns
    # IMPORTANT: This must run AFTER building bounds filtering
    # CRITICAL: User said "ถ้ามันใกล้เสา ไม่ต้องปักก็ได้" - skip points near columns!
    if columns is not None and len(columns) > 0 and pts:
        filtered_pts = []
        original_count = len(pts)
        # Calculate buffer distance: USER REQUEST - "ให้วาดวงกลมรัศมี 6.4เมตร แล้วห้ามปัก ในเขตนั้น"
        # Use EXACTLY 6.4 meters radius for column exclusion zones
        # This ensures detectors are placed OUTSIDE the 6.4m radius circle, not ON or near columns
        # Detect units from spacing_units magnitude: if > 100, likely mm; else meters
        buffer_radius_m = 6.4  # EXACTLY 6.4 meters radius as requested by user
        if spacing_units > 100:
            # Likely mm units, so 6.4m = 6400mm
            buffer_distance = buffer_radius_m * 1000
        else:
            # Likely meters
            buffer_distance = buffer_radius_m
        
        # Convert buffer to meters for debug output
        buffer_m = buffer_distance / (1000.0 if spacing_units > 100 else 1.0)
        
        print(f"   🔍 Checking {original_count} detector points against {len(columns)} columns")
        print(f"   ⚠️  CRITICAL: Skipping ALL points within {buffer_m:.2f} m radius of any column")
        print(f"   ⚠️  User: 'ให้วาดวงกลมรัศมี 2เมตร แล้วห้ามปัก ในเขตนั้น' - using 2.0m radius exclusion zone")
        
        overlaps_count = 0
        overlap_details = []  # Store details for debugging
        
        for (x, y) in pts:
            pt = Point(x, y)
            # Check if point is inside any column (with buffer)
            overlaps_column = False
            closest_col_idx = None
            closest_distance = float('inf')
            
            for idx, col_poly in enumerate(columns):
                try:
                    # USER REQUEST: "ให้วาดวงกลมรัศมี 2เมตร แล้วห้ามปัก ในเขตนั้น"
                    # Check if point is within 2m radius circle around column CENTROID
                    # Get column centroid (center point)
                    col_centroid = col_poly.centroid
                    col_center = Point(col_centroid.x, col_centroid.y)
                    
                    # Calculate distance from point to column center
                    dist_to_center = pt.distance(col_center)
                    
                    # Check 1: If point is within 2m radius circle from column center, reject it
                    # This is the PRIMARY check - user wants 2m radius exclusion zone
                    if dist_to_center < buffer_distance:
                        overlaps_column = True
                        overlaps_count += 1
                        if dist_to_center < closest_distance:
                            closest_distance = dist_to_center
                            closest_col_idx = idx
                        break
                    
                    # Check 2: Also check if point is inside column polygon itself (safety check)
                    if col_poly.contains(pt):
                        overlaps_column = True
                        overlaps_count += 1
                        closest_col_idx = idx
                        closest_distance = 0.0
                        break
                    
                    # Check 4: Additional safety check - check if point is within column bounds + buffer
                    col_bounds = col_poly.bounds
                    col_minx, col_miny, col_maxx, col_maxy = col_bounds
                    # Expand bounds by buffer
                    if (col_minx - buffer_distance <= x <= col_maxx + buffer_distance and
                        col_miny - buffer_distance <= y <= col_maxy + buffer_distance):
                        # Point is within expanded bounds, check distance again
                        dist = col_poly.distance(pt)
                        if dist < buffer_distance:
                            overlaps_column = True
                            overlaps_count += 1
                            if dist < closest_distance:
                                closest_distance = dist
                                closest_col_idx = idx
                            break
                except Exception as e:
                    # If ANY check fails, assume overlap for safety (better safe than sorry)
                    overlaps_column = True
                    overlaps_count += 1
                    closest_col_idx = idx
                    break
            
            if overlaps_column:
                # Store overlap details for first few overlaps
                if len(overlap_details) < 10:
                    overlap_details.append((x, y, closest_col_idx, closest_distance))
            else:
                filtered_pts.append((x, y))
        
        filtered_count = len(filtered_pts)
        removed = original_count - filtered_count
        
        # Always print debug info when columns are present
        if removed > 0:
            print(f"   ✅ SKIPPED {removed} detector point(s) that are too close to columns (buffer: {buffer_m:.2f} m)")
            print(f"      (User instruction: 'ถ้ามันใกล้เสา ไม่ต้องปักก็ได้' - skipping points near columns)")
            if overlap_details:
                print(f"      Sample skipped points (first {min(5, len(overlap_details))}):")
                for ox, oy, col_idx, dist in overlap_details[:5]:
                    dist_m = dist / (1000.0 if spacing_units > 100 else 1.0)
                    print(f"         Point ({ox:.1f}, {oy:.1f}) too close to column #{col_idx+1} (distance: {dist_m:.2f} m) - SKIPPED")
        elif original_count > 0:
            # No points filtered - this might indicate columns aren't being detected properly
            print(f"   ⚠️  WARNING: {original_count} detector points checked, but NONE were filtered (columns: {len(columns)})")
            print(f"      This might mean:")
            print(f"      1. Columns are not overlapping detector points (good!)")
            print(f"      2. OR columns are not being detected properly (bad!)")
            print(f"      3. OR columns list is not being passed correctly (bad!)")
            # Show sample detector points and columns for debugging
            if len(pts) > 0 and len(columns) > 0:
                sample_pt = Point(pts[0][0], pts[0][1])
                sample_col = columns[0]
                dist_to_col = sample_col.distance(sample_pt)
                dist_m = dist_to_col / (1000.0 if spacing_units > 100 else 1.0)
                print(f"      DEBUG: First detector at ({pts[0][0]:.1f}, {pts[0][1]:.1f}), distance to first column: {dist_m:.2f} m")
                print(f"      DEBUG: Buffer distance is {buffer_m:.2f} m - if distance < buffer, point should be filtered")
                if dist_m < buffer_m:
                    print(f"      ⚠️  ERROR: Point should have been filtered but wasn't! This is a bug!")
                else:
                    print(f"      ✅ Point is far enough from column (distance {dist_m:.2f} m > buffer {buffer_m:.2f} m)")
        
        pts = filtered_pts
        
        # FINAL VERIFICATION: Double-check that no points are inside columns
        # This is a safety check to catch any points that might have slipped through
        final_check_count = 0
        for (x, y) in pts:
            pt = Point(x, y)
            for col_poly in columns:
                try:
                    if col_poly.contains(pt):
                        final_check_count += 1
                        print(f"   ⚠️  ERROR: Point ({x:.1f}, {y:.1f}) is INSIDE a column but wasn't filtered! This is a bug!")
                        break
                except Exception:
                    pass
        
        if final_check_count > 0:
            print(f"   ⚠️  WARNING: {final_check_count} point(s) passed filtering but are still inside columns!")
            # Remove these points
            final_filtered = []
            for (x, y) in pts:
                pt = Point(x, y)
                is_inside = False
                for col_poly in columns:
                    try:
                        if col_poly.contains(pt):
                            is_inside = True
                            break
                    except Exception:
                        pass
                if not is_inside:
                    final_filtered.append((x, y))
            pts = final_filtered
            print(f"   ✅ Removed {final_check_count} point(s) that were inside columns")
    elif columns is not None and len(columns) == 0:
        # Columns list is empty
        if pts:
            print(f"   ⚠️  Warning: No columns detected, {len(pts)} detector points placed without column filtering")
            print(f"   ⚠️  This is why detectors are still overlapping columns!")
        buffer_distance = buffer_distance_default
    elif columns is None:
        # Columns not provided
        if pts:
            print(f"   ⚠️  ERROR: Columns list is None, {len(pts)} detector points placed without column filtering")
            print(f"   ⚠️  This is why detectors are still overlapping columns!")
        buffer_distance = buffer_distance_default
    
    return pts, buffer_distance


def write_output_dxf(src_doc, out_path: Path, points_by_room: List[Dict], offset_x: float = 0.0, offset_y: float = 0.0,
                     coverage_radius: Optional[float] = None, coverage_radius_m: Optional[float] = None,
                     columns: Optional[List[Polygon]] = None, buffer_distance: Optional[float] = None):
    doc = src_doc
    msp = doc.modelspace()
    
    # Debug: Check if columns and buffer_distance are provided
    print(f"🔍 write_output_dxf: columns={columns is not None}, columns_count={len(columns) if columns else 0}, buffer_distance={buffer_distance}")

    # Calculate appropriate symbol size based on drawing extents
    try:
        bbox = msp.bbox()
        max_dim = max(bbox.size.x, bbox.size.y)
    except Exception:
        # Fallback to calculating from entities
        minx, miny, maxx, maxy = _bbox_fallback(msp)
        max_dim = max(maxx - minx, maxy - miny)
    
    # Symbol size: about 0.3-0.5% of drawing dimension (visible but not overwhelming)
    symbol_radius = max_dim * 0.004  # 0.4%
    cross_size = symbol_radius * 1.5
    
    # Text height: about 0.2% of drawing dimension for radius labels
    text_height = max_dim * 0.002

    layer_name = "SMOKE_DETECTORS"
    try:
        if layer_name not in doc.layers:
            doc.layers.add(name=layer_name, color=1)  # Red color
    except Exception:
        try:
            doc.layers.add(layer_name)
        except Exception:
            pass
    
    coverage_layer_name = "SMOKE_COVERAGE"
    coverage_text_layer_name = "SMOKE_COVERAGE_TEXT"
    if coverage_radius and coverage_radius > 0:
        try:
            if coverage_layer_name not in doc.layers:
                # Use indexed green (ACI color 3) for visibility
                doc.layers.add(name=coverage_layer_name, color=3)
        except Exception:
            try:
                doc.layers.add(coverage_layer_name)
            except Exception:
                pass
        
        # Create layer for radius text labels
        try:
            if coverage_text_layer_name not in doc.layers:
                doc.layers.add(name=coverage_text_layer_name, color=3)
        except Exception:
            try:
                doc.layers.add(coverage_text_layer_name)
            except Exception:
                pass

    blk_name = "SMOKE_DET_SYMBOL"
    if blk_name not in doc.blocks:
        blk = doc.blocks.new(name=blk_name)
        # Circle for detector
        blk.add_circle(center=(0, 0), radius=symbol_radius, dxfattribs={"color": 1})
        # Crosshair
        blk.add_line((-cross_size, 0), (cross_size, 0), dxfattribs={"color": 1})
        blk.add_line((0, -cross_size), (0, cross_size), dxfattribs={"color": 1})

    # USER REQUEST: "ให้วาดวงกลมรัศมี 2เมตร" - Draw 2m radius circles around columns FIRST
    # Draw circles BEFORE filtering points so they're always visible
    if columns is not None and len(columns) > 0:
        print(f"   📐 Drawing 6.4m radius circles around {len(columns)} columns...")
        # Create a layer for column buffer zones
        column_buffer_layer = "COLUMN_BUFFER_ZONES"
        try:
            if column_buffer_layer not in doc.layers:
                doc.layers.add(name=column_buffer_layer, color=6)  # Magenta color for visibility
                print(f"   ✅ Created layer '{column_buffer_layer}' for column buffer zones")
        except Exception:
            try:
                doc.layers.add(column_buffer_layer)
                print(f"   ✅ Created layer '{column_buffer_layer}' for column buffer zones (fallback)")
            except Exception:
                print(f"   ⚠️  Warning: Could not create layer '{column_buffer_layer}'")
        
        # Calculate buffer distance for visualization
        # USER REQUEST: "ให้วาดวงกลมรัศมี 6.4เมตร" - Use EXACTLY 6.4 meters radius
        if buffer_distance is not None and buffer_distance > 0:
            buffer_vis_units = buffer_distance
            print(f"   📏 Using buffer_distance from parameter: {buffer_vis_units} units")
        else:
            # Fallback: use 6.4m radius
            try:
                bbox = msp.bbox()
                max_dim = max(bbox.size.x, bbox.size.y)
            except Exception:
                minx, miny, maxx, maxy = _bbox_fallback(msp)
                max_dim = max(maxx - minx, maxy - miny)
            if max_dim > 2000:
                buffer_vis_units = 6400  # 6.4m in mm
            else:
                buffer_vis_units = 6.4  # 6.4m
            print(f"   📏 Using fallback buffer_distance: {buffer_vis_units} units (max_dim: {max_dim:.1f})")
        
        # Convert to meters for display
        try:
            bbox = msp.bbox()
            max_dim = max(bbox.size.x, bbox.size.y)
        except Exception:
            minx, miny, maxx, maxy = _bbox_fallback(msp)
            max_dim = max(maxx - minx, maxy - miny)
        buffer_vis_m = buffer_vis_units / (1000.0 if max_dim > 2000 else 1.0)
        
        print(f"   ⚠️  User: 'ให้วาดวงกลมรัศมี 6.4เมตร แล้วห้ามปัก ในเขตนั้น' - drawing circles on layer 'COLUMN_BUFFER_ZONES'")
        print(f"   📐 Buffer radius: {buffer_vis_units} units = {buffer_vis_m:.2f} m")
        
        # Draw 6.4m radius circles around each column
        circles_drawn = 0
        for col_idx, col_poly in enumerate(columns):
            try:
                # Get column centroid
                centroid = col_poly.centroid
                print(f"   🔵 Column #{col_idx+1}: centroid=({centroid.x:.2f}, {centroid.y:.2f}), radius={buffer_vis_units:.2f}")
                
                # Draw circle with EXACTLY 6.4m radius around column center
                circle = msp.add_circle(center=(centroid.x, centroid.y), radius=buffer_vis_units,
                                      dxfattribs={"layer": column_buffer_layer})
                circles_drawn += 1
                print(f"      ✅ Circle drawn successfully")
            except Exception as e:
                print(f"   ⚠️  ERROR: Could not draw circle for column #{col_idx+1}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"   ✅ Successfully drew {circles_drawn} circles around {len(columns)} columns")
    
    # FINAL SAFETY CHECK: Filter out any points that are inside columns before writing
    # This is a last-resort check to ensure NO detectors are placed on columns
    # CRITICAL: User showed image with detector overlapping column - MUST prevent this!
    filtered_points_by_room = []
    if columns is not None and len(columns) > 0:
        total_points = sum(len(r['points']) for r in points_by_room)
        print(f"🔍 Final safety check: Verifying {total_points} detector points against {len(columns)} columns...")
        print(f"   ⚠️  CRITICAL: Removing ANY points that are inside columns before writing to DXF!")
        removed_count = 0
        for room in points_by_room:
            filtered_points = []
            for (x, y) in room["points"]:
                # Apply offset to get final position
                adjusted_x = x + offset_x
                adjusted_y = y + offset_y
                pt = Point(adjusted_x, adjusted_y)
                
                # Check if point is inside any column - STRICT CHECK
                is_inside_column = False
                for col_idx, col_poly in enumerate(columns):
                    try:
                        # Check 1: Direct containment
                        if col_poly.contains(pt):
                            is_inside_column = True
                            removed_count += 1
                            print(f"   ⚠️  REMOVED: Point ({adjusted_x:.1f}, {adjusted_y:.1f}) is INSIDE column #{col_idx+1} - skipping!")
                            break
                        
                        # Check 2: Distance check - if point is very close to column, reject it
                        dist_to_col = col_poly.distance(pt)
                        if dist_to_col < 0.1:  # If within 0.1 units, reject it
                            is_inside_column = True
                            removed_count += 1
                            print(f"   ⚠️  REMOVED: Point ({adjusted_x:.1f}, {adjusted_y:.1f}) is too close to column #{col_idx+1} (distance: {dist_to_col:.3f}) - skipping!")
                            break
                        
                        # Check 3: Also check with tiny buffer to catch points on edge
                        tiny_buffer = 0.1  # Small buffer to catch points on edge
                        if col_poly.buffer(tiny_buffer, resolution=16).contains(pt):
                            is_inside_column = True
                            removed_count += 1
                            print(f"   ⚠️  REMOVED: Point ({adjusted_x:.1f}, {adjusted_y:.1f}) is ON/INSIDE column #{col_idx+1} edge - skipping!")
                            break
                    except Exception as e:
                        # If check fails, assume inside for safety
                        is_inside_column = True
                        removed_count += 1
                        print(f"   ⚠️  REMOVED: Point ({adjusted_x:.1f}, {adjusted_y:.1f}) - check failed, assuming inside column - skipping!")
                        break
                
                if not is_inside_column:
                    filtered_points.append((x, y))
            
            filtered_points_by_room.append({
                **room,
                "points": filtered_points
            })
        
        if removed_count > 0:
            print(f"   ✅ Removed {removed_count} detector point(s) that were inside/on columns (final safety check)")
        else:
            print(f"   ✅ All {total_points} detector points passed final safety check (none inside columns)")
        points_by_room = filtered_points_by_room
    elif columns is None:
        print(f"   ⚠️  WARNING: columns is None in write_output_dxf - cannot perform final safety check!")
        print(f"   ⚠️  This might be why detectors are still overlapping columns!")
    elif len(columns) == 0:
        print(f"   ⚠️  WARNING: columns list is empty in write_output_dxf - cannot perform final safety check!")
        print(f"   ⚠️  This might be why detectors are still overlapping columns!")
        print(f"   ⚠️  No columns were detected - detectors may overlap columns!")

    total_placed = 0
    for room in points_by_room:
        for (x, y) in room["points"]:
            # Apply offset to align with main drawing
            adjusted_x = x + offset_x
            adjusted_y = y + offset_y
            msp.add_blockref(blk_name, insert=(adjusted_x, adjusted_y), dxfattribs={"layer": layer_name})
            if coverage_radius and coverage_radius > 0:
                # Add coverage circle
                msp.add_circle(center=(adjusted_x, adjusted_y), radius=coverage_radius,
                               dxfattribs={"layer": coverage_layer_name})
                
                # Add radius text label if radius in meters is provided
                if coverage_radius_m is not None and coverage_radius_m > 0:
                    # Position text at top-right of circle (45 degrees from center)
                    text_offset = coverage_radius * 1.1  # Slightly outside the circle
                    text_x = adjusted_x + text_offset * 0.707  # cos(45°)
                    text_y = adjusted_y + text_offset * 0.707  # sin(45°)
                    
                    # Format radius text: "R: X.XX m"
                    radius_text = f"R: {coverage_radius_m:.2f} m"
                    msp.add_text(
                        radius_text,
                        dxfattribs={
                            "layer": coverage_text_layer_name,
                            "height": text_height,
                            "color": 3  # Green to match circle
                        }
                    ).set_placement((text_x, text_y))
            total_placed += 1

    doc.saveas(out_path.as_posix())
    
    return total_placed


def write_csv(csv_path: Path, points_by_room: List[Dict]):
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["room_index", "room_name", "x", "y"])
        for room in points_by_room:
            idx = room["index"]
            name = room.get("name", "")
            for (x, y) in room["points"]:
                w.writerow([idx, name, f"{x:.3f}", f"{y:.3f}"])


def generate_pdf_preview_simple(pdf_path: Path, dxf_with_detectors_path: Path):
    """Generate a simple PDF preview (smoke detectors only for large files)."""
    
    print("💡 Note: For best quality, open the DXF file directly in AutoCAD/DWG viewer")
    print("   and print to PDF. This gives you full quality with all details.")
    print("")
    print(f"   The DXF file with detectors: {dxf_with_detectors_path}")
    
    # Skip PDF generation for now as it's too slow for large files
    # User should use their CAD software to export to PDF
    return


def _bbox_fallback(msp):
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    def upd(x, y):
        nonlocal minx, miny, maxx, maxy
        minx = min(minx, x); miny = min(miny, y)
        maxx = max(maxx, x); maxy = max(maxy, y)
    try:
        for e in msp:
            dxftype = e.dxftype()
            if dxftype in ("LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "SPLINE", "ELLIPSE", "POINT"):
                try:
                    box = e.bbox()
                    if box is None:
                        continue
                    (x1, y1, _), (x2, y2, _) = box.extmin, box.extmax
                    upd(x1, y1); upd(x2, y2)
                except Exception:
                    if dxftype in ("LWPOLYLINE", "POLYLINE"):
                        try:
                            if dxftype == "LWPOLYLINE":
                                pts = [(v[0], v[1]) for v in e.get_points()]
                            else:
                                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                            for x, y in pts:
                                upd(x, y)
                        except Exception:
                            pass
    except Exception:
        pass
    if minx == float("inf"):
        return 0.0, 0.0, 0.0, 0.0
    return minx, miny, maxx, maxy


def inspect_dxf(doc):
    msp = doc.modelspace()
    from collections import defaultdict
    counts = defaultdict(int)
    areas = defaultdict(float)

    for e in msp.query("LWPOLYLINE"):
        try:
            if e.closed:
                pts = [(v[0], v[1]) for v in e.get_points()]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    counts[e.dxf.layer] += 1
                    areas[e.dxf.layer] += poly.area
        except Exception:
            pass

    for e in msp.query("POLYLINE"):
        try:
            if e.is_closed:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    counts[e.dxf.layer] += 1
                    areas[e.dxf.layer] += poly.area
        except Exception:
            pass

    try:
        box = msp.bbox()
        w = box.size.x
        h = box.size.y
        max_dim = max(w, h)
    except Exception:
        minx, miny, maxx, maxy = _bbox_fallback(msp)
        w = maxx - minx
        h = maxy - miny
        max_dim = max(w, h)

    if max_dim > 2000:
        units_guess = "mm"
    elif max_dim > 50:
        units_guess = "m"
    else:
        units_guess = "m"

    print("=== INSPECT REPORT ===")
    print(f"Approx drawing size (units): W={w:.2f}, H={h:.2f}, max={max_dim:.2f}")
    print(f"Units guess: {units_guess}")
    print(f"Closed polygons by layer:")
    for layer, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {layer}: {cnt} closed polys, total area={areas[layer]:.2f}")
    print("======================")


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    
    # Auto-generate output paths if not specified
    if args.out_path is None:
        out_path = in_path.parent / f"{in_path.stem}_with_detectors.dxf"
    else:
        out_path = Path(args.out_path)
    
    if args.pdf_path is None and not args.no_pdf:
        pdf_path = in_path.parent / f"{in_path.stem}_preview.pdf"
    else:
        pdf_path = Path(args.pdf_path) if args.pdf_path else None
    
    csv_path = Path(args.csv_path) if args.csv_path else None

    if not in_path.exists():
        print(f"❌ Input DXF not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("🔥 Smoke Detector Auto-Placer")
    print("=" * 60)
    print(f"📁 Reading: {in_path}")
    
    try:
        doc = ezdxf.readfile(in_path.as_posix())
    except Exception as e:
        print(f"❌ Failed to read DXF: {e}", file=sys.stderr)
        sys.exit(1)

    if args.inspect:
        try:
            inspect_dxf(doc)
        except Exception as e:
            print(f"❌ Inspect failed: {e}", file=sys.stderr)
        sys.exit(0)

    # Auto-detect units if not specified
    if args.units is None:
        detected_units = auto_detect_units(doc)
        print(f"🔍 Auto-detected units: {detected_units}")
        units = detected_units
    else:
        units = args.units
        print(f"📏 Using specified units: {units}")
    
    # Auto-detect room layers if not specified
    if args.room_layers is None:
        detected_layers = auto_detect_room_layers(doc)
        print(f"🔍 Auto-detected room layers: {', '.join(detected_layers)}")
        room_layers = detected_layers
    else:
        room_layers = [s.strip().upper() for s in args.room_layers.split(",") if s.strip()]
        print(f"📋 Using specified layers: {', '.join(room_layers)}")
    
    spacing_m, spacing_note = compute_spacing(args.standard, args.spacing)
    print(f"📐 {spacing_note}")
    print(f"⚙️  Standard: {args.standard}, Grid: {args.grid}")

    scale = unit_scale(units)
    spacing_units = spacing_m * scale
    margin_units = args.margin * scale

    # USER REQUEST: "วงกลมรัศมีจะล้อคที่ 6.4เมตร"
    # Lock coverage radius at 6.4 meters
    coverage_radius_m = 6.4
    if args.coverage_radius and args.coverage_radius > 0:
        # Allow override if explicitly specified, but default to 6.4m
        coverage_radius_m = args.coverage_radius
    print(f"📐 Coverage radius locked at: {coverage_radius_m:.2f} m (user request)")

    coverage_radius_units_effective = coverage_radius_m * scale if coverage_radius_m else None
    coverage_radius_units_draw = coverage_radius_units_effective if args.coverage_circles else None

    if args.coverage_circles:
        if args.coverage_radius and args.coverage_radius > 0:
            print(f"🟢 Coverage circles enabled (radius {coverage_radius_m:.2f} m)")
        else:
            print(f"🟢 Coverage circles enabled (default radius {coverage_radius_m:.2f} m)")
    elif args.coverage_radius:
        print("⚠️  Coverage radius specified but coverage circles disabled; using it for coverage checks only.")

    rooms = dxf_to_room_polygons(doc, room_layers)
    if not rooms:
        print("❌ No closed room polygons found on specified layers.", file=sys.stderr)
        print("💡 Try running with --inspect to see available layers.")
        sys.exit(1)

    names_map: Dict[int, str] = {}
    if args.room_name_layer:
        names_map = map_room_names(doc, [p for _, p in rooms], args.room_name_layer.upper())

    print(f"🏠 Found {len(rooms)} rooms")
    
    # Create building bounds for filtering
    building_bounds = create_building_bounds(doc)
    if building_bounds:
        print(f"🏢 Created building bounds for filtering")
    
    # Extract columns/pillars to avoid placing detectors on them
    # USER REQUEST: "ถ้าเจอเสาแบบนี้ไม่ว่าจะขนาดไหน ให้วาดวงรอบตัวเสา และห้ามปักจุดในวงนั้น"
    # So we detect ALL rectangular shapes regardless of size (within reasonable limits)
    print(f"🔍 Detecting columns/pillars...")
    print(f"   ⚠️  CRITICAL: Detecting ALL rectangular shapes (any size) that might be columns")
    print(f"   ⚠️  User: 'ถ้าเจอเสาแบบนี้ไม่ว่าจะขนาดไหน' - detecting columns of any size")
    columns = extract_columns(doc, room_layers, min_area_m2=0.01, max_area_m2=100.0)
    if columns and len(columns) > 0:
        print(f"🏛️  Found {len(columns)} columns/pillars to avoid")
        print(f"   ⚠️  CRITICAL: These columns will be used to filter detector placement")
        # Debug: show first few column areas and dimensions
        if len(columns) <= 10:
            for i, col in enumerate(columns):
                area_m2 = col.area / (scale ** 2)
                bounds = col.bounds
                width_m = (bounds[2] - bounds[0]) / scale
                height_m = (bounds[3] - bounds[1]) / scale
                center_x = (bounds[0] + bounds[2]) / 2
                center_y = (bounds[1] + bounds[3]) / 2
                print(f"   Column {i+1}: area={area_m2:.2f} m², size={width_m:.2f}×{height_m:.2f} m, center=({center_x:.1f}, {center_y:.1f})")
        elif len(columns) > 10:
            # Show summary for many columns
            total_area = sum(col.area for col in columns) / (scale ** 2)
            avg_area = total_area / len(columns)
            print(f"   (showing summary: {len(columns)} columns, avg area={avg_area:.2f} m²)")
            # Show first 5 for reference
            print(f"   First 5 columns:")
            for i, col in enumerate(columns[:5]):
                area_m2 = col.area / (scale ** 2)
                bounds = col.bounds
                center_x = (bounds[0] + bounds[2]) / 2
                center_y = (bounds[1] + bounds[3]) / 2
                print(f"      Column {i+1}: area={area_m2:.2f} m², center=({center_x:.1f}, {center_y:.1f})")
    else:
        print(f"⚠️  No columns detected - detectors may overlap columns")
        print(f"💡 Try running with --inspect to see available layers and entities")
        print(f"💡 Columns are detected as small rectangular shapes (0.05-20 m²) that are not room layers")
        print(f"💡 If you see square/rectangular shapes in your drawing, they might be:")
        print(f"      - On room layers (excluded from detection)")
        print(f"      - Too large or too small (outside 0.05-20 m² range)")
        print(f"      - Not closed polygons (must be LWPOLYLINE/POLYLINE with closed=True)")
    
    # Auto-detect or use specified offset
    if args.offset_x is not None and args.offset_y is not None:
        offset_x = args.offset_x
        offset_y = args.offset_y
        print(f"📍 Using specified offset: ({offset_x:.0f}, {offset_y:.0f})")
    else:
        offset_x, offset_y = auto_detect_offset(doc, rooms, room_layers)
    
    points_by_room: List[Dict] = []
    buffer_distance_for_drawing = 0.0  # Initialize buffer for drawing
    print(f"📊 Processing {len(rooms)} rooms with column filtering...")
    if columns and len(columns) > 0:
        print(f"   ✅ Using {len(columns)} detected columns for filtering")
        print(f"   ⚠️  CRITICAL: All detector points will be checked against these columns")
        print(f"   ⚠️  User instruction: 'ให้วาดวงกลมรัศมี 6.4เมตร แล้วห้ามปัก ในเขตนั้น' - using 6.4m radius exclusion zone")
    else:
        print(f"   ⚠️  WARNING: No columns detected - detectors may overlap columns!")
        print(f"   ⚠️  This might be why detectors are still overlapping columns!")
    for i, (layer, poly) in enumerate(rooms):
        area_m2 = poly.area / (scale ** 2)
        if area_m2 < args.min_area:
            continue
        # Pass columns to place_detectors_in_room - MUST pass columns list!
        if columns is None:
            print(f"   ⚠️  ERROR: columns is None for room {i+1}! This will cause detectors to overlap columns!")
        elif len(columns) == 0:
            print(f"   ⚠️  WARNING: columns list is empty for room {i+1}! Detectors may overlap columns!")
        else:
            print(f"   Room {i+1}: Checking detector points against {len(columns)} columns...")
        pts, room_buffer_distance = place_detectors_in_room(poly, spacing_units, margin_units, args.grid, building_bounds, columns)
        if room_buffer_distance > buffer_distance_for_drawing:
            buffer_distance_for_drawing = room_buffer_distance
        
        # USER REQUEST: "ไม่จำเป็นต้องถี่ ถ้าดูแล้ววงมันซ้อนเยอะเกินจำเป็นก็ไม่ต้องปัก แต่ก็ยังต้องทำให้ครอบคลุม"
        # Remove detectors with excessive overlap (overlap > 50% of coverage circle)
        if len(pts) > 1 and coverage_radius_units_effective and coverage_radius_units_effective > 0:
            original_count = len(pts)
            pts = remove_excessive_overlaps(pts, coverage_radius_units_effective, max_overlap_ratio=0.5)
            if len(pts) < original_count:
                print(f"   🗑️  Room {i+1}: Removed {original_count - len(pts)} detector(s) with excessive overlap (kept {len(pts)})")
        
        points_by_room.append({
            "index": i,
            "layer": layer,
            "name": names_map.get(i, ""),
            "points": pts,
        })

    extra_points = []
    if coverage_radius_units_effective and coverage_radius_units_effective > 0 and building_bounds:
        extra_points = fill_coverage_gaps(points_by_room, coverage_radius_units_effective, building_bounds)
        if extra_points:
            print(f"➕ Added {len(extra_points)} additional detector(s) to cover uncovered zones")
            points_by_room.append({
                "index": len(points_by_room),
                "layer": "COVERAGE_FILL",
                "name": "Coverage Gap",
                "points": extra_points,
            })
    
    total = sum(len(r["points"]) for r in points_by_room)
    rooms_processed = sum(1 for r in points_by_room if r.get("layer") != "COVERAGE_FILL")
    
    # Calculate buffer distance for column filtering
    # USER REQUEST: "ให้วาดวงกลมรัศมี 6.4เมตร แล้วห้ามปัก ในเขตนั้น"
    # Use EXACTLY 6.4 meters radius for column exclusion zones
    buffer_radius_m = 6.4  # EXACTLY 6.4 meters radius as requested
    if spacing_units > 100:
        buffer_distance_for_columns = buffer_radius_m * 1000  # 6400mm
    else:
        buffer_distance_for_columns = buffer_radius_m  # 6.4m
    print(f"📐 Column exclusion zone: {buffer_radius_m:.2f} m radius circles (user: 'ให้วาดวงกลมรัศมี 6.4เมตร แล้วห้ามปัก ในเขตนั้น')")
    
    # Write output DXF with offset
    print(f"💾 Saving DXF with detectors: {out_path}")
    total_placed = write_output_dxf(doc, out_path, points_by_room, offset_x, offset_y, 
                                     coverage_radius_units_draw, coverage_radius_m if args.coverage_circles else None,
                                     columns, buffer_distance_for_drawing)
    
    if total_placed != total:
        print(f"⚠️  Warning: Expected {total} but placed {total_placed} detectors")
    
    # Write CSV if requested
    if csv_path:
        print(f"📊 Saving CSV: {csv_path}")
        write_csv(csv_path, points_by_room)
    
    # Note about PDF generation
    if pdf_path and not args.no_pdf:
        print("")
        print("=" * 60)
        generate_pdf_preview_simple(pdf_path, out_path)
        print("=" * 60)

    print("=" * 60)
    print("✅ COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   • Rooms found: {len(rooms)}")
    print(f"   • Rooms processed (≥ {args.min_area} m²): {rooms_processed}")
    print(f"   • Total detectors placed: {total}")
    if rooms_processed:
        print(f"   • Average detectors per room: {total/rooms_processed:.1f}")
    if coverage_radius_units_effective:
        if args.coverage_circles:
            print(f"   • Coverage circles drawn: {coverage_radius_m:.2f} m radius")
        else:
            print(f"   • Coverage radius for validation: {coverage_radius_m:.2f} m")
    if extra_points:
        print(f"   • Coverage gap fills added: {len(extra_points)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
