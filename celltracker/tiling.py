import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from .config import DUP_IOU_SUPPRESS, MIN_AREA
from .geometry import simplify_poly, compute_iou

def tile_image(img, tile_size, overlap):
    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tiles.append((x, y, img[y:y2, x:x2]))
            
    return tiles

def stitch_polygons(polys_per_tile, tile_boxes):
    all_polys = []
    for (x, y), polys in zip(tile_boxes, polys_per_tile):
        for poly in polys:
            try:
                shifted = affinity.translate(poly, xoff=x, yoff=y)
                all_polys.append(shifted)
            except Exception:
                continue
                
    return all_polys

def suppress_duplicates(polys, iou_dup=DUP_IOU_SUPPRESS):
    if not polys:
        return []
    result = []
    
    for p in polys:
        if p is None or p.is_empty:
            continue
        p = simplify_poly(p)
        
        if p.area < MIN_AREA:
            continue
            
        merged = False
        for k, q in enumerate(result):
            iou = compute_iou(p, q)
            
            if iou > iou_dup:
                try:
                    u = q.union(p)
                    if not u.is_empty and isinstance(u, Polygon):
                        result[k] = simplify_poly(u)
                        merged = True
                        break
                except Exception:
                    pass
                    
        if not merged:
            result.append(p)
            
    return result

def merge_overlapping(polys, iou_merge=0.35, max_passes=3):
    if not polys:
        return []
    current = polys[:]
    
    for _ in range(max_passes):
        used = [False] * len(current)
        new_list = []
        changed = False
        for i, p in enumerate(current):
            if used[i] or p is None or p.is_empty:
                continue
            acc = p
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                q = current[j]
                
                if q is None or q.is_empty:
                    continue
                    
                iou = compute_iou(acc, q)
                if iou >= iou_merge:
                    try:
                        u = acc.union(q)
                        if not u.is_empty and isinstance(u, Polygon):
                            acc = simplify_poly(u)
                            used[j] = True
                            changed = True
                    except Exception:
                        pass
                        
            used[i] = True
            new_list.append(acc)
        current = new_list
        
        if not changed:
            break
            
    return current
