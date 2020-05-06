

def hex_to_rgb(hex: str):
    h = hex.lstrip('#')
    assert(len(h) == 6)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

