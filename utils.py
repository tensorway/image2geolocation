import math 

def great_circle_distance_relative(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    φ1 = lat1 * math.pi/180; # φ, λ in radians
    φ2 = lat2 * math.pi/180
    Δφ = (lat2-lat1) * math.pi/180
    Δλ = (lon2-lon1) * math.pi/180

    a = math.sin(Δφ/2) * math.sin(Δφ/2) + \
            math.cos(φ1) * math.cos(φ2) * \
            math.sin(Δλ/2) * math.sin(Δλ/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return c

def great_circle_distance(point1, point2):
    R = 6371; # km
    c = great_circle_distance_relative(point1, point2)
    d = R * c; # in km
    return d
