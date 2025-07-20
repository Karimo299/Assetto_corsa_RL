import numpy as np
import math
import os
import csv

# Try to import shapely for polygon operations
try:
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import unary_union, polygonize_full
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def load_track_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(script_dir, 'csv/austria.csv')

    left_barrier = []
    right_barrier = []

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            right_barrier.append((float(row['x_right']), float(row['z_right'])))
            left_barrier.append((float(row['x_left']), float(row['z_left'])))

    # Convert to NumPy arrays
    left_barrier = np.array(left_barrier)
    right_barrier = np.array(right_barrier)

    # Reverse the left barrier to go around the track properly
    left_barrier_reversed = left_barrier[::-1]

    # Combine right barrier (forward) + left barrier (reversed) to form one loop
    track_loop = np.vstack((right_barrier, left_barrier_reversed))

    # Close the loop by adding the first point at the end
    track_loop = np.vstack((track_loop, track_loop[0]))

    return track_loop, left_barrier, right_barrier


def create_continuous_track_polygon(all_barriers):
    if not SHAPELY_AVAILABLE:
        return None

    # Ensure the polygon is closed by adding the first point at the end
    if not np.array_equal(all_barriers[0], all_barriers[-1]):
        all_barriers = np.vstack((all_barriers, all_barriers[0]))

    print("Creating continuous track polygon with points:", all_barriers)
    # Create the polygon
    track_polygon = Polygon(all_barriers)
    return track_polygon


def is_car_off_track(x, z, left_polygon, right_polygon):
    if not SHAPELY_AVAILABLE:
        return False
    point = Point(x, z)
    return not left_polygon.contains(point) or right_polygon.intersects(point)


def calculate_ray_endpoint(car_pos, heading, angle, left_polygon, right_polygon):
    """
    Calculate the end point of a ray at a given angle relative to the car's heading.
    The ray has a fixed length of 100 units, but will be shortened if it intersects with the track border.
    """
    if not SHAPELY_AVAILABLE:
        return car_pos[0] + 100 * math.cos(heading + math.radians(angle)), car_pos[1] + 100 * math.sin(heading + math.radians(angle))

    # Calculate the direction of the ray based on the car's heading and the given angle
    ray_direction = heading + math.radians(angle)  # Convert angle to radians and add to heading
    ray_length = 10000  # Fixed length of 10000 units

    # Calculate the end point of the ray
    end_x = car_pos[0] + ray_length * math.cos(ray_direction)
    end_z = car_pos[1] + ray_length * math.sin(ray_direction)

    # Create a LineString for the ray
    ray_line = LineString([(car_pos[0], car_pos[1]), (end_x, end_z)])

    # Check for intersection with the track polygon
    intersection_left = ray_line.intersection(left_polygon.boundary)
    intersection_right = ray_line.intersection(right_polygon.boundary)

    if intersection_left.is_empty:
        intersection = intersection_right
    elif intersection_right.is_empty:
        intersection = intersection_left
    else:
        # Compare distances and choose the closer intersection
        point_left = min(intersection_left.geoms, key=lambda p: p.distance(Point(car_pos[0], car_pos[1]))) if intersection_left.geom_type == 'MultiPoint' else intersection_left
        point_right = min(intersection_right.geoms, key=lambda p: p.distance(Point(car_pos[0], car_pos[1]))) if intersection_right.geom_type == 'MultiPoint' else intersection_right
        intersection = point_left if point_left.distance(Point(car_pos[0], car_pos[1])) < point_right.distance(Point(car_pos[0], car_pos[1])) else point_right

    if intersection.is_empty:
        # No intersection, return the original endpoint
        return end_x, end_z
    else:
        # Intersection found, return the intersection point
        if intersection.geom_type == 'Point':
            return intersection.x, intersection.y
        elif intersection.geom_type == 'MultiPoint':
            # If multiple intersection points, return the closest one to the car
            closest_point = min(intersection.geoms, key=lambda p: p.distance(Point(car_pos[0], car_pos[1])))
            return closest_point.x, closest_point.y
        else:
            # Handle other geometry types (e.g., LineString) if necessary
            return end_x, end_z

