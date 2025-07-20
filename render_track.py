import pygame
import numpy as np
import csv
import os
import math
import sys
from sim_info import ray_angles, get_car_details
import track_utils  # Import the new utilities module


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

_, left_barrier, right_barrier = track_utils.load_track_data()
print(left_barrier, right_barrier)
left_polygon = track_utils.create_continuous_track_polygon(left_barrier)
right_polygon = track_utils.create_continuous_track_polygon(right_barrier)

in_lap = False



class TrackRenderer:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("AC Track Renderer")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.zoom = 1.0  # Initial zoom level
        self.font = pygame.font.Font(None, 24)  # Initialize font for text rendering

    def draw_track(self, car_pos):
        """Draw the track with proper zoom and offset applied."""

        # Calculate offset with zoom to center car
        zoom_offset_x = WINDOW_WIDTH // 2 - (car_pos[0] * self.zoom)
        zoom_offset_y = WINDOW_HEIGHT // 2 - (car_pos[1] * self.zoom)

        # zoomed_track_points = [(p[0] * self.zoom + zoom_offset_x, p[1] * self.zoom + zoom_offset_y)
        #                        for p in track_polygon.exterior.coords]
        # pygame.draw.polygon(self.screen, (0, 128, 255), zoomed_track_points, 2)

        zoomed_right_points = [(p[0] * self.zoom + zoom_offset_x, p[1] * self.zoom + zoom_offset_y)
                               for p in right_polygon.exterior.coords]
        pygame.draw.polygon(self.screen, (0, 128, 255), zoomed_right_points, 2)

        zoomed_left_points = [(p[0] * self.zoom + zoom_offset_x, p[1] * self.zoom + zoom_offset_y)
                              for p in left_polygon.exterior.coords]
        pygame.draw.polygon(self.screen, (0, 128, 255), zoomed_left_points, 2)

    def draw_car(self):
        """Draw the car at the center of the screen."""
        car_radius = 3
        pygame.draw.circle(self.screen, (255, 0, 0), (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2), car_radius)

    def handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom in/out with mouse wheel
                zoom_factor = 1.1  # Adjust this value for faster/slower zooming
                if event.y > 0:  # Scroll up to zoom in
                    self.zoom *= zoom_factor
                elif event.y < 0:  # Scroll down to zoom out
                    self.zoom /= zoom_factor
                # Limit zoom range to prevent extreme values
                self.zoom = max(0.1, min(10.0, self.zoom))

    def render_text(self, text, position, color=(255, 255, 255)):
        """Helper method to render text on the screen."""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def draw_rays(self, car_pos, heading):
        """
        Draw rays at the specified angles relative to the car's heading.
        """
        for angle in ray_angles:
            # Calculate the end point of the ray
            end_x, end_z = track_utils.calculate_ray_endpoint(car_pos, heading, angle, left_polygon, right_polygon)

            # Draw the ray
            zoom_offset_x = WINDOW_WIDTH // 2 - (car_pos[0] * self.zoom)
            zoom_offset_y = WINDOW_HEIGHT // 2 - (car_pos[1] * self.zoom)
            start_pos = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)  # Car is at the center of the screen
            end_pos = (end_x * self.zoom + zoom_offset_x, end_z * self.zoom + zoom_offset_y)
            pygame.draw.line(self.screen, (255, 255, 0), start_pos, end_pos, 1)

    def render(self):
        global in_lap
        car_pos, heading, speed, gas, brake, steerAngle, normalizedCarPosition, laps = get_car_details()

        norm_pos = normalizedCarPosition
        # Fill screen with black
        self.screen.fill((0, 0, 0))

        # Draw track and car
        self.draw_track(car_pos)
        self.draw_car()
        self.draw_rays(car_pos, heading)
        # Display car details as text
        self.render_text(f"Position: ({car_pos[0]:.1f}, {car_pos[1]:.1f})", (10, 10))
        self.render_text(f"Heading: {heading:.2f} rad", (10, 40))
        self.render_text(f"Speed: {speed:.1f} km/h", (10, 70))
        self.render_text(f"FPS: {self.clock.get_fps():.1f}", (WINDOW_WIDTH - 120, 40))
        self.render_text(f"Gas: {gas:.2f}", (10, 100))
        self.render_text(f"Brake: {brake:.2f}", (10, 130))
        self.render_text(f"Steer Angle: {steerAngle:.2f} rad", (10, 160))
        on_track = not track_utils.is_car_off_track(car_pos[0], car_pos[1], left_polygon, right_polygon)
        color = (0, 255, 0) if on_track else (255, 0, 0)
        self.render_text(f"On Track: {on_track}", (10, 190), color)

        if normalizedCarPosition < 0.1 and normalizedCarPosition >= 0 and not in_lap:
            in_lap = True
            print("Lap started")

        if not in_lap:
            norm_pos = 0
        self.render_text(f"Normalized Pos: {norm_pos:.3f}", (10, 220))
        self.render_text(f"In lap: {in_lap}", (10, 250))

        
        # Draw zoom level information
        self.render_text(f"Zoom: {self.zoom:.1f}x", (WINDOW_WIDTH - 120, 10), (200, 200, 200))

        # Handle events
        self.handle_events()

        # Update display
        pygame.display.flip()
        # Control frame rate
        self.clock.tick(60)


if __name__ == "__main__":
    renderer = TrackRenderer()
    while True:
        renderer.render()
