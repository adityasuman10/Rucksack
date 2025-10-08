import pygame
import sys
import math
from queue import PriorityQueue
import random
pygame.init()

# Get screen dimensions
screen_info = pygame.display.Info()
WIDTH, HEIGHT = screen_info.current_w, screen_info.current_h

# Constants
GRID_SIZE = 20  # Size of each grid cell
WHITE = (255, 255, 255)
BLACK = (100, 100, 100)
START_COLOR = (0, 255, 0)
END_COLOR = (255, 0, 0)
PATH_COLOR = (0,255,0)

OPEN_SET_COLOR = (0,0,255)
CLOSED_SET_COLOR = (255,0,0)

HIGHLIGHT_COLOR = (255, 255, 0)
GRID_LINE_COLOR = (200, 200, 200)  # Light gray for grid lines

# Pygame setup
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("A* Algorithm Maze Challenge")
clock = pygame.time.Clock()


class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.color = WHITE
        self.neighbors = []
        self.width = GRID_SIZE
        self.x = col * GRID_SIZE
        self.y = row * GRID_SIZE

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        for dr, dc in directions:
            if 0 <= self.row + dr < len(grid) and 0 <= self.col + dc < len(grid[0]):
                neighbor = grid[self.row + dr][self.col + dc]
                if neighbor.color != BLACK:
                    self.neighbors.append(neighbor)

    def __lt__(self, other):
        return False


def heuristic(node1, node2):
    return math.sqrt((node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2)


def astar(grid, start, end):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic(start, end)

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = open_set.get()[1]
        if current == end:
            reconstruct_path(came_from, end)
            end.color = END_COLOR
            return came_from

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, end)
                if neighbor not in [item[1] for item in open_set.queue]:
                    open_set.put((f_score[neighbor], neighbor))
                    neighbor.color = OPEN_SET_COLOR

        draw(grid, start, end)
        if current != start:
            current.color = CLOSED_SET_COLOR
    return False


def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        current.color = PATH_COLOR


def draw(grid, start, end):
    for row in grid:
        for node in row:
            node.draw()

    # Draw grid lines
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    # Vertical lines
    for x in range(0, cols * GRID_SIZE, GRID_SIZE):
        pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, rows * GRID_SIZE))
    # Horizontal lines
    for y in range(0, rows * GRID_SIZE, GRID_SIZE):
        pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (cols * GRID_SIZE, y))

    # Redraw start and end to cover grid lines
    start.draw()
    end.draw()
    pygame.display.flip()
    clock.tick(30)


def create_grid(rows, cols):
    grid = [[Node(row, col) for col in range(cols)] for row in range(rows)]
    for col in range(cols):
        if col not in [5, cols - 6]:
            grid[0][col].color = BLACK
            grid[rows - 1][col].color = BLACK

    for row in range(rows):
        if row not in [10, rows - 11]:
            grid[row][0].color = BLACK
            grid[row][cols - 1].color = BLACK

    # Inner maze structure
    for row in range(10, rows - 10):
        for col in [10, 20, 30]:
            if col < cols:
                grid[row][col].color = BLACK
    for row in [15, 25]:
        for col in [10, 20, 30]:
            if col < cols:
                grid[row][col].color = WHITE

    for col in range(10, cols - 10):
        for row in [5, 15, 25, rows - 6]:
            if row < rows:
                grid[row][col].color = BLACK
    for col in [12, 22, 28]:
        for row in [5, 15, 25, rows - 6]:
            if row < rows:
                grid[row][col].color = WHITE

    # Diagonal barriers
    for i in range(8, 20):
        if i < rows and cols - i < cols:
            grid[i][cols - i].color = BLACK
        if i + 10 < rows and cols - i - 15 < cols:
            grid[i + 10][cols - i - 15].color = BLACK

    # Random obstacles
    for _ in range(25):
        row = random.randint(5, rows - 5)
        col = random.randint(5, cols - 5)
        if (row, col) not in [(5, 5), (rows - 6, cols - 6)]:  # Updated end point
            grid[row][col].color = BLACK

    return grid


def main():
    rows = HEIGHT // GRID_SIZE
    cols = WIDTH // GRID_SIZE
    grid = create_grid(rows, cols)

    start = grid[5][5]
    start.color = START_COLOR

    # Randomly select end point
    while True:
        end_row = random.randint(0, rows - 1)
        end_col = random.randint(0, cols - 1)
        if grid[end_row][end_col].color != BLACK and (end_row, end_col) != (5, 5):
            end = grid[end_row][end_col]
            end.color = END_COLOR
            break

    for row in grid:
        for node in row:
            node.update_neighbors(grid)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Allow exiting with ESC key
                    running = False

        came_from = astar(grid, start, end)

        if came_from:
            current = end
            while current in came_from:
                current.color = HIGHLIGHT_COLOR
                draw(grid, start, end)
                current = came_from[current]
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()