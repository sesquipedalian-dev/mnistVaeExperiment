import pygame 

# constants
WHITE = (255, 255, 255)
TRANS = (1, 1, 1)


class Slider():
    def __init__(self, i, maxi, mini, x, y, w, h):
        self.maxi = maxi
        self.mini = mini
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.surf = pygame.surface.Surface((w, h))
        self.hit = False
        self.i = i
        self.font = pygame.font.SysFont("Helvetica", 16)

    def true_i(self):
        return self.i

    def draw(self, settings, screen):
        name = "PCA #" + str(self.i)

        txt_surf = self.font.render(name, 1, WHITE)
        txt_rect = txt_surf.get_rect(center=(self.w / 2, 13))

        s = 70
        if self.i % 2 + (self.i // 2) % 2 == 1:
            s = 100
        self.surf.fill((s, s, s))
        pygame.draw.rect(self.surf, (220, 220, 220), [10, 30, self.w - 20, 5], 0)
        for g in range(7):
            pygame.draw.rect(self.surf, (s + 50, s + 50, s + 50), [9 + (self.w - 20) / 6 * g, 40, 2, 5], 0)

        self.surf.blit(txt_surf, txt_rect)

        button_surf = pygame.surface.Surface((10, 20))
        button_surf.fill(TRANS)
        button_surf.set_colorkey(TRANS)
        pygame.draw.rect(button_surf, WHITE, [0, 0, 10, 20])

        surf = self.surf.copy()

        v = min(max(settings[self.i], -9999), 9999)
        pos = (10 + int((v - self.mini) / (self.maxi - self.mini) * (self.w - 20)), 33)
        button_rect = button_surf.get_rect(center=pos)
        surf.blit(button_surf, button_rect)
        button_rect.move_ip(self.x, self.y)

        screen.blit(surf, (self.x, self.y))

    def move(self, settings, state):
        settings[self.i] = (pygame.mouse.get_pos()[0] - self.x - 10) / 130 * (self.maxi - self.mini) + self.mini
        if settings[self.i] < self.mini:
            settings[self.i] = self.mini
        if settings[self.i] > self.maxi:
            settings[self.i] = self.maxi
        state.shouldCalculateImage = True

