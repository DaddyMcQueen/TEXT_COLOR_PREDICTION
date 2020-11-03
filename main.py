import numpy as np
import pygame
import pickle
from constants import IN_NODES
from constants import HIDDEN_NODES
from constants import OUT_NODES
from constants import SAMPLES
from constants import WIN_WIDTH
from constants import WIN_HEIGHT
from constants import WHITE
from constants import BLACK
from constants import RED
from constants import GREEN
from constants import BLUE
from NeuralNetwork import NeuralNetwork

class sliders(object):

	def __init__(self):
		pygame.font.init()
		self.font = pygame.font.SysFont("comicsans", 50)
		self.reset_rgb()

	def reset_rgb(self):
		self.r = np.random.randint(256)
		self.g = np.random.randint(256)
		self.b = np.random.randint(256)
		self.choice = 0

	def draw_win(self, win):
		pygame.draw.rect(win, WHITE, [ 0, 0, WIN_WIDTH, WIN_HEIGHT])

		self.bar_x = (WIN_WIDTH - 512) / 2
		self.red_x = int(self.bar_x + self.r * 2)
		self.red_y = int(WIN_HEIGHT - 110)
		self.green_x = int(self.bar_x + self.g * 2)
		self.green_y = int(WIN_HEIGHT - 70)
		self.blue_x = int(self.bar_x + self.b * 2)
		self.blue_y = int(WIN_HEIGHT - 30)
		pygame.draw.rect(win, RED, [self.bar_x, WIN_HEIGHT - 120, 512, 20])
		pygame.draw.rect(win, GREEN, [self.bar_x, WIN_HEIGHT - 80, 512, 20])
		pygame.draw.rect(win, BLUE, [self.bar_x, WIN_HEIGHT - 40, 512, 20])
		pygame.draw.circle(win, BLACK, (self.red_x, self.red_y), 10)
		pygame.draw.circle(win, BLACK, (self.green_x, self.green_y), 10)
		pygame.draw.circle(win, BLACK, (self.blue_x, self.blue_y), 10)

		text = self.font.render(str(self.r), 1, BLACK)
		win.blit(text, (self.bar_x - 70, self.red_y - 15))
		text = self.font.render(str(self.g), 1, BLACK)
		win.blit(text, (self.bar_x - 70, self.green_y - 15))
		text = self.font.render(str(self.b), 1, BLACK)
		win.blit(text, (self.bar_x - 70, self.blue_y - 15))

		bg_color = (self.r, self.g, self.b)
		pygame.draw.rect(win, bg_color, [0, 0, WIN_WIDTH, WIN_HEIGHT - 200])

		text = self.font.render("WHITE TEXT", 1, WHITE)
		win.blit(text, (0 + 1/2 * text.get_width(), WIN_HEIGHT - 580))
		text = self.font.render("BLACK TEXT", 1, BLACK)
		win.blit(text, ( WIN_WIDTH / 2 + 1/2 * text.get_width(), WIN_HEIGHT - 580))

		if self.choice:
			pygame.draw.circle(win, BLACK, (int(WIN_WIDTH / 4), int(WIN_HEIGHT/ 4)), 50)
		else:
			pygame.draw.circle(win, BLACK, (int(WIN_WIDTH * 3 / 4), int(WIN_HEIGHT/ 4)), 50)

		pygame.draw.rect(win, BLACK, [0, 400, WIN_WIDTH, 10])
		pygame.draw.rect(win, BLACK, [WIN_WIDTH / 2, 0, 10, WIN_HEIGHT - 200])

	def update_rgb(self, mx, my):
		valid = True

		mx = (mx - self.bar_x)
		my = int((my - (WIN_HEIGHT - 120)) / 40)

		if (mx > 510) or (mx < 0):
			valid = False
		
		if (my == 0) and valid:
			self.r = int(mx / 2)

		if (my == 1) and valid:
			self.g = int(mx / 2)

		if (my == 2) and valid:
			self.b = int(mx / 2)
		

def main():

	new_network = False
	continue_training = False
	train_network = False
	test_network = True
	
	if new_network:
		nn = NeuralNetwork(IN_NODES, HIDDEN_NODES, OUT_NODES)
		train_network = True
	else:
		pickel_in = open("nn.pickel", "rb")
		nn = pickle.load(pickel_in)
		if continue_training == False:
			train_network = False
		else:
			train_network = True

	if train_network:
		for i in range(SAMPLES):
			rgb_in = np.random.randint(256, size = 3)
			squished_in = rgb_in/255
			nn.train(squished_in)
		pickle_out = open("nn.pickel", "wb")
		pickle.dump(nn, pickle_out)
		pickle_out.close()

	if test_network:
		run = True
		testing = True
		win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
		pygame.draw.rect(win, WHITE, [ 0, 0, WIN_WIDTH, WIN_HEIGHT])
		inputs = sliders()

		while run:
			inputs.draw_win(win)
			pygame.display.update()
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False
					testing = False
					pygame.quit()
			
				if event.type == pygame.MOUSEBUTTONDOWN:
					clicked = True

					while clicked:	
						for event in pygame.event.get():
							if event.type == pygame.MOUSEBUTTONUP:
								clicked = False
						mx, my = pygame.mouse.get_pos()
						inputs.update_rgb(mx, my)

			rgb = [inputs.r / 255 , inputs.g / 255, inputs.b / 255]
			output = nn.test_nn(rgb)
			if output >= 0.5:
				inputs.choice = 0
			else:
				inputs.choice = 1
main()