import pygame
import random
import sys
import math
import neat
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import warnings
import multiprocessing
import pickle

width = 1300
height = 1100
bg = (213, 193, 154, 255)

generation = 0

class Car:

	# Список доступных автомобилей, выбирается случайным образом каждый раз
	car_sprites = ("Audi", "Black_viper", "Orange", "Police", "Taxi")

	def __init__(self):
		self.random_sprite()

		self.angle = 0
		self.speed = 5

		self.radars = []
		self.collision_points = []

		self.is_alive = True
		self.goal = False
		self.distance = 0
		self.time_spent = 0

	def random_sprite(self):
		self.car_sprite = pygame.image.load('sprites/' + random.choice(self.car_sprites) + '.png')
		self.car_sprite = pygame.transform.scale(self.car_sprite,
			(math.floor(self.car_sprite.get_size()[0]/2), math.floor(self.car_sprite.get_size()[1]/2)))
		self.car = self.car_sprite

		# Пересчет
		self.pos = [650, 930]
		self.compute_center()

	def compute_center(self):
		self.center = (self.pos[0] + (self.car.get_size()[0]/2), self.pos[1] + (self.car.get_size()[1] / 2))

	def draw(self, screen):
		screen.blit(self.car, self.pos)
		self.draw_radars(screen)

	def draw_center(self, screen):
		pygame.draw.circle(screen, (0,72,186), (math.floor(self.center[0]), math.floor(self.center[1])), 5)

	def draw_radars(self, screen):
		for r in self.radars:
			p, d = r
			pygame.draw.line(screen, (183,235,70), self.center, p, 1)
			pygame.draw.circle(screen, (183,235,70), p, 5)

# Входные данные (5 радаров -90, -45, 0, 45, 90 градусов от центра машинки)
	def compute_radars(self, degree, road):
		length = 0
		x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
		y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

		while not road.get_at((x, y)) == bg and length < 300:
			length = length + 1
			x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
			y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

		dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
		self.radars.append([(x, y), dist])

# Выходные данные (повороты налево и направо)
	def compute_collision_points(self):
		self.compute_center()
		lw = 65
		lh = 65

		lt = [self.center[0] + math.cos(math.radians(360 - (self.angle + 20))) * lw, self.center[1] + math.sin(math.radians(360 - (self.angle + 20))) * lh]
		rt = [self.center[0] + math.cos(math.radians(360 - (self.angle + 160))) * lw, self.center[1] + math.sin(math.radians(360 - (self.angle + 160))) * lh]
		lb = [self.center[0] + math.cos(math.radians(360 - (self.angle + 200))) * lw, self.center[1] + math.sin(math.radians(360 - (self.angle + 200))) * lh]
		rb = [self.center[0] + math.cos(math.radians(360 - (self.angle + 340))) * lw, self.center[1] + math.sin(math.radians(360 - (self.angle + 340))) * lh]

		self.collision_points = [lt, rt, lb, rb]

	def draw_collision_points(self, road, screen):
		if not self.collision_points:
			self.compute_collision_points()

		for p in self.collision_points:
			if(road.get_at((int(p[0]), int(p[1]))) == bg):
				pygame.draw.circle(screen, (255,0,0), (int(p[0]), int(p[1])), 5)
			else:
				pygame.draw.circle(screen, (15,192,252), (int(p[0]), int(p[1])), 5)

	def check_collision(self, road):
		self.is_alive = True

		for p in self.collision_points:
			try:
				if road.get_at((int(p[0]), int(p[1]))) == bg:
					self.is_alive = False
					break
			except IndexError:
				self.is_alive = False

	def rotate(self, angle):
		orig_rect = self.car_sprite.get_rect()
		rot_image = pygame.transform.rotate(self.car_sprite, angle)
		rot_rect = orig_rect.copy()
		rot_rect.center = rot_image.get_rect().center
		rot_image = rot_image.subsurface(rot_rect).copy()

		self.car = rot_image

	def get_data(self):
		radars = self.radars
		data = [0, 0, 0, 0, 0]

		for i, r in enumerate(radars):
			data[i] = int(r[1] / 50)

		return data

	def get_reward(self):
		return self.distance / 50.0

	def update(self, road):
		# Устанавливаем некоторую фиксированную скорость
		self.speed = 5

		# Поворот
		self.rotate(self.angle)

		# Движение
		self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
		if self.pos[0] < 20:
			self.pos[0] = 20
		elif self.pos[0] > width - 120:
			self.pos[0] = width - 120

		self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
		if self.pos[1] < 20:
			self.pos[1] = 20
		elif self.pos[1] > height - 120:
			self.pos[1] = height - 120

		# Обновляем расстояние и затраченное время
		self.distance += self.speed
		self.time_spent += 1

		# Вычислить/проверить точки столкновения и создать радары
		self.compute_collision_points()
		self.check_collision(road)

		self.radars.clear()
		for d in range(-90, 120, 45):
			self.compute_radars(d, road)

start = False

# !!!!!!!!!!!!Реализация алгоритма NEAT!!!!!!!!!!!!

def run_generation(genomes, config):

	nets = []
	cars = []

	# Инициирование геномов
	for i, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		g.fitness = 0 # каждый геном ничего не умеет в начале

		# Инициируем автомобили
		cars.append(Car())

	# Инициируем игру
	pygame.init()
	screen = pygame.display.set_mode((width, height))
	clock = pygame.time.Clock()
	road = pygame.image.load('sprites/road1.png')

	font = pygame.font.SysFont("Roboto", 40)
	heading_font = pygame.font.SysFont("Roboto", 80)

	# Дорожная петля
	global generation
	global start
	generation += 1

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit(0)
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					start = True

		if not start:
			continue

		# Вводим данные каждой машины
		for i, car in enumerate(cars):
			output = nets[i].activate(car.get_data())
			i = output.index(max(output))

			if i == 0:
				car.angle += 5
			elif i == 1:
				car.angle = car.angle
			elif i == 2:
				car.angle -= 5

		# Теперь нужно обновить машину и установить фитнес (обучение) (только для живых машин)

		cars_left = 0
		p_i = 0
		max_distance = 0
		n_o = 0
		b_c = 0
		for i, car in enumerate(cars):
			if car.is_alive:
				cars_left += 1
				car.update(road)
				genomes[i][1].fitness += car.get_reward() # новый фитнес (также известный как успех экземпляра автомобиля)
				if max_distance < genomes[i][1].fitness:
					max_distance = genomes[i][1].fitness
					n_o = nets[i].activate(car.get_data())
					p_i = car.get_data()
					b_c = i
		perfect_input = p_i
		best_distance = max_distance
		nise_output = n_o
		best_car = b_c
		# print("Удивительно", kery)
		# Проверяем, уехали ли  машины
		if not cars_left:
			break

		# Отображаем материал
		screen.blit(road, (0, 0))

		for car in cars:
			if car.is_alive:
				car.draw(screen)
				car.draw_center(screen)
				car.draw_collision_points(road, screen)

		# Вывод информации на экран
		label = heading_font.render("Поколение: " + str(generation), True, (73, 168, 70))
		label_rect = label.get_rect()
		label_rect.center = (width / 1.5, 300)
		screen.blit(label, label_rect)

		label = font.render("Машин осталось: " + str(cars_left), True, (51, 59, 70))
		label_rect = label.get_rect()
		label_rect.center = (width / 1.5, 375)
		screen.blit(label, label_rect)

		label = font.render("Входные данные: " + str(perfect_input), True, (51, 59, 70))
		label_rect = label.get_rect()
		label_rect.center = (width / 1.5, 450)
		screen.blit(label, label_rect)

		label = font.render("Наилучшая машина с номером " + str(best_car), True, (51, 59, 70))
		label_rect = label.get_rect()
		label_rect.center = (width / 1.5, 525)
		screen.blit(label, label_rect)

		label = font.render("Пройденное расстояние: " + str(round(best_distance)), True, (51, 59, 70))
		label_rect = label.get_rect()
		label_rect.center = (width / 1.5, 600)
		screen.blit(label, label_rect)

		label = font.render("Выходные данные: " + str(nise_output), True, (51, 59, 70))
		label_rect = label.get_rect()
		label_rect.center = (width / 2, 800)
		screen.blit(label, label_rect)
		pygame.display.flip()
		clock.tick(0)

if __name__ == "__main__":
	# Настраиваем конфиг
	config_path = "./config-feedforward.txt"
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

	# Инизиализируем NEAT
	p = neat.Population(config)

	# Для небольшого окошка вывода на консоли
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	# Для вывода рисунка топологии ИНС
	pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), run_generation)

	# Запускаем алгоритм
	winner = p.run(run_generation, 100)

	# Сохраняем победителей
	with open('winner', 'wb') as f:
		pickle.dump(winner, f)

	print("Победители!", winner)

	node_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '0',6: '1',7: '2',}

	def plot_stats(statistics, filename="avg_fitness.png"):
		""" График средней и лучшей физической подготовки популяции """
		generationII = range(len(statistics.most_fit_genomes))
		best_fitness = [c.fitness for c in statistics.most_fit_genomes]
		avg_fitness = np.array(statistics.get_fitness_mean())

		plt.plot(generationII, avg_fitness, 'b-', label="average")
		plt.plot(generationII, best_fitness, 'r-', label="best")

		plt.title("Population's average and best fitness")
		plt.xlabel("Generations (Start to 0 (0 = 1, 1 = 2, ...)")
		plt.ylabel("Fitness")
		plt.grid()
		plt.legend(loc="best")
		plt.savefig(filename)
		plt.show()

		plt.close()


	def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
				 node_colors=None, fmt='svg'):
		""" Получаем геном и рисуем нейронную сеть с произвольной топологией """
		# Attributes for network nodes.
		if graphviz is None:
			warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
			return

		# If requested, use a copy of the genome which omits all components that won't affect the output.
		if prune_unused:
			if show_disabled:
				warnings.warn("show_disabled has no effect when prune_unused is True")

			genome = genome.get_pruned_copy(config.genome_config)

		if node_names is None:
			node_names = {}

		assert type(node_names) is dict

		if node_colors is None:
			node_colors = {}

		assert type(node_colors) is dict

		node_attrs = {
			'shape': 'circle',
			'fontsize': '9',
			'height': '0.2',
			'width': '0.2'}

		dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

		inputs = set()
		for k in config.genome_config.input_keys:
			inputs.add(k)
			name = node_names.get(k, str(k))
			input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
			dot.node(name, _attributes=input_attrs)

		outputs = set()
		for k in config.genome_config.output_keys:
			outputs.add(k)
			name = node_names.get(k, str(k))
			node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

			dot.node(name, _attributes=node_attrs)

		for n in genome.nodes.keys():
			if n in inputs or n in outputs:
				continue

			attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
			dot.node(str(n), _attributes=attrs)

		for cg in genome.connections.values():
			if cg.enabled or show_disabled:
				input, output = cg.key
				a = node_names.get(input, str(input))
				b = node_names.get(output, str(output))
				style = 'solid' if cg.enabled else 'dotted'
				color = 'green' if cg.weight > 0 else 'red'
				width = str(0.1 + abs(cg.weight / 5.0))
				dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

		dot.render(filename, view=view)

		return dot

	draw_net(config, winner, view=True, node_names=node_names,
			 filename="winner-feedforward.gv")
	plot_stats(stats)
