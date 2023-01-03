import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import pyro
import pyro.distributions as dist
import torch

import sys; sys.path.append('..')
from pyro_webppl import viz

# imports for physics example:
import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse

import pymunk
import pymunk.pyglet_util
from pymunk import Vec2d

from copy import deepcopy
from pyro.infer import Importance, EmpiricalMarginal


class Block:
    def __init__(self, width=None, height=None, x=0, y=0, is_first=False, is_static=False):
        self.w = width
        self.h = height
        self.x = x
        self.y = y
        self.is_first = is_first
        self.is_static = is_static
        
        if not self.w:
            self.w = np.random.randint(50, 100)
        if not self.h:
            self.h = np.random.randint(50, 100)

class Simulate:
    def __init__(self, world=None, num_blocks=5, dims={}):
        self.world = world
        self.num_blocks = num_blocks
        self.dims = dims

        vals = torch.tensor([1.0, 5.0, 10.0], dtype=torch.float)
        log_probs = torch.log(torch.tensor([0.2, 0.6, 0.2], dtype=torch.float))
        noise_width = pyro.sample("noise_width", dist.Empirical(samples=vals, log_weights=log_probs))
        self.noise_width = float(noise_width.item())

        # create the simulation world
        self.create_world()
        
    def create_world(self):
        # Create the simulation world

        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0.0, -200.0)
        self.space.sleep_time_threshold = 1

        # add the floor - other walls/bounds could be added here
        static_lines = [
            pymunk.Segment(self.space.static_body, Vec2d(-self.dims["WORLD_WIDTH"], self.dims["FLOOR_HEIGHT"]), Vec2d(self.dims["WORLD_WIDTH"]*2, self.dims["FLOOR_HEIGHT"]), 1),
        ]

        for l in static_lines:
            l.friction = 0.3
        self.space.add(*static_lines)

        # create the blocks
        if not self.world:
            prev_block = self.add_block(None)
            for idx in range(1, self.num_blocks):
                next_block = self.add_block(idx, prev_block=prev_block)
                prev_block = next_block    

        else:
            for idx, item in enumerate(self.world):
                self.add_block(idx, new_block=item)

        return 1

    def add_block(self, idx, prev_block=None, new_block=None):
        # function for creating and positioning blocks

        if new_block == None:
            block = Block()
            if prev_block == None:
                block.is_first = True
                block.x = int(self.dims["WORLD_WIDTH"]/2)
                block.y = self.dims["FLOOR_HEIGHT"] + block.h/2

            else:
                block.x = np.random.randint(prev_block.x - prev_block.w/2, prev_block.x + prev_block.w/2)
                block.y = prev_block.y + prev_block.h/2 + block.h/2

        else:
            block = deepcopy(new_block)
            vals = torch.tensor(list(range(int(np.floor(block.x-self.noise_width/2)), int(np.ceil(block.x+self.noise_width/2))+1)), dtype=torch.float)
            log_probs = torch.log(torch.ones(vals.size(), dtype=torch.float))
            Block_x = pyro.sample("Block_{}_x".format(idx), dist.Empirical(samples=vals, log_weights=log_probs))
            block.x = int(Block_x.item())


        # pymunk box coords are at Centre of Mass !
        mass = block.w*block.h/100
        moment = pymunk.moment_for_box(mass, (block.w, block.h))
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(block.x, block.y)
        shape = pymunk.Poly.create_box(body, (block.w, block.h))
        shape.friction = 0.3
        shape.color = (np.random.randint(50,150), np.random.randint(50,150), np.random.randint(100,255), 255)
        self.space.add(body, shape)
        return block
        
    def update(self, dt):
        # Here we use a very basic way to keep a set space.step dt.
        # For a real game its probably best to do something more complicated.
        step_dt = 1 / 250.0
        x = 0
        while x < dt:
            x += step_dt
            self.space.step(step_dt)

class Animate(pyglet.window.Window):
    # main class for animation

    def __init__(self, world=None, num_blocks=5, dims={}):

        self.world=world
        self.num_blocks=num_blocks
        self.dims=dims
        
        #set the window size
        self.window = pyglet.window.Window.__init__(self, width=self.dims["WORLD_WIDTH"], height= self.dims["WORLD_HEIGHT"], vsync=False)
        self.set_caption("2D Boxes")
        
        # sets how frequently self.update gets called once sim starts
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)
        # display FPS
        self.fps_display = pyglet.window.FPSDisplay(self)

        # create the simulation world
        self.simulation = Simulate(self.world, self.num_blocks, self.dims)
        self.simulation.create_world()

        # add draw options
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES

        # add some text
        self.label = pyglet.text.Label('Hit R to reset', font_name='Times New Roman', font_size=16, x=10, y=self.dims["WORLD_HEIGHT"]-25)

    def update(self, dt):
        self.simulation.update(dt)

    def on_key_press(self, symbol, modifiers):
        
        if symbol == key.R:
            self.clear()
            self.simulation = Simulate(self.world, self.num_blocks, dims=self.dims)
            self.simulation.create_world()

        elif symbol == key.ESCAPE:
            self.close()
            pyglet.app.exit()
            
        elif symbol == pyglet.window.key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                "box2d_vertical_stack.png"
            )

    def on_draw(self):
        self.clear()
        # self.fps_display.draw()
        self.simulation.space.debug_draw(self.draw_options)
        self.label.draw()

def does_tower_fall(initial_w, final_w):
    """
    Compare the final state of a tower to its initial state and determine whether or not it stayed standing. 
    """
    # here we compare the intial and final world states and see whether the highest block is roughly in the same place
    def highest_y(w):
        return max([b.y for b in w])

    def approx_equal(a, b):
        return (abs(a - b) < 1)

    return not approx_equal(highest_y(initial_w), highest_y(final_w))


def run_simulation(world, dims):
    """
    Run one iteration of a physics simulation for a given world. Returns the outcome of whether or not the tower was stable.
    """
    initial_world = deepcopy(world)
    simulation = Simulate(initial_world, dims=dims)
    # probs want to make this a while loop that stops when everything stops moving
    for i in range(1000):
        simulation.update(1/60.0)

    final_world = []
    for body in simulation.space._bodies:
        final_world.append(Block(x=body.position.x, y=body.position.y))

    S = not does_tower_fall(initial_world, final_world)
    return S

def infer_physics(model, conditions: dict, sites: list, num_samples: int):
    """
    Function to perform importance sampling on our physics model. Returns EmpiricalMarginal distribution.
    """
    conditioned_model = pyro.condition(model, conditions)
    importance = Importance(conditioned_model, guide=None, num_samples=num_samples)
    trace_posterior_conditioned = importance.run()
    emp_marginal = EmpiricalMarginal(trace_posterior=trace_posterior_conditioned, sites=sites)
    return emp_marginal

def draw_world(world: list, dims: dict, figsize=(4,6)):
    fig, ax = plt.subplots(1, figsize=figsize)
    for block in world:
        # Rectangle((bottomleftx, bottomlefty), width, height)
        ax.add_patch(Rectangle((block.x-block.w/2, block.y-block.h/2), block.w, block.h, linewidth=1, edgecolor='k'))
        
    ax.set_xlim(0, dims["WORLD_WIDTH"])
    ax.set_ylim(0, dims["WORLD_HEIGHT"])
    ax.axhline(dims["FLOOR_HEIGHT"], color='k')
    plt.show()