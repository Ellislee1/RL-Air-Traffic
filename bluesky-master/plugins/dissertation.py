"""
    Ellis Thompson - Undergraduate Dissertation BSc Swansea University
    Based on the system by Marc Brittian (https://github.com/marcbrittain/bluesky/tree/master)
    May 2020
"""

""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """

# Import the global bluesky objects. Uncomment the ones you need
import random
from time import time

import geopy.distance as geopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from bluesky import navdb, scr, settings, sim, stack, tools, traf
from bluesky.tools import geo
from bluesky.tools.aero import ft

from Diss_Agent.atc_agent import Mult_Agent

EPISODES = 1


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global positions
    global max_ac
    global agent
    global spawn_queue
    global times
    global active_ac
    global total_ac
    global ac_routes
    global update_timer
    global success_counter
    global collision_counter
    global total_sucess
    global total_collision
    global episode_counter
    global no_states
    global previous_observation
    global previous_action
    global observation
    global number_of_actions
    global start
    global intruders
    global best_reward
    global mean
    mean = []

    try:
        positions = np.load('routes/sim3.npy')
    except:
        # Sim 1
        #positions = np.array([[46.3, -20.7, 0, 47, -20.7], [47, -20.7, 180, 46.3, -20.7]])
        # Sim 2
        #positions = np.array([[46.3, -20.7, 0, 47, -20.7], [47, -20.7, 180, 46.3, -20.7], [46.65, -20.201, 270, 46.65, -21.2055]])
        # Sim 3
        positions = np.array([[47.3, -21.4, 128.029, 46.65,-17], [46, -21.4, 51.434, 46.65, -17], [46.65, -21.6, 90, 46.65, -17]])
        np.save("routes/sim3.npy", positions)
        positions = np.load('routes/sim3.npy')
    
    max_ac = 5000
    active_ac = 0
    total_ac = 0
    # 5  states: lat, lon, alt, route, vs
    no_states = 5
    number_of_actions = 3
    intruders = 2
    agent = Mult_Agent(no_states, number_of_actions,number_of_actions, intruders, positions)
    times = [20, 25, 30]
    spawn_queue = random.choices(times, k=positions.shape[0])
    ac_routes = np.zeros(max_ac)
    update_timer = 0
    success_counter = 0
    collision_counter = 0
    total_sucess = []
    total_collision = []
    episode_counter = 0
    observation = {}
    previous_observation = {}
    previous_action = {}
    best_reward = -10e5

    # Start the sim
    # stack.stack('OP')
    # # Fast forward the sim
    # stack.stack('FF')
    # Start episode timer
    start = time()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DISS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 12.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        # 'reset':         reset
        }

    stackfunctions = {
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    global positions
    global max_ac
    global agent
    global spawn_queue
    global times
    global active_ac
    global total_ac
    global ac_routes
    global update_timer
    global success_counter
    global collision_counter
    global previous_observation
    global previous_action
    global observation
    global no_states

    if total_ac < max_ac:
        if total_ac == 0:
            for i in range(len(positions)):

                spawn_ac(total_ac, positions[i])

                ac_routes[total_ac] = i

                total_ac += 1
                active_ac += 1
        else:
            for k in range(len(spawn_queue)):
                if update_timer == spawn_queue[k]:
                    spawn_ac(total_ac, positions[k])
                    ac_routes[total_ac] = k

                    total_ac += 1
                    active_ac += 1

                    spawn_queue[k] = update_timer + random.choices(times, k=1)[0]
                
                if total_ac == max_ac:
                    break
    
    terminal_ac = np.zeros(len(traf.id), dtype=int)
    for i in range(len(traf.id)):
        T_state, T_type = agent.update(traf, i, ac_routes)
        
        call_sig = traf.id[i]

        if T_state == True:
                       
            stack.stack('DEL {}'.format(call_sig))
            active_ac -= 1
            if T_type == 1:
                collision_counter += 1
            if T_type == 2:
                success_counter += 1
            
            terminal_ac[i] = 1

            try:
                agent.store(previous_observation[call_sig], previous_action[call_sig], [np.zeros(previous_observation[call_sig][0].shape), (previous_observation[call_sig][1].shape)], traf, call_sig, ac_routes, T_type)
            except Exception as e:
                print(f'ERROR: {e}')
            
            del previous_observation[call_sig]

    if active_ac == 0 and max_ac == total_ac:
        reset()
        return
    
    if active_ac == 0 and total_ac != max_ac:
        update_timer += 1
        return
    
    if not len(traf.id) == 0:
        next_action = {}
        state = np.zeros((len(traf.id), 5))

        non_T_ids = np.array(traf.id)[terminal_ac != 1]
        
        indexes = np.array([int(x[4:]) for x in traf.id])
        route = ac_routes[indexes]
        state[:,0] = traf.lat
        state[:, 1] = traf.lon
        state[:, 2] = traf.alt
        state[:, 3] = route
        state[:, 4] = traf.vs
        
        normal_state, context = get_normals_states(state, traf, ac_routes, next_action, no_states, terminal_ac, agent, previous_observation, observation)

        if len(context) == 0:
            update_timer += 1
            return


        policy, values = agent.act(normal_state, context)

        # if (episode_counter + 1) % 20 == 0:
        #     print(policy)

        for j in range(len(non_T_ids)):
            id_ = non_T_ids[j]

            if not id_ in previous_observation.keys():
                previous_observation[id_] = [normal_state[j], context[j]]
                
            if not id_ in observation.keys() and id_ in previous_action.keys():
                observation[id_] = [normal_state[j], context[j]]

                agent.store(previous_observation[id_], previous_action[id_], observation[id_], traf, id_, ac_routes)

                previous_observation[id_] = observation[id_]

                del observation[id_]
            
            action = np.random.choice(agent.no_actions, 1, p=policy[j].flatten())[0]
    
            index = traf.id2idx(id_)
            new_alt = agent.alts[action]
            
            stack.stack('ALT {}, {}'.format(id_, new_alt))
            
            next_action[id_] = action
        
        previous_action = next_action
    print('Total Success: {} | Total Conflicts: {}'.format(success_counter, collision_counter))
    update_timer += 1



def spawn_ac(_id, ac_details):
    # Uncomment for Sim 3 
    lat, lon, hdg, glat, glon = ac_details
    # Uncomment for Sim 1 and 2
    # lat, lon, hdg, glat, glon = ac_details
    #speed = 251
    speed = np.random.randint(251,340)
    
    stack.stack('CRE SWAN{}, A320, {}, {}, {}, 28000,{}'.format(_id, lat, lon, hdg, speed))

    # Sim 3 only
    stack.stack('ADDWPT SWAN{} {}, {}'.format(_id, 46.65, -20.201))
    # All Sims
    stack.stack('ADDWPT SWAN{} {}, {}'.format(_id,glat,glon))

def dist_goal(_id):
    global ac_routes
    global positions
    
    _id = traf.id2idx(_id)

    olat = traf.lat[_id]
    olon = traf.lon[_id]
    ilat,ilon = traf.ap.route[_id].wplat[0],traf.ap.route[_id].wplon[0]

    dist = geo.latlondist(olat,olon,ilat,ilon)/geo.nm
    return dist

def get_distance_matrix_ac(_id):
    size = traf.lat.shape[0]
    distances = geo.latlondist_matrix(np.repeat(traf.lat[_id], size), np.repeat(traf[_id].lon, size), np.tile(traf.lat, size), np.tile(traf.lon, size)).reshape(size, size)
    
    return distances






def reset():
    global positions
    global max_ac
    global agent
    global spawn_queue
    global times
    global active_ac
    global total_ac
    global ac_routes
    global update_timer
    global success_counter
    global collision_counter
    global total_sucess
    global total_collision
    global episode_counter
    global no_states
    global previous_observation
    global previous_action
    global observation
    global number_of_actions
    global start
    global intruders
    global best_reward
    global mean

    # if (episode_counter + 1) % 5 == 0:
    #     print("\n\n---------- TRAINING ----------\n\n")
    #     agent.train()
    #     print("\n\n---------- COMPLETE ----------\n\n")

    end = time()

    print(end-start)
    goals_made = success_counter

    total_sucess.append(success_counter)
    total_collision.append(collision_counter)

    mean_success = np.mean(total_sucess)
    mean.append(mean_success)


    success_counter = 0
    collision_counter = 0


    update_timer = 0
    total_ac = 0
    active_ac = 0

    spawn_queue = random.choices([20,25,30],k=positions.shape[0])


    previous_action = {}
    ac_routes = np.zeros(max_ac,dtype=int)
    previous_observation = {}
    observation = {}

    t_success = np.array(total_sucess)
    t_coll = np.array(total_collision)
    # np.save('results/sim3Suc.npy',t_success)
    # np.save('results/sim3Col.npy', t_coll)
    
    final = np.array(list(zip(total_sucess, total_collision, mean)))

    # np.savetxt('results/sim3.csv', final, delimiter=',', fmt='%.5f')

    # if EPISODES > 150:
    #     df = pd.DataFrame(t_success)
    #     if float(df.rolling(150,150).mean().max()) >= best_reward:
    #         agent.save(_type='Sim_3 default',highest=True)
    #         best_reward = float(df.rolling(150,150).mean().max())


    # agent.save(_type='Sim_3 default')


    print("Episode: {} | Reward: {} | Best Reward: {} | Mean Success: {}".format(episode_counter,goals_made,best_reward, mean_success))


    episode_counter += 1

    if episode_counter == EPISODES:
        stack.stack('STOP')
        
        collision_mean = np.mean(total_collision)
        success_mean = np.mean(total_sucess)

        plt.plot(total_sucess, label='Successes', color='green')
        plt.axhline(y=success_mean, color='green', linestyle='dotted', label='Mean Successes')
        plt.plot(total_collision, label='Collisions', color='red')
        plt.axhline(y=collision_mean, color='red', linestyle='dotted', label='Mean Collisions')
        plt.xlabel('Episode')
        plt.ylabel('# Aircraft')
        plt.xlim(0, EPISODES-1)
        plt.ylim(0, max_ac)
        plt.title('Sim 1')
        plt.legend()
        
        plt.show()

    stack.stack('IC multi_agent.scn')

    start = time()
        

def get_normals_states(state, traf, ac_routes, next_action, no_states, terminal_ac, agent, previous_observation, observation):
    number_of_aircraft = traf.lat.shape[0]

    normal_state = np.zeros((len(terminal_ac[terminal_ac != 1]), 5))
    
    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1, 1)
    
    distancematrix = geo.latlondist_matrix(np.repeat(state[:, 0], number_of_aircraft), np.repeat(state[:, 1], number_of_aircraft), np.tile(state[:, 0], number_of_aircraft), np.tile(state[:, 1], number_of_aircraft)).reshape(number_of_aircraft, number_of_aircraft)
    
    sort = np.array(np.argsort(distancematrix, axis=1))
    total_closest_states = []
    routecount = 0
    i = 0
    j = 0
    
    max_agents = 1

    count = 0

    for i in range(distancematrix.shape[0]):
        if terminal_ac[i] == 1:
            continue

        r = int(state[i][4])
        
        normal_state[count,:] = agent.normalize_state(state[i], id_=traf.id[i])
        
        closest_states = []
        count += 1

        routecount = 0
        intruder_count = 0

        for j in range(len(sort[i])):
            index = int(sort[i, j])
            
            if i == index:
                continue

            if terminal_ac[index] == 1:
                continue

            route = int(state[index][4])

            if route == r and routecount == 2:
                continue

            if route == r:
                routecount += 1
            
            if distancematrix[i, index] > 100:
                continue

            max_agents = max(max_agents, j)

            if len(closest_states) == 0:
                closest_states = np.array([traf.lat[index], traf.lon[index], traf.tas[index], traf.alt[index], route, traf.ax[index]])
                
                closest_states = agent.normalize_context(normal_state[count - 1], closest_states, state[i], id_=traf.id[index])
            else:
                adding = np.array([traf.lat[index], traf.lon[index], traf.tas[index], traf.alt[index], route, traf.ax[index]])

                adding = agent.normalize_context(normal_state[count - 1], adding, state[i], id_=traf.id[index])

                closest_states = np.append(closest_states, adding, axis=1)

            intruder_count += 1
            
            if intruder_count == agent.num_intruders:
                break
        
        if len(closest_states) == 0:
            closest_states = np.zeros(5).reshape(1,1, 5)

        if len(total_closest_states) == 0:
            total_closest_states = closest_states
        else:
            total_closest_states = np.append(tf.keras.preprocessing.sequence.pad_sequences(total_closest_states, agent.num_intruders, dtype='float32'), tf.keras.preprocessing.sequence.pad_sequences(closest_states, agent.num_intruders, dtype='float32'), axis=0)
    return normal_state, total_closest_states
