import pvm
import matplotlib
import matplotlib.pyplot as plt
import stock_dataset as ds
import numpy as np
import torch
import portfolio_env_pvm as portfolio_env
import os
import pickle
import risk_metrics as rm
import pandas as pd


def train_model(model, optimizer, env, env_eq, criterion, epsilon=0.05, num_epoch=20,
                num_batches=16, batch_size=8, beta=0.01, device='cpu'):
    
    train_losses = []
    param_history = []
    
    model = model.to(device)
    
    for epoch in range(num_epoch):
        # Need to reset memory each time, and so dataset as well
        memory = pvm.PVM(num_steps=env.data.shape[2],
                         num_stocks=env.data.shape[1], beta=beta, device=device)
        data = ds.StockDataset(memory, env, env_eq, split='train', device=device)
        X_l, w_prev_l, p_val_l = [], [], []
        daily_ret_l, p_val_eq_l = [], []

        for i in range(num_batches):
            optimizer.zero_grad()
            t_start = memory.draw(batch_size)
            total_risk = np.zeros(3)
            loss = 0

            # Build lists of values for batches
            X, w_prev, p_val_prev, daily_ret, p_val_prev_eq = data[t_start]
            for j in range(batch_size):
                action = model(X.view(1,3,16,50).transpose(1,3), w_prev)
                if np.random.rand() < epsilon:
                    action = torch.rand(action.shape, requires_grad=True)
                    action = action / torch.sum(action)
                if loss == 0:
                    loss = criterion(action, w_prev, p_val_prev, daily_ret,
                                     p_val_prev_eq, data.get_index())
                else:
                    loss += criterion(action, w_prev, p_val_prev, daily_ret, p_val_prev_eq,
                                      data.get_index())
                memory.update_weight(t_start+j, action.detach())
                X, w_prev, p_val_prev, daily_ret, p_val_prev_eq, reward, done = data.step(action)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            param_history.append([p.detach() for p in model.parameters()])
    return model, train_losses, param_history

def validate_model(model, alpha=0.01):
    env_eval = portfolio_env.PortfolioEnv(split='valtest')
    X, w_prev, p_val_prev = env_eval.reset()
    daily_ret = X[-1]
    X = X[:-1]
    
    weights_eval = [w_prev.detach().tolist()]
    p_vals_eval = [p_val_prev.item()]
    rewards_eval = [0]
    risk_history = []

    env_eval_eq = portfolio_env.PortfolioEnv(alpha=alpha, split='valtest', eq=True)
    state_eq = env_eval_eq.reset()

    p_vals_eval_eq = [state_eq[2].item()]
    rewards_eval_eq = [0]

    done = False

    while not done:
        with torch.no_grad():
            action = model(X.view(1,3,16,50).transpose(1,3), w_prev)
            state, reward, done = env_eval.step(action)
            weights_eval.append(state[1].tolist())
            p_vals_eval.append(state[2])
            rewards_eval.append(reward)
            risk_history.append(rm.get_all_risks(action, env_eval.idx, reward))

            state_eq, reward_eq, _ = env_eval_eq.step()
            p_vals_eval_eq.append(state_eq[2])
            rewards_eval_eq.append(reward_eq)
    weights_eval = np.array(weights_eval)
    risk_history = np.array(risk_history)
    
    stocks = ['Cash', 'AAPL', 'XOM' ,'VMC', 'BA', 'AMZN', 'TGT', 'WMT', 'KO', 'UNH', 'JPM', 'GOOGL',
          'STT', 'MSFT', 'VZ', 'XEL', 'SPG']
    data = {}
    for i, s in enumerate(stocks):
        data[s] = weights_eval[:,i]
    weights_eval = pd.DataFrame(data)
    
    return weights_eval, p_vals_eval, rewards_eval, risk_history, p_vals_eval_eq, rewards_eval_eq

def plot_risks(risks):
    
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12) 
    fig, ax = plt.subplots(2,2,figsize=(10,7))

    ax[0,0].plot(risks[:,0])
    ax[0,0].set_title('Beta', size=13)
    ax[0,0].set_xlabel('Time', size=12)
    ax[0,0].set_ylabel('Value', size=12)

    ax[0,1].plot(risks[:,1])
    ax[0,1].set_title('StDev of returns', size=13)
    ax[0,1].set_xlabel('Time', size=12)
    ax[0,1].set_ylabel('Value', size=12)

    ax[1,0].plot(risks[:,2])
    ax[1,0].set_title('Sharpe', size=13)
    ax[1,0].set_xlabel('Time', size=12)
    ax[1,0].set_ylabel('Value', size=12)

    plt.tight_layout()
    plt.show()
    
def plot_weights(weights):
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12) 
    fig = plt.figure(figsize=(10,6))
    
    for col in weights.columns:
        plt.plot(weights[col].values, label=col)
    plt.xlabel('Time', size=13)
    plt.ylabel('Weight', size=13)
    plt.title('Weights', size=13)
    plt.legend()
    plt.show()
    
def plot_stats(rewards, rewards_eq, vals, vals_eq):
    matplotlib.rc('xtick', labelsize=11) 
    matplotlib.rc('ytick', labelsize=11) 
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    
    ax[0].plot(rewards, label='Agent')
    ax[0].plot(rewards_eq, label='Equiweighted')
    ax[0].set_xlabel('Time', size=13)
    ax[0].set_ylabel('Reward', size=13)
    ax[0].set_title('Weights', size=13)
    ax[0].legend()
    
    ax[1].plot(vals, label='Agent')
    ax[1].plot(vals_eq, label='Equiweighted')
    ax[1].hlines(xmin=0,xmax=len(vals) ,y=10000, color='r')
    ax[1].set_xlabel('Time', size=13)
    ax[1].set_ylabel('Portfolio value', size=13)
    ax[1].legend()
    
    plt.show()
    
def save_items(items, names, folder):
    if not os.path.isdir(f'results/{folder}'):
        os.mkdir(f'results/{folder}')
    for item, name in zip(items, names):
        with open(f'results/{folder}/{name}', 'wb') as f:
            pickle.dump(item, f)