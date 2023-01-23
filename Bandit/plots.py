import matplotlib.pyplot as plt

def testbed(df, fig, title, k=10, run=0):
    fig.violinplot(dataset = [df[df.run == run][df.actions == r]["rewards"].values for r in range(k)])

    for k, v in enumerate(df[df.run == run].tail(1).loc[0].Q):
        fig.hlines(y=v, xmin=k+1-0.2, xmax=k+1+0.2, linewidth=1, color='r')
    
    for k, v in enumerate(df[df.run == run].tail(1).loc[0].q_star):
        fig.hlines(y=v, xmin=k+1-0.2, xmax=k+1+0.2, linewidth=1, color='b')

    fig.set_title(title)
    fig.yaxis.grid(True)
    fig.set_xlabel('Actions')
    fig.set_ylabel('Rewards')
    fig.set_xticks(ticks=range(11))


def avgPerformance(df, fig, title):
    fig[0].plot(df[["steps", "rewards"]].groupby(['steps']).mean())
    df["bestAct"] = df["actions"] == df["optimal_actions"]
    fig[1].plot(df[['steps', "bestAct"]].astype(int).groupby(['steps']).mean()*100)

    fig[0].set_title(title)
    fig[0].set_xlabel('Steps')
    fig[0].set_ylabel('Avg Reward')
    fig[1].set_xlabel('Steps')
    fig[1].set_ylabel('% Optimal Action')
    fig[1].set_yticks(ticks=range(0, 101, 20))