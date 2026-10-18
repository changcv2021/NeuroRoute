"""Render recorded Go1 trajectories and tracking plots."""

def render_trace(env,trace,output):
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=mujoco.MjData(env.mj_model)
    renderer=mujoco.Renderer(env.mj_model,height=240,width=320)
    selected=[]
    with imageio.get_writer(output/'scripted_seed41001.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for step in range(0,len(trace['qpos']),2):
            data.qpos[:]=trace['qpos'][step,0]; data.qvel[:]=trace['qvel'][step,0]
            mujoco.mj_forward(env.mj_model,data)
            renderer.update_scene(data,camera='track')
            frame=renderer.render()
            writer.append_data(frame)
            if step in (50,150,300,400,550,650,800,900):
                selected.append((step,frame.copy()))
    renderer.close()
    fig,axes=plt.subplots(4,2,figsize=(8,11),constrained_layout=True)
    for ax,(step,frame) in zip(axes.flat,selected):
        ax.imshow(frame); ax.axis('off')
        ax.set_title(f't={step*.02:.1f}s command={trace["command"][step,0].round(2)}')
    fig.savefig(output/'scripted_contact_sheet.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
    time=np.arange(len(trace['qpos']))*.02
    for i,label in enumerate(['vx / m/s','vy / m/s','yaw / rad/s']):
        actual=trace['actual_velocity'][:,0,i] if i<2 else trace['actual_yaw_rate'][:,0]
        axes[i].plot(time,trace['command'][:,0,i],label='Command')
        axes[i].plot(time,actual,label='Actual');axes[i].set_ylabel(label);axes[i].grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Time / s')
    fig.savefig(output/'scripted_tracking.png',dpi=160);plt.close(fig)
