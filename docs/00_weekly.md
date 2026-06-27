# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

## Week 1 — 2026-06-10
**Attended this week's meeting:** Yes (attend kick off meeting)

### Progress this week
- Set up repository from the FURP template.
- read the relevant paper of VLN-CE, knowing the basic concept and the main obstacle we are facing.

### Challenges & blockers
- Do not know what to do next, need instruction, need partner.

### Next steps
- attend next meeting and make plan.

**Hours spent (optional):** 6 hours

**Links (optional):** none


### Week 2 — 2026/6/19

**Attended this week's meeting:** Yes

**Progress this week**
-Finish a simple smoke test,evidence:<img width="865" height="282" alt="image" src="https://github.com/user-attachments/assets/d2b8c4e5-adba-4387-a761-4d12115ef657" />


**Challenges & blockers**
-can not visualise the result

**Next steps**
-do further test focus not just on can it work but how it can work better

**Hours spent (optional):** 6h

**Links (optional):** <img width="865" height="282" alt="image" src="https://github.com/user-attachments/assets/f0f00434-d546-4419-bfeb-496c44b2dadd" />
command history:(habitat) scyjh11@subicomputer1:~$ history | tail -50
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

obs = sim.get_sensor_observations()
img = Image.fromarray(obs["color_sensor"])
img.save("/tmp/smoke_test.png")
print("Smoke test passed! Screenshot saved to /tmp/smoke_test.png")
sim.close()
EOF

   33  sudo apt-get install -y libosmesa6-dev
   34  python ~/smoke_test.py
   35  conda install -y -c aihabitat -c conda-forge habitat-sim=0.3.3 headless
   36  python ~/smoke_test.py
   37  conda activate habitat
   38  HABITAT_SIM_RENDERER=osmesa python ~/smoke_test.py
   39  conda remove habitat-sim --force -y
   40  conda install -c conda-forge -c aihabitat habitat-sim=0.3.3 headless_osmesa -y
   41  cat > ~/smoke_test.py << 'EOF'
import habitat_sim
import numpy as np

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "/home/scyjh11/habitat-data/scene_datasets/habitat-test-scenes/apartment_1.glb"
backend_cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = []

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

agent = sim.initialize_agent(0)
state = agent.get_state()
print(f"Smoke test passed!")
print(f"Agent starting position: {state.position}")
print(f"Scene loaded: apartment_1.glb")
sim.close()
EOF

   42  python ~/smoke_test.py
   43  conda activate habitat
   44  conda list | grep habitat
   45  conda install -c aihabitat -c conda-forge habitat-sim=0.3.3 headless -y
   46  conda list | grep habitat
   47  python ~/smoke_test.py
   48  history | tail -50

### Week 3 — 2026/6/26

**Attended this week's meeting:** Yes

**Progress this week**
-根据老师的指导准备给预训练模型做evaluation 申请matterport3d数据集 但还没收到回复

<img width="1262" height="1168" alt="image" src="https://github.com/user-attachments/assets/47001ddf-1a17-4f42-8567-5a01c97d86b2" />

等待的同时注册autodl 并在网上寻找现成的数据据链接


<img width="1958" height="694" alt="image" src="https://github.com/user-attachments/assets/5e51afee-632d-4976-a042-e6e9a5d6fc25" />
将ETPNav的开源算法克隆到本地


<img width="2038" height="1276" alt="image" src="https://github.com/user-attachments/assets/b7bdea45-bddd-4647-9a41-eb256c0309b3" />

将从网上找来的可行的预训练权重pt文件 和R2R任务数据集下载到本地的D盘位置

租借autodl机器开始跑
<img width="2132" height="776" alt="image" src="https://github.com/user-attachments/assets/5e73c8ca-3d68-4e54-87a3-050a4037e868" />
并将本地下载好的数据集和模型上传到租的机器上

**Challenges & blockers**
-找不到matterport数据集 写邮件给他 他拒绝我 但是后面在百度网盘上发现了别人下载好的 于是下载下来用了

**Next steps*

跑预训练模型

**Hours spent (optional):**  6h

**Links (optional):** none

