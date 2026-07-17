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
并将本地下载好的数据集和模型上传到租的机器上 已经完成 正在在组的机器上安装依赖环境 但是感觉在本地电脑上安装依赖环境很快很顺 在租的电脑上就很慢 很卡

**Challenges & blockers**
-找不到matterport数据集 写邮件给他 他拒绝我 但是后面在百度网盘上发现了别人下载好的 于是下载下来用了
 我在那个租的机器上想装habitat sim 结果很难装上 一直显示在solving environment 导致我配不好环境 跑不起来预训练模型
 如图卡死
 <img width="2328" height="360" alt="image" src="https://github.com/user-attachments/assets/3620df9c-7492-45ea-a416-f26b10882e9c" />
 事实证明 是我不够有耐心 还是装下来了最后
但是当我实际输入运行指令的时候 报错多到爆炸 根本调不完 现在在我修了一万个bug之后比如丢包 少字段 版本不匹配各种恶心问题 场景终于是成功加载出来了 实例也被创建出来了
<img width="2376" height="1124" alt="image" src="https://github.com/user-attachments/assets/3f624b5d-b895-40e2-8afc-762ab9d5095d" />
<img width="1170" height="864" alt="image" src="https://github.com/user-attachments/assets/68eab806-4cb3-44be-843f-33e31972717e" />



**Next steps*

跑预训练模型

**Hours spent (optional):**  6h

**Links (optional):** none


### Week 4 — 2026/7/8

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

<img width="1254" height="914" alt="image" src="https://github.com/user-attachments/assets/7f179b82-88fb-4d6a-acd8-59142e7bf863" />

<img width="2270" height="1080" alt="image" src="https://github.com/user-attachments/assets/ad8d322d-ea90-4ef9-9b82-a918bafacaeb" />


**Challenges & blockers**
- 依旧版本不匹配 后面装了旧版本的python库 貌似解决了问题因为ai默认都装最新版本的 所以前面一直有很多冲突
- 本来以为版本弄完可以跑eval了 发现一直卡死再初始化阶段 检查才发现居然之前下的预训练模型居然缺失了 但是发现如果整个下载的话下载不下 于是清理了很多日志等 与跑eval无关的文件 最后成功下载下来了磁盘只有30G 确实太紧张

**Next steps**
- 再次跑eval 感觉总应该没啥问题了吧 如果还有问题继续改

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._


### Week 5 — 2026/7/15

**Attended this week's meeting:** Yes

**Progress this week**
- 确实预训练模型 成功下载 但还是卡在初始阶段 他说找不到我下载的matterport3d数据集 我正在重新传

**Challenges & blockers**
- <img width="2334" height="1042" alt="image" src="https://github.com/user-attachments/assets/d2371106-da20-451d-a481-a5808fce4eff" />

从目前掌握的情况看,大致是这样:

已经具备的:

conda 环境(etpnav)已装好,依赖跑通过
预训练权重 pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt 有了
eval checkpoint ckpt.iter12000.pth 也下载完了
数据集 annotation(R2R_VLNCE_v1-2 及预处理版本的 .json.gz)已经在容器里
connectivity_graphs.pkl 也在
还差的,按顺序:

场景文件(.glb) —— 这是最关键的缺口。

**Next steps**
- 前几周就是因为版本问题卡死在这上面 不降级永远都跑不通 代码一堆错误 现在克隆到另一个机子上面 版本降级了 但是少很多文件 预训练模型啊 checkpoint啊 但是现在基本上补齐了 就差一个场景数据集了

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._


