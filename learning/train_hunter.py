"""
ライブラリ
pip install mujoco gymnasium jax jaxlib optax chex opencv-python

実行例

python learning/train_hunter.py \
--xml_path /home/hachix/repos/smart_one/mujoco_playground/assets/pdd.xml \
--torso_name base_link \
--save_dir out/frames --save_every 10 \
--video_path out/run.mp4 --video_fps 30 \
--push_every 400 --push_duration 20 --push_fmin 100 --push_fmax 200

使い方メモ
- 倒れ判定（z < 0.6）やトルクスケール（action*0.5）はモデルに合わせて微調整してください。
- 学習が不安定なら horizon=512、epochs=3、lr=1e-3→3e-4 など試してみてくださ
"""


import argparse, os, functools
import numpy as np
import gymnasium as gym
import mujoco
import jax, jax.numpy as jnp, optax

# ========= 環境（立位 + ランダム外乱 + 画像/動画保存 + カメラ安全化） =========
class StandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}
    def __init__(self, xml_path, torso="base_link", frame_skip=5,
                 camera=None, W=480, H=360, save_dir=None, save_every=10,
                 video_path=None, video_fps=30,
                 push_every=400, push_dur=20, fmin=80., fmax=180., horiz_only=True):
        # --- モデル読み込み ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # --- 胴体（基準ボディ） ---
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso)
        if self.torso < 0:
            self.torso = 0
            name0 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, 0)
            print(f"[WARN] body '{torso}' not found; fallback to id 0 ('{name0}')")

        self.frame_skip = frame_skip

        # --- Renderer と カメラ（名前/ID フォールバック） ---
        self.renderer = mujoco.Renderer(self.model, W, H)

        cam_name = camera
        if cam_name is not None:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                available = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                             for i in range(self.model.ncam)]
                print(f"[WARN] camera '{cam_name}' not found. available={available}")
                if self.model.ncam > 0:
                    cam_id = 0
                    cam_name = available[0]
                else:
                    cam_id = -1
                    cam_name = None
        else:
            if self.model.ncam > 0:
                cam_id = 0
                cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
            else:
                cam_id = -1
                cam_name = None

        self.camera = cam_name       # 新API向け（名前 or None）
        self.camera_id = int(cam_id) # 旧API向け（数値ID, -1: free）

        if self.model.ncam > 0:
            cams = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]
            print(f"[INFO] cameras in model: {cams}, using: {self.camera or 'free'} (id={self.camera_id})")
        else:
            print("[INFO] no cameras in model, using free camera (id=-1)")

        # --- 画像/動画保存 ---
        self.save_dir, self.save_every = save_dir, max(1, save_every)
        self.video_path, self.video_fps = video_path, video_fps
        self.writer = None
        if self.save_dir: os.makedirs(self.save_dir, exist_ok=True)

        # --- action/obs ---
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.model.nu,), dtype=np.float32)
        obs_dim = self.model.nq + self.model.nv + 3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        # --- 外乱（押し） ---
        self.push_every, self.push_dur = max(1, push_every), max(1, push_dur)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.horiz_only = horiz_only
        self._remain = 0
        self._force = np.zeros(3, np.float32)

        # --- その他 ---
        self.qpos_init = self.model.key_qpos.copy() if self.model.nkey > 0 else None
        self._last_xy = np.zeros(2, np.float32)
        self.dt = float(self.model.opt.timestep) * float(self.frame_skip)
        self.gstep = 0

    def _up(self):
        xmat = self.data.xmat[self.torso].reshape(3,3)
        return xmat[:,2]
    def _obs(self):
        return np.concatenate([self.data.qpos.ravel(), self.data.qvel.ravel(), self._up()], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        if self.qpos_init is not None and len(self.qpos_init)==self.model.nq:
            self.data.qpos[:] = self.qpos_init
        else:
            self.data.qpos[:] = 0; self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._last_xy = self.data.xpos[self.torso,:2].copy()
        self._remain = 0
        self.data.xfrc_applied[:,:] = 0.0
        return self._obs(), {}

    def _maybe_start_push(self):
        if self.gstep % self.push_every != 0: return
        if self.horiz_only:
            v = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), 0.0], np.float32)
        else:
            v = np.random.normal(size=3).astype(np.float32)
        v /= (np.linalg.norm(v)+1e-8)
        mag = np.random.uniform(self.fmin, self.fmax)
        self._force = v * mag
        self._remain = self.push_dur

    def step(self, a):
        a = np.clip(a, -1, 1); self.data.ctrl[:] = a * 0.5
        self._maybe_start_push()
        for _ in range(self.frame_skip):
            if self._remain > 0:
                # body合力: 力のみ（作用点トルクは簡略化）
                self.data.xfrc_applied[self.torso,:3] = self._force
            mujoco.mj_step(self.model, self.data)
            if self._remain > 0:
                self._remain -= 1
                if self._remain == 0: self.data.xfrc_applied[self.torso,:] = 0.0
        if self._remain<=0: self.data.xfrc_applied[self.torso,:] = 0.0

        obs = self._obs()
        up = float(np.dot(obs[-3:], np.array([0,0,1]))); up = np.clip(up, -1, 1)
        vel_pen = 0.01 * float(np.sum(self.data.qvel**2))
        tor_pen = 0.001 * float(np.sum(self.data.ctrl**2))
        xy = self.data.xpos[self.torso,:2]
        sp = np.linalg.norm((xy - self._last_xy)/self.dt)
        self._last_xy = xy.copy()
        sp_pen = 0.05 * float(sp)
        rew = 1.6*up - vel_pen - tor_pen - sp_pen
        z = self.data.xpos[self.torso,2]
        terminated = z < 0.65   # base_link 初期z=0.88 想定。調整可。
        info = {"upright": up, "speed_xy": sp, "push": self._remain>0}
        self._maybe_save_frame()
        self.gstep += 1
        return obs, rew, terminated, False, info

    # --- 画像取得（新旧API対応） ---
    def get_rgb(self):
        try:
            # 新API: camera に name/None が使える
            return self.renderer.render(self.data, camera=self.camera)
        except TypeError:
            # 旧API: update_scene(..., camera=<int>) → render()
            self.renderer.update_scene(self.data, camera=self.camera_id)  # -1: free
            return self.renderer.render()
        except ValueError:
            self.renderer.update_scene(self.data, camera=self.camera_id)
            return self.renderer.render()

    def _maybe_save_frame(self):
        rgb = None
        if self.save_dir and (self.gstep % self.save_every == 0):
            rgb = self.get_rgb()
            if rgb is not None:
                import cv2
                cv2.imwrite(os.path.join(self.save_dir, f"step_{self.gstep:06d}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self.video_path:
            import cv2
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                frame = self.get_rgb()
                h,w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.video_path, fourcc, self.video_fps, (w,h))
                if not self.writer.isOpened():
                    alt = os.path.splitext(self.video_path)[0]+".avi"
                    self.writer = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"XVID"), self.video_fps, (w,h))
                    self.video_path = alt
            if rgb is None: rgb = self.get_rgb()
            if rgb is not None:
                self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release(); self.writer = None

# ========= 極小 PPO（JAX） =========
class Cfg:
    seed=0; total_steps=100_000; horizon=1024
    mb=64; epochs=5; gamma=0.99; lam=0.95; clip=0.2; vf=0.5; ent=0.0; lr=3e-4

def mlp(sizes, key):
    ps=[]; keys=jax.random.split(key, len(sizes)-1)
    for k,(m,n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w=jax.random.normal(k,(m,n))/jnp.sqrt(m); b=jnp.zeros((n,))
        ps.append((w,b))
    return ps
def fwd(ps,x):
    for w,b in ps[:-1]: x=jnp.tanh(x@w+b)
    w,b=ps[-1]; return x@w+b
@functools.partial(jax.jit, static_argnums=())
def pi(params, obs):
    mean=fwd(params["pi"], obs); std=jnp.exp(params["logstd"]); return mean, std
@jax.jit
def vf(params, obs): return jnp.squeeze(fwd(params["v"], obs), -1)
def logp_gauss(m,s,a): return -0.5*jnp.sum(((a-m)**2)/(s**2)+2*jnp.log(s)+jnp.log(2*jnp.pi),axis=-1)

@jax.jit
def loss_fn(params, batch, clip, vf_c, ent_c):
    m,s = pi(params, batch["obs"])
    lp  = logp_gauss(m,s,batch["act"])
    rat = jnp.exp(lp - batch["logp_old"])
    adv = batch["adv"]
    pg  = -jnp.mean(jnp.minimum(rat*adv, jnp.clip(rat,1-clip,1+clip)*adv))
    v   = vf(params, batch["obs"])
    vloss=jnp.mean((batch["ret"]-v)**2)
    ent = jnp.mean(0.5*jnp.sum(1+2*jnp.log(s)+jnp.log(2*jnp.pi),axis=-1))
    return pg + vf_c*vloss - ent_c*ent, (pg, vloss, ent)

def make_update(optimizer):
    # optimizer をクロージャに閉じ込めて、jit に「関数引数」を渡さない
    @functools.partial(jax.jit, static_argnums=(3,))
    def update(params, opt_state, batch, cfg: Cfg):
        (loss,(pg,vl,ent)),grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, cfg.clip, cfg.vf, cfg.ent
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pg, vl, ent
    return update

def init_agent(obs, act, key):
    k1,k2,_ = jax.random.split(key,3)
    return {
        "pi": mlp([obs,128,128,act], k1),
        "v" : mlp([obs,128,128,1], k2),
        "logstd": jnp.zeros((act,))
    }

# ========= 学習ループ =========
def train(args):
    env = StandEnv(args.xml_path, torso=args.torso_name, frame_skip=5,
                   camera=args.camera_name, W=args.vision_width, H=args.vision_height,
                   save_dir=args.save_dir, save_every=args.save_every,
                   video_path=args.video_path, video_fps=args.video_fps,
                   push_every=args.push_every, push_dur=args.push_duration,
                   fmin=args.push_fmin, fmax=args.push_fmax, horiz_only=not args.push_allow_vertical)
    cfg=Cfg; key=jax.random.PRNGKey(cfg.seed)
    obs_dim=env.observation_space.shape[0]; act_dim=env.action_space.shape[0]
    params=init_agent(obs_dim, act_dim, key)

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)
    update = make_update(optimizer)

    obs,_=env.reset(); ep_ret=0.0; ep_len=0; g=0

    while g < cfg.total_steps:
        H=cfg.horizon
        buf=dict(
            obs=np.zeros((H,obs_dim),np.float32),
            act=np.zeros((H,act_dim),np.float32),
            logp=np.zeros((H,),np.float32),
            rew=np.zeros((H,),np.float32),
            val=np.zeros((H+1,),np.float32),
            done=np.zeros((H,),np.float32),
        )
        for t in range(H):
            mean,std = jax.device_get(pi(params, jnp.asarray(obs)))
            v       = float(jax.device_get(vf(params, jnp.asarray(obs))))
            act     = np.random.normal(mean, std).astype(np.float32)
            lp      = float(jax.device_get(logp_gauss(jnp.asarray(mean), jnp.asarray(std), jnp.asarray(act))))
            buf["obs"][t]=obs; buf["act"][t]=act; buf["logp"][t]=lp; buf["val"][t]=v
            obs, r, term, trunc, info = env.step(act)
            buf["rew"][t]=r; d=float(term or trunc); buf["done"][t]=d
            ep_ret+=r; ep_len+=1; g+=1
            if d:
                print(f"[Episode] return={ep_ret:.2f}, len={ep_len}")
                obs,_=env.reset(); ep_ret=0.0; ep_len=0
            if g>=cfg.total_steps: break
        buf["val"][-1] = float(jax.device_get(vf(params, jnp.asarray(obs))))
        # GAE
        adv=np.zeros_like(buf["rew"],np.float32); last=0.0
        for t in reversed(range(len(buf["rew"]))):
            nonterm=1.0-buf["done"][t]
            delta=buf["rew"][t]+cfg.gamma*buf["val"][t+1]*nonterm-buf["val"][t]
            last=delta+cfg.gamma*cfg.lam*nonterm*last; adv[t]=last
        ret=adv+buf["val"][:-1]
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        dataset={"obs":jnp.asarray(buf["obs"]),"act":jnp.asarray(buf["act"]),
                 "logp_old":jnp.asarray(buf["logp"]),"adv":jnp.asarray(adv),"ret":jnp.asarray(ret)}
        # PPO update
        N=len(buf["rew"]); idx=np.arange(N)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0,N,cfg.mb):
                mb=idx[s:s+cfg.mb]; batch={k:v[mb] for k,v in dataset.items()}
                params,opt_state,loss,pg,vl,ent = update(params,opt_state,batch,cfg)
        print(f"update {g//cfg.horizon:4d} | loss {float(loss):.3f} | pg {float(pg):.3f} | vf {float(vl):.3f}")

    env.close()
    if env.video_path: print(f"[Video] saved to: {env.video_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--torso_name", default="base_link")   # あなたのXMLに合わせたデフォルト
    # Vision/保存
    ap.add_argument("--camera_name", default=None, help="MuJoCo <camera name=...>. 未指定/不在なら自動フォールバック")
    ap.add_argument("--vision_width", type=int, default=480)
    ap.add_argument("--vision_height", type=int, default=360)
    ap.add_argument("--save_dir", default="out/frames")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--video_path", default="out/run.mp4")
    ap.add_argument("--video_fps", type=int, default=30)
    # 外乱（押し）
    ap.add_argument("--push_every", type=int, default=400)
    ap.add_argument("--push_duration", type=int, default=20)
    ap.add_argument("--push_fmin", type=float, default=100.0)
    ap.add_argument("--push_fmax", type=float, default=200.0)
    ap.add_argument("--push_allow_vertical", action="store_true")
    args=ap.parse_args()
    train(args)
