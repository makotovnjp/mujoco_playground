"""
ライブラリ
uv pip install -U mujoco gymnasium jax jaxlib optax chex opencv-python
uv pip install -U tf-nightly tf2onnx

実行例
python learning/train_hunter.py \
--xml_path /home/hachix/repos/smart_one/mujoco_playground/assets/pdd.xml \
--torso_name base_link \
--save_dir out/frames --save_every 10 \
--video_path out/run.mp4 --video_fps 30 \
--push_every 400 --push_duration 20 --push_fmin 100 --push_fmax 200 \
--export_onnx out/policy.onnx --onnx_opset 17 \
--spawn_z 0.9 --settle_steps 600 --settle_hits 3

使い方メモ
- 観測: [各モーターのトルク (ctrl) | IMU(orientation, gyro, accel)]
- 行動: 各モーターのトルク [Nm]（ctrlrange に合わせて自動スケール）
- 倒れ判定（z < 0.6〜0.7）や報酬係数はモデルに合わせて微調整
"""


import argparse, os, functools
import numpy as np
import gymnasium as gym
import mujoco
import jax, jax.numpy as jnp, optax

# ========= 環境（観測=トルク+IMU / 行動=トルク[-1..1]→ctrlrange 変換 / 着地セトル付） =========
class HunterEnv(gym.Env):
    def __init__(self, 
        xml_path, torso="base_link", frame_skip=5,
        camera=None, W=480, H=360, save_dir=None, save_every=10,
        video_path=None, video_fps=30,
        push_every=400, push_dur=20, fmin=80., fmax=180., horiz_only=True,
        # 着地関連
        spawn_z=0.9, settle_steps=600, settle_contact_hits=3):

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # 胴体
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso)
        if self.torso < 0:
            self.torso = 0
            name0 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, 0)
            print(f"[WARN] body '{torso}' not found; fallback to id 0 ('{name0}')")

        self.frame_skip = frame_skip

        # Renderer / Camera（名前 or ID, 無ければ free）
        self.renderer = mujoco.Renderer(self.model, W, H)
        cam_name = camera
        if cam_name is not None:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                available = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                             for i in range(self.model.ncam)]
                print(f"[WARN] camera '{cam_name}' not found. available={available}")
                if self.model.ncam > 0: cam_id, cam_name = 0, available[0]
                else: cam_id, cam_name = -1, None
        else:
            if self.model.ncam > 0:
                cam_id, cam_name = 0, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
            else:
                cam_id, cam_name = -1, None
        self.camera = cam_name
        self.camera_id = int(cam_id)
        if self.model.ncam > 0:
            cams = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]
            print(f"[INFO] cameras in model: {cams}, using: {self.camera or 'free'} (id={self.camera_id})")
        else:
            print("[INFO] no cameras in model, using free camera (id=-1)")

        # 保存
        self.save_dir, self.save_every = save_dir, max(1, save_every)
        self.video_path, self.video_fps = video_path, video_fps
        self.writer = None
        if self.save_dir: os.makedirs(self.save_dir, exist_ok=True)

        # 行動のスケール（ctrlrange）
        if getattr(self.model, "actuator_ctrlrange", None) is not None and self.model.nu > 0:
            cr = np.array(self.model.actuator_ctrlrange, dtype=np.float32)
            self.act_low  = cr[:,0]; self.act_high = cr[:,1]
        else:
            self.act_low  = -np.ones(self.model.nu, np.float32)
            self.act_high =  np.ones(self.model.nu, np.float32)
        self.act_mid   = (self.act_low + self.act_high) / 2.0
        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.model.nu,), dtype=np.float32)

        # IMU センサー（存在するものだけ使用）
        self.imu_names = ["orientation", "angular-velocity", "linear-acceleration"]
        self._imu_slices = []
        for nm in self.imu_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, nm)
            if sid >= 0:
                adr = int(self.model.sensor_adr[sid]); dim = int(self.model.sensor_dim[sid])
                self._imu_slices.append((nm, adr, dim))
            else:
                print(f"[WARN] IMU sensor '{nm}' not found; skipping.")
        self.imu_dim = sum(dim for _,_,dim in self._imu_slices)

        # 観測 = [現在のモータートルク(ctrl) | IMU]
        obs_dim = int(self.model.nu + self.imu_dim)
        self.obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        # 外乱
        self.push_every, self.push_dur = max(1, push_every), max(1, push_dur)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.horiz_only = horiz_only
        self._remain = 0
        self._force = np.zeros(3, np.float32)

        # 着地用設定
        self.spawn_z = float(spawn_z)
        self.settle_steps = int(settle_steps)
        self.settle_contact_hits = int(settle_contact_hits)

        # 地面ジオム候補（plane 全部 + name=ground があれば追加）
        self.ground_ids = set(int(g) for g in range(self.model.ngeom)
                              if int(self.model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE))
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if gid >= 0:
            self.ground_ids.add(int(gid))
        if not self.ground_ids:
            print("[WARN] no plane geoms detected; settle will accept any contact.")

        # ルート自由関節の qpos アドレス（slide x/y/z と ball quat）
        self._root_idx = {"x":None, "y":None, "z":None, "quat":None}
        for j in range(self.model.njnt):
            if int(self.model.jnt_bodyid[j]) != self.torso:
                continue
            jtype = int(self.model.jnt_type[j])
            adr   = int(self.model.jnt_qposadr[j])
            if jtype == mujoco.mjtJoint.mjJNT_SLIDE:
                ax = self.model.jnt_axis[j]
                if abs(ax[0]) > 0.9: self._root_idx["x"] = adr
                if abs(ax[1]) > 0.9: self._root_idx["y"] = adr
                if abs(ax[2]) > 0.9: self._root_idx["z"] = adr
            elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                self._root_idx["quat"] = adr  # 4要素

        # その他
        self.qpos_init = self.model.key_qpos.copy() if self.model.nkey > 0 else None
        self._last_xy = np.zeros(2, np.float32)
        self.dt = float(self.model.opt.timestep) * float(self.frame_skip)
        self.gstep = 0

    # ---- ルート姿勢をセット ----
    def _set_root_pose(self, x=0.0, y=0.0, z=None, quat=(1.0,0.0,0.0,0.0)):
        z = self.spawn_z if z is None else float(z)
        if self._root_idx["x"] is not None: self.data.qpos[self._root_idx["x"]] = x
        if self._root_idx["y"] is not None: self.data.qpos[self._root_idx["y"]] = y
        if self._root_idx["z"] is not None: self.data.qpos[self._root_idx["z"]] = z
        qi = self._root_idx["quat"]
        if qi is not None:
            self.data.qpos[qi:qi+4] = np.asarray(quat, dtype=np.float64)

    # ---- 着地が確認できるまで数百ステップ回す ----
    def _settle_to_ground(self):
        ctrl_backup = self.data.ctrl.copy()
        self.data.ctrl[:] = self.act_mid
        hits = 0
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)
            touched = False
            if self.data.ncon > 0:
                for c in range(self.data.ncon):
                    con = self.data.contact[c]
                    if not self.ground_ids:
                        touched = True
                        break
                    if int(con.geom1) in self.ground_ids or int(con.geom2) in self.ground_ids:
                        touched = True
                        break
            if touched:
                hits += 1
                if hits >= self.settle_contact_hits:
                    break
        self.data.ctrl[:] = ctrl_backup

    # ---- IMU 読み出し ----
    def _read_imu(self):
        parts = []
        for _, adr, dim in self._imu_slices:
            parts.append(self.data.sensordata[adr:adr+dim])
        return np.concatenate(parts, dtype=np.float32) if parts else np.zeros((0,), np.float32)

    # ---- 観測 ----
    def _obs(self):
        torques = self.data.ctrl.copy().astype(np.float32)
        imu = self._read_imu()
        return np.concatenate([torques, imu], dtype=np.float32)

    # ---- 外乱 ----
    def _maybe_start_push(self):
        if self.gstep % self.push_every != 0: return
        v = (np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), 0.0], np.float32)
             if self.horiz_only else np.random.normal(size=3).astype(np.float32))
        v /= (np.linalg.norm(v)+1e-8)
        self._force = v * np.random.uniform(self.fmin, self.fmax)
        self._remain = self.push_dur

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        if self.qpos_init is not None and len(self.qpos_init)==self.model.nq:
            self.data.qpos[:] = self.qpos_init
        else:
            self.data.qpos[:] = 0; self.data.qvel[:] = 0
            self._set_root_pose(z=self.spawn_z, quat=(1.0,0.0,0.0,0.0))
        self.data.ctrl[:] = self.act_mid
        mujoco.mj_forward(self.model, self.data)

        # ★ 着地セトル
        self._settle_to_ground()

        self._last_xy = self.data.xpos[self.torso,:2].copy()
        self._remain = 0
        self.data.xfrc_applied[:,:] = 0.0
        return self._obs(), {}

    def step(self, a):
        a = np.clip(a, -1, 1)
        torque_cmd = self.act_mid + self.act_scale * a
        self.data.ctrl[:] = torque_cmd

        self._maybe_start_push()
        for _ in range(self.frame_skip):
            if self._remain > 0:
                self.data.xfrc_applied[self.torso,:3] = self._force
            mujoco.mj_step(self.model, self.data)
            if self._remain > 0:
                self._remain -= 1
                if self._remain == 0: self.data.xfrc_applied[self.torso,:] = 0.0
        if self._remain<=0: self.data.xfrc_applied[self.torso,:] = 0.0

        xmat = self.data.xmat[self.torso].reshape(3,3)
        up = float(np.dot(xmat[:,2], np.array([0,0,1]))); up = np.clip(up, -1, 1)
        vel_pen = 0.01 * float(np.sum(self.data.qvel**2))
        tor_pen = 0.0005 * float(np.sum(self.data.ctrl**2))
        xy = self.data.xpos[self.torso,:2]
        sp = np.linalg.norm((xy - self._last_xy)/self.dt); self._last_xy = xy.copy()
        sp_pen = 0.05 * float(sp)
        rew = 1.6*up - vel_pen - tor_pen - sp_pen

        z = self.data.xpos[self.torso,2]
        terminated = z < 0.65
        info = {"upright": up, "speed_xy": sp, "push": self._remain>0}

        self._maybe_save_frame()
        self.gstep += 1
        return self._obs(), rew, terminated, False, info

    # 可視化（新旧API両対応）
    def get_rgb(self):
        try:
            return self.renderer.render(self.data, camera=self.camera)
        except TypeError:
            self.renderer.update_scene(self.data, camera=self.camera_id)
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
                frame = self.get_rgb(); h,w = frame.shape[:2]
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

# ========= PPO（JAX） =========
class Cfg:
    seed=0; total_steps=10000; horizon=1024
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
    return {"pi": mlp([obs,128,128,act], k1),
            "v" : mlp([obs,128,128,1], k2),
            "logstd": jnp.zeros((act,))}

# ========= ONNX（遅延インポート / NumPy 2 互換） =========
def export_policy_value_onnx(params, obs_dim: int, act_mid: np.ndarray, act_scale: np.ndarray,
                             onnx_path: str, opset: int = 17):
    """
    入力 : obs [B, obs_dim] (float32)
    出力 : torque_mean [B, nu] (Nm), value [B], logstd [nu]
    """
    try:
        from jax.experimental import jax2tf
        import tensorflow as tf
        import tf2onnx
    except Exception as e:
        print("[ONNX] Missing/old deps:", e)
        print(" -> Try: uv pip install -U tf-nightly tf2onnx")
        raise

    act_mid_j = jnp.asarray(act_mid,  dtype=jnp.float32)
    act_scl_j = jnp.asarray(act_scale, dtype=jnp.float32)

    def jax_model(x):
        mean_raw, _ = pi(params, x.astype(jnp.float32))
        torque_mean = act_mid_j + act_scl_j * jnp.tanh(mean_raw)  # [-1,1]→トルク
        v = vf(params, x.astype(jnp.float32))
        logstd = params["logstd"]
        return torque_mean, v, logstd

    tried_dynamic = True
    try:
        tf_fn = jax2tf.convert(jax_model, with_gradient=False, polymorphic_shapes=["(b,{})".format(obs_dim)])
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, obs_dim], dtype=tf.float32, name="obs")],
                     autograph=False)
        def tf_func(obs):
            torque_mean, v, logstd = tf_fn(obs)
            return {"torque_mean": torque_mean, "value": v, "logstd": logstd}
        concrete = tf_func.get_concrete_function()
        print("[ONNX] traced with dynamic batch dimension.")
    except Exception as e:
        print(f"[ONNX] dynamic-batch trace failed: {e}\n[ONNX] fallback to fixed batch=1.")
        tried_dynamic = False
        tf_fn = jax2tf.convert(jax_model, with_gradient=False)
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, obs_dim], dtype=tf.float32, name="obs")],
                     autograph=False)
        def tf_func(obs):
            torque_mean, v, logstd = tf_fn(obs)
            return {"torque_mean": torque_mean, "value": v, "logstd": logstd}
        concrete = tf_func.get_concrete_function()

    out_dir = os.path.dirname(onnx_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    tf2onnx.convert.from_function(concrete_function=concrete, opset=opset, output_path=onnx_path)
    print(f"[ONNX] saved to: {onnx_path} (opset={opset}, input shape: [{'dynamic B' if tried_dynamic else 'B=1'}, {obs_dim}])")

# ========= 学習ループ =========
def train(args):
    env = HunterEnv(args.xml_path, torso=args.torso_name, frame_skip=5,
                   camera=args.camera_name, W=args.vision_width, H=args.vision_height,
                   save_dir=args.save_dir, save_every=args.save_every,
                   video_path=args.video_path, video_fps=args.video_fps,
                   push_every=args.push_every, push_dur=args.push_duration,
                   fmin=args.push_fmin, fmax=args.push_fmax, horiz_only=not args.push_allow_vertical,
                   spawn_z=args.spawn_z, settle_steps=args.settle_steps, settle_contact_hits=args.settle_hits)
    
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

    # ONNX（必要なときだけ）
    if args.export_onnx:
        export_policy_value_onnx(params, obs_dim, env.act_mid, env.act_scale, args.export_onnx, args.onnx_opset)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--torso_name", default="base_link")
    # Vision/保存
    ap.add_argument("--camera_name", default=None)
    ap.add_argument("--vision_width", type=int, default=480)
    ap.add_argument("--vision_height", type=int, default=360)
    ap.add_argument("--save_dir", default="out/frames")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--video_path", default="out/run.mp4")
    ap.add_argument("--video_fps", type=int, default=30)
    # 外乱（押し）
    ap.add_argument("--push_every", type=int, default=400)
    ap.add_argument("--push_duration", type=int, default=20)
    ap.add_argument("--push_fmin", type=float, default=100.0)
    ap.add_argument("--push_fmax", type=float, default=200.0)
    ap.add_argument("--push_allow_vertical", action="store_true")
    # 着地パラメータ
    ap.add_argument("--spawn_z", type=float, default=0.9)
    ap.add_argument("--settle_steps", type=int, default=600)
    ap.add_argument("--settle_hits", type=int, default=3)
    # ONNX
    ap.add_argument("--export_onnx", default=None, help="保存先（例: out/policy.onnx）")
    ap.add_argument("--onnx_opset", type=int, default=17)
    args=ap.parse_args()
    train(args)
