import streamlit as st
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import time

st.title('Diffusion Model Sampling')
st.caption('このデモプログラムはDiffusion Modelのサンプリング手法を2次元座標データセットで検証したものです．アルゴリズムの詳細は会誌([KaiRA公式HP](https://kyoto-kaira.github.io/assets/docs/NF_2023.pdf)からダウンロード可)の「画像生成」の章をご覧ください．')

st.markdown('''
## 準備
まずデータの準備をします．
scikit-learnのmake_moons関数を用いて1000個の点をサンプリングします．
各点は2次元座標で表され，0か1のどちらかのクラスに属しています．
このデータセットに含まれるようなデータを，デノイジングスコアマッチングによってサンプリングします．
\nなお，普通Diffusion Modelではスコアをニューラルネット等で計算しますが，今回のデータ規模ではデータセットから正確なスコアを求めることが可能になります．
''')

# dataset config
N = 1000
D = 2
data, label = make_moons(n_samples=N, noise=0.05, shuffle=True, random_state=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data[label==0][:, 0], data[label==0][:, 1], marker=".", label=0)
ax.scatter(data[label==1][:, 0], data[label==1][:, 1], marker=".", label=1)
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-1.0, 1.5])
ax.grid()
ax.legend()

st.pyplot(fig)

# sampling config
T = 1000
betas = np.linspace(0.00085, 0.012, T)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas)
betas_cumprod = 1 - alphas_cumprod
sigmas = ((1 - alphas_cumprod) / alphas_cumprod)**0.5


def get_score_(X, x, t, var_preserve=True):
    if var_preserve:
        mu = alphas_cumprod[t]**0.5 * X.reshape(1, -1, D)
        sigma2 = betas_cumprod[t]
    else:
        mu = X.reshape(1, -1, D)
        sigma2 = sigmas[t]**2

    x = x.reshape(-1, 1, D)
    p = np.exp(-0.5 * ((x - mu)**2).sum(-1, keepdims=True) / sigma2)
    d = - (x - mu) / sigma2**0.5
    return (p * d).sum(1) / p.sum(1)


def get_score(x, t, c=None, guidance_scale=1.0, var_preserve=True):
    if c is None:
        return get_score_(data, x, t, var_preserve)

    score_cond = get_score_(data[label==c], x, t, var_preserve)
    score_uncond = get_score_(data, x, t, var_preserve)
    return score_uncond + guidance_scale * (score_cond - score_uncond)


def ddpm(x, t, t_prev, c=None, guidance_scale=1.0):
    noise = - get_score(x, t, c, guidance_scale, var_preserve=True)
    mu = alphas[t]**(-0.5) * (x - betas[t]/betas_cumprod[t]**0.5 * noise)

    if t_prev == 0:
        return mu
    else:
        return mu + np.random.randn(*mu.shape) * betas[t]**0.5


def ddim(x, t, t_prev, c=None, guidance_scale=1.0):
    noise = - get_score(x, t, c, guidance_scale, var_preserve=True)
    x0 = (x - betas_cumprod[t]**0.5 * noise) / alphas_cumprod[t]**0.5
    return alphas_cumprod[t_prev]**0.5 * x0 + betas_cumprod[t_prev]**0.5 * noise


def dpm2(x, t, t_prev, c=None, guidance_scale=1.0):
    noise = - get_score(x, t, c, guidance_scale, var_preserve=True)
    alpha_t = alphas_cumprod[t]**0.5
    alpha_t_prev = alphas_cumprod[t_prev]**0.5
    sigma_t = (1 - alphas_cumprod[t])**0.5
    sigma_t_prev = (1 - alphas_cumprod[t_prev])**0.5
    lambda_t = np.log(alpha_t / sigma_t)
    lambda_t_prev = np.log(alpha_t_prev / sigma_t_prev)
    sigma_lambda = np.exp(-0.5 * (lambda_t + lambda_t_prev))
    s_i = np.argmin((sigmas - sigma_lambda)**2)
    h_i = lambda_t_prev - lambda_t
    alpha_s = alphas_cumprod[s_i]**0.5
    sigma_s = (1 - alphas_cumprod[s_i])**0.5
    u_i = (alpha_s / alpha_t) * x - sigma_s * (np.exp(0.5*h_i) - 1) * noise
    noise = - get_score(u_i, s_i, c, guidance_scale, var_preserve=True)
    return (alpha_t_prev / alpha_t) * x - sigma_t_prev * (np.exp(h_i) - 1) * noise


st.markdown('''
## サンプリングの可視化
実際に逆拡散過程を計算することでサンプリングを行います．ここでは200個のデータを生成します．
''')

N_sampling = st.slider('**:footprints:サンプリングステップ数**\n\n計算時間はサンプリングステップ数に比例します．', 1, 1000)
sampler = st.selectbox('**:toolbox:サンプリング手法**', ('DDPM', 'DDIM', 'DPM2'))
st.markdown('''
- **DDPM**…サンプリングステップ数が500未満だと生成データの品質が悪くなります．
- **DDIM**…サンプリングステップ数が100未満だと生成データの品質が悪くなります．
- **DPM2**…サンプリングステップ数が20未満だと生成データの品質が悪くなります．
''')
cond = st.selectbox(
    '**:chains:条件指定**\n\n0か1を選んだ場合は，選ばれたクラスのデータを生成するよう，Classifier-free guidanceを用いてサンプリングが行われます．', 
    ('None', '0', '1'))
guidance_scale = st.slider(
    '**:guide_dog: guidance scale**\n\nClassifier-free guidanceのパラメータです．1より大きい場合はClassifier-free guidanceが有効になります．値を大きくすると，より条件に合ったデータが生成されるようになりますが，生成データの多様性が小さくなります．', 
    0.0, 100.0, step=0.1)

N_data = 200
t_step = T // N_sampling

if sampler == "DDPM":
    sampler = ddpm
elif sampler == "DDIM":
    sampler = ddim
elif sampler == "DPM2":
    sampler = dpm2

if cond in ["0", "1"]:
    cond = int(cond)
else:
    cond = None

x_init = np.random.randn(N_data, D)
trajectory = [x_init]

start_time = time.time()

t = T - 1
bar = st.progress(0, text="サンプリング中...")
while t > 0:
    t_prev = max(0, t - t_step)
    x_prev = sampler(trajectory[-1], t, t_prev, cond, guidance_scale)
    trajectory.append(x_prev)
    t = t_prev
    bar.progress((T-t)/T, text=f"サンプリング中... {(T-t)/T*100:.0f}%")
    
trajectory = np.array(trajectory)
finish_time = time.time()

st.success(f'''
- 計算時間:stopwatch: ... {finish_time - start_time:.1f} 秒
- サンプリングステップ数:footprints: ... {N_sampling}
- サンプリング手法:toolbox: ... {sampler.__name__}
- 条件指定:chains: ... {cond}
- guidance scale:guide_dog: ... {guidance_scale}
''')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data[label==0][:, 0], data[label==0][:, 1], marker=".", label=0)
ax.scatter(data[label==1][:, 0], data[label==1][:, 1], marker=".", label=1)
ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], marker="*", color="black")
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-1.0, 1.5])
ax.legend()
ax.grid()

st.pyplot(fig)
