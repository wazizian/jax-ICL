import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

# Problem setup
d = 8
C = 128.0 * 2 * 0.1
nus = [3.0, 5.0, 10.0, jnp.inf]
x_grid = jnp.linspace(0.0, 4.0, 500)
n_samples = 200_000
key = jax.random.PRNGKey(0)

# ---------- log densities ----------
def log_pdf_student_var1_per_coord(y, nu, is_nu_inf):
    """
    Log pdf of a univariate Student-t with df=nu, scaled to variance 1.
    If nu = inf -> standard normal.
    """
    if is_nu_inf:
        return -0.5*jnp.log(2*jnp.pi) - 0.5*(y**2)

    s = jnp.sqrt((nu - 2.0)/nu)  # variance-normalizing scale
    t = y / s
    # log c = ln Γ((ν+1)/2) - ln Γ(ν/2) - 0.5 ln(νπ) - ln s
    logc = (jax.lax.lgamma((nu+1.0)/2.0)
            - jax.lax.lgamma(nu/2.0)
            - 0.5*jnp.log(nu*jnp.pi)
            - jnp.log(s))
    # -(ν+1)/2 * ln(1 + t^2/ν)
    logtail = -0.5*(nu+1.0)*jnp.log1p((t*t)/nu)
    return logc + logtail

def log_pdf_student_var1(y, nu, is_nu_inf):
    # sum over independent coordinates
    return jnp.sum(log_pdf_student_var1_per_coord(y, nu, is_nu_inf), axis=-1)

def log_pdf_normal_mean1(y, mean):
    # q_x is N(mean, I): log q = -d/2 log(2π) - 0.5 ||y-mean||^2
    return -0.5*y.shape[-1]*jnp.log(2*jnp.pi) - 0.5*jnp.sum((y-mean)**2, axis=-1)

# ---------- IS estimator ----------
@partial(jax.jit, static_argnums=(1, 2))
def neg_log_mgf_IS_for_nu(base_Z, nu, is_nu_inf, C, x_grid):
    """
    base_Z ~ N(0, I_d) with shape (n_samples, d)
    For each x, set proposal q_x = N(x*1, I_d), sample theta = base_Z + x,
    and estimate E_p[exp(-C ||theta - x||^2)] via importance sampling.
    Returns vector over x_grid of -log expectation.
    """
    n = base_Z.shape[0]
    ones = jnp.ones((1, base_Z.shape[1]))

    def per_x(x):
        theta = base_Z + x * ones  # proposal samples ~ N(x, I)
        # log weights: log p_nu(theta) - log q_x(theta)
        log_w = log_pdf_student_var1(theta, nu, is_nu_inf) - log_pdf_normal_mean1(theta, x*ones)
        # log integrand: -C ||theta - x||^2 + log_w
        quad = jnp.sum((theta - x*ones)**2, axis=-1)
        log_integrand = -C * quad + log_w
        # log-mean-exp
        m = jax.scipy.special.logsumexp(log_integrand) - jnp.log(n)
        return -m  # negative log expectation

    return jax.vmap(per_x)(x_grid)

# ---------- Analytic Normal baseline (optional) ----------
@jax.jit
def neg_log_expect_normal(d, C, x_grid):
    # Z ~ N(0, I_d), mu = x*1:
    # E[exp(-C ||Z - mu||^2)] = (1 + 2C)^(-d/2) * exp(-C ||mu||^2 / (1+2C))
    factor = (1.0 + 2.0*C)
    term1 = 0.5 * d * jnp.log(factor)
    mu_norm2 = d * (x_grid**2)
    term2 = (C / factor) * mu_norm2
    return term1 + term2

# ---------- Decrease at infinity ----------
import jax
import jax.numpy as jnp
from jax.scipy.special import betainc

# ---------- Student-t: CDF (standard), log-PDF (scaled), and interval log-mass ----------

@jax.jit
def _t_cdf(x, df):
    """CDF of standard Student-t (loc=0, scale=1). Handles df=+inf (Normal)."""
    # Normal branch
    norm_cdf = 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))
    is_inf = jnp.isinf(df)

    # Student-t branch using regularized incomplete beta
    z = df / (df + x * x)
    I = betainc(df / 2.0, 0.5, z)         # regularized incomplete beta
    cdf_t = jnp.where(x >= 0, 1.0 - 0.5 * I, 0.5 * I)

    return jnp.where(is_inf, norm_cdf, cdf_t)

@jax.jit
def _t_logpdf_scaled(x, scale, df):
    """
    log pdf of Student-t with loc=0, scale>0, df>0 (df=+inf gives Normal).
    """
    inf_res = -0.5*jnp.log(2*jnp.pi) - jnp.log(scale) - 0.5*(x/scale)**2

    s = scale
    t = x / s
    # log c = ln Γ((ν+1)/2) - ln Γ(ν/2) - 0.5 ln(νπ) - ln s
    logc = (jax.lax.lgamma((df + 1.0) / 2.0)
            - jax.lax.lgamma(df / 2.0)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(s))
    # tail term
    logtail = -0.5 * (df + 1.0) * jnp.log1p((t * t) / df)
    notinf_res = logc + logtail
    return jnp.where(jnp.isinf(df), inf_res, notinf_res)

@jax.jit
def _t_log_interval_prob(a, b, scale, df, tiny=1e-300, switch_thresh=1e-12):
    """
    log P(X in [a,b]) for Student-t(loc=0,scale,df).
    Uses CDF difference when stable; otherwise switches to pdf(mid)*length.
    """
    a_s, b_s = a / scale, b / scale
    diff = jnp.clip(_t_cdf(b_s, df) - _t_cdf(a_s, df), 0.0, 1.0)

    # Stable branch selection
    # If diff is not extremely small, use log(diff).
    log_diff = jnp.log(jnp.maximum(diff, tiny))

    # Otherwise approximate via log pdf at midpoint + log length
    mid = 0.5 * (a + b)
    log_len = jnp.log(jnp.maximum(b - a, tiny))
    log_pdf_mid = _t_logpdf_scaled(mid, scale, df)
    log_small = log_pdf_mid + log_len

    return jnp.where(diff > switch_thresh, log_diff, log_small)

# ---------- Main function: log sum_j pi([βj, βj+δ])^α with log-domain stabilization ----------

@jax.jit
def log_sum_student_interval_prob_alpha(beta, delta, alpha, scale, df, d, max_abs_j=2000):
    """
    Returns log( Σ_j P([βj, βj+δ])^α ), computed stably in the log domain.
    """
    js = jnp.arange(-max_abs_j, max_abs_j + 1)
    a = beta * js
    b = a + delta
    new_alpha = alpha * d  # scale α by d for d-dimensional case

    # vectorized log interval masses
    log_p = jax.vmap(lambda aa, bb: _t_log_interval_prob(aa, bb, scale, df))(a, b)

    # compute log Σ exp(α * log p) = logsumexp(α * log p)
    return jax.scipy.special.logsumexp(new_alpha * log_p)


# ---------- Run ----------
# One base pool reused for all x (variance reduction)
base_key, = jax.random.split(key, 1)
base_Z = jax.random.normal(base_key, (n_samples, d))  # ~ N(0, I)

results = {}
for nu in nus:
    if jnp.isinf(nu):
        # You can either use IS like others, or the exact closed form:
        y = neg_log_expect_normal(d, C, x_grid)
        # If you want IS as well for ν=∞, uncomment:
        # y = neg_log_mgf_IS_for_nu(base_Z, nu, C, x_grid)
    else:
        y = neg_log_mgf_IS_for_nu(base_Z, float(nu), False, C, x_grid)
    z = y + 0.01* log_sum_student_interval_prob_alpha(0.5, 0.1, 2, jnp.sqrt((nu - 2.0) / nu), nu, d) 
    results[nu] = y / C

# ---------- Plot ----------
plt.figure(figsize=(12, 8))
for nu in nus:
    label = "Normal (ν=∞)" if jnp.isinf(nu) else f"ν={int(nu)}"
    plt.plot(x_grid, results[nu], label=label, linewidth=2)

plt.xlabel("Task shift")
plt.ylabel("Theory bound on ICL error")
plt.title("Theory bound for ICL error as a function of task shift")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("visu2.png", dpi=150, bbox_inches='tight')
plt.show()


C_grid = jnp.linspace(1, C, 200)
x_small_grid = jnp.linspace(0.0, 3.0, 10)
neg_log_expect_normal_vmaped = jax.vmap(partial(neg_log_expect_normal, d), in_axes=(0, None))
neg_log_mgf_IS_for_nu_vmaped = jax.vmap(partial(neg_log_mgf_IS_for_nu, base_Z), in_axes=(None, None, 0, None))


for nu in nus:
    key, subkey = jax.random.split(key)
    plt.figure(figsize=(12, 8))
    if jnp.isinf(nu):
        all_y_vals = neg_log_expect_normal_vmaped(C_grid, x_small_grid)
    else:
        all_y_vals = neg_log_mgf_IS_for_nu_vmaped(float(nu), False, C_grid, x_small_grid)
    all_y_vals = all_y_vals / C_grid[:, None]  # normalize by C
    for i, x in enumerate(x_small_grid):
        plt.plot(C_grid.flatten(), all_y_vals[:, i], label=f"Shift={x:.1f}", linewidth=2)

    title_str = "Normal (ν=∞)" if jnp.isinf(nu) else f"ν={int(nu)}"
    plt.xlabel("Context length")
    plt.ylabel("Theory bound on ICL error")
    plt.title(fr"Theory bound for ICL error as a function of context length for {title_str}")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"visu2_nu_{int(nu) if not jnp.isinf(nu) else 'inf'}.png", dpi=150, bbox_inches='tight')
