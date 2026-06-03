# Librerías importadas
import numpy as np              # Librería para cálculo numérico
import matplotlib.pyplot as plt # Librería para poder dibujar gráficas
from scipy.integrate import odeint
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Constantes físicas y parámetros
Cf = 1.5      # kJ/kg/K
Ca = 1.0      # kJ/kg/K
H = 4000.0    # kJ/kg
h = 0.05      # kJ/(m^3 s K)
rhof = 800.0  # kg/m^3
Rf0 = 0.01    # porosity / fuel fraction
Tinf = 300.0  # K
A = 0.003     # 1/s
Tpc = 600.0   # K
R = 287.058   # J cte de los gases ideales
Patm = 1.0e5  # Pa = J/m^3

EMISSIVITY = 0.9
SIGMA = 5.670374419e-11   # kJ/(m^2 s K^4)
crad = 1.0

# Modos de operación
VOLATILIZATION_PRODUCTS = 0   # 0 (Modelo Simplificado), 1 (Modelo Completo)
COOLING_MODEL = 0             # 0 (convección) o 1 (convección + radiación)
REACTION_TYPE = 0             # no se cambia


# Selección automática del nombre del caso
def case_name():
    if VOLATILIZATION_PRODUCTS == 0:
        if COOLING_MODEL == 0:
            return "Simplificado"
        elif COOLING_MODEL == 1:
            return "Simplificado_ConvRad"

    elif VOLATILIZATION_PRODUCTS == 1:
        if COOLING_MODEL == 0:
            return "Completo"
        elif COOLING_MODEL == 1:
            return "CompletoConvRad"

    raise ValueError("Combinación de modelos no válida.")


def save_figure(fig, filename):
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {output_path}")


# Definiciones para rho_a, rho_cp
def rhoa(T):
    if VOLATILIZATION_PRODUCTS == 0:
        rhoa = 1.0      # cte
    elif VOLATILIZATION_PRODUCTS == 1:
        rhoa = Patm / (R * T)  # variable con T
    return rhoa


def rho_cp(T, Y):
    return rhoa(T) * Ca * (1 - Rf0) + rhof * Rf0 * Cf


# Definición de Psi
def psi_rate(A, T, Tpc):
    T = np.asarray(T)
    s = np.where(T < Tpc, 0.0, 1.0)

    if REACTION_TYPE == 0:
        return s * A
    else:
        return s * A * np.exp(-Tpc / T)


# Término de enfriamiento
def cooling_term(T):
    """
    Término E(T,Y) asociado a las pérdidas por enfriamiento.
    """
    if COOLING_MODEL == 1:    # Enfriamiento como convección + radiación
        b = crad * EMISSIVITY * SIGMA * (T * T + Tinf * Tinf) * (T + Tinf)
        return (h + b) * (T - Tinf)
    else:                     # Enfriamiento como convección
        return h * (T - Tinf)


# Término de enfriamiento por radiación
def radiation_term(T):
    return crad * EMISSIVITY * SIGMA * (T * T + Tinf * Tinf) * (T + Tinf) * (T - Tinf)


# Término de enfriamiento por convección
def convection_term(T):
    return h * (T - Tinf)


# Término de productos de volatilización
def volatilization_term(T, Y):
    """
    Término C(T,Y) asociado a los productos de volatilización.
    """
    T = np.asarray(T)
    Y = np.asarray(Y)

    if VOLATILIZATION_PRODUCTS == 1:
        psi = psi_rate(A, T, Tpc)
        return rhof * Rf0 * T * (Cf - Ca) * psi * Y

    return np.zeros_like(T, dtype=float)


# Término de reacción
def reaction_term(T, Y):
    """
    Término R(T,Y) asociado a la reacción de combustión.
    """
    return psi_rate(A, T, Tpc) * rhof * Rf0 * H * Y


# Sistema de ecuaciones
def rhs_T(T, Y):
    return (reaction_term(T, Y) - cooling_term(T) + volatilization_term(T, Y)) / rho_cp(T, Y)


def rhs_Y(T, Y):
    return -psi_rate(A, T, Tpc) * Y


def system(y, t):
    T, Y = y
    return [rhs_T(T, Y), rhs_Y(T, Y)]


# Condiciones iniciales
T0 = 670.0
Y0 = 1.0
y0 = [T0, Y0]


# ---------------------------------------------------------------------------------------------------------
# SIMULACIÓN PARA GRÁFICAS DE FLUJOS TÉRMICOS
# ---------------------------------------------------------------------------------------------------------

# Discretización temporal
t = np.linspace(0, 2000, 5000)

# Resolución por integración
sol = odeint(system, y0, t)
T_sol = sol[:, 0]
Y_sol = sol[:, 1]

# Evaluación de las soluciones
cooling_t = cooling_term(T_sol)
convection_t = convection_term(T_sol)
radiation_t = radiation_term(T_sol)
volatilization_t = volatilization_term(T_sol, Y_sol)
reaction_t = reaction_term(T_sol, Y_sol)
net_rate = reaction_t - cooling_t + volatilization_t


# ---------------------------------------------------------------------------------------------------------
# GRÁFICA DE FLUJOS TÉRMICOS
# ---------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

ax.set_xlim(0, 1600)
ax.plot(t, reaction_t, label="Término de reacción", color="#E74C3C", lw=2)
ax.plot(t, -cooling_t, label="Término de enfriamiento", color="#3498DB", lw=1.5, ls="--")

# Para mostrar por separado convección y radiación, descomentar:
# ax.plot(t, -convection_t, label="Término de enfriamiento \npor convección", color="#5E95FF", lw=1.5, ls="--")
# ax.plot(t, -radiation_t, label="Término de enfriamiento \npor radiación", color="#1263FF", lw=1.5, ls="--")

if VOLATILIZATION_PRODUCTS == 1:
    ax.plot(
        t,
        volatilization_t,
        label="Término de variación \ntemporal del combustible",
        color="green",
        lw=1.5,
        ls=":"
    )

ax.plot(t, net_rate, label="Tasa Neta", color="#8E44AD", lw=2, alpha=0.6)

ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Flujo térmico (kW/m$^2$)")  # Se normaliza la tercera dimensión por una longitud l = 1 m.
ax.legend(loc="upper right", bbox_to_anchor=(1, 1.03), frameon=False)
ax.grid(False)

fig.tight_layout()
save_figure(fig, f"ContribucionTerminos_{case_name()}.png")


# ---------------------------------------------------------------------------------------------------------
# SIMULACIÓN PARA EVOLUCIÓN TEMPORAL Y ESPACIO DE FASES
# ---------------------------------------------------------------------------------------------------------

# Discretización temporal
t = np.linspace(0, 2000, 20001)

# Resolución del sistema
sol = odeint(system, y0, t)


# ---------------------------------------------------------------------------------------------------------
# GRÁFICA DE EVOLUCIÓN TEMPORAL DE T E Y
# ---------------------------------------------------------------------------------------------------------

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(t, sol[:, 0], "r", label="T(t)")
ax1.set_xlabel("t")
ax1.set_ylabel("T(K)")
ax1.legend()

ax2.plot(t, sol[:, 1], "r", label="Y(t)")
ax2.set_xlabel("t")
ax2.set_ylabel("Y")
ax2.legend()

fig.tight_layout()
save_figure(fig, f"EvolucionTemporal_{case_name()}.png")


# ---------------------------------------------------------------------------------------------------------
# GRÁFICA DE LA TRAYECTORIA INDIVIDUAL EN EL ESPACIO DE FASES
# ---------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(sol[:, 0], sol[:, 1], "k")
ax.plot(sol[0, 0], sol[0, 1], "or", label="initial")
ax.plot([200, np.max(sol[:, 0])], [1, 1], "k--")
ax.plot([Tinf, Tinf], [0, 1], "g--", label="$T_{inf}$")
ax.plot([Tpc, Tpc], [0, 1], "b--", label="$T_{pc}$")

ax.set_xlabel("T(K)")
ax.set_ylabel("Y")
ax.set_ylim([0, 1.1])
ax.legend()

fig.tight_layout()
save_figure(fig, f"Trayectoria_EspacioFases_{case_name()}.png")


# ---------------------------------------------------------------------------------------------------------
# GRÁFICA DEL ESPACIO DE FASES
# ---------------------------------------------------------------------------------------------------------

N = 100
T_start, T_end = 601, 1600
Y_start, Y_end = 0.0, 1.0

Tpts = np.linspace(T_start, T_end, N)
Ypts = np.linspace(Y_start, Y_end, N)
TT, YY = np.meshgrid(Tpts, Ypts)

uu = rhs_T(TT, YY)
vv = rhs_Y(TT, YY)

norm = TwoSlopeNorm(vmin=np.min(uu), vcenter=0.0001, vmax=np.max(uu))

fig, ax = plt.subplots(figsize=(6, 4))
ax.streamplot(TT, YY, uu, vv, density=1.4, color=uu, cmap="seismic", norm=norm)

print(np.max(uu), np.min(uu))

T_start2, T_end2 = 200, 599
Tpts2 = np.linspace(T_start2, T_end2, N)
TT2, YY2 = np.meshgrid(Tpts2, Ypts)

uu2 = rhs_T(TT2, YY2)
vv2 = rhs_Y(TT2, YY2)

norm2 = TwoSlopeNorm(vmin=np.min(uu2), vcenter=0.0, vmax=np.max(uu2))

ax.streamplot(TT2, YY2, uu2, vv2, density=1.4, color=uu2, cmap="seismic", norm=norm2)


# Ecuación de la curva de temperaturas máximas según el modo de operación
def Y_tipping(T):
    T = np.asarray(T, dtype=float)

    if COOLING_MODEL == 1:
        b = crad * EMISSIVITY * SIGMA * (T**2 + Tinf**2) * (T + Tinf)
        q_num = (h + b) * (T - Tinf)
    else:
        q_num = h * (T - Tinf)

    psi = psi_rate(A, T, Tpc)

    if VOLATILIZATION_PRODUCTS == 0:
        denom = psi * rhof * Rf0 * H
    elif VOLATILIZATION_PRODUCTS == 1:
        denom = psi * rhof * Rf0 * (H + T * (Cf - Ca))

    with np.errstate(divide="ignore", invalid="ignore"):
        Ytip = q_num / denom

    return np.where(T >= Tpc, Ytip, np.nan)


Ttip = np.linspace(601, T_end, 400)
Ytip = Y_tipping(Ttip)

ax.plot([Tinf, Tinf], [0, 1], "g--", label="$T_{\infty}$")
ax.plot([Tpc, Tpc], [0, 1], "--", color="tab:orange", label="$T_{pc}$")
ax.plot(Ttip, Ytip, "--", color="tab:gray", label="Curva de T$_{máx}$")

ax.set_xlabel("T (K)")
ax.set_ylabel("Y")
ax.set_xlim([T_start2, T_end])
ax.set_ylim([0, 1])
ax.legend(loc="upper right")

ax.tick_params(axis="y", which="both", right=True, labelright=True)
ax.tick_params(axis="x", which="both", top=True)

fig.tight_layout()
save_figure(fig, f"EspacioFases_{case_name()}.png")