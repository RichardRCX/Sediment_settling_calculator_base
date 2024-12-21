from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from sklearn.impute import SimpleImputer
import joblib
from tensorflow.keras.models import load_model
import sediment_type
import ref_water
import os
import gc

# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# 加载模型和预处理器 (仅加载一次)
single_scaler = joblib.load('Particles_single_checkpoint/Particles_7features_scaler.pkl')
single_rf_model = joblib.load('Particles_single_checkpoint/Particles_7features_rf_model.pkl')
single_kan_model = load_model('Particles_single_checkpoint/Particles_7features_kan_model.keras')

multiple_scaler = joblib.load('Particles_single_checkpoint_1600/Particles_5features_scaler.pkl')
multiple_rf_model = joblib.load('Particles_single_checkpoint_1600/Particles_5features_rf_model.pkl')
multiple_kan_model = load_model('Particles_single_checkpoint_1600/Particles_5features_kan_model.keras')

# Constants
m = 6.0  # power-law coefficient
kappa = 0.41  # von Karman constant
FS = 18  # Font size for plots

plt.rcParams['font.size'] = FS
plt.rcParams['figure.dpi'] = 200

def calculate_sediment_properties(velocity, water_depth, diameter):
    """计算沉积特性"""
    Um = (m + 1) / m * velocity
    shear_velocity = Um * kappa / m
    water = ref_water.water(20)
    sand_particle = sediment_type.sand(water, diameter)
    settling_velocity = np.abs(sand_particle.ws[0])
    Ro_beta = min(3, 1 + 2 * (settling_velocity / shear_velocity) ** 2)
    Ro = settling_velocity / (Ro_beta * kappa * shear_velocity)
    return shear_velocity, settling_velocity, Ro

@app.route("/", methods=["GET", "POST"])
def index():
    """主页面路由"""
    calculated_values, plot_filename = None, None
    mode = "single"
    velocity, water_depth, diameter = None, None, None

    if request.method == "POST":
        try:
            # 获取输入数据
            mode = request.form.get("mode", "single")
            velocity = float(request.form.get("feature1", 0))
            water_depth = float(request.form.get("feature2", 0))
            diameter = float(request.form.get("feature3", 0)) if mode == "single" else None
        except ValueError:
            return "请输入有效的数字值！"

        # 单粒子预测
        if mode == "single" and velocity and water_depth and diameter:
            shear_velocity, settling_velocity, Ro = calculate_sediment_properties(velocity, water_depth, diameter)
            calculated_values = {'shear_velocity': shear_velocity, 'settling_velocity': settling_velocity, 'Ro': Ro}
            features = np.array([velocity, water_depth, diameter, shear_velocity, settling_velocity, Ro]).reshape(1, -1)
            features_scaled = single_scaler.transform(features)
            features_imputed = SimpleImputer(strategy="mean").fit_transform(features_scaled)

            rf_predictions = single_rf_model.predict(features_imputed)
            kan_predictions = single_kan_model.predict(features_imputed)

            # 绘制分布图
            plot_filename = create_plot(rf_predictions, kan_predictions, "plot_single.png")

        # 多粒子预测
        elif mode == "multiple" and velocity and water_depth:
            features = np.array([[velocity, water_depth]])
            features_scaled = multiple_scaler.transform(features)
            features_imputed = SimpleImputer(strategy="mean").fit_transform(features_scaled)

            rf_predictions = multiple_rf_model.predict(features_imputed)
            kan_predictions = multiple_kan_model.predict(features_imputed)

            # 绘制分布图
            plot_filename = create_plot(rf_predictions, kan_predictions, "plot_multiple.png")

        # 清理未使用变量
        gc.collect()

        return render_template("index.html", plot_url=url_for("static", filename=plot_filename),
                               calculated_values=calculated_values, mode=mode)

    return render_template("index.html", plot_url=None, mode=mode)

def create_plot(rf_predictions, kan_predictions, filename):
    """生成分布图"""
    scale_kan = np.log(kan_predictions[:, 0])
    shape_kan = np.sqrt(np.log(1 + (kan_predictions[:, 1] / kan_predictions[:, 0]) ** 2))
    scale_rf = np.log(rf_predictions[:, 0])
    shape_rf = np.sqrt(np.log(1 + (rf_predictions[:, 1] / rf_predictions[:, 0]) ** 2))

    x_kan = np.linspace(0.01, lognorm(scale=scale_kan, s=shape_kan).ppf(0.999), 500)
    x_rf = np.linspace(0.01, lognorm(scale=scale_rf, s=shape_rf).ppf(0.999), 500)

    plt.figure(figsize=(8, 6))
    plt.plot(x_kan, lognorm.pdf(x_kan, s=shape_kan, scale=scale_kan), 'r-', lw=2, label="KAN")
    plt.plot(x_rf, lognorm.pdf(x_rf, s=shape_rf, scale=scale_rf), 'g--', lw=2, label="RF")
    plt.xlabel("Distance (m)")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    filepath = f"static/{filename}"
    plt.savefig(filepath)
    plt.close()
    return filename

if __name__ == "__main__":
    app.run(debug=True)
