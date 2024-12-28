# Import necessary modules
from flask import Flask, render_template, request, url_for
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from sklearn.impute import SimpleImputer
import sediment_type
import ref_water
import os

# Set environment to avoid using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# Load scaler and RF model for single particle
single_scaler = joblib.load('Particles_single_checkpoint/Particles_7features_scaler.pkl')
single_rf_model = joblib.load('Particles_single_checkpoint/Particles_7features_rf_model.pkl')

# Load scaler and RF model for multiple particles
multiple_scaler = joblib.load('Particles_single_checkpoint_1600/Particles_5features_scaler.pkl')
multiple_rf_model = joblib.load('Particles_single_checkpoint_1600/Particles_5features_rf_model.pkl')

# Constants
m = 6.0  # power-law coefficient
kappa = 0.41  # von Karman constant

# Set font settings
FS = 16
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = FS
plt.rcParams['axes.labelsize'] = FS
plt.rcParams['axes.titlesize'] = FS
plt.rcParams['legend.fontsize'] = FS
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS
plt.rcParams['figure.dpi'] = 500

def calculate_sediment_properties(velocity, water_depth, diameter):
    Um = (m + 1) / m * velocity
    shear_velocity = Um * kappa / m
    water = ref_water.water(20)
    sand_particle = sediment_type.sand(water, diameter)
    settling_velocity = np.abs(sand_particle.ws[0])
    Ro_beta = min(3, 1 + 2 * (settling_velocity / shear_velocity) ** 2)
    Ro = settling_velocity / (Ro_beta * kappa * shear_velocity)

    if Ro < 2.5:
        return shear_velocity, settling_velocity, Ro, False
    return shear_velocity, settling_velocity, Ro, True

@app.route("/", methods=["GET", "POST"])
def index():
    calculated_values = None
    plot_filename = None
    velocity, water_depth, diameter = None, None, None
    mode = "single"
    suspension_message = None

    if request.method == "POST":
        mode = request.form['mode']
        velocity = request.form.get('feature1')
        water_depth = request.form.get('feature2')
        diameter = request.form.get('feature3', "")

        try:
            velocity = float(velocity) if velocity else None
            water_depth = float(water_depth) if water_depth else None
            diameter = float(diameter) if diameter else None
        except ValueError:
            return "Please enter valid numeric values."

        # Check if parameters are within the valid range
        if (velocity is None or velocity < 0.1 or velocity > 1.0 or
            water_depth is None or water_depth < 0.5 or water_depth > 10.0 or
            diameter is None or diameter < 0.00001 or diameter > 0.02):
            suspension_message = "parameters outside the range! No output plot."
            # Create a simple plot to indicate parameter error
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
            plt.axis('off')
            plot_filename = 'parameter_error_plot.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()
            return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                   calculated_values=None, mode=mode, velocity=velocity,
                                   water_depth=water_depth, diameter=diameter,
                                   suspension_message=suspension_message)

        if mode == "single" and velocity and water_depth and diameter:
            shear_velocity, settling_velocity, Ro, valid = calculate_sediment_properties(velocity, water_depth,
                                                                                         diameter)
            if not valid:
                suspension_message = "particles are suspended! No output plot."
                # Create a simple plot to indicate suspension
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
                plt.axis('off')
                plot_filename = 'suspension_plot.png'
                plt.savefig(f'static/{plot_filename}')
                plt.close()
                return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                       calculated_values=None, mode=mode, velocity=velocity,
                                       water_depth=water_depth, diameter=diameter,
                                       suspension_message=suspension_message)

            calculated_values = {'shear_velocity': shear_velocity, 'settling_velocity': settling_velocity, 'Ro': Ro}
            predict_features = np.array(
                [velocity, water_depth, diameter, shear_velocity, settling_velocity, Ro]).reshape(1, -1)
            predict_features_scaled = single_scaler.transform(predict_features)
            predict_features_imputed = SimpleImputer(strategy='mean').fit_transform(predict_features_scaled)
            rf_predictions = single_rf_model.predict(predict_features_imputed)
            rf_predictions[:, 1]=rf_predictions[:, 1]**2
            # Plot
            scale_rf, shape_rf = np.log(rf_predictions[:, 0]), np.sqrt(
                np.log(1 + (rf_predictions[:, 1] / rf_predictions[:, 0]) ** 2))
            plt.figure(figsize=(8, 6))
            x = np.linspace(0.01, lognorm(scale=scale_rf, s=shape_rf).ppf(0.9999), 1000)
            x2 = np.exp(x)
            plt.plot(x2, lognorm.pdf(x, s=shape_rf, scale=scale_rf) / 3, 'g--', lw=2, label='lognorm_RF')

            plt.text(
                0.9, 0.75,
                f'$\mu$ (RF): {scale_rf[0]:.3f}\n$\sigma$ (RF): {shape_rf[0]:.3f}\n'
                f'Median (RF): {rf_predictions[0, 0]:.3f} m\nVariance (RF): {rf_predictions[0, 1]:.3f} m²',
                transform=plt.gca().transAxes,
                fontsize=FS,
                verticalalignment='top',
                horizontalalignment='right'
            )

            plt.xlabel("Distance (m)")
            plt.ylabel("PDF")
            plt.legend()
            plot_filename = 'plot_single.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()

        elif mode == "multiple" and velocity and water_depth:
            predict_features = np.array([[velocity, water_depth]])
            predict_features_scaled = multiple_scaler.transform(predict_features)
            predict_features_imputed = SimpleImputer(strategy='mean').fit_transform(predict_features_scaled)
            rf_predictions = multiple_rf_model.predict(predict_features_imputed)

            # Plot
            scale_rf, shape_rf = np.log(rf_predictions[:, 0]), np.sqrt(
                np.log(1 + (rf_predictions[:, 1] / rf_predictions[:, 0]) ** 2))
            plt.figure(figsize=(8, 6))
            x = np.linspace(lognorm(scale=scale_rf, s=shape_rf).ppf(0.001),
                            lognorm(scale=scale_rf, s=shape_rf).ppf(0.95), 1000)
            x2 = np.exp(x)
            plt.plot(x2, lognorm.pdf(x, s=shape_rf, scale=scale_rf) / 3, 'g--', lw=2, label='lognorm_RF')

            plt.text(
                0.9, 0.75,
                f'$\mu$ (RF): {scale_rf[0]:.3f}\n$\sigma$ (RF): {shape_rf[0]:.3f}\n'
                f'Median (RF): {rf_predictions[0, 0]:.3f} m\nVariance (RF): {rf_predictions[0, 1]:.3f} m²',
                transform=plt.gca().transAxes,
                fontsize=FS,
                verticalalignment='top',
                horizontalalignment='right'
            )

            plt.xlabel("Distance (m)")
            plt.ylabel("PDF")
            plt.legend()
            plot_filename = 'plot_multiple.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()

        return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                               calculated_values=calculated_values, mode=mode, velocity=velocity,
                               water_depth=water_depth, diameter=diameter, suspension_message=suspension_message)

    return render_template("index.html", plot_url=None, mode=mode, velocity=None, water_depth=None, diameter=None,
                           suspension_message=None)

if __name__ == "__main__":
    app.run(debug=True)
