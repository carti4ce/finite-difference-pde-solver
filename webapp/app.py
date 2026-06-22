"""Streamlit webapp for PDE solver demo"""

import streamlit as st
import numpy as np
from initial_conditions import create_initial_condition
from solvers import solve_heat_equation, solve_laplace_equation
from finite_differences.utils import quick_plot_2d, animated_plot_2d
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="PDE Solver", layout="wide")
st.title("Finite Difference PDE Solver")

# UI: Equation and dimension selection
col1, col2 = st.columns(2)

with col1:
    equation = st.segmented_control(
        "**Equation:**",
        ["Heat Equation", "Laplace's Equation"],
        selection_mode="single",
    )

with col2:
    if equation:
        dimension = st.segmented_control(
            "**Dimension:**",
            ["1D", "2D"],
            selection_mode="single",
        )
    else:
        dimension = None

# UI: Grid bounds
if dimension:
    st.subheader("Domain Bounds")
    col_bounds_x, col_bounds_y = st.columns(2)

    with col_bounds_x:
        x_bounds = st.slider(
            "X bounds",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 1.0),
            step=0.1,
        )

    if dimension == "2D":
        with col_bounds_y:
            y_bounds = st.slider(
                "Y bounds",
                min_value=0.0,
                max_value=10.0,
                value=(0.0, 1.0),
                step=0.1,
            )
    else:
        y_bounds = (0.0, 1.0)

    # UI: Grid resolution
    st.subheader("Grid Resolution")
    col_res_x, *col_res_y = st.columns(2 if dimension == "2D" else [1])

    with col_res_x:
        nx = st.slider("Number of points (x)", min_value=10, max_value=200, value=50)

    if dimension == "2D":
        with col_res_y[0]:
            ny = st.slider("Number of points (y)", min_value=10, max_value=200, value=50)
    else:
        ny = 1



    # ****              Heat Equation Logic            ****



    if equation == "Heat Equation":
        st.subheader("Heat Equation Parameters")

        col_alpha, col_ic = st.columns(2)

        with col_alpha:
            alpha = st.slider(
                "Diffusivity (α)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )

        with col_ic:
            ic_shape = st.selectbox(
                "Initial condition shape",
                ["Gaussian", "Uniform", "Sine Wave", "Random"],
            )

        ic_intensity = st.slider(
            "Initial condition intensity",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )

        # Time stepping
        st.subheader("Time Stepping")

        col_preset, col_override = st.columns(2)

        with col_preset:
            preset = st.selectbox(
                "Preset",
                ["Coarse (10 steps)", "Medium (50 steps)", "Fine (200 steps)"],
            )
            preset_map = {
                "Coarse (10 steps)": 10,
                "Medium (50 steps)": 50,
                "Fine (200 steps)": 200,
            }
            num_steps = preset_map[preset]

        with col_override:
            use_custom = st.checkbox("Custom time stepping")

        if use_custom:
            col_dt, col_steps = st.columns(2)
            with col_dt:
                dt_custom = st.number_input("dt (time step)", value=0.001, format="%.6f")
            with col_steps:
                num_steps_custom = st.number_input(
                    "Number of steps", value=50, min_value=1
                )
            num_steps = num_steps_custom
            dt = dt_custom
        else:
            dt = None

        anim_container = st.container()
        with anim_container:
            generate_animation = st.segmented_control(
                "**Generate Animation:**",
                ["True", "False"],
                selection_mode="single",
    )

        if st.button("Solve Heat Equation", key="generate_heat"):
            with st.spinner("Solving..."):
                # Create initial condition
                grid_shape = (nx,) if dimension == "1D" else (nx, ny)
                grid_bounds = x_bounds if dimension == "1D" else (x_bounds, y_bounds)

                ic_shape_map = {
                    "Gaussian": "gaussian",
                    "Uniform": "uniform",
                    "Sine Wave": "sine_wave",
                    "Random": "random",
                }

                u0 = create_initial_condition(
                    grid_shape=grid_shape,
                    grid_bounds=grid_bounds,
                    shape=ic_shape_map[ic_shape],
                    intensity=ic_intensity,
                    dimension=dimension,
                )

                # Solve
                time_array, solution_history = solve_heat_equation(
                    dimension=dimension,
                    grid_bounds=grid_bounds,
                    initial_condition_array=u0,
                    alpha=alpha,
                    dt=dt,
                    num_steps=num_steps,
                    bc_value=0.0,
                )

            # Display results
            st.success("Solution computed!")

            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.subheader("Final Solution")
                fig_final, _ = quick_plot_2d(
                    solution_history[-1],
                    vmin_global=solution_history.min(),
                    vmax_global=solution_history.max(),
                    title="Final Temperature Distribution",
                    xlabel="x",
                    ylabel="y" if dimension == "2D" else "",
                    cbarlabel="Temperature",
                )
                st.pyplot(fig_final)
                plt.close(fig_final)

            with col_plot2:
                st.subheader("Initial Condition")
                fig_initial, _ = quick_plot_2d(
                    solution_history[0],
                    vmin_global=solution_history.min(),
                    vmax_global=solution_history.max(),
                    title="Initial Temperature Distribution",
                    xlabel="x",
                    ylabel="y" if dimension == "2D" else "",
                    cbarlabel="Temperature",
                )
                st.pyplot(fig_initial)
                plt.close(fig_initial)

            # Animation

            if generate_animation == "True":
                st.subheader("Solution Evolution")
                with st.spinner("Creating animation..."):
                    try:
                        anim, fig = animated_plot_2d(
                            solution_history,
                            title="Heat Equation Evolution",
                            xlabel="x",
                            ylabel="y" if dimension == "2D" else "",
                            cbarlabel="Temperature",
                            interval_ms=50,
                            verbose=True
                        )

                        # Save animation to bytes buffer
                        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                            tmp_path = tmp.name

                        anim.save(tmp_path, writer="pillow", fps=20)

                        with open(tmp_path, "rb") as f:
                            gif_bytes = f.read()

                        os.remove(tmp_path)

                        # Display inline
                        st.image(gif_bytes, caption="Heat Equation Evolution")

                        # Offer Download
                        st.download_button(
                            label="Download Animation (GIF)",
                            data=gif_bytes,
                            file_name="pde_solution.gif",
                            mime="image/gif",
                        )
                        st.info("Animation saved. Click download button to get the GIF.")
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Animation creation failed: {e}")



    # ****              Laplace's Equation Logic            ****



    elif equation == "Laplace's Equation":
        st.subheader("Boundary Conditions (Note: constant boundary conditions must be equal for PDE to be well-posed, function BC coming soon)")

        if dimension == "1D":
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                bc_x_left = st.number_input(
                    "Left boundary (x=x_min)", value=0.0, step=0.1
                )
            with col_x2:
                bc_x_right = st.number_input(
                    "Right boundary (x=x_max)", value=0.0, step=0.1
                )
            bc_dict = {"x_left": bc_x_left, "x_right": bc_x_right}

        else:  # 2D
            col1, col2 = st.columns(2)
            with col1:
                bc_x_left = st.number_input(
                    "Left boundary (x=x_min)", value=0.0, step=0.1
                )
                bc_x_right = st.number_input(
                    "Right boundary (x=x_max)", value=0.0, step=0.1
                )
            with col2:
                bc_y_bottom = st.number_input(
                    "Bottom boundary (y=y_min)", value=0.0, step=0.1
                )
                bc_y_top = st.number_input(
                    "Top boundary (y=y_max)", value=0.0, step=0.1
                )
            bc_dict = {
                "x_left": bc_x_left,
                "x_right": bc_x_right,
                "y_bottom": bc_y_bottom,
                "y_top": bc_y_top,
            }

        if st.button("Solve Laplace's Equation"):
            with st.spinner("Solving..."):
                grid_bounds = x_bounds if dimension == "1D" else (x_bounds, y_bounds)

                solution = solve_laplace_equation(
                    dimension=dimension,
                    grid_bounds=grid_bounds,
                    boundary_conditions=bc_dict,
                )

            st.success("Solution computed!")

            st.subheader("Solution")
            fig, ax = quick_plot_2d(
                solution,
                title="Laplace Equation Solution",
                xlabel="x",
                ylabel="y" if dimension == "2D" else "",
                cbarlabel="Potential",
            )
            st.pyplot(fig)
            plt.close(fig)

else:
    st.info("Select an equation and dimension to get started!")
