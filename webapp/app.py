import streamlit as st

eq_choices = ["Heat Equation", "Laplace's Equation"]
dimension_choices = ["1D", "2D"]
boundary_choices = ["Dirichlet", "Neumann (WIP)"]

equation_cont, dimension_cont = st.columns(2)

selected_eq = equation_cont.segmented_control(
    "Equation: ", eq_choices, selection_mode="single"
)

selected_dim = boundary_type = initial_cond = alpha = None
boundary_vals = []
grid = []

grid_cont_x, grid_cont_y = st.columns(2)
boundary_cont = st.container()
boundary_cont_x, boundary_cont_y = st.columns(2)
initial_cond = st.empty()

if selected_eq != None:
    selected_dim = dimension_cont.segmented_control("Dimension: ", dimension_choices, selection_mode="single")

    if selected_dim != None:

        equation_cont.write("Please select bounds for cartesian grid:")
        grid.append(grid_cont_x.slider("X bounds: ", min_value=0.0, max_value=10.0, value=(0.0,1.0)))
        if selected_dim == "2D": grid.append(grid_cont_y.slider("Y bounds: ", min_value=1.0, max_value=10.0, value=(0.0,1.0)))

        with boundary_cont.container():
            boundary_type = st.segmented_control("Type of boundary condition", boundary_choices, selection_mode="single")

        if boundary_type == "Dirichlet":
                with boundary_cont.container(): st.write("Enter boundary conditions for fixed endpoints:")
                boundary_vals.append(boundary_cont_x.number_input("X1", value=0.0))
                boundary_vals.append(boundary_cont_x.number_input("X2", value=0.0))
                if selected_dim == "2D":
                    boundary_vals.append(boundary_cont_y.number_input("Y1", value=0.0))
                    boundary_vals.append(boundary_cont_y.number_input("Y2", value=0.0))