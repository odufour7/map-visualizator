import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

from shapely import Polygon
import glob
from plotly.graph_objs import Figure

import pedpy


def plot_trajectories(
    trajectory_data: pedpy.TrajectoryData,
    framerate: int,
    uid: int,
    show_direction: int,
    walkable_area: pedpy.WalkableArea,
) -> go.Figure:
    fig = go.Figure()
    c1, c2, c3 = st.columns((1, 1, 1))
    data = trajectory_data.data
    num_agents = len(np.unique(data["id"]))
    colors = {
        1: "magenta",  # Assuming 1 is for female
        2: "green",  # Assuming 2 is for male
        3: "black",  # non binary
        4: "blue",
    }
    x_exterior, y_exterior = walkable_area.polygon.exterior.xy
    x_exterior = list(x_exterior)
    y_exterior = list(y_exterior)

    directions = assign_direction_number(data)
    # For each unique id, plot a trajectory
    if uid is not None:
        df = data[data["id"] == uid]
        direction = directions.loc[directions["id"] == uid, "direction_number"].iloc[0]

        color_choice = colors[direction]
        fig.add_trace(
            go.Scatter(
                x=df["x"][::framerate],
                y=df["y"][::framerate],
                line=dict(color=color_choice),
                marker=dict(color=color_choice),
                mode="lines",
                name=f"ID {uid}",
            )
        )
    else:
        for uid, df in data.groupby("id"):

            direction = directions.loc[
                directions["id"] == uid, "direction_number"
            ].iloc[0]

            if show_direction is None:
                color_choice = colors[direction]
                fig.add_trace(
                    go.Scatter(
                        x=df["x"][::framerate],
                        y=df["y"][::framerate],
                        line=dict(color=color_choice),
                        marker=dict(color=color_choice),
                        mode="lines",
                        name=f"ID {uid}",
                    )
                )
            else:
                if direction == show_direction:
                    color_choice = colors[direction]
                    fig.add_trace(
                        go.Scatter(
                            x=df["x"][::framerate],
                            y=df["y"][::framerate],
                            line=dict(color=color_choice),
                            marker=dict(color=color_choice),
                            mode="lines",
                            name=f"ID {uid}",
                        )
                    )

    # geometry
    fig.add_trace(
        go.Scatter(
            x=x_exterior,
            y=y_exterior,
            mode="lines",
            line=dict(color="red"),
            name="geometry",
        )
    )
    xmin = np.min(x_exterior) - 0.1
    xmax = np.max(x_exterior) + 0.1
    ymin = np.min(y_exterior) - 0.1
    ymax = np.max(y_exterior) + 0.1
    count_direction = ""
    for direction in [1, 2, 3, 4]:
        count = directions[directions["direction_number"] == direction].shape[0]
        count_direction += "Direction: " + str(direction) + ": " + str(count) + ". "
    fig.update_layout(
        title=f" Trajectories: {num_agents}. {count_direction}",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y"),  # , range=[xmin, xmax]),
        yaxis=dict(scaleratio=1),  # , range=[ymin, ymax]),
        showlegend=False,
    )
    return fig


def plot_time_series(density: pd.DataFrame, speed: pd.DataFrame, fps: int) -> go.Figure:

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            rf"$\mu= {np.mean(density.density):.2f}\; \pm {np.std(density.density):.2f}\; 1/m^2$",
            rf"$\mu= {np.mean(speed):.2f}\;\pm {np.std(speed):.2f}\; m/s$",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=density.index / fps,
            y=density.density,
            line=dict(color="blue"),
            marker=dict(color="blue"),
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=speed.index / fps,
            y=speed,
            line=dict(color="blue"),
            marker=dict(color="blue"),
            mode="lines",
        ),
        row=1,
        col=2,
    )

    rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    rmax = 5  # np.max(data["instantaneous_density"]) + 0.5
    vmax = np.max(speed) + 0.5
    fig.update_layout(
        xaxis_title=r"$t\; / s$",
        title_font=dict(size=20),
        showlegend=False,
    )
    fig.update_yaxes(
        range=[rmin, rmax],
        title_text=r"$\rho\; /\; 1/m^2$",
        title_font=dict(size=20),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[rmin, vmax],
        title_text=r"$v\; /\; m/s$",
        title_font=dict(size=20),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=r"$t\; / s$", title_font=dict(size=20), row=1, col=2)
    fig.update_xaxes(title_text=r"$t\; / s$", title_font=dict(size=20), row=1, col=1)
    return fig


def plot_fundamental_diagram(
    country, density: pd.DataFrame, speed: pd.DataFrame
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=density[::50],
            y=speed[::50],
            marker=dict(color="blue"),
            mode="markers",
        ),
    )

    # rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    # rmax = np.max(density) + 0.5
    vmin = np.min(speed) - 0.05
    vmax = np.max(speed) + 0.05
    fig.update_layout(
        title=f"Country: {country}",
        xaxis_title="Density / 1/m/m",
        showlegend=False,
    )
    fig.update_yaxes(range=[vmin, vmax], title_text="Speed / m/s")
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
    )

    return fig


def plot_fundamental_diagram_all(country_data) -> go.Figure:
    fig = go.Figure()

    rmax = -1
    vmax = -1

    colors_const = ["blue", "red", "green", "magenta", "black"]
    marker_shapes = ["circle", "square", "diamond", "cross", "x-thin"]  # Example shapes

    colors = {}
    for country, color in zip(country_data.keys(), colors_const):
        colors[country] = color

    for i, (country, (density, speed)) in enumerate(country_data.items()):
        fig.add_trace(
            go.Scatter(
                x=density[::50],
                y=speed[::50],
                marker=dict(
                    color=colors[country],
                    opacity=0.5,
                    symbol=marker_shapes[i % len(marker_shapes)],
                ),
                mode="markers",
                name=f"{country}",
                showlegend=True,
            )
        )
        rmax = max(rmax, np.max(density))
        vmax = max(vmax, np.max(speed))
        vmin = min(vmax, np.min(speed))

    vmax += 0.05
    rmax += 0.05
    vmin -= 0.05

    # vmax = 2.0
    fig.update_yaxes(range=[vmin, vmax], title_text=r"$v\; / \frac{m}{s}$")
    fig.update_xaxes(
        title_text=r"$\rho / m^{-2}$",
        scaleanchor="y",
        scaleratio=1,
    )

    return fig


def plot_x_y(x, y, title, xlabel, ylabel, threshold=0):

    x = np.unique(x)
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"<b>{title}</b>"],
        x_title=xlabel,
        y_title=ylabel,
    )
    if threshold:
        trace_threshold = go.Scatter(
            x=[x[0], x[-1]],
            y=[threshold, threshold],
            mode="lines",
            showlegend=True,
            name="Social Distance = 1.5 m",
            line=dict(width=4, dash="dash", color="gray"),
        )
        fig.append_trace(trace_threshold, row=1, col=1)

    trace = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        showlegend=False,
        line=dict(width=3, color="blue"),
        fill="none",
    )

    fig.append_trace(trace, row=1, col=1)
    return fig


def assign_direction_number(agent_data):
    """
    Assigns a direction number to each agent based on their main direction of motion.

    Parameters:
    - agent_data (DataFrame): A DataFrame with columns 'id', 'frame', 'x', 'y', representing
      agent IDs, frame numbers, and their positions at those frames.

    Returns:
    - A DataFrame with an additional 'direction_number' column.
    """
    # Group by agent ID and calculate the difference in position
    direction_numbers = []
    for agent_id, group in agent_data.groupby("id"):
        start_pos = group.iloc[0]  # Starting position
        end_pos = group.iloc[-1]  # Ending position

        delta_x = end_pos["x"] - start_pos["x"]
        delta_y = end_pos["y"] - start_pos["y"]

        # Determine primary direction of motion
        if abs(delta_x) > abs(delta_y):
            # Motion is primarily horizontal
            direction_number = (
                3 if delta_x > 0 else 4
            )  # East if delta_x positive, West otherwise
        else:
            # Motion is primarily vertical
            direction_number = (
                1 if delta_y > 0 else 2
            )  # North if delta_y positive, South otherwise

        direction_numbers.append((agent_id, direction_number))

    # Create a DataFrame from the direction numbers
    direction_df = pd.DataFrame(direction_numbers, columns=["id", "direction_number"])

    # Merge the direction DataFrame with the original agent_data DataFrame
    # result_df = pd.merge(agent_data, direction_df, on='id')

    return direction_df


def show_fig(fig: Figure, html: bool = False, height: int = 500) -> None:
    """Workaround function to show figures having LaTeX-Code.

    Args:
        fig (Figure): A Plotly figure object to display.
        html (bool, optional): Flag to determine if the figure should be shown as HTML. Defaults to False.
        height (int, optional): Height of the HTML component if displayed as HTML. Defaults to 500.

    Returns:
        None
    """
    if not html:
        st.plotly_chart(fig)
    else:
        st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=height)