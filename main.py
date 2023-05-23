import vtk
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkRenderingCore import vtkLight

import vtkmodules.vtkCommonDataModel as vtkCommonDataModel
import numpy as np

colors = vtkNamedColors()


def main():
    # Read DEM data using numpy
    dem_filename = "EarthEnv-DEM90_N60E010.bil"
    dem_data = np.fromfile(dem_filename, dtype=np.int16)
    dem_data = dem_data.reshape((6000, 6000)).astype(float)

    # Create a structured grid from the DEM data
    grid = vtkCommonDataModel.vtkStructuredGrid()
    grid.SetDimensions(6000, 6000, 1)

    vtk_points = vtk.vtkPoints()
    vtk_scalars = vtk.vtkFloatArray()

    # Assign the DEM data to the scalars array
    for x in range(6000):
        for y in range(6000):
            vtk_points.InsertNextPoint(x, y, dem_data[x, y])
            vtk_scalars.InsertNextValue(dem_data[x, y])

    vtk_structured_grid = vtk.vtkStructuredGrid()
    vtk_structured_grid.SetDimensions(6000, 6000, 1)
    vtk_structured_grid.SetPoints(vtk_points)
    vtk_structured_grid.GetPointData().SetScalars(vtk_scalars)
    vtk_structured_grid.Modified()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(vtk_structured_grid)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Set up the renderer
    renderer = vtkmodules.vtkRenderingCore.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("White"))

    # Add a light source to the scene
    light = vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(1, 1, 1)
    renderer.AddLight(light)

    # Set up the render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName('VTK - Labo 4')

    # Set up the interactor
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())
    interactor.Initialize()
    interactor.Start()


if __name__ == '__main__':
    main()
