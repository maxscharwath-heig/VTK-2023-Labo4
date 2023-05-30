import vtk
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindowInteractor,
)

colors = vtkNamedColors()


def read_structured_grid(filename):
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def main():
    # Read the structured grid
    vtk_structured_grid = read_structured_grid('DEM90.vtk')

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(vtk_structured_grid)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a texture map
    texture_map = vtk.vtkTexture()
    image_reader = vtk.vtkJPEGReader()
    image_reader.SetFileName("glider_map.jpg")
    image_reader.Update()
    texture_map.SetInputDataObject(image_reader.GetOutput())

    # Apply the texture map to the actor
    actor.SetTexture(texture_map)

    # Set up the renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("White"))

    # Set up the render window
    render_window = vtk.vtkRenderWindow()
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
