from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from scipy.spatial import distance
import platform

class Triangle(QPolygonF):
    def __init__(self,d=None):
        super(Triangle, self).__init__()

        self.append(QPointF(-d/2.,d/3))
        self.append(QPointF(0.,-d*2/3))
        self.append(QPointF(d/2.,d/3))
        self.append(QPointF(-d/2.,d/3))

class Tetrahedre(vtk.vtkUnstructuredGrid):
    def __init__(self, d=None, pos = None, scale = None):
        vtk.vtkUnstructuredGrid.__init__(self, )
        # Make a tetrahedron.
        points = vtk.vtkPoints()
        points.InsertNextPoint(pos[0]*scale[0]+0   ,pos[1]*scale[1]+0      ,pos[2]*scale[2]+2*d/3)
        points.InsertNextPoint(pos[0]*scale[0]-d/2 ,pos[1]*scale[1]-d/3  ,pos[2]*scale[2]-d/3)
        points.InsertNextPoint(pos[0]*scale[0]+d/2 ,pos[1]*scale[1]-d/3  ,pos[2]*scale[2]-d/3)
        points.InsertNextPoint(pos[0]*scale[0]+0   ,pos[1]*scale[1]+2*d/3  ,pos[2]*scale[2]-d/3)
        points.InsertNextPoint(pos[0]*scale[0]+0   ,pos[1]*scale[1]+0      ,pos[2]*scale[2]+2*d/3)

        self.SetPoints(points)

        ids = vtk.vtkIdList()
        ids.SetNumberOfIds(5)
        for i  in range(5) :
          ids.SetId(i, i)

        cellArray =vtk.vtkCellArray()
        cellArray.InsertNextCell(ids)

        self.SetCells(vtk.VTK_TETRA, cellArray)


class MouseInteractorHighLightActor2(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.parent = parent
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver("RightButtonPressEvent", self.RightButtonPressEvent)
        if platform.system() == 'Darwin':
            self.facteurzoom = 2
        else:
            self.facteurzoom = 1

    def leftButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()
        clickPos = [p * self.facteurzoom for p in clickPos]

        pickerActor = vtk.vtkPropPicker()
        pickerActor.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        # get the new
        NewPickedActor = pickerActor.GetActor()
        if NewPickedActor:

            picker = vtk.vtkCellPicker()
            picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
            pos = [p/self.parent.scaling_x for p in picker.GetPickPosition()]
            CellPosition = self.parent.parent.CellPosition

            CellDistances = distance.cdist(CellPosition, [pos], 'euclidean')

            self.parent.cellReceOfInterest = None
            self.parent.cellEmitOfInterest = np.argmin(CellDistances)
            if self.GetInteractor().GetControlKey() and len(self.parent.selected_cells) >0:
                cell = np.argmin(CellDistances)
                if cell in self.parent.selected_cells:
                    self.parent.selected_cells.remove(cell)
                elif self.parent.List_Neurone_type[cell] == self.parent.List_Neurone_type[self.parent.selected_cells[0]]:
                    self.parent.selected_cells.append(cell)
            else:
                self.parent.selected_cells = [np.argmin(CellDistances)]
            if self.parent.globalGUI:
                self.parent.send_selected_cell()
                self.parent.draw_Lines()
            else:
                if len(self.parent.selected_cells) == 1:
                    self.parent.send_selected_cell_2()
            self.parent.draw_BoundingBox()

        self.OnLeftButtonDown()
        return

    def RightButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()
        clickPos = [p * self.facteurzoom for p in clickPos]
        pickerActor = vtk.vtkPropPicker()
        pickerActor.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        # get the new
        NewPickedActor = pickerActor.GetActor()

        if NewPickedActor:
            picker = vtk.vtkCellPicker()
            picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
            pos = [p / self.parent.scaling_x for p in picker.GetPickPosition()]
            CellPosition = self.parent.parent.CellPosition

            CellDistances = distance.cdist(CellPosition, [pos], 'euclidean')
            self.parent.cellEmitOfInterest = None
            self.parent.cellReceOfInterest = np.argmin(CellDistances)

            if self.GetInteractor().GetControlKey() and len(self.parent.selected_cells) >0:
                cell = np.argmin(CellDistances)
                if cell in self.parent.selected_cells:
                    self.parent.selected_cells.remove(cell)
                elif self.parent.List_Neurone_type[cell] == self.parent.List_Neurone_type[self.parent.selected_cells[0]]:
                    self.parent.selected_cells.append(cell)
            else:
                self.parent.selected_cells = [np.argmin(CellDistances)]
            if self.parent.globalGUI:
                self.parent.send_selected_cell()
                self.parent.draw_Lines()
            else:
                if len(self.parent.selected_cells) == 1:
                    self.parent.send_selected_cell_2()
            self.parent.draw_BoundingBox()

        self.OnRightButtonDown()
        return

class Graph_viewer3D_VTK(QMainWindow):
    def __init__(self, parent=None, withline = True, globalGUI = True, scale = 50):
        super(Graph_viewer3D_VTK, self).__init__(parent)

        if platform.system() == 'Darwin':
            self.facteurzoom = 1.05
        else:
            self.facteurzoom = 1.25

        self.globalGUI=globalGUI
        self.parent=parent
        self.withline = withline


        # self.scene = QGraphicsScene(self)
        # self.setScene(self.scene)

        self.frame = QFrame()

        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(.1, .1, .1)
        # light = self.ren.MakeLight()
        # light.SetAmbientColor(1, 1, 1)
        # light.SetDiffuseColor(1, 1, 1)
        # light.SetSpecularColor(1, 1, 1)
        # self.ren.AddLight(light)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)


        # interactor = vtk.vtkRenderWindowInteractor()
        # interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        # interactor.Start()


        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        style = MouseInteractorHighLightActor2(self)
        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)

        # Create source
        self.scaling_x = scale
        self.scaling_y = scale
        self.scaling_z = scale

        self.cellEmitOfInterest = None
        self.cellReceOfInterest = None


        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show()
        self.iren.Initialize()
        self.iren.Start()

        self.selected_cells = []

        self.List_of_lines = []
        self.List_of_lines_mappers = []
        self.List_of_lines_actors = []
        self.List_of_forms = []
        self.List_of_forms_mappers = []
        self.List_of_forms_actors = []
        self.List_of_electrodes_actors = []
        self.List_of_boundingbox_actors = []
        self.List_of_axes_actors = []

        self.LinewidthfromGUI = 5
        self.radiuswidthfromGUI = 50
        self.scalewidthfromGUI = 50
    #
    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.vtkWidget.Finalize()     ############################ importan

    def set_center(self):
        self.ren.ResetCamera()

    def draw_Lines(self):
        linewidth_E = self.LinewidthfromGUI
        color = [1, 1, 1]

        if len(self.List_of_lines_actors) > 0:
            for actor in self.List_of_lines_actors:
                self.ren.RemoveActor(actor)

        self.List_of_lines = []
        self.List_of_lines_mappers = []
        self.List_of_lines_actors = []
        if not self.cellEmitOfInterest == None:
            r = self.cellEmitOfInterest
            pos_emet = [self.CellPosition[r][0] * self.scaling_x, self.CellPosition[r][1] * self.scaling_y, self.CellPosition[r][2] * self.scaling_z]
            for e, cells in enumerate(self.ConnectivityMatrix):
                if r in cells:
                    pos_rece = [self.CellPosition[e][0] * self.scaling_x, self.CellPosition[e][1] * self.scaling_y, self.CellPosition[e][2] * self.scaling_z]
                    lineSource = vtk.vtkLineSource()
                    lineSource.SetPoint1(pos_emet)
                    lineSource.SetPoint2(pos_rece)
                    lineSource.Update()

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(lineSource.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetLineWidth(linewidth_E)
                    actor.GetProperty().SetColor(color)

                    self.ren.AddActor(actor)

                    self.List_of_lines.append(lineSource)
                    self.List_of_lines_mappers.append(mapper)
                    self.List_of_lines_actors.append(actor)


        elif not self.cellReceOfInterest == None:
            l = self.cellReceOfInterest
            pos_rece = [self.CellPosition[l][0] * self.scaling_x, self.CellPosition[l][1] * self.scaling_y, self.CellPosition[l][2] * self.scaling_z]
            for r in self.ConnectivityMatrix[l]:
                pos_emet = [self.CellPosition[r][0] * self.scaling_x, self.CellPosition[r][1] * self.scaling_y, self.CellPosition[r][2] * self.scaling_z]
                lineSource = vtk.vtkLineSource()
                lineSource.SetPoint1(pos_emet)
                lineSource.SetPoint2(pos_rece)
                lineSource.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(lineSource.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetLineWidth(linewidth_E)
                actor.GetProperty().SetColor(color)

                self.ren.AddActor(actor)

                self.List_of_lines.append(lineSource)
                self.List_of_lines_mappers.append(mapper)
                self.List_of_lines_actors.append(actor)

    def draw_BoundingBox(self):
        if len(self.List_of_boundingbox_actors) > 0:
            for actor in self.List_of_boundingbox_actors:
                self.ren.RemoveActor(actor)

        self.List_of_boundingbox_actors=[]
        atleastone = False
        appendFilter = vtk.vtkAppendPolyData()

        for i, cell in enumerate(self.List_Neurone_type):

            # color = [255, 0, 0]
            # Colors = vtk.vtkUnsignedCharArray()
            # Colors.SetNumberOfComponents(3)
            # Colors.SetName("Colors")
            # Colors.InsertNextTuple3(color[0], color[1], color[2])

            if i in self.selected_cells:
                atleastone = True
                outline = vtk.vtkOutlineFilter()
                try:
                    outline.SetInputData(self.List_of_forms[i].GetOutput())
                except:
                    outline.SetInputData(self.List_of_forms[i])
                outline.Update()
                # Cellarray = outline.GetOutput().GetPolys().GetNumberOfCells()
                # for c in range(Cellarray):
                #     Colors.InsertNextTuple3(color[0], color[1], color[2])
                # outline.GetOutput().GetCellData().SetScalars(Colors)
                # outline.Update()
                appendFilter.AddInputData(outline.GetOutput())

        if atleastone:
            appendFilter.Update()
            cleanFilter = vtk.vtkCleanPolyData()
            cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
            cleanFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.ScalarVisibilityOn()
            mapper.SetInputConnection(cleanFilter.GetOutputPort())
            mapper.SetColorModeToDirectScalars()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.ren.AddActor(actor)
            self.List_of_boundingbox_actors.append(actor)


    def draw_Electrode(self):
        if len(self.List_of_electrodes_actors) > 0:
            for actor in self.List_of_electrodes_actors:
                self.ren.RemoveActor(actor)
        if self.electrode_disk[0] == 'Point':
            source = vtk.vtkSphereSource()
            source.SetCenter(self.electrode_pos[0] * self.scaling_x, self.electrode_pos[1] * self.scaling_y, self.electrode_pos[2] * self.scaling_z)
            source.SetRadius(10*self.radiuswidthfromGUI)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 1)

            self.ren.AddActor(actor)
            self.List_of_electrodes_actors.append(actor)
        elif self.electrode_disk[0] == 'Disk':
            source = vtk.vtkCylinderSource()
            # source.SetCenter(self.electrode_pos[0] * self.scaling_x, self.electrode_pos[1] * self.scaling_y, self.electrode_pos[2] * self.scaling_z)
            source.SetCenter(0, 0, 0)
            source.SetRadius( self.electrode_disk[1] * self.scaling_x)
            source.SetResolution(100)
            source.SetHeight(10)

            # v1 = np.array(self.electrode_pos) - np.array([self.electrode_pos[1],self.electrode_pos[1], self.electrode_pos[2]-1])
            # v1 /= np.linalg.norm(v1)
            # v1 = (0, 0, 1)
            # v2 = (0, 1, 0)
            # vp = np.cross(v2, v1)
            #
            # angle = np.arcsin(np.linalg.norm(vp))
            # angle_deg = self.electrode_disk[2] * angle / np.pi

            trans = vtk.vtkTransform()
            trans.PostMultiply()
            trans.RotateX(self.electrode_disk[2]+90)
            trans.RotateY(self.electrode_disk[3])
            trans.Translate(self.electrode_pos[0] * self.scaling_x, self.electrode_pos[1] * self.scaling_y, self.electrode_pos[2] * self.scaling_z)

            trans_filter = vtk.vtkTransformPolyDataFilter()
            trans_filter.SetInputConnection(source.GetOutputPort())
            trans_filter.SetTransform(trans)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(trans_filter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 1)

            self.ren.AddActor(actor)
            self.List_of_electrodes_actors.append(actor)

        elif self.electrode_disk[0] == 'Cylinder':
            source = vtk.vtkCylinderSource()
            # source.SetCenter(self.electrode_pos[0] * self.scaling_x, self.electrode_pos[1] * self.scaling_y, self.electrode_pos[2] * self.scaling_z)
            source.SetCenter(0, 0, 0)
            source.SetRadius( self.electrode_disk[1] * self.scaling_x)
            source.SetResolution(100)
            source.SetHeight(self.electrode_disk[4] * self.scaling_x)

            # v1 = np.array(self.electrode_pos) - np.array([self.electrode_pos[1],self.electrode_pos[1], self.electrode_pos[2]-1])
            # v1 /= np.linalg.norm(v1)
            # v1 = (0, 0, 1)
            # v2 = (0, 1, 0)
            # vp = np.cross(v2, v1)
            #
            # angle = np.arcsin(np.linalg.norm(vp))
            # angle_deg = self.electrode_disk[2] * angle / np.pi

            trans = vtk.vtkTransform()
            trans.PostMultiply()
            trans.RotateX(self.electrode_disk[2]+90)
            trans.RotateY(self.electrode_disk[3])
            trans.Translate(self.electrode_pos[0] * self.scaling_x, self.electrode_pos[1] * self.scaling_y, self.electrode_pos[2] * self.scaling_z - (self.electrode_disk[4] * self.scaling_x)/2)

            trans_filter = vtk.vtkTransformPolyDataFilter()
            trans_filter.SetInputConnection(source.GetOutputPort())
            trans_filter.SetTransform(trans)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(trans_filter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 1)

            self.ren.AddActor(actor)
            self.List_of_electrodes_actors.append(actor)



    def draw_Shperes(self):
        if len(self.List_of_forms_actors) > 0:
            for actor in self.List_of_forms_actors:
                self.ren.RemoveActor(actor)

        self.List_of_forms = []
        self.List_of_forms_mappers = []
        self.List_of_forms_actors = []
        appendFilter = vtk.vtkAppendPolyData()
        for i, cell in enumerate(self.List_Neurone_type):
            if i == self.cellEmitOfInterest or i == self.cellReceOfInterest:
                linewidth = 2
            else:
                linewidth = 1

            color = QColor(self.List_Colors[i]).getRgb()
            # color = [c/255 for c in color[:3]]
            Colors = vtk.vtkUnsignedCharArray()
            Colors.SetNumberOfComponents(3)
            Colors.SetName("Colors")
            Colors.InsertNextTuple3(color[0], color[1], color[2])

            source = vtk.vtkSphereSource()
            source.SetCenter(self.CellPosition[i][0] * self.scaling_x, self.CellPosition[i][1] * self.scaling_y, self.CellPosition[i][2] * self.scaling_z)
            source.SetRadius(10.0 * self.radiuswidthfromGUI)
            source.Update()
            Cellarray = source.GetOutput().GetPolys().GetNumberOfCells()
            for c in range(Cellarray):
                Colors.InsertNextTuple3(color[0], color[1], color[2])

            source.GetOutput().GetCellData().SetScalars(Colors)
            source.Update()
            self.List_of_forms.append(source)
            appendFilter.AddInputData(source.GetOutput())
        appendFilter.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOn()
        mapper.SetInputConnection(cleanFilter.GetOutputPort())
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.ren.AddActor(actor)
        self.List_of_forms_actors.append(actor)

    def setScales(self,scalewidthfromGUI):
        self.scaling_x = scalewidthfromGUI
        self.scaling_y = scalewidthfromGUI
        self.scaling_z = scalewidthfromGUI

    def draw_axes(self):
        if len(self.List_of_axes_actors) > 0:
            for actor in self.List_of_axes_actors:
                self.ren.RemoveActor(actor)

        self.List_of_axes_actors = []
        transform = vtk.vtkTransform()
        transform.Translate(-self.scaling_x, -self.scaling_y, 0)
        axes = vtk.vtkAxesActor()
        # x = np.max(self.CellPosition,axis=0)
        x = (1,1,1)
        axes.SetTotalLength(x[0]* self.scaling_x, x[1]* self.scaling_y, x[2]* self.scaling_z)
        axes.SetUserTransform(transform)
        self.ren.AddActor(axes)

    def draw_Graph(self,):
        try:
            self.List_Neurone_type = self.parent.List_Neurone_type
            # List_Names = self.parent.List_Names
            self.List_Colors = self.parent.List_Colors
            self.CellPosition = self.parent.CellPosition
            self.ConnectivityMatrix = self.parent.ConnectivityMatrix
            self.electrode_pos = self.parent.electrode_pos
            self.electrode_disk = self.parent.electrode_disk

            self.ren.RemoveAllViewProps()

            self.draw_Lines()
            self.draw_Shperes()
            self.draw_BoundingBox()
            self.draw_Electrode()
            # self.draw_axes()


            self.iren.GetRenderWindow().Render()

            self.set_center()
        except:
            pass

    def Render(self, ):
        self.iren.GetRenderWindow().Render()

    def send_selected_cell(self):
        self.parent.update_ModXNMM_from_VTKgraph(self.selected_cells[0])


    def send_selected_cell_2(self):
        self.parent.NewModifyXNMM.ckick_from_VTK(self.selected_cells[0])




