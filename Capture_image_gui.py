import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from Capture_reconstruct_func import reconstrct_aplha_shapes, reconstrct_poisson_surface, reconstruct_ball_pivoting, \
    get_scene_pcd_from_camera, remove_statistical_outlier, remove_radius_outlier, crop_func, zoom_image, \
    down_sample_uniform, crop_function2
from apscheduler.schedulers.background import BackgroundScheduler

"""
UI to capture scene point cloud and reconstruct surface with three different algorithms.
These three algorithms are provided by open3d
http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
The UI provides support to change parameters of these reconstruction algorithms, along 
with several other features like saving the point cloud or the mesh reconstructed.
"""


class Settings:
    """
    Add default values of the parameters of the reconstruction functions.
    These values will be displayed in the UI panel. The user can change these values,
    and they will be updated in the backend.
    """

    def __init__(self):
        self.show_recon_panel = False
        self.radius = 0.05
        self.nb_points = 16
        self.std_ratio = 2.0
        self.nb_neighbors = 20
        self.show_axes = False
        self.show_scene_panel = True
        self.auto_update = False

        '''Surface reconstruction default parameters 
        http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html '''

        # Parameters for Alpha shapes
        self.alpha = 0.03

        # Parameters Ball pivoting
        self.factor = 2
        self.radii = np.array([0, 0, 0], dtype=object)

        # Parameters for Poisson surface reconstruction
        self.depth = 9
        self.width = 0
        self.scale = 1
        self.linear_fit = False
        self.n_threads = - 1


class CaptureScene:
    """
    Here, we design the complete UI, its panels and functions what a certain button will do when clicked.
    """
    SAVE_MESH_TO_FILE = 1
    SAVE_PCD_TO_FILE = 2
    MENU_QUIT = 3
    SHOW_ADD_SCENE_PANEL = 4
    SHOW_SURFACE_RECON_PANEL = 5

    def __init__(self):
        self.job = None
        self.mesh = None
        self.pcd = None
        self.settings = Settings()

        # Create UI window
        self.window = gui.Application.instance.create_window(
            "Capture scene from 3D camera", 1080, 768)

        # Add scene widget to the UI window
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        # Set dimensions of the windows and widgets according to the font size
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        # Create a separate panel for settings and controls
        self._add_scene_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # self._surface_recon_panel.frame()
        # --------------------------------------------------------------------------------
        # ================================================================================
        # Create a collapsable menu in the settings panel
        self.sched = BackgroundScheduler()
        self.sched.start()

        # Button to capture scene point cloud
        self._add_scene_pcd_button = gui.Button("Capture scene point cloud")
        self._add_scene_pcd_button.set_on_clicked(self._on_button_add_pcd)
        self._add_scene_panel.add_fixed(separation_height)
        self._add_scene_panel.add_child(self._add_scene_pcd_button)

        self._update_scene = gui.Checkbox("Auto update scene")
        self._update_scene.set_on_checked(self._on_auto_update_scene)
        self._add_scene_panel.add_fixed(separation_height)
        self._add_scene_panel.add_child(self._update_scene)

        mouse_ctrls = gui.CollapsableVert("Mouse controls", 0.25 * em,
                                          gui.Margins(em, 0, 0, 0))
        mouse_ctrls.set_is_open(False)
        # Buttons to set mouse controls to move the scene in the window
        self._arcball_button = gui.Button("Arc-ball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        mouse_ctrls.add_child(gui.Label("Mouse controls"))
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        mouse_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        mouse_ctrls.add_child(h)
        mouse_ctrls.add_fixed(separation_height)

        # Add a checkbox to show axes
        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        mouse_ctrls.add_fixed(separation_height)
        mouse_ctrls.add_child(self._show_axes)

        # Add the collapsable control menu as a child to the settings panel
        self._add_scene_panel.add_fixed(separation_height)
        self._add_scene_panel.add_child(mouse_ctrls)
        # ===================================================================================
        # ===================================================================================
        '''crop_pcd_ctrls = gui.CollapsableVert("Crop point-cloud controls", 0,
                                             gui.Margins(em, 0, 0, 0))
        crop_pcd_ctrls.set_is_open(False)
        self._crop_pcd_button = gui.Button("Crop point-cloud")
        self._crop_pcd_button.set_on_clicked(self.on_button_crop_pcd)
        crop_pcd_ctrls.add_fixed(separation_height)
        crop_pcd_ctrls.add_child(self._crop_pcd_button)
        self._add_scene_panel.add_fixed(separation_height)
        self._add_scene_panel.add_child(crop_pcd_ctrls)'''
        # ===================================================================================
        # ===================================================================================
        # Button to remove statistical outlier
        downsampling_ctrls = gui.CollapsableVert("Downsample point-cloud controls", 0,
                                                 gui.Margins(em, 0, 0, 0))
        downsampling_ctrls.set_is_open(False)
        self._statistical_outlier_removal_button = gui.Button("Statistical outlier removal")
        self._statistical_outlier_removal_button.set_on_clicked(self._on_button_statistical_outlier_removal)
        downsampling_ctrls.add_fixed(separation_height)
        downsampling_ctrls.add_child(self._statistical_outlier_removal_button)
        # Button to remove radius outlier
        self._radius_outlier_removal_button = gui.Button("Radius outlier removal")
        self._radius_outlier_removal_button.set_on_clicked(self._on_button_radius_outlier_removal)
        downsampling_ctrls.add_fixed(separation_height)
        downsampling_ctrls.add_child(self._radius_outlier_removal_button)
        self._add_scene_panel.add_fixed(separation_height)
        self._add_scene_panel.add_child(downsampling_ctrls)

        # ===================================================================================
        # ===================================================================================
        # ===================================================================================
        # ===================================================================================
        self._surface_recon_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # ----------------------------------------------------------------------------------
        # Add collapsable menu for surface reconstruction from point cloud using Alpha shapes
        alpha_shapes = gui.CollapsableVert("Alpha shapes method", 0,
                                           gui.Margins(em, 0, 0, 0))
        alpha_shapes.set_is_open(False)
        self._alpha_value = gui.Slider(gui.Slider.DOUBLE)
        self._alpha_value.set_limits(0.0, 1.1)
        self._alpha_value.set_on_value_changed(self._on_alpha_value)
        alpha_shapes.add_child(gui.Label("Alpha"))
        alpha_shapes.add_child(self._alpha_value)
        alpha_shapes.add_fixed(separation_height)

        # Add a button to reconstruct surface
        self._alpha_recontrctn_button = gui.Button("Create surface with alpha shapes")
        self._alpha_recontrctn_button.set_on_clicked(self._on_button_alpha_rconstrctn)
        alpha_shapes.add_fixed(separation_height)
        alpha_shapes.add_child(self._alpha_recontrctn_button)
        # Add the collapsable menu as a child to the srfc_recnstrctn_ctrls
        self._surface_recon_panel.add_fixed(separation_height)
        self._surface_recon_panel.add_child(alpha_shapes)

        # ===================================================================================
        # ===================================================================================
        # Add collapsable menu for surface reconstruction from point cloud using Ball pivoting
        ball_pivoting = gui.CollapsableVert("Ball pivoting method", 0,
                                            gui.Margins(em, 0, 0, 0))
        ball_pivoting.set_is_open(False)
        self._factor_value = gui.NumberEdit(gui.NumberEdit.INT)
        self._factor_value.set_on_value_changed(self._on_factor_value)
        ball_pivoting.add_child(gui.Label("Factor of roh"))
        ball_pivoting.add_child(self._factor_value)
        ball_pivoting.add_fixed(separation_height)

        self._radii_value = gui.VectorEdit()
        # Not accepting values for radii
        # self._radii_value.set_on_value_changed(self._on_radii_value)
        ball_pivoting.add_child(gui.Label("Radii values"))
        ball_pivoting.add_child(self._radii_value)
        ball_pivoting.add_fixed(separation_height)

        # Add a button to reconstruct surface
        self._ball_pivoting_button = gui.Button("Create surface with Ball pivoting")
        self._ball_pivoting_button.set_on_clicked(self._on_button_ball_pivoting)
        ball_pivoting.add_fixed(separation_height)
        ball_pivoting.add_child(self._ball_pivoting_button)
        ball_pivoting.add_fixed(separation_height)

        # Add the collapsable menu as a child to the srfc_recnstrctn_ctrls
        self._surface_recon_panel.add_fixed(separation_height)
        self._surface_recon_panel.add_child(ball_pivoting)

        # ===================================================================================
        # ===================================================================================
        # Add collapsable menu for surface reconstruction from point cloud using Poisson surface reconstruction
        poisson_surface = gui.CollapsableVert("Poisson surface reconstruction", 0,
                                              gui.Margins(em, 0, 0, 0))
        poisson_surface.set_is_open(False)
        self._depth_value = gui.NumberEdit(gui.NumberEdit.INT)
        self._depth_value.set_on_value_changed(self._on_depth_value)
        poisson_surface.add_child(gui.Label("Depth"))
        poisson_surface.add_child(self._depth_value)
        poisson_surface.add_fixed(separation_height)

        self._width_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._width_value.set_on_value_changed(self._on_width_value)
        poisson_surface.add_child(gui.Label("Width"))
        poisson_surface.add_child(self._width_value)
        poisson_surface.add_fixed(separation_height)

        self._scale_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._scale_value.set_on_value_changed(self._on_scale_value)
        poisson_surface.add_child(gui.Label("Scale"))
        poisson_surface.add_child(self._scale_value)
        poisson_surface.add_fixed(separation_height)

        self._linear_fit = gui.Checkbox("Linear fit")
        self._linear_fit.set_on_checked(self._on_linear_fit)
        poisson_surface.add_child(self._linear_fit)
        poisson_surface.add_fixed(separation_height)

        self._n_threads = gui.Slider(gui.Slider.INT)
        self._n_threads.set_limits(-5, 5)
        self._n_threads.set_on_value_changed(self._on_n_threads)
        poisson_surface.add_child(gui.Label("n threads"))
        poisson_surface.add_child(self._n_threads)
        poisson_surface.add_fixed(separation_height)

        # Add a button to reconstruct surface
        self._poisson_surface_button = gui.Button("Create surface with Poisson reconstruction")
        self._poisson_surface_button.set_on_clicked(self._on_poisson_surface_button)
        poisson_surface.add_fixed(separation_height)
        poisson_surface.add_child(self._poisson_surface_button)
        poisson_surface.add_fixed(separation_height)

        # Add the collapsable menu as a child to the srfc_recnstrctn_ctrls
        self._surface_recon_panel.add_fixed(separation_height)
        self._surface_recon_panel.add_child(poisson_surface)
        # ---------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------
        # ========================================================================================
        # Add children to UI window, such as:
        # 1. Scene
        # 2. Settings panel
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self._surface_recon_panel)
        self.window.add_child(self._add_scene_panel)
        self.apply_settings()

        # Create menubar on UI window
        if gui.Application.instance.menubar is None:
            # Add file menu
            debug_menu = gui.Menu()
            debug_menu.add_item("Save mesh", CaptureScene.SAVE_MESH_TO_FILE)
            debug_menu.add_item("Save pcd", CaptureScene.SAVE_PCD_TO_FILE)

            debug_menu.add_separator()
            debug_menu.add_item("Quit", CaptureScene.MENU_QUIT)

            # Add settings menu
            settings_menu = gui.Menu()
            settings_menu.add_item("Show add scene panel",
                                   CaptureScene.SHOW_ADD_SCENE_PANEL)
            settings_menu.set_checked(CaptureScene.SHOW_ADD_SCENE_PANEL, self.settings.show_scene_panel)
            settings_menu.add_item("Show surface reconstruction panel",
                                   CaptureScene.SHOW_SURFACE_RECON_PANEL)
            settings_menu.set_checked(CaptureScene.SHOW_SURFACE_RECON_PANEL, False)
            menu = gui.Menu()

            menu.add_menu("File", debug_menu)
            menu.add_menu("Settings", settings_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.

        self.window.set_on_menu_item_activated(CaptureScene.SAVE_MESH_TO_FILE,
                                               self._on_menu_save_mesh)
        self.window.set_on_menu_item_activated(CaptureScene.SAVE_PCD_TO_FILE,
                                               self._on_menu_save_pcd)
        self.window.set_on_menu_item_activated(CaptureScene.MENU_QUIT,
                                               self._on_menu_quit)
        self.window.set_on_menu_item_activated(CaptureScene.SHOW_ADD_SCENE_PANEL, self._on_menu_toggle_add_scene_panel)
        self.window.set_on_menu_item_activated(CaptureScene.SHOW_SURFACE_RECON_PANEL,
                                               self._on_menu_toggle_surface_recon_panel)

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will lay out
        # the grandchildren.
        r = self.window.content_rect
        self.scene.frame = r
        width = 17 * layout_context.theme.font_size
        height_scene_panel = min(
            r.height,
            self._add_scene_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._add_scene_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                               height_scene_panel)
        height_recon_panel = min(r.height,
                                 self._surface_recon_panel.calc_preferred_size(
                                     layout_context, gui.Widget.Constraints()).height)
        self._surface_recon_panel.frame = gui.Rect(r.get_right() - width, (r.y * 2) + height_scene_panel, width,
                                                   height_recon_panel)

    def _on_menu_toggle_add_scene_panel(self):
        self._add_scene_panel.visible = not self._add_scene_panel.visible
        gui.Application.instance.menubar.set_checked(
            CaptureScene.SHOW_ADD_SCENE_PANEL, self._add_scene_panel.visible)
        self.settings.show_scene_panel = not self.settings.show_scene_panel

    def _on_menu_toggle_surface_recon_panel(self):
        self._surface_recon_panel.visible = not self._surface_recon_panel.visible
        gui.Application.instance.menubar.set_checked(
            CaptureScene.SHOW_SURFACE_RECON_PANEL, self._surface_recon_panel.visible)
        self.settings.show_recon_panel = not self.settings.show_recon_panel

    def _on_alpha_value(self, value):
        self.settings.alpha = float(value)
        self.apply_settings()

    def _on_radii_value(self, value):
        self.settings.radii = value
        self.apply_settings()

    def _on_factor_value(self, value):
        self.settings.factor = int(value)
        self.apply_settings()

    def _on_depth_value(self, value):
        self.settings.depth = int(value)
        self.apply_settings()

    def _on_width_value(self, value):
        self.settings.width = float(value)
        self.apply_settings()

    def _on_scale_value(self, value):
        self.settings.scale = float(value)
        self.apply_settings()

    def _on_linear_fit(self, value):
        self.settings.linear_fit = value
        self.apply_settings()

    def _on_n_threads(self, value):
        # todo: something wrong if you try to change in it in UI
        self.settings.n_threads = value
        self.apply_settings()

    def _set_mouse_mode_rotate(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def apply_settings(self):
        self.scene.scene.show_axes(self.settings.show_axes)
        self._show_axes.checked = self.settings.show_axes
        self._alpha_value.double_value = self.settings.alpha
        self._factor_value.int_value = self.settings.factor
        self._radii_value.vector_value = np.array(self.settings.radii, dtype=object)
        self._depth_value.int_value = self.settings.depth
        self._width_value.double_value = self.settings.width
        self._scale_value.double_value = self.settings.scale
        self._linear_fit.checked = self.settings.linear_fit
        self._n_threads.int_value = self.settings.n_threads
        self._add_scene_panel.visible = self.settings.show_scene_panel
        self._surface_recon_panel.visible = self.settings.show_recon_panel
        self.auto_update(self.settings.auto_update)
        self._update_scene.checked = self.settings.auto_update

    def _on_button_add_pcd(self):
        print('take picture')
        mat = rendering.MaterialRecord()
        if self.pcd is None:
            temp = True
        else:
            temp = False
        self.pcd = get_scene_pcd_from_camera()
        self.scene.scene.clear_geometry()
        if temp:
            bbox = zoom_image(self.pcd)
            self.scene.setup_camera(60, bbox, [0, 0, 0])
        self.scene.scene.add_geometry("3D Scene", self.pcd, mat)

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self.apply_settings()

    def _on_auto_update_scene(self, show):
        self.settings.auto_update = show
        self.apply_settings()

    def auto_update(self, enable):
        if enable:
            self.job = self.sched.add_job(self._on_button_add_pcd, 'interval', seconds=10)
            print('Auto-update enabled')
        else:
            if self.job:
                self.job.remove()
                print('Auto-update disabled')

    def _on_button_statistical_outlier_removal(self):
        mat = rendering.MaterialRecord()
        self.pcd = remove_statistical_outlier(self.pcd, self.settings.nb_neighbors, self.settings.std_ratio)
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("3D Scene", self.pcd, mat)

    def _on_button_radius_outlier_removal(self):
        mat = rendering.MaterialRecord()
        self.pcd = remove_radius_outlier(self.pcd, self.settings.nb_points, self.settings.radius)
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("3D Scene", self.pcd, mat)

    def _on_menu_save_mesh(self):
        if self.mesh is not None:
            o3d.io.write_triangle_mesh("Camera_images_ply/output_mesh.ply", self.mesh)
        else:
            pass

    '''def on_button_crop_pcd(self):
        # todo: issue: gui closes after closing the cropping window

        mat = rendering.MaterialRecord()

        crop_function2(gui.Application.instance, self.pcd)

        #crop_function2(self.pcd)
        #self.pcd = crop_function2(self.pcd)
        #self.scene.scene.clear_geometry()
        #self.scene.scene.add_geometry("3D Scene", self.pcd, mat)'''

    def _on_menu_save_pcd(self):
        if self.pcd is not None:
            o3d.io.write_point_cloud("output_pcd/output_pcd.pcd", self.pcd)
        else:
            pass

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_button_alpha_rconstrctn(self):
        mesh = reconstrct_aplha_shapes(self.pcd, self.settings.alpha)
        mat = rendering.MaterialRecord()
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("3D Scene", mesh, mat)
        self.mesh = mesh

    def _on_button_ball_pivoting(self):
        mat = rendering.MaterialRecord()
        radii, mesh = reconstruct_ball_pivoting(self.pcd, self.settings.factor)
        self.settings.radii = radii
        self.apply_settings()
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("3D Scene", mesh, mat)
        self.mesh = mesh

    def _on_poisson_surface_button(self):
        mesh = reconstrct_poisson_surface(self.pcd, self.settings.depth, self.settings.width, self.settings.scale,
                                          self.settings.linear_fit, self.settings.n_threads)
        mat = rendering.MaterialRecord()
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("3D Scene", mesh, mat)
        self.mesh = mesh


def main():
    gui.Application.instance.initialize()
    CaptureScene()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
