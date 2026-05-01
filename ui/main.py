import os
import time
import warnings
import gradio as gr
import roop.globals
import roop.metadata
import roop.utilities as util
import ui.globals as uii

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.livecam_tab import livecam_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_environment():
    if roop.globals.CFG.output_folder and os.path.isdir(roop.globals.CFG.output_folder):
        roop.globals.output_path = roop.globals.CFG.output_folder
    else:
        roop.globals.output_path = ""
    
    if not roop.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

def run():
    from roop.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)
    if roop.globals.CFG.provider == "cuda" and util.has_cuda_device() == False:
       roop.globals.CFG.provider = "cpu"

    roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    gputype = util.get_device()
    if gputype == 'cuda':
        util.print_cuda_info()
        
    print(f'Using provider {roop.globals.execution_providers} - Device:{gputype}')
    
    run_server = True
    uii.ui_restart_server = False
    mycss = """
        span {color: var(--block-info-text-color)}
        #fixedheight {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
        .image-container.svelte-1l6wqyv {height: 100%}
        #action_buttons_row {
            position: sticky !important;
            top: 0;
            z-index: 1000;
        }
        #target_history_dropdown span {
            white-space: nowrap !important;
        }
        #target_history_dropdown .wrap {
            min-width: 100% !important;
        }
    """

    while run_server:
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = False if server_name == '0.0.0.0' else True
        with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}', theme=roop.globals.CFG.selected_theme, css=mycss, delete_cache=(60, 86400)) as ui:
            with gr.Row(variant='compact'):
                    gr.Markdown(f"### [{roop.metadata.name} {roop.metadata.version}](https://github.com/C0untFloyd/roop-unleashed)")
                    gr.HTML(util.create_version_html(), elem_id="versions")

            with gr.Column(visible=not bool(roop.globals.output_path)) as output_setup_col:
                gr.Markdown("### ⚠️ Output Directory Required\nPlease select or enter a directory where processed files will be saved.")
                with gr.Row():
                    out_dir_path = gr.Textbox(label="Output Path", value=roop.globals.output_path, placeholder="e.g. C:/Users/Documents/RoopOutput", interactive=True)
                    bt_browse_out = gr.Button("📁 Browse", size="sm")
                bt_confirm_out = gr.Button("✅ Save and Start", variant="primary")

            with gr.Tabs(visible=bool(roop.globals.output_path)) as main_tabs:
                faceswap_tab()
                livecam_tab()
                facemgr_tab()
                extras_tab()
                settings_tab()

            def on_confirm_output(path):
                if path and os.path.isdir(path):
                    roop.globals.CFG.output_folder = path
                    roop.globals.CFG.save()
                    roop.globals.output_path = path
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    gr.Warning("The specified path is not a valid directory!")
                    return gr.update(visible=True), gr.update(visible=False)

            bt_browse_out.click(fn=lambda: util.browse_directory(initial_dir=roop.globals.CFG.output_folder), outputs=[out_dir_path])
            bt_confirm_out.click(fn=on_confirm_output, inputs=[out_dir_path], outputs=[output_setup_col, main_tabs])
        launch_browser = roop.globals.CFG.launch_browser

        uii.ui_restart_server = False
        allowed_paths = [os.getcwd(), os.path.abspath('data')]
        if roop.globals.output_path:
            allowed_paths.append(roop.globals.output_path)
        if "TEMP" in os.environ:
            allowed_paths.append(os.environ["TEMP"])

        try:
            ui.queue().launch(inbrowser=launch_browser, server_name=server_name, server_port=server_port, share=roop.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True, allowed_paths=allowed_paths)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)

