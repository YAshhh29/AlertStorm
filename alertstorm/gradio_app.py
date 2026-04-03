import gradio as gr
from typing import List, Set, Dict
from server.alertstorm_environment import AlertstormEnvironment
from models import AlertstormAction
import json
import os
from datetime import datetime

HISTORY_FILE = "alertstorm_history.json"

def log_to_session(session_id: str, history: list):
    if not session_id:
        return
    db = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                db = json.load(f)
            if not isinstance(db, dict):
                db = {}
        except Exception:
            pass
    db[session_id] = history
    with open(HISTORY_FILE, "w") as f:
        json.dump(db, f, indent=4)

def get_session_list():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            db = json.load(f)
        if isinstance(db, dict):
            return sorted(list(db.keys()), reverse=True)
        return []
    except Exception:
        return []

def load_session(session_id: str):
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            db = json.load(f)
        if isinstance(db, dict):
            return db.get(session_id, [])
        return []
    except Exception:
        return []

NODE_POSITIONS = {
    "API_Gateway": (390, 50),
    "Auth_Service": (175, 150),
    "Order_Service": (600, 150),
    "User_DB": (100, 250),
    "Redis_Cache": (250, 250),
    "Inventory_DB": (450, 250),
    "Stripe_API": (600, 250),
    "Notification_Service": (750, 250)
}

EDGES = [
    ("API_Gateway", "Auth_Service"),
    ("API_Gateway", "Order_Service"),
    ("Auth_Service", "User_DB"),
    ("Auth_Service", "Redis_Cache"),
    ("Order_Service", "Inventory_DB"),
    ("Order_Service", "Stripe_API"),
    ("Order_Service", "Notification_Service")
]

def render_svg(active_alerts: List[str], investigated: Set[str], root_causes: List[str] = None) -> str:
    svg = ['<svg viewBox="0 0 850 320" style="width:100%; height:auto; font-family: sans-serif; background-color: transparent;">']
    
    for frm, to in EDGES:
        x1, y1 = NODE_POSITIONS[frm]
        x2, y2 = NODE_POSITIONS[to]
        is_alert_path = to in active_alerts or frm in active_alerts
        
        if is_alert_path and root_causes:
            color = "#10b981"
            width = "2"
            dash = ""
        elif is_alert_path:
            color = "#EE4C2C"
            width = "2"
            dash = 'stroke-dasharray="4 4"'
        else:
            color = "#475569"
            width = "1"
            dash = ""
            
        svg.append(f'<line x1="{x1}" y1="{y1+15}" x2="{x2}" y2="{y2-15}" stroke="{color}" stroke-width="{width}" {dash} />')
    
    for node, (x, y) in NODE_POSITIONS.items():
        fill = "#1e293b"
        stroke = "#64748b"
        text_color = "#e2e8f0"
        stroke_width = "1.5"
        
        if root_causes and node in root_causes:
            fill = "#064e3b"
            stroke = "#10b981"
            stroke_width = "4"
        elif root_causes and node in active_alerts:
            fill = "#064e3b"
            stroke = "#10b981"
            stroke_width = "2"
        elif node in active_alerts:
            fill = "#450a0a"
            stroke = "#EE4C2C"
            text_color = "#fca5a5"
        elif node in investigated:
            fill = "#422006"
            stroke = "#FFD21E"
            text_color = "#fef08a"
            
        label = node.replace("_", " ")
        
        svg.append(f'''
        <g transform="translate({x}, {y})">
            <rect x="-60" y="-15" width="120" height="30" rx="15" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"></rect>
            <text x="0" y="4" font-size="11" fill="{text_color}" text-anchor="middle" font-weight="500" style="letter-spacing: 0.5px;">{label}</text>
        </g>
        ''')
        
    svg.append('</svg>')
    return "".join(svg)

ENT_NODE_POSITIONS = {
    "API_Gateway": (900, 50),
    "Auth_Service": (320, 200),
    "Order_Service": (800, 200),
    "CDN_Edge": (1200, 200),
    "GraphQL_Server": (1500, 200),
    "User_DB_Primary": (150, 350),
    "User_Cache_Cluster": (350, 350),
    "OAuth_Provider_External": (550, 350),
    "Inventory_Service": (750, 350),
    "Payment_Gateway": (950, 350),
    "Message_Queue": (1150, 350),
    "Static_Assets_S3": (1350, 350),
    "Product_Catalog": (1550, 350),
    "Recommendation_Engine": (1750, 350),
    "EBS_Volume_A": (80, 500),
    "RDS_Proxy": (220, 500),
    "Redis_Node_1": (360, 500),
    "Redis_Node_2": (500, 500),
    "Inventory_DB": (680, 500),
    "ElasticSearch_Index": (820, 500),
    "Stripe_API": (960, 500),
    "Fraud_Detection_AI": (1100, 500),
    "Kafka_Broker_Leader": (1240, 500),
    "Zookeeper_Ensemble": (1380, 500),
    "MongoDB_Cluster": (1520, 500),
    "Image_Optimizer": (1660, 500),
    "Spark_Job_Runner": (1800, 500)
}

from server.alertstorm_environment import ENTERPRISE_DEPENDENCY_GRAPH

def render_ent_svg(active_alerts: List[str], investigated: Set[str], root_causes: List[str] = None) -> str:
    svg = ['<svg viewBox="0 0 1900 600" style="width:100%; height:auto; font-family: sans-serif; background-color: transparent;">']
    
    edges = []
    for parent, children in ENTERPRISE_DEPENDENCY_GRAPH.items():
        for child in children:
            edges.append((parent, child))
            
    for frm, to in edges:
        x1, y1 = ENT_NODE_POSITIONS[frm]
        x2, y2 = ENT_NODE_POSITIONS.get(to, (0,0))
        is_alert_path = to in active_alerts or frm in active_alerts
        
        if is_alert_path and root_causes:
            color = "#10b981"; width = "3"; dash = ""
        elif is_alert_path:
            color = "#EE4C2C"; width = "3"; dash = 'stroke-dasharray="6 6"'
        else:
            color = "#475569"; width = "1.5"; dash = ""
            
        svg.append(f'<line x1="{x1}" y1="{y1+15}" x2="{x2}" y2="{y2-15}" stroke="{color}" stroke-width="{width}" {dash} />')
    
    for node, (x, y) in ENT_NODE_POSITIONS.items():
        fill = "#1e293b"; stroke = "#64748b"; text_color = "#e2e8f0"; stroke_width = "1.5"
        
        if root_causes and node in root_causes:
            fill = "#064e3b"; stroke = "#10b981"; stroke_width = "4"
        elif root_causes and node in active_alerts:
            fill = "#064e3b"; stroke = "#10b981"; stroke_width = "2"
        elif node in active_alerts:
            fill = "#450a0a"; stroke = "#EE4C2C"; text_color = "#fca5a5"
        elif node in investigated:
            fill = "#422006"; stroke = "#FFD21E"; text_color = "#fef08a"
            
        label = node.replace("_", " ")
        svg.append(f'''
        <g transform="translate({x}, {y})">
            <rect x="-70" y="-15" width="140" height="30" rx="15" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"></rect>
            <text x="0" y="4" font-size="10" fill="{text_color}" text-anchor="middle" font-weight="500" style="letter-spacing: 0px;">{label}</text>
        </g>
        ''')
        
    svg.append('</svg>')
    return "".join(svg)


def create_app():
    theme = gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Geist"), "sans-serif"]
    )
    
    with gr.Blocks(title="AlertStorm Agent Dashboard") as app:
        env_state = gr.State(None)
        chat_history = gr.State([])
        investigated_nodes = gr.State(set())
        session_id_state = gr.State(None)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                # 🌪️ AlertStorm OVERSIGHT
                Advanced Microservice Failure Simulation Console
                """)
            with gr.Column(scale=0, min_width=150):
                btn_theme = gr.Button("🌓 Toggle Dark Mode", size="sm")
        
        with gr.Tabs():
            with gr.Tab("Standard Environment (8 Nodes)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        graph_html = gr.HTML(render_svg([], set(), None), label="Dependency DAG")
                        
                        with gr.Row():
                            target_dropdown = gr.Dropdown(
                                choices=list(NODE_POSITIONS.keys()),
                                label="Target Service(s)",
                                value=["API_Gateway"],
                                multiselect=True
                            )
                        with gr.Row():
                            btn_investigate = gr.Button("🔍 View Logs")
                            btn_suppress = gr.Button("🔇 Dismiss Alert")
                            btn_resolve = gr.Button(" Declare Root Cause", variant="primary")
                            
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(label="Network Audit Log", height=380, avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"))
                        btn_share = gr.Button("🔗 Generate Shareable Log")
                        btn_reset = gr.Button("🚨 Trigger Outage", variant="stop")
            
            with gr.Tab("Enterprise Environment (29 Nodes)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ent_graph_html = gr.HTML(render_ent_svg([], set(), None), label="Enterprise 29-Node DAG")
                        
                        with gr.Row():
                            ent_target_dropdown = gr.Dropdown(
                                choices=list(ENTERPRISE_DEPENDENCY_GRAPH.keys()),
                                label="Target Service(s)",
                                value=["API_Gateway"],
                                multiselect=True
                            )
                        with gr.Row():
                            ent_btn_investigate = gr.Button("🔍 View Logs")
                            ent_btn_suppress = gr.Button("🔇 Dismiss Alert")
                            ent_btn_resolve = gr.Button(" Declare Root Cause", variant="primary")
                            
                    with gr.Column(scale=1):
                        ent_chatbot = gr.Chatbot(label="Enterprise Telemetry Stream", height=500, avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"))
                        ent_btn_reset = gr.Button("🚨 Trigger Enterprise Outage", variant="stop")
            
            with gr.Tab("Session History"):
                gr.Markdown("Select a previous RCA session date to view its exact event trace (LLM-style). Everything is strictly localized.")
                
                with gr.Row():
                    session_dropdown = gr.Dropdown(choices=get_session_list(), label="Historical Incident Sessions")
                    btn_refresh_mem = gr.Button("🔄 Refresh List")
                
                historical_chatbot = gr.Chatbot(label="Archived Incident Trace", height=450, avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"))
                
                btn_refresh_mem.click(lambda: gr.update(choices=get_session_list()), None, session_dropdown)
                session_dropdown.change(load_session, session_dropdown, historical_chatbot)

        share_output = gr.Markdown(visible=False)

        def generate_shareable_report(history):
            if not history:
                return gr.update(value="No logs yet.", visible=True)
            text = "### 📋 AlertStorm Incident Report\n\n```text\n"
            for msg in history:
                role = msg.get("role", "SYSTEM").upper()
                content = msg.get("content", "")
                text += f"[{role}] {content}\n\n"
            text += "```"
            return gr.update(value=text, visible=True)

        btn_share.click(generate_shareable_report, chat_history, share_output)
                
        def do_reset(history, investigated):
            import server.alertstorm_environment as env_module
            env_module.TASK_OVERRIDE = "standard_medium"
            new_env = AlertstormEnvironment()
            obs = new_env.reset()
            active = [a['service'] for a in obs.active_alerts]
            
            sess_id = f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            history = [{"role": "assistant", "content": "System Warning: Cascading failure injected into the Goldilocks Graph."}]
            for a in obs.active_alerts:
                history.append({"role": "assistant", "content": f"🚨 {a['service']}: {a['type']}"})
                
            log_to_session(sess_id, history)
            return new_env, history, history, render_svg(active, set(), None), set(), sess_id
            
        def do_ent_reset(history, investigated):
            import server.alertstorm_environment as env_module
            env_module.TASK_OVERRIDE = "enterprise_medium"
            new_env = AlertstormEnvironment()
            obs = new_env.reset()
            active = [a['service'] for a in obs.active_alerts]
            
            sess_id = f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            history = [{"role": "assistant", "content": "Global Architecture Warning: 29-Node topological failure detected."}]
            for a in obs.active_alerts:
                history.append({"role": "assistant", "content": f"🚨 {a['service']}: {a['type']}"})
                
            log_to_session(sess_id, history)
            return new_env, history, history, render_ent_svg(active, set(), None), set(), sess_id

        def do_investigate(sess_id, env, target, history, investigated):
            if not env or not target:
                return env, history, history, render_svg([], investigated, None), investigated
                
            t_list = target if isinstance(target, list) else [target]
            action = AlertstormAction(action_type="investigate", targets=t_list, confidence=1.0)
            obs = env.step(action)
            investigated.update(t_list)
            
            history.append({"role": "user", "content": f"Investigate {', '.join(t_list)}"})
            if obs.reward > 0:
                history.append({"role": "assistant", "content": f"🔍 Logs: {obs.recent_logs} [Reward: +{obs.reward}]"})
            else:
                history.append({"role": "assistant", "content": f"🔍 Logs: {obs.recent_logs}"})
            log_to_session(sess_id, history)
            
            active = [a['service'] for a in obs.active_alerts]
            return env, history, history, render_svg(active, investigated, None), investigated

        def do_suppress(sess_id, env, target, history, investigated):
            if not env or not target:
                return env, history, history, render_svg([], investigated, None), investigated
                
            t_list = target if isinstance(target, list) else [target]
            action = AlertstormAction(action_type="suppress_alert", targets=t_list, confidence=1.0)
            obs = env.step(action)
            
            history.append({"role": "user", "content": f"Suppress {', '.join(t_list)}"})
            if obs.reward > 0:
                history.append({"role": "assistant", "content": f"🔇 {obs.recent_logs} [Reward: +{obs.reward}]"})
            else:
                history.append({"role": "assistant", "content": f"🔇 {obs.recent_logs}"})
            log_to_session(sess_id, history)
            
            active = [a['service'] for a in obs.active_alerts]
            return env, history, history, render_svg(active, investigated, None), investigated

        def do_resolve(sess_id, env, target, history, investigated):
            if not env or not target:
                return env, history, history, render_svg([], investigated, None), investigated
                
            t_list = target if isinstance(target, list) else [target]
            action = AlertstormAction(action_type="propose_root_cause", targets=t_list, confidence=1.0)
            obs = env.step(action)
            
            t_str = ", ".join(t_list)
            if obs.recent_logs and obs.recent_logs.startswith("SUCCESS"):
                history.append({"role": "user", "content": f"Propose Root Cause: {t_str}"})
                history.append({"role": "assistant", "content": f"🏆 {obs.recent_logs} [Reward: +{obs.reward}]"})
                root_col = t_list
            else:
                history.append({"role": "user", "content": f"Propose Root Cause: {t_str}"})
                history.append({"role": "assistant", "content": f"❌ {obs.recent_logs} [Reward: {obs.reward}]"})
                root_col = None
                
            log_to_session(sess_id, history)
            active = [a['service'] for a in obs.active_alerts]
            return env, history, history, render_svg(active, investigated, root_col), investigated
            
        # Enterprise Handlers mapped to the Custom Responsive SVG
        def do_ent_investigate(sess_id, env, target, history, investigated):
            if not env or not target: return env, history, history, render_ent_svg([], investigated, None), investigated
            t_list = target if isinstance(target, list) else [target]
            obs = env.step(AlertstormAction(action_type="investigate", targets=t_list, confidence=1.0))
            investigated.update(t_list)
            history.append({"role": "user", "content": f"Investigate {', '.join(t_list)}"})
            history.append({"role": "assistant", "content": f"🔍 Logs: {obs.recent_logs} [Reward: +{obs.reward}]"})
            return env, history, history, render_ent_svg([a['service'] for a in obs.active_alerts], investigated, None), investigated

        def do_ent_suppress(sess_id, env, target, history, investigated):
            if not env or not target: return env, history, history, render_ent_svg([], investigated, None), investigated
            t_list = target if isinstance(target, list) else [target]
            obs = env.step(AlertstormAction(action_type="suppress_alert", targets=t_list, confidence=1.0))
            history.append({"role": "user", "content": f"Suppress {', '.join(t_list)}"})
            history.append({"role": "assistant", "content": f"🔇 {obs.recent_logs} [Reward: +{obs.reward}]"})
            return env, history, history, render_ent_svg([a['service'] for a in obs.active_alerts], investigated, None), investigated

        def do_ent_resolve(sess_id, env, target, history, investigated):
            if not env or not target: return env, history, history, render_ent_svg([], investigated, None), investigated
            t_list = target if isinstance(target, list) else [target]
            obs = env.step(AlertstormAction(action_type="propose_root_cause", targets=t_list, confidence=1.0))
            success = obs.recent_logs and obs.recent_logs.startswith("SUCCESS")
            history.append({"role": "user", "content": f"Propose Root Cause: {', '.join(t_list)}"})
            history.append({"role": "assistant", "content": f"{'🏆' if success else '❌'} {obs.recent_logs} [Reward: {obs.reward}]"})
            return env, history, history, render_ent_svg([a['service'] for a in obs.active_alerts], investigated, t_list if success else None), investigated

        toggle_js = """
        function() {
            document.body.classList.toggle('dark');
        }
        """

        btn_theme.click(None, None, None, js=toggle_js)
        btn_reset.click(do_reset, [chat_history, investigated_nodes], [env_state, chatbot, chat_history, graph_html, investigated_nodes, session_id_state])
        btn_investigate.click(do_investigate, [session_id_state, env_state, target_dropdown, chat_history, investigated_nodes], [env_state, chatbot, chat_history, graph_html, investigated_nodes])
        btn_suppress.click(do_suppress, [session_id_state, env_state, target_dropdown, chat_history, investigated_nodes], [env_state, chatbot, chat_history, graph_html, investigated_nodes])
        btn_resolve.click(do_resolve, [session_id_state, env_state, target_dropdown, chat_history, investigated_nodes], [env_state, chatbot, chat_history, graph_html, investigated_nodes])
        
        ent_btn_reset.click(do_ent_reset, [chat_history, investigated_nodes], [env_state, ent_chatbot, chat_history, ent_graph_html, investigated_nodes, session_id_state])
        ent_btn_investigate.click(do_ent_investigate, [session_id_state, env_state, ent_target_dropdown, chat_history, investigated_nodes], [env_state, ent_chatbot, chat_history, ent_graph_html, investigated_nodes])
        ent_btn_suppress.click(do_ent_suppress, [session_id_state, env_state, ent_target_dropdown, chat_history, investigated_nodes], [env_state, ent_chatbot, chat_history, ent_graph_html, investigated_nodes])
        ent_btn_resolve.click(do_ent_resolve, [session_id_state, env_state, ent_target_dropdown, chat_history, investigated_nodes], [env_state, ent_chatbot, chat_history, ent_graph_html, investigated_nodes])
        
    return app

if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Geist"), "sans-serif"]
    )
    
    custom_css = """
    .gradio-container {
        font-family: 'Geist', sans-serif !important;
        transition: background-color 0.4s ease;
    }
    
    /* Interactive glowing buttons */
    button.primary {
        box-shadow: 0 4px 15px rgba(6, 104, 225, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    button.primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(6, 104, 225, 0.4);
    }
    button.stop {
        box-shadow: 0 4px 15px rgba(238, 76, 44, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    button.stop:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(238, 76, 44, 0.4);
    }
    
    /* PURE BLACK DARK MODE */
    .dark .gradio-container {
        background: #000000 !important;
    }
    .dark .svelte-container {
        border-color: rgba(255, 255, 255, 0.08) !important;
        background: #000000 !important;
    }
    .dark .panel {
        background: #0a0a0a !important;
    }
    """
    
    app = create_app()
    app.launch(server_name="127.0.0.1", theme=theme, css=custom_css, share=True)
