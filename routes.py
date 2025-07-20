# routes.py
from flask import Blueprint, request, jsonify, Response, send_from_directory
import json
import threading
import queue
import traceback
from urllib.parse import quote
from datetime import datetime
import platform
from pathlib import Path
import base64

# Import from our local modules
from config import MODELS_BASE_DIR
import services
from utils import find_models, build_markdown_from_messages, format_data_to_markdown
from model_manager import ModelManager

# --- Globals ---
# One global instance of the model manager
model_manager = ModelManager()
# Global to cache the currently loaded translation file from the frontend
current_translation = {}
# Get OpenVINO version once
ov_version = model_manager.Get_OV_Version()

# --- Blueprint Setup ---
api_blueprint = Blueprint('api', __name__)

# --- Helper Functions ---
def _get_translation(key: str, **kwargs):
    """Safely gets a nested translation string from the cached dictionary."""
    if not current_translation:
        # Fallback if no language has been set by the frontend yet
        return key

    try:
        keys = key.split('.')
        value = current_translation
        for k in keys:
            value = value[k]
        
        if kwargs and isinstance(value, str):
            return value.format(**kwargs)
        return value if isinstance(value, str) else key
    except (KeyError, TypeError):
        # Fallback to the key itself if not found
        print(f"Warning: Translation key not found: {key}")
        return key


def _get_all_params_for_tool(tool_config):
    """
    Intelligently gets the parameters for a tool that should be shown to the AI.
    It excludes implementation details like 'script_body'.
    For RSS tools, it dynamically adds feed info to the description.
    """
    params = tool_config.get('params', {}).copy()
    description = tool_config.get('description', '')

    # For script-based tools, add their special parameters to the description for the AI
    if tool_config.get('type') == 'python_script':
        params['script_body'] = _get_translation('systemPrompt.toolParams.pythonScriptBody')
        params['pip_packages'] = _get_translation('systemPrompt.toolParams.pythonPipPackages')
    elif tool_config.get('type') == 'js_script':
        params['script_body'] = _get_translation('systemPrompt.toolParams.jsScriptBody')

    # For the RSS tool, append the list of available feeds to its description.
    if tool_config.get('editorType') == 'rss' and tool_config.get('config', {}).get('feeds'):
        feeds = tool_config['config']['feeds']
        feed_names = [f"`{feed.get('name')}`" for feed in feeds if feed.get('name')]
        if feed_names:
            feeds_str = ", ".join(feed_names)
            description += f"\n{_get_translation('systemPrompt.availableFeeds')}: {feeds_str}"
            
    return params, description

def _build_tool_prompt(active_tools, mcp_tools):
    """Builds the tool instructions part of the system prompt from full tool objects in a simple list format."""
    if not active_tools and not mcp_tools:
        return ""

    tool_definitions = []
    
    # Process custom-defined tools
    for tool in active_tools:
        params, description = _get_all_params_for_tool(tool)
        tool_def_parts = [
            f"  - **{_get_translation('systemPrompt.toolId')}:** `{tool.get('id')}`",
            f"    - **{_get_translation('systemPrompt.description')}:** {description}"
        ]
        param_lines = []
        if params:
            for name, param_description in params.items():
                param_lines.append(f"      - `{name}`: {param_description}")
        if param_lines:
            tool_def_parts.append(f"    - **{_get_translation('systemPrompt.parameters')}:**")
            tool_def_parts.extend(param_lines)
        else:
            tool_def_parts.append(f"    - **{_get_translation('systemPrompt.parameters')}:** {_get_translation('systemPrompt.noParameters')}")
        tool_definitions.append("\n".join(tool_def_parts))
        
    # Process discovered MCP tools
    for tool in mcp_tools:
        tool_def_parts = [
            f"  - **{_get_translation('systemPrompt.toolId')}:** `{tool.get('name')}`",
            f"    - **{_get_translation('systemPrompt.description')}:** {tool.get('description')}"
        ]
        param_lines = []
        schema = tool.get('inputSchema', {})
        params = schema.get('properties', {})
        if params:
            for name, param_schema in params.items():
                param_lines.append(f"      - `{name}`: {param_schema.get('description', 'No description.')}")
        if param_lines:
            tool_def_parts.append(f"    - **{_get_translation('systemPrompt.parameters')}:**")
            tool_def_parts.extend(param_lines)
        else:
            tool_def_parts.append(f"    - **{_get_translation('systemPrompt.parameters')}:** {_get_translation('systemPrompt.noParameters')}")
        tool_definitions.append("\n".join(tool_def_parts))
    
    tool_list_str = "\n".join(tool_definitions)

    tool_preamble = _get_translation('systemPrompt.toolPreamble')
    tool_structure = _get_translation('systemPrompt.toolStructure')
    tool_notes = _get_translation('systemPrompt.toolNotes')
    available_tools_header = _get_translation('systemPrompt.availableTools')

    return f"{tool_preamble}\n\n{tool_structure}\n\n{tool_notes}\n\n{available_tools_header}\n{tool_list_str}"

def _build_system_info_prompt(system_info):
    """Builds the system information block for the system prompt."""
    if not system_info: return ""
    
    lines = [_get_translation('systemPrompt.systemInfo') + ":"]
    if system_info.get("os"):
        lines.append(f"- {_get_translation('systemPrompt.os')}: {system_info.get('os')}")
    
    current_time_str = _get_translation('systemPrompt.currentTime', time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append(f"- {current_time_str}")
        
    return "\n".join(lines)


# --- Static & Info Endpoints ---
@api_blueprint.route('/static/models/<path:path>')
def serve_model_static(path):
    return send_from_directory(MODELS_BASE_DIR, path)

@api_blueprint.route('/api/models', methods=['GET'])
def get_models_api():
    local_models = find_models()
    return jsonify(local_models)

@api_blueprint.route('/api/system_info', methods=['GET'])
def system_info_api():
    return jsonify({
        "os": platform.platform(),
        "ov_genai_version": ov_version,
    })

@api_blueprint.route('/api/heartbeat', methods=['GET'])
def heartbeat_api():
    """A lightweight endpoint to check if the backend is responsive."""
    return jsonify({"status": "ok"}), 200

@api_blueprint.route('/api/set_language', methods=['POST'])
def set_language_api():
    """Receives a base64 encoded JSON string of translations and caches it."""
    global current_translation
    data = request.json
    messages_b64 = data.get('messages_b64')
    if not messages_b64:
        return jsonify({"success": False, "error": "Missing 'messages_b64' payload."}), 400
    
    try:
        # Decode the base64 string to a JSON string
        decoded_json_str = base64.b64decode(messages_b64).decode('utf-8')
        # Parse the JSON string into a Python dictionary
        current_translation = json.loads(decoded_json_str)
        print(f"Backend language updated successfully.")
        return jsonify({"success": True}), 200
    except (base64.binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error processing language data: {e}")
        return jsonify({"success": False, "error": f"Invalid language data payload: {e}"}), 400


# --- Core Model & Chat Endpoints ---

@api_blueprint.route('/api/load_model', methods=['POST'])
def load_model_api():
    data = request.json
    model_id = data.get('model_id')
    device = data.get('device', 'CPU')
    if not model_id: return jsonify({"success": False, "error": "Model ID is required."}), 400
    model_path_str = str(MODELS_BASE_DIR / model_id)
    success = model_manager.load_model(model_path_str, device)
    if success: return jsonify({"success": True, "message": f"Model '{model_id}' loaded."})
    else: return jsonify({"success": False, "error": "Failed to load model."}), 500

@api_blueprint.route('/api/download_model_stream', methods=['POST'])
def download_model_stream_api():
    data = request.json
    model_id = data.get("model_id")
    source = data.get("source")
    if not model_id or not source:
        return Response(json.dumps({"type": "error", "content": "Missing model_id or source"}), status=400, mimetype='application/json')
    def generate_events():
        try:
            for progress_update in services.download_model_from_source(model_id, source):
                 yield f"data: {json.dumps(progress_update)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    return Response(generate_events(), mimetype='text/event-stream')

@api_blueprint.route('/api/chat_stream', methods=['POST'])
def chat_stream_api():
    if not model_manager.is_loaded():
        return Response(json.dumps({"error": "No model loaded"}), status=503, mimetype='application/json')

    data = request.json
    history = data.get('history', [])
    generation_config = data.get('generationConfig', {})
    image_data_urls = data.get('images', [])
    system_prompt = data.get('systemPrompt')
    active_tools = data.get('tools', [])
    mcp_tools = data.get('mcp_tools', [])
    memory_turns = data.get('memoryTurns')
    max_message_length = data.get('maxMessageLength', 0)
    system_info = data.get('system_info', {})
    carry_tool_results = data.get('carryToolResults', False)

    def generate_events():
        q = queue.Queue()
        stop_event = threading.Event()
        
        def generation_task():
            try:
                images = [services.image_from_data_url(url) for url in image_data_urls if url]
                tokenizer = model_manager.get_tokenizer()
                if not tokenizer: raise RuntimeError("Could not get tokenizer.")
                
                if memory_turns is not None and memory_turns > 0:
                    if len(history) > memory_turns * 2:
                        history_to_use = history[-(memory_turns * 2):]
                    else:
                        history_to_use = history
                else:
                    history_to_use = history

                info_prompt = _build_system_info_prompt(system_info)
                tool_instructions = _build_tool_prompt(active_tools, mcp_tools)

                final_system_prompt_parts = [system_prompt, info_prompt, tool_instructions]
                final_system_prompt = "\n".join(filter(None, final_system_prompt_parts))
                
                formatted_history = []
                if final_system_prompt:
                    formatted_history.append({"role": "system", "content": final_system_prompt})
                
                # Find the last user message to potentially re-prompt the AI
                last_user_message_content = ""
                for msg in reversed(history_to_use):
                    if msg.get("role") == "user" and msg.get("content"):
                        last_user_message_content = msg.get("content")
                        break

                was_last_turn_tool = False
                
                processed_history = []
                for msg in history_to_use:
                    # Create a new message object to avoid modifying the original
                    new_msg = msg.copy()
                    
                    # Apply truncation to all messages
                    content = new_msg.get("content", "")
                    if max_message_length and len(content) > max_message_length:
                        new_msg['content'] = content[:max_message_length] + "... (truncated)"
                    
                    processed_history.append(new_msg)

                for msg in processed_history:
                    content = msg.get("content", "")
                    attachments = msg.get("attachments", [])

                    if attachments:
                        attachment_texts = [f"[{_get_translation('ChatPanel.Input.attachedFile')}: {att.get('path')}]" for att in attachments]
                        content = f"{content}\n{' '.join(attachment_texts)}".strip()

                    if msg.get("role") == "user":
                        was_last_turn_tool = False
                        if content:
                            formatted_history.append({"role": "user", "content": content})
                    
                    elif msg.get("role") == "assistant":
                        tool_calls = msg.get("toolCalls")
                        if tool_calls:
                            # This turn was a tool call
                            was_last_turn_tool = True
                            tool_call_payloads = []
                            for call in tool_calls:
                                tool_call_payloads.append({
                                    "id": call.get("name"),
                                    "args": call.get("args", {})
                                })
                            
                            formatted_history.append({
                                "role": "assistant",
                                "content": f"<tool_code>{json.dumps(tool_call_payloads[0])}</tool_code>" if len(tool_call_payloads) == 1 else f"<tool_code>{json.dumps(tool_call_payloads)}</tool_code>"
                            })
                            
                            # The result MUST be sent back to the AI for it to know what happened.
                            # The `carryToolResults` toggle should only affect if this result is also
                            # shown in the UI and long-term memory, not the immediate reasoning loop.
                            for call in tool_calls:
                                if call.get("result"):
                                    result_content = call["result"].get("content", "No result content.")
                                    preamble = _get_translation(
                                        'systemPrompt.toolResultHeader', 
                                        toolId=call.get("name")
                                    )
                                    formatted_history.append({
                                        "role": "assistant",
                                        "content": f"{preamble}\n\n{result_content}"
                                    })
                        elif content:
                            # Regular assistant message
                            was_last_turn_tool = False
                            if carry_tool_results or not msg.get('toolCalls'):
                                formatted_history.append({"role": "assistant", "content": content})
                
                # If the last turn was a tool call and we have the user's question, add the re-prompt.
                if was_last_turn_tool and last_user_message_content:
                    re_prompt = _get_translation('systemPrompt.rePrompt', userQuestion=last_user_message_content)
                    formatted_history.append({"role": "user", "content": re_prompt})

                prompt_text = tokenizer.apply_chat_template(formatted_history, add_generation_prompt=True)
                # print(f"--- Final Prompt to Model ---\n{prompt_text}\n---------------------------")

                model_manager.stream_generate(
                    prompt=prompt_text, images=images, queue=q, 
                    stop_event=stop_event, **(generation_config or {})
                )
            except Exception as e:
                q.put({"type": "error", "content": str(e)})
                traceback.print_exc()
                q.put(None)

        thread = threading.Thread(target=generation_task)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is None: break
                if stop_event.is_set(): break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            stop_event.set()
            thread.join()

    return Response(generate_events(), mimetype='text/event-stream')


# --- MCP Server Endpoints ---
@api_blueprint.route('/api/mcp/start', methods=['POST'])
def mcp_start_api():
    config = request.json.get('config')
    if not config:
        return jsonify({"success": False, "error": "Missing MCP server config"}), 400
    try:
        pid, discovered_tools = services.start_mcp_server(config)
        return jsonify({"success": True, "pid": pid, "tools": discovered_tools})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@api_blueprint.route('/api/mcp/stop', methods=['POST'])
def mcp_stop_api():
    config = request.json.get('config')
    if not config:
        return jsonify({"success": False, "error": "Missing MCP server config"}), 400
    try:
        services.stop_mcp_server(config)
        return jsonify({"success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- Tool & Export Endpoints ---

def _execute_tool_logic(tool_id, tool_args, tool_config):
    """Helper function to contain the tool execution and formatting logic."""
    tool_type = tool_config.get("type")
    editor_type = tool_config.get("editorType")

    # This maps a tool_id directly to a function in the services module
    # It's a security measure to prevent calling arbitrary functions.
    TOOL_FUNCTION_MAP = {
        "system_shell": services.system_shell,
        "write_local_file": services.write_local_file,
        "read_web_page": services.read_web_page,
        "get_rss_news": services.get_rss_news,
        "read_local_file": services.read_local_file,
    }
    
    raw_result = None

    if tool_type == 'text':
        raw_result = tool_config.get("config", {}).get("content", "")
    elif tool_type == 'http':
        raw_result = services.http_request(config=tool_config, **tool_args)
    elif tool_type == 'python_script':
        # The AI's script takes precedence.
        script_body = tool_args.get('script_body')
        
        # If the AI provides pip_packages, use that. If not, use an empty list.
        # Do not fall back to the tool's default config.
        pip_packages = tool_args.get('pip_packages', [])

        user_args = tool_args.copy()
        user_args.pop('script_body', None)
        user_args.pop('pip_packages', None)
        
        if not script_body:
            raise ValueError("No python script was found in the arguments.")

        raw_result = services.execute_python_script(
            script_body=script_body,
            pip_packages=pip_packages,
            **user_args
        )
    elif tool_type == 'mcp':
        server_id = tool_config.get("serverId")
        if not server_id:
             raise ValueError(f"Tool config for MCP tool '{tool_id}' is missing its 'serverId'.")
        raw_result = services.run_mcp_tool(server_id, tool_id, tool_args)

    elif tool_type == 'backend':
        if editor_type == 'alias':
            target_tool_id = tool_config.get("config", {}).get("targetTool")
            args_template_str = tool_config.get("config", {}).get("argsTemplate", "{}")
            if not target_tool_id:
                raise ValueError(f"Alias tool '{tool_id}' has no targetTool defined.")
            if target_tool_id not in TOOL_FUNCTION_MAP:
                 raise FileNotFoundError(f"Alias target tool implementation '{target_tool_id}' not found or not whitelisted.")
            new_args_str = args_template_str
            for key, value in tool_args.items():
                new_args_str = new_args_str.replace(f'{{{{{key}}}}}', str(value))
            try:
                final_args = json.loads(new_args_str)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse args template for alias '{tool_id}' after substitution. Result: {new_args_str}")
            tool_function = TOOL_FUNCTION_MAP[target_tool_id]
            raw_result = tool_function(**final_args)
        else:
            if tool_id in TOOL_FUNCTION_MAP:
                tool_function = TOOL_FUNCTION_MAP[tool_id]
                # Pass config for tools like RSS that need it
                if tool_id == 'get_rss_news':
                    raw_result = tool_function(**tool_args, config=tool_config.get("config", {}))
                else:
                    raw_result = tool_function(**tool_args)
            else:
                raise FileNotFoundError(f"Backend tool implementation for '{tool_id}' not found or not whitelisted.")
    else:
        raise NotImplementedError(f"Tool type '{tool_type or 'None'}' with ID '{tool_id}' is not handled by the backend.")

    return format_data_to_markdown(raw_result)


@api_blueprint.route('/api/execute_tool', methods=['POST'])
def execute_tool_api():
    data = request.json
    tool_id, tool_args, tool_config = data.get("tool"), data.get("args", {}), data.get("config", {})
    if not tool_id:
        return jsonify({"error": "tool ID is required"}), 400
    try:
        result = _execute_tool_logic(tool_id, tool_args, tool_config)
        return jsonify({"result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/api/export_html', methods=['POST'])
def export_html_api():
    data = request.get_json()
    messages = data.get('messages', [])
    title = data.get('title', 'Chat Export')
    html_content = services.export_to_html(messages, title)
    return Response(
        html_content,
        mimetype="text/html",
        headers={"Content-disposition": "attachment; filename=chat_export.html"}
    )
    
@api_blueprint.route('/api/clear_cache', methods=['POST'])
def clear_cache_api():
    # In a real app, you might clear specific caches (e.g., a database cache).
    # For this example, we'll just return a success message.
    # If you have specific cache directories, you could clear them here.
    # For example:
    # import shutil
    # cache_dir = Path("./cache")
    # if cache_dir.exists():
    #     shutil.rmtree(cache_dir)
    #     cache_dir.mkdir()
    print("Received request to clear cache. (No specific server cache to clear)")
    return jsonify({"success": True, "message": "Server-side cache cleared (if any)."}), 200

    

    
