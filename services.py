# services.py
import os
import json
import io
import contextlib
import subprocess
import platform
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, quote
import base64
import asyncio
from pathlib import Path
from PIL import Image, ExifTags
from xml.dom import minidom
import tempfile
import sys
import threading
import queue
from typing import Dict, Any

# Defer winrt imports to prevent memory issues on startup
IS_WINDOWS_OCR_AVAILABLE = sys.platform == "win32"

try:
    import requests
    TOOLS_LIBS_AVAILABLE = True
    print("Tool libraries (requests) loaded successfully.")
except ImportError:
    print("WARNING: 'requests' library not found. Web search and RSS tools will be unavailable.")
    print("Please run 'pip install requests' to enable them.")
    TOOLS_LIBS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    WEB_EXTRACTION_LIBS_AVAILABLE = True
    print("Web extraction library (BeautifulSoup4) loaded successfully.")
except ImportError:
    print("\nWARNING: 'beautifulsoup4' library not found. The 'read_web_page' tool will be unavailable.")
    print("Please run 'pip install beautifulsoup4' to enable it.\n")
    WEB_EXTRACTION_LIBS_AVAILABLE = False

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    from huggingface_hub import snapshot_download as hf_snapshot_download, HfApi
    DOWNLOAD_LIBS_AVAILABLE = True
    print("Model download libraries (modelscope, huggingface_hub) loaded successfully.")
except ImportError:
    print("\nWARNING: 'modelscope' or 'huggingface_hub' library not found. Model downloading will be unavailable.")
    print("Please run 'pip install modelscope huggingface_hub' to enable it.\n")
    DOWNLOAD_LIBS_AVAILABLE = False


# Import from our local modules
from config import MODELS_BASE_DIR
from utils import build_markdown_from_messages, format_data_to_markdown

# --- MCP Globals ---
# These need to be global to be accessible by the API endpoints,
# as Flask runs each request in its own context.
MCP_PROCESSES: Dict[str, subprocess.Popen] = {}
MCP_PENDING_REQUESTS: Dict[str, Dict[int, queue.Queue]] = {}
MCP_REQUEST_ID_COUNTERS: Dict[str, int] = {}
MCP_LOCKS = {
    'processes': threading.Lock(),
    'requests': threading.Lock(),
    'counters': threading.Lock(),
}

def mcp_stdout_reader(server_id: str, proc: subprocess.Popen):
    """
    Thread target to continuously read stdout from an MCP process and put
    JSON-RPC responses into the correct queues.
    """
    for line in iter(proc.stdout.readline, ''):
        line = line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
            req_id = response.get('id')
            
            with MCP_LOCKS['requests']:
                # The queue might have been removed already if a timeout occurred
                if server_id in MCP_PENDING_REQUESTS and req_id in MCP_PENDING_REQUESTS[server_id]:
                    MCP_PENDING_REQUESTS[server_id][req_id].put(response)
                else:
                    # It's possible to receive a response after a timeout, which is okay.
                    print(f"[MCP RX WORKER WARNING {server_id}]: Received response for unknown/timed-out request ID {req_id}")

        except json.JSONDecodeError:
            # Ignore non-json lines like startup messages from cmd
            print(f"[MCP RX NON-JSON from {server_id}]: {line}")
            pass
        except Exception as e:
            print(f"[MCP RX WORKER ERROR {server_id}]: {e}")
    print(f"[MCP RX WORKER {server_id}]: stdout stream closed.")


def start_mcp_server(config: Dict[str, Any]):
    """Starts an MCP server as a subprocess."""
    server_id = config.get('id')
    if not server_id:
        raise ValueError("MCP config is missing an 'id'.")
    
    server_type = config.get('type')
    if server_type != 'process':
        raise NotImplementedError(f"MCP server type '{server_type}' is not supported.")

    with MCP_LOCKS['processes']:
        if server_id in MCP_PROCESSES and MCP_PROCESSES[server_id].poll() is None:
            raise RuntimeError(f"MCP server '{config.get('name')}' is already running.")

        if platform.system() == "Windows":
             command_parts = ['cmd', '/c'] + [config['command']] + config['args']
        else:
            command_parts = [config['command']] + config['args']
        
        # Merge system environment with custom env from config
        process_env = os.environ.copy()
        user_env = config.get('env')
        if user_env and isinstance(user_env, dict):
            process_env.update(user_env)
            print(f"Starting MCP server with custom environment variables.")

        print(f"Starting MCP server '{config.get('name')}': {' '.join(command_parts)}")

        proc = subprocess.Popen(
            command_parts,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            env=process_env
        )
        MCP_PROCESSES[server_id] = proc
        
        stdout_thread = threading.Thread(target=mcp_stdout_reader, args=(server_id, proc), daemon=True)
        stdout_thread.start()

    try:
        init_params = {
            "protocolVersion": "1.0",
            "capabilities": {},
            "clientInfo": { "name": "天问.OV", "version": "0.1.0" }
        }
        _ = _send_mcp_request(server_id, 'initialize', init_params)
        
        tools_response = _send_mcp_request(server_id, 'tools/list', {})
        discovered_tools = tools_response.get('tools', [])
        
        print(f"MCP server '{config.get('name')}' started with PID {proc.pid} and exposed {len(discovered_tools)} tools.")
        return proc.pid, discovered_tools
        
    except Exception as e:
        print(f"Error during MCP handshake for '{config.get('name')}': {e}")
        proc.kill()
        with MCP_LOCKS['processes']:
            if server_id in MCP_PROCESSES:
                del MCP_PROCESSES[server_id]
        raise


def stop_mcp_server(config: Dict[str, Any]):
    """Stops a running MCP server process."""
    server_id = config.get('id')
    with MCP_LOCKS['processes']:
        proc = MCP_PROCESSES.pop(server_id, None)
    if proc and proc.poll() is None:
        print(f"Stopping MCP server '{config.get('name')}' (PID: {proc.pid})")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Process for '{config.get('name')}' did not terminate gracefully, killing.")
            proc.kill()
    else:
        print(f"MCP server '{config.get('name')}' was not running.")

def _send_mcp_request(server_id: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to send a JSON-RPC request to a specific MCP server and get the response."""
    with MCP_LOCKS['counters']:
        req_id = MCP_REQUEST_ID_COUNTERS.get(server_id, 0) + 1
        MCP_REQUEST_ID_COUNTERS[server_id] = req_id
    
    request_payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

    with MCP_LOCKS['processes']:
        proc = MCP_PROCESSES.get(server_id)
        if not proc or proc.poll() is not None:
            raise RuntimeError(f"MCP server with ID '{server_id}' is not running.")
    
    response_queue = queue.Queue(maxsize=1)
    with MCP_LOCKS['requests']:
        if server_id not in MCP_PENDING_REQUESTS:
            MCP_PENDING_REQUESTS[server_id] = {}
        MCP_PENDING_REQUESTS[server_id][req_id] = response_queue

    try:
        print(f"[MCP TX -> {server_id}]: {json.dumps(request_payload)}")
        proc.stdin.write(json.dumps(request_payload) + '\n')
        proc.stdin.flush()
        
        response = response_queue.get(timeout=60)
        
        print(f"[MCP RX <- {server_id}]: {json.dumps(response)}")
        
        if 'error' in response:
            error_details = response['error']
            raise RuntimeError(f"MCP request '{method}' failed: {error_details.get('message', 'Unknown error')} (Code: {error_details.get('code')})")
        
        return response.get('result', {})

    finally:
        with MCP_LOCKS['requests']:
            if server_id in MCP_PENDING_REQUESTS and req_id in MCP_PENDING_REQUESTS[server_id]:
                del MCP_PENDING_REQUESTS[server_id][req_id]


def run_mcp_tool(server_id: str, tool_name: str, params: Dict[str, Any]) -> Any:
    """Runs a specific tool on an MCP server."""
    print(f"Running MCP tool '{tool_name}' on server '{server_id}' with params: {params}")

    result = _send_mcp_request(
        server_id,
        'tools/call',
        {"name": tool_name, "arguments": params}
    )
    # Format the final result to markdown before returning
    return format_data_to_markdown(result)


# --- Model Download Service ---

def download_model_from_source(model_id, source):
    """Downloads a model from the specified source, streaming progress."""
    if not DOWNLOAD_LIBS_AVAILABLE:
        raise ImportError("Model downloading libraries are not installed. Please run 'pip install modelscope huggingface_hub'.")

    local_dir = MODELS_BASE_DIR / model_id
    local_dir.mkdir(parents=True, exist_ok=True)
    validation_file = "config.json"
    
    yield {"type": "progress", "percentage": 0, "status_text": f"Verifying {model_id}..."}

    try:
        if source == "modelscope":
             ms_snapshot_download(model_id, allow_patterns=[validation_file], cache_dir=str(local_dir.parent), local_dir=str(local_dir))
        elif source == "huggingface":
            hf_snapshot_download(model_id, allow_patterns=[validation_file], local_dir=str(local_dir), local_dir_use_symlinks=False)
        else:
            raise ValueError(f"Unknown download source: {source}")
        
        yield {"type": "progress", "percentage": 10, "status_text": "开始下载. Starting full download..."}
        
        if source == "modelscope":
             yield {"type": "progress", "percentage": 10, "status_text": "从魔搭社区下载中,进度不显示..."}
             ms_snapshot_download(model_id, cache_dir=str(local_dir.parent), local_dir=str(local_dir))
        elif source == "huggingface":
            yield {"type": "progress", "percentage": 10, "status_text": "Downloading from Hugging Face..."}
            hf_snapshot_download(model_id, local_dir=str(local_dir), local_dir_use_symlinks=False, resume_download=True)
    except Exception as e:
        yield {"type": "error", "content": f"Download failed: {str(e)}"}
        return

    config_path = local_dir / validation_file
    model_details = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        avatar_path = local_dir / 'avatar.png'
        avatar_url = f'/static/models/{quote(model_id)}/avatar.png' if avatar_path.exists() else None
        vision_model_file = local_dir / 'openvino_vision_embeddings_model.bin'
        model_type = 'multimodal' if vision_model_file.exists() else 'language'

        model_details = {
            "id": model_id, "name": config_data.get("model_type"),
            "description": config_data.get("architectures"),
            "avatarUrl": avatar_url, "status": "local", "modelType": model_type,
            "details": config_data
        }

    yield {"type": "progress", "percentage": 100, "status_text": "Download complete!"}
    yield {"type": "success", "content": f"Model {model_id} downloaded successfully.", "model_details": model_details}

# --- Helper Functions ---
def image_from_data_url(data_url: str):
    """Converts a data URL to a PIL Image."""
    if not data_url or not data_url.startswith('data:image'): return None
    try:
        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        from PIL import Image
        from io import BytesIO
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"Error converting data URL to image: {e}")
        return None

# --- Tool Functions ---

def system_shell(command: str, **kwargs):
    """Executes a shell command and returns its output, indicating success or failure."""
    try:
        # For Windows, use 'cmd /c' to properly handle commands. For others, use '/bin/sh -c'.
        if platform.system() == "Windows":
            # Using 'gbk' encoding for Windows CMD to handle non-UTF8 characters often found in system paths/outputs.
            process = subprocess.run(['cmd', '/c', command], capture_output=True, text=True, encoding='gbk', errors='ignore', shell=False, timeout=30)
        else:
            process = subprocess.run(['/bin/sh', '-c', command], capture_output=True, text=True, encoding='utf-8', shell=False, timeout=30)
        
        # Check the return code to determine if the command was successful.
        if process.returncode == 0:
            # If the command succeeded, return its standard output.
            # If there's no output, return a clear success message.
            return process.stdout or "Command executed successfully with no output."
        else:
            # If the command failed, return a detailed error message.
            return f"Error executing command:\nReturn Code: {process.returncode}\nStderr:\n{process.stderr}\nStdout:\n{process.stdout}"
            
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        # Catch any other exceptions during process execution.
        return f"Error executing command: {e}"


def _generate_variable_declarations(args: dict) -> str:
    """
    Generates Python code to declare variables based on the provided arguments.
    This makes the arguments directly available as variables in the user's script.
    """
    declarations = []
    for name, value in args.items():
        # Use repr() to get a safe, quoted string representation of the value.
        py_value = repr(value)
        declarations.append(f"{name} = {py_value}")

    if not declarations:
        return ""
        
    return "\n".join(declarations) + "\n"


def _run_command_in_venv(venv_python_path: Path, command: list):
    """
    Executes a command within a specified virtual environment and returns the output.
    This is a helper function focused on command execution.
    """
    if not venv_python_path.exists():
        raise FileNotFoundError(f"Venv Python executable not found: {venv_python_path}")

    # Normalize command to ensure correct interpreter usage
    if command and command[0].lower() in ['python', 'pip']:
        executable = str(venv_python_path)
        if command[0].lower() == 'pip':
            full_command = [executable, "-m", "pip"] + command[1:]
        else: # command[0] == 'python'
            full_command = [executable] + command[1:]
    else:
        full_command = command

    # Use 'gbk' on Windows for console output, otherwise 'utf-8'
    # Use errors='ignore' as a fallback for problematic characters.
    encoding = 'gbk' if platform.system() == "Windows" else 'utf-8'
    
    try:
        result = subprocess.run(
            full_command,
            check=True,
            capture_output=True,
            text=True,
            encoding=encoding,
            errors='ignore',
            timeout=300 # 5-minute timeout to prevent stuck scripts
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        error_output = f"Command failed with exit code {e.returncode}\n"
        error_output += f"--- STDOUT ---\n{e.stdout}\n"
        error_output += f"--- STDERR ---\n{e.stderr}\n"
        return False, None, error_output
    except subprocess.TimeoutExpired as e:
        error_output = f"Command timed out after {e.timeout} seconds.\n"
        error_output += f"--- STDOUT ---\n{e.stdout}\n"
        error_output += f"--- STDERR ---\n{e.stderr}\n"
        return False, None, error_output
    except Exception as e:
        return False, None, f"An unexpected error occurred while running command: {e}\n{traceback.format_exc()}"


def execute_python_script(script_body: str, pip_packages: list[str] = None, **kwargs):
    """
    Creates a temporary, sandboxed virtual environment, installs dependencies,
    injects parameters as variables, and executes a Python script.
    """
    if pip_packages is None: pip_packages = []
    
    # Inject parameters provided by AI as global variables into the script
    param_declarations = _generate_variable_declarations(kwargs)
    full_script_body = f"{param_declarations}\n{script_body}"
    
    print(f"--- Full Script Body for Execution ---\n{full_script_body}\n--------------------------------------")
    
    with tempfile.TemporaryDirectory(prefix="py_exec_sandbox_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        venv_path = temp_dir / "venv"
        
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture):
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True, capture_output=True, text=True, encoding='utf-8'
                )
            except subprocess.CalledProcessError as e:
                return f"Fatal: Failed to create environment.\n{e.stderr}"

            venv_python_path = venv_path / "Scripts" / "python.exe" if sys.platform == "win32" else venv_path / "bin" / "python"
            
            if pip_packages:
                print(f"Installing packages: {', '.join(pip_packages)}")
                success, _, stderr = _run_command_in_venv(venv_python_path, ["pip", "install"] + pip_packages)
                if not success:
                    return f"Fatal: Failed to install pip packages.\n{stderr}"
                print("Packages installed successfully.")
            else:
                print("No packages to install.")
            
            print(f"\n--- Executing Script ---")
            success, stdout, stderr = _run_command_in_venv(venv_python_path, ["python", "-c", full_script_body])

            final_output = output_capture.getvalue()
            final_output += f"\n--- Script Execution Result ---\n"
            if success:
                final_output += f"Script executed successfully.\n--- STDOUT ---\n{stdout}"
                if stderr:
                    final_output += f"--- STDERR ---\n{stderr}"
            else:
                final_output += f"Script execution failed.\n--- ERROR ---\n{stderr}"

            return final_output
        
def write_local_file(path: str, content: str, **kwargs):
    """Writes content to a local file, overwriting if it exists."""
    try:
        file_path = Path(path)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing to file {path}: {e}"


async def _run_ocr_on_bytes_async(image_bytes: bytes, lang: str = None) -> str:
    """
    Uses the Windows OCR API and manually joins results to avoid extra spaces.
    Lazy-loads winrt to prevent memory issues on startup.
    """
    try:
        import winrt.windows.media.ocr as ocr
        import winrt.windows.graphics.imaging as imaging
        from winrt.windows.storage.streams import InMemoryRandomAccessStream, DataWriter
        from winrt.windows.globalization import Language
    except ImportError:
        return " [Windows OCR libraries (winrt) not available on this system] "
        
    try:
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(image_bytes)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)

        decoder = await imaging.BitmapDecoder.create_async(stream)
        software_bitmap = await decoder.get_software_bitmap_async()

        engine = None
        if lang:
            try:
                language = Language(lang)
                if ocr.OcrEngine.is_language_supported(language):
                    engine = ocr.OcrEngine.try_create_from_language(language)
                else:
                    return f" [OCR language '{lang}' is not supported. Please install it via Windows Settings.] "
            except Exception:
                 return f" [Invalid OCR language code: '{lang}'] "
        if not engine:
            engine = ocr.OcrEngine.try_create_from_user_profile_languages()
        if not engine:
            return " [OCR engine could not be initialized. Check language packs.] "

        result = await engine.recognize_async(software_bitmap)

        recognized_lines = []
        if result.lines is not None:
            for line in result.lines:
                line_text_parts = [word.text for word in line.words]
                recognized_lines.append("".join(line_text_parts))

        final_text = "\n".join(recognized_lines)
        
        return final_text
    
    except Exception as e:
        return f" [OCR Error: {e}] "

    

def _get_image_metadata(img_path: Path) -> dict:
    """Extracts image metadata like dimensions and camera info."""
    info = {}
    try:
        with Image.open(img_path) as img:
            info["Size (WxH)"] = f"{img.width}x{img.height}"
            info["Format"] = img.format

            exif_data = img._getexif()
            if exif_data:
                exif_tags = {v: k for k, v in ExifTags.TAGS.items()}
                make = exif_data.get(exif_tags.get("Make"))
                model = exif_data.get(exif_tags.get("Model"))
                if make:
                    info["Camera Make"] = str(make).strip()
                if model:
                    info["Camera Model"] = str(model).strip()
    except Exception:
        pass
    return info


def read_local_file(path: str, max_chars: int = 3000, ocr_lang: str = None, **kwargs):
    """
    Reads a local file, with enhanced handling for images, PDFs, etc.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found at {path}"
        if not file_path.is_file():
            return f"Error: Path {path} is a directory, not a file."

        suffix = file_path.suffix.lower()
        content = ""
        info = {}

        if suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'] and IS_WINDOWS_OCR_AVAILABLE:
            info.update(_get_image_metadata(file_path))
            file_bytes = file_path.read_bytes()
            ocr_text = asyncio.run(_run_ocr_on_bytes_async(file_bytes, lang=ocr_lang))
            content = f"--- Recognized Text (OCR) ---\n{ocr_text}"
            info["Content Source"] = f"Windows OCR (Lang: {ocr_lang or 'auto'})"
        elif suffix == '.xml':
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                rough_string = ET.tostring(root, 'utf-8')
                reparsed = minidom.parseString(rough_string)
                pretty_xml = reparsed.toprettyxml(indent="  ")
                content = pretty_xml
                info["Content Source"] = "Parsed XML"
            except ET.ParseError as e:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                info["Warning"] = f"XML parsing failed ({e}), read as plain text."
        else:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            info["Content Source"] = "Plain Text"

        total_chars = len(content)
        truncated_content = content[:max_chars]
        
        info_str = "\n".join([f"- {key}: {value}" for key, value in info.items()])

        return (
            f"File: {path}\n"
            f"Size: {file_path.stat().st_size} bytes\n"
            f"Important Info:\n{info_str}\n\n"
            f"Total Characters (after processing): {total_chars}\n\n"
            f"--- Start of Content (first {max_chars} chars) ---\n{truncated_content}"
        )

    except Exception as e:
        return f"Error processing file {path}: {e}"

def read_web_page(url: str, **kwargs):
    """(DEPRECATED) Fetches and returns the text content of a given URL. Use the frontend version for better results."""
    if not WEB_EXTRACTION_LIBS_AVAILABLE: return "Web extraction library (BeautifulSoup4) is not installed."
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['github.com', 'gitee.com', 'gitcode.com'] and '/blob/' in parsed_url.path:
            raw_url = url.replace('/blob/', '/raw/')
            response = requests.get(raw_url, headers=headers)
        else:
            response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        for script in soup(["script", "style", "nav", "footer", "aside"]): script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e: return f"Error processing page: {e}"

def get_rss_news(feed_names: list, config: dict, **kwargs):
    """Fetches latest news from configured RSS feeds and formats as Markdown."""
    if not TOOLS_LIBS_AVAILABLE or not WEB_EXTRACTION_LIBS_AVAILABLE: return "Required libraries not installed."
    available_feeds = {feed['name']: feed['url'] for feed in config.get('feeds', [])}
    markdown_output = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    if not feed_names: return "Please provide a list of feed names. Available: " + ", ".join(f"`{name}`" for name in available_feeds.keys())
    for name in feed_names:
        if name in available_feeds:
            try:
                response = requests.get(available_feeds[name], headers=headers, timeout=10)
                response.raise_for_status()
                # Use lxml-xml for parsing XML-based feeds
                soup = BeautifulSoup(response.content, 'lxml-xml')
                items = soup.find_all('item') or soup.find_all('entry')
                markdown_output.append(f"### {name}\n")
                count = 0
                for item in items[:5]:
                    title_tag, link_tag = item.find('title'), item.find('link')
                    if title_tag and link_tag:
                        link = link_tag.get('href') or link_tag.text
                        markdown_output.append(f"- [{title_tag.text.strip()}]({link.strip()})")
                        count += 1
                if count == 0: markdown_output.append("- No articles found.\n")
            except Exception as e: markdown_output.append(f"### Error fetching {name}\n- {e}\n")
        else: markdown_output.append(f"### Error: Feed '{name}' not found\n")
    return "\n".join(markdown_output)

def http_request(config: dict, **kwargs):
    """Performs a generic HTTP request based on tool config and returns JSON or text."""
    if not TOOLS_LIBS_AVAILABLE:
        return {"error": "HTTP tool unavailable: 'requests' library not installed."}

    args = kwargs
    
    http_config = config.get("config", {})
    
    method = http_config.get("method", "GET").upper()
    url = http_config.get("url")
    headers_str = http_config.get("headers", "{}")
    body_template_str = http_config.get("bodyTemplate", "{}")

    if not url:
        raise ValueError("URL is missing in the tool configuration.")
    try:
        headers = json.loads(headers_str) if headers_str.strip() else {}
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for headers: {headers_str}")

    for key, value in args.items():
        url = url.replace(f"{{{{{key}}}}}", quote(str(value)))
    
    original_proxies = {
        'HTTP_PROXY': os.environ.pop('HTTP_PROXY', None),
        'HTTPS_PROXY': os.environ.pop('HTTPS_PROXY', None),
        'http_proxy': os.environ.pop('http_proxy', None),
        'https_proxy': os.environ.pop('https_proxy', None),
    }

    try:
        response = None
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=20)
        elif method == "POST":
            body_str = body_template_str
            for key, value in args.items():
                body_str = body_str.replace(f"{{{{{key}}}}}", str(value))
            
            try:
                # Assuming the final body is JSON
                body_data = json.loads(body_str) if body_str.strip() else {}
                response = requests.post(url, headers=headers, json=body_data, timeout=20)
            except json.JSONDecodeError:
                # If not JSON, send as raw data
                response = requests.post(url, headers=headers, data=body_str, timeout=20)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
        else:
            return response.text
            
    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP Request Failed: {e}"}
    finally:
        # Restore the original proxy settings
        for key, value in original_proxies.items():
            if value:
                os.environ[key] = value

# --- Export Functions ---
def export_to_html(messages, title):
    """Generates a styled HTML string from a list of messages, including tool calls."""
    body_content = ""
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        html_content = ""

        # Render collapsible thoughts block
        if msg.get('collapsibleBlocks'):
            for block in msg.get('collapsibleBlocks'):
                if block.get('title') == 'thought':
                    thoughts = block.get('content', '')
                    # Use <details> which is collapsed by default
                    html_content += f"""
                    <details class="tool-call">
                        <summary>思考过程</summary>
                        <pre><code>{thoughts}</code></pre>
                    </details>
                    """

        # Render tool calls, also collapsed by default
        if msg.get('toolCalls'):
            for call in msg.get('toolCalls'):
                tool_name = call.get('name', 'Unknown Tool')
                tool_args = json.dumps(call.get('args', {}), indent=2, ensure_ascii=False)
                tool_result_obj = call.get('result', {})
                tool_result = tool_result_obj.get('content', 'No result.')
                
                html_content += f"""
                <div class="tool-call">
                    <details>
                        <summary><strong>Calling Tool:</strong> <code>{tool_name}</code></summary>
                        <div class="tool-details">
                            <strong>Arguments:</strong>
                            <pre><code>{tool_args}</code></pre>
                            <strong>Result:</strong>
                            <pre><code>{tool_result}</code></pre>
                        </div>
                    </details>
                </div>
                """

        # Render main message content if it exists
        if content:
             html_content += f"<div class=\"markdown-content\">{content}</div>"

        # Determine CSS class and header based on role
        if role == 'user':
            css_class = "message-user"
            header = "<h4><strong>我:</strong></h4>"
        else:
            css_class = "message-assistant"
            header = "<h4><strong>AI:</strong></h4>"

        body_content += f'<div class="message-block {css_class}">{header}{html_content}</div>\n<hr>\n'

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; background-color: #f9f9f9; color: #333; }}
            h1 {{ color: #1a1a1a; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }}
            h4 {{ margin-bottom: 0.5em; }}
            pre {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 14px; }}
            code {{ font-family: inherit; }}
            hr {{ border: 0; border-top: 1px solid #ddd; margin: 2em 0; }}
            .message-block {{ margin-bottom: 1.5em; }}
            .message-user {{ }}
            .message-assistant {{ }}
            .markdown-content {{ white-space: pre-wrap; }}
            .tool-call {{ border: 1px solid #e0e0e0; border-radius: 8px; margin: 1em 0; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
            .tool-call summary {{ cursor: pointer; padding: 12px; font-weight: 500; background-color: #f7f7f7; border-radius: 8px 8px 0 0; }}
            .tool-call summary:hover {{ background-color: #efefef; }}
            .tool-call[open] > summary {{ border-bottom: 1px solid #e0e0e0; }}
            .tool-details {{ padding: 0 12px 12px 12px; }}
            .tool-details strong {{ display: block; margin-top: 10px; margin-bottom: 5px; color: #555; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <hr>
        {body_content}
    </body>
    </html>
    """
    

