
# utils.py
from config import MODELS_BASE_DIR
import json

def find_models():
    """Recursively scans the model directory for 'config.json' and creates a list of models."""
    if not MODELS_BASE_DIR.is_dir():
        print(f"WARNING: Model directory not found at {MODELS_BASE_DIR}")
        return []
        
    models_list = []
    # Use glob to find all config.json files recursively
    for config_path in MODELS_BASE_DIR.glob('**/config.json'):
        model_dir = config_path.parent
        model_id = model_dir.relative_to(MODELS_BASE_DIR).as_posix()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse config for {model_id}: {e}")
            config_data = {}

        # Check for avatar.png in the model folder
        avatar_path = model_dir / 'avatar.png'
        avatar_url = f'/static/models/{model_id}/avatar.png' if avatar_path.exists() else None

        # Determine model type
        vision_model_file = model_dir / 'openvino_vision_embeddings_model.bin'
        model_type = 'multimodal' if vision_model_file.exists() else 'language'

        models_list.append({
            "id": model_id,
            "name": config_data.get("name", model_id.split('/')[-1]),
            "description": config_data.get("description", f"ID: {model_id}"),
            "avatarUrl": avatar_url,
            "status": "local",
            "modelType": model_type,
            "details": config_data
        })
        
    # Sort models for consistent ordering
    models_list.sort(key=lambda x: x['name'])
    return models_list


def build_markdown_from_messages(messages):
    """Builds a complete Markdown string from a list of messages for export."""
    message_blocks = []
    for message in messages:
        role = message.get('role')
        content = message.get('content', '')
        
        thoughts_block_content = ""
        # The frontend now parses thoughts into this structure
        if message.get('collapsibleBlocks'):
            for block in message.get('collapsibleBlocks'):
                if block.get('title') == 'thought':
                    thoughts = block.get('content', '')
                    # Use HTML <details> for collapsible sections in the final HTML export
                    thoughts_block_content += f"<details><summary>思考过程</summary><pre><code>{thoughts}</code></pre></details>\n\n"

        prefix_user = "#### **我:**"
        prefix_ai = "#### **AI:**"
        
        block_content = ""
        if role == 'user':
            block_content = content
        else:
            # Prepend thoughts to the assistant's content
            block_content = f"{thoughts_block_content}{content}"

        block = f"{prefix_user if role == 'user' else prefix_ai}\n\n{block_content.strip()}"
        message_blocks.append(block)
        
    return "\n\n---\n\n".join(message_blocks)


def format_data_to_markdown(data):
    """
    Intelligently formats a Python object (dict, list, str) into a readable Markdown string.
    """
    
    if isinstance(data, str):
        # If it's already a string that looks like markdown, return it.
        # Otherwise, wrap it in a code block.
        if data.strip().startswith(('#', '-', '```', '*')):
            return data
        return f"```text\n{data}\n```"

    if not isinstance(data, (dict, list)):
        return f"```text\n{str(data)}\n```"

    try:
        # Case A: It's a list of dictionaries (most common for APIs)
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            markdown_parts = []
            for item in data:
                # Heuristically find a title and a link
                title = item.get('title') or item.get('name')
                link = item.get('link') or item.get('url')
                description = item.get('description') or item.get('desc') or item.get('summary')

                if title and link:
                    line = f"- **[{title}]({link})**"
                elif title:
                    line = f"- **{title}**"
                else:
                    # If no obvious title, just present the first value as a header if possible
                    first_key = next(iter(item), None)
                    if first_key:
                        line = f"- **{item[first_key]}**"
                    else: # empty dict
                        line = "- (Empty item)"

                markdown_parts.append(line)
                if description:
                    markdown_parts.append(f"  - {description}")
            
            return "\n".join(markdown_parts)

        # Case B: It's a dictionary that might wrap a list (e.g., {'code': 200, 'data': [...]})
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            return format_data_to_markdown(data['data'])

        # Fallback: Pretty-print the JSON in a code block for any other structure
        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
        return f"```json\n{pretty_json}\n```"

    except Exception:
        # Super safe fallback in case of any formatting error
        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
        return f"```json\n{pretty_json}\n```"
