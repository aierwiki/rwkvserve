"""
Chat Templates for RWKVServe API
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent
DEFAULT_TEMPLATE = "rwkv_default.jinja"


def get_template_path(template_name: str) -> Path:
    """Get the full path for a template file."""
    return TEMPLATES_DIR / template_name


def get_default_template() -> str:
    """Get the default template content."""
    template_path = get_template_path(DEFAULT_TEMPLATE)
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def list_available_templates():
    """List all available template names."""
    templates = []
    for file in TEMPLATES_DIR.glob("*.jinja"):
        templates.append(file.name)
    return templates
