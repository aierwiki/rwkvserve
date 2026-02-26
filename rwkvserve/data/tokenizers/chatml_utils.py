"""
ChatML Tokenizer utility module.

Provides systematic ChatML format support including:
1. Configuration validation
2. Initialization checks
3. Runtime verification
4. Error handling
"""

import os
import json
from typing import Optional, Dict, Any, List, Tuple
from jinja2 import Template, TemplateError


CHATML_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}

CHATML_ROLES = ["system", "user", "assistant"]


class ChatMLConfigError(Exception):
    """ChatML configuration error."""
    pass


class ChatMLValidationError(Exception):
    """ChatML validation error."""
    pass


def validate_chatml_tokenizer(tokenizer) -> Tuple[bool, List[str]]:
    """
    Validate the completeness of a tokenizer's ChatML configuration.

    Returns:
        (is_valid, error_messages): Whether valid, and list of error messages.
    """
    errors = []

    if not hasattr(tokenizer, 'chat_template'):
        errors.append("tokenizer is missing chat_template attribute")
        return False, errors

    if tokenizer.chat_template is None:
        errors.append("tokenizer.chat_template is None")
        return False, errors

    if '<|im_start|>' not in tokenizer.chat_template:
        errors.append("chat_template does not contain ChatML marker <|im_start|>")
        return False, errors

    if '<|im_end|>' not in tokenizer.chat_template:
        errors.append("chat_template does not contain ChatML marker <|im_end|>")
        return False, errors

    if not hasattr(tokenizer, 'additional_special_tokens'):
        errors.append("tokenizer is missing additional_special_tokens attribute")
        return False, errors

    im_start_token = CHATML_TOKENS["im_start"]
    if im_start_token not in getattr(tokenizer, 'additional_special_tokens', {}):
        try:
            token_ids = tokenizer.encode(im_start_token)
            if not token_ids:
                errors.append(f"Cannot encode ChatML token: {im_start_token}")
        except Exception as e:
            errors.append(f"Error encoding ChatML token {im_start_token}: {e}")

    im_end_token = CHATML_TOKENS["im_end"]
    if im_end_token not in getattr(tokenizer, 'additional_special_tokens', {}):
        try:
            token_ids = tokenizer.encode(im_end_token)
            if not token_ids:
                errors.append(f"Cannot encode ChatML token: {im_end_token}")
        except Exception as e:
            errors.append(f"Error encoding ChatML token {im_end_token}: {e}")

    if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
        errors.append("Warning: tokenizer.eos_token is not set")

    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        errors.append("Warning: tokenizer.pad_token is not set")

    try:
        template = Template(tokenizer.chat_template)
        test_messages = [{"role": "user", "content": "test"}]
        template.render(messages=test_messages, add_generation_prompt=False)
    except TemplateError as e:
        errors.append(f"chat_template Jinja2 syntax error: {e}")
    except Exception as e:
        errors.append(f"chat_template render test failed: {e}")

    critical_errors = [e for e in errors if not e.startswith("Warning:")]
    return len(critical_errors) == 0, errors


def setup_chatml_tokenizer(
    tokenizer,
    chat_template_path: Optional[str] = None,
    verify: bool = True,
    raise_on_error: bool = True
) -> Dict[str, Any]:
    """
    Systematically configure a tokenizer for ChatML format support.

    Args:
        tokenizer: RWKVTokenizer instance.
        chat_template_path: Path to ChatML template file (optional).
        verify: Whether to verify after configuration.
        raise_on_error: Whether to raise on error.

    Returns:
        Dict with setup results and diagnostics.
    """
    results = {
        "success": False,
        "added_tokens": 0,
        "template_set": False,
        "warnings": [],
        "errors": []
    }

    try:
        chatml_tokens = [CHATML_TOKENS["im_start"], CHATML_TOKENS["im_end"]]
        special_tokens_dict = {
            "additional_special_tokens": chatml_tokens,
            "eos_token": CHATML_TOKENS["im_end"],
            "pad_token": CHATML_TOKENS["im_end"],
        }

        added_count = tokenizer.add_special_tokens(special_tokens_dict)
        results["added_tokens"] = added_count

        if chat_template_path and os.path.exists(chat_template_path):
            with open(chat_template_path, 'r', encoding='utf-8') as f:
                chatml_template = f.read()
            tokenizer.chat_template = chatml_template
            results["template_set"] = True
            results["template_source"] = "file"
        else:
            chatml_template = """{% for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
            tokenizer.chat_template = chatml_template
            results["template_set"] = True
            results["template_source"] = "inline"
            if chat_template_path:
                results["warnings"].append(f"Template file not found: {chat_template_path}, using inline template")

        if verify:
            is_valid, validation_errors = validate_chatml_tokenizer(tokenizer)
            if not is_valid:
                critical_errors = [e for e in validation_errors if not e.startswith("Warning:")]
                warnings = [e for e in validation_errors if e.startswith("Warning:")]
                results["errors"].extend(critical_errors)
                results["warnings"].extend(warnings)

                if critical_errors and raise_on_error:
                    raise ChatMLConfigError(
                        f"ChatML configuration validation failed:\n" + "\n".join(critical_errors)
                    )
            else:
                results["success"] = True

        return results

    except Exception as e:
        error_msg = f"Error during ChatML configuration: {e}"
        results["errors"].append(error_msg)
        if raise_on_error:
            raise ChatMLConfigError(error_msg) from e
        return results


def verify_chatml_formatting(
    tokenizer,
    test_messages: Optional[List[Dict[str, str]]] = None,
    raise_on_error: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that ChatML formatting works correctly.

    Args:
        tokenizer: Configured tokenizer.
        test_messages: Test messages (defaults to standard test).
        raise_on_error: Whether to raise on error.

    Returns:
        (is_valid, diagnostics)
    """
    diagnostics = {
        "success": False,
        "errors": [],
        "warnings": [],
        "test_results": {}
    }

    if test_messages is None:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

    try:
        try:
            formatted_text = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            diagnostics["test_results"]["format_without_prompt"] = {
                "success": True,
                "output_length": len(formatted_text),
                "contains_im_start": "<|im_start|>" in formatted_text,
                "contains_im_end": "<|im_end|>" in formatted_text,
            }
        except Exception as e:
            diagnostics["errors"].append(f"apply_chat_template (no generation prompt) failed: {e}")
            diagnostics["test_results"]["format_without_prompt"] = {"success": False, "error": str(e)}

        try:
            formatted_text_with_prompt = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            diagnostics["test_results"]["format_with_prompt"] = {
                "success": True,
                "output_length": len(formatted_text_with_prompt),
                "contains_im_start": "<|im_start|>" in formatted_text_with_prompt,
                "contains_im_end": "<|im_end|>" in formatted_text_with_prompt,
                "ends_with_assistant": formatted_text_with_prompt.rstrip().endswith("<|im_start|>assistant"),
            }
        except Exception as e:
            diagnostics["errors"].append(f"apply_chat_template (with generation prompt) failed: {e}")
            diagnostics["test_results"]["format_with_prompt"] = {"success": False, "error": str(e)}

        try:
            formatted_text = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            token_ids = tokenizer.encode(formatted_text)
            decoded_text = tokenizer.decode(token_ids)

            diagnostics["test_results"]["tokenization"] = {
                "success": True,
                "token_count": len(token_ids),
                "roundtrip_match": formatted_text.strip() == decoded_text.strip(),
            }

            if not diagnostics["test_results"]["tokenization"]["roundtrip_match"]:
                diagnostics["warnings"].append(
                    "Tokenization roundtrip does not match exactly (may be normal for small differences)"
                )
        except Exception as e:
            diagnostics["errors"].append(f"Tokenization test failed: {e}")
            diagnostics["test_results"]["tokenization"] = {"success": False, "error": str(e)}

        try:
            im_start_ids = tokenizer.encode(CHATML_TOKENS["im_start"])
            im_end_ids = tokenizer.encode(CHATML_TOKENS["im_end"])

            diagnostics["test_results"]["special_token_ids"] = {
                "im_start_ids": im_start_ids,
                "im_end_ids": im_end_ids,
                "eos_token_id": getattr(tokenizer, 'eos_token_id', None),
                "pad_token_id": getattr(tokenizer, 'pad_token_id', None),
            }
        except Exception as e:
            diagnostics["errors"].append(f"Special token ID check failed: {e}")

        diagnostics["success"] = len(diagnostics["errors"]) == 0

        if not diagnostics["success"] and raise_on_error:
            raise ChatMLValidationError(
                f"ChatML formatting validation failed:\n" + "\n".join(diagnostics["errors"])
            )

        return diagnostics["success"], diagnostics

    except Exception as e:
        diagnostics["errors"].append(f"Verification error: {e}")
        if raise_on_error:
            raise ChatMLValidationError(f"ChatML verification error: {e}") from e
        return False, diagnostics


def print_chatml_diagnostics(tokenizer, verbose: bool = True):
    """
    Print ChatML configuration diagnostic information.

    Args:
        tokenizer: Tokenizer to diagnose.
        verbose: Whether to print detailed information.
    """
    print("\n" + "=" * 80)
    print("ChatML Tokenizer Diagnostics")
    print("=" * 80)

    is_valid, errors = validate_chatml_tokenizer(tokenizer)
    print(f"\nConfiguration: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"   {error}")

    format_valid, format_diag = verify_chatml_formatting(tokenizer, raise_on_error=False)
    print(f"\nFormatting: {'OK' if format_valid else 'ERROR'}")

    if verbose:
        print("\nDetailed test results:")
        for test_name, result in format_diag.get("test_results", {}).items():
            print(f"\n   {test_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"     - {key}: {value}")

    print("\nSpecial Token Info:")
    print(f"   - im_start: {CHATML_TOKENS['im_start']}")
    print(f"   - im_end: {CHATML_TOKENS['im_end']}")
    print(f"   - eos_token: {getattr(tokenizer, 'eos_token', 'N/A')}")
    print(f"   - pad_token: {getattr(tokenizer, 'pad_token', 'N/A')}")
    print(f"   - eos_token_id: {getattr(tokenizer, 'eos_token_id', 'N/A')}")
    print(f"   - pad_token_id: {getattr(tokenizer, 'pad_token_id', 'N/A')}")

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("\nChat Template Preview:")
        template_preview = tokenizer.chat_template[:200] + "..." if len(tokenizer.chat_template) > 200 else tokenizer.chat_template
        print(f"   {template_preview}")

    print("\n" + "=" * 80 + "\n")
