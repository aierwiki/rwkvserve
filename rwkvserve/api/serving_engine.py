"""
Base Serving Engine

Provides common functionality and base class for API services:
- Model and tokenizer management
- Error handling
- Request ID generation
- Parameter validation
"""

import asyncio
import torch
import uuid
from typing import Optional

from rwkvserve.api.protocol import ErrorResponse


class ServingEngine:
    """
    Base serving engine providing shared functionality for API services.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        model_name: str = "rwkv-7",
        served_model_name: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.device = next(model.parameters()).device
    
    def create_error_response(
        self,
        message: str,
        error_type: str = "BadRequestError",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> ErrorResponse:
        return ErrorResponse(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )
    
    def generate_request_id(self, prefix: str = "req") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:16]}"
    
    def _count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def validate_request_params(
        self,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
    ) -> Optional[ErrorResponse]:
        if temperature < 0.0 or temperature > 2.0:
            return self.create_error_response(
                message="temperature must be between 0.0 and 2.0",
                param="temperature",
            )
        
        if top_p < 0.0 or top_p > 1.0:
            return self.create_error_response(
                message="top_p must be between 0.0 and 1.0",
                param="top_p",
            )
        
        if max_tokens is not None and max_tokens < 1:
            return self.create_error_response(
                message="max_tokens must be at least 1",
                param="max_tokens",
            )
        
        return None
