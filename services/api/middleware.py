"""Security and rate limiting middleware."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.
    
    Limits requests per IP address to prevent abuse.
    """
    
    def __init__(
        self,
        app: any,
        *,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ) -> None:
        """Initialize rate limiter.
        
        Args:
            app: FastAPI application.
            requests_per_minute: Maximum requests per minute per IP.
            requests_per_hour: Maximum requests per hour per IP.
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests: dict[str, list[float]] = defaultdict(list)
        self.hour_requests: dict[str, list[float]] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old entries
        current_time = time.time()
        self._clean_old_entries(client_ip, current_time)
        
        # Check rate limits
        if len(self.minute_requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
        
        if len(self.hour_requests[client_ip]) >= self.requests_per_hour:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hourly rate limit exceeded. Please try again later.",
            )
        
        # Record request
        self.minute_requests[client_ip].append(current_time)
        self.hour_requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response
    
    def _clean_old_entries(self, client_ip: str, current_time: float) -> None:
        """Remove old entries from rate limit tracking."""
        # Remove entries older than 1 minute
        self.minute_requests[client_ip] = [
            t for t in self.minute_requests[client_ip]
            if current_time - t < 60
        ]
        
        # Remove entries older than 1 hour
        self.hour_requests[client_ip] = [
            t for t in self.hour_requests[client_ip]
            if current_time - t < 3600
        ]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

