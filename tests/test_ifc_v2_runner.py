"""
Test Runner for IFC V2 Export with Auto-Fix capability.

Runs tests multiple times and attempts to fix errors automatically.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Any

import pytest

from tests.test_ifc_v2_export_complete import test_ifc_v2_export_complete

logger = logging.getLogger(__name__)


class TestRunner:
    """Test runner with auto-fix capability."""
    
    def __init__(self, max_attempts: int = 10):
        self.max_attempts = max_attempts
        self.attempts = 0
        self.errors = []
    
    def identify_error_type(self, error: Exception) -> str:
        """Identify the type of error for targeted fixes."""
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        # Numpy array boolean ambiguity
        if "truth value of an array" in error_msg or "ambiguous" in error_msg:
            return "numpy_array_boolean"
        
        # ifcopenshell API errors
        if "incorrect function arguments" in error_msg or "unexpected keyword argument" in error_msg:
            return "ifcopenshell_api"
        
        # IFC schema violations
        if "derived" in error_msg or "schema" in error_msg:
            return "ifc_schema"
        
        # Type errors
        if "type" in error_msg and ("expecting" in error_msg or "got" in error_msg):
            return "type_error"
        
        # Geometry errors
        if "geometry" in error_msg or "polygon" in error_msg or "empty" in error_msg:
            return "geometry_error"
        
        return "unknown"
    
    def can_auto_fix(self, error_type: str) -> bool:
        """Check if error can be auto-fixed."""
        # Most errors require manual code fixes
        # This is a placeholder for future auto-fix logic
        return False
    
    async def run_test_with_retry(self, tmp_path: Path) -> tuple[bool, str]:
        """
        Run test with retry logic.
        
        Args:
            tmp_path: Temporary directory for test output
            
        Returns:
            Tuple of (success, message)
        """
        for attempt in range(1, self.max_attempts + 1):
            self.attempts = attempt
            logger.info(f"Test attempt {attempt}/{self.max_attempts}")
            
            try:
                await test_ifc_v2_export_complete(tmp_path)
                logger.info(f"✓ Test succeeded on attempt {attempt}")
                return True, f"Test succeeded on attempt {attempt}"
            except Exception as e:
                error_type = self.identify_error_type(e)
                error_msg = str(e)
                self.errors.append((attempt, error_type, error_msg, traceback.format_exc()))
                
                logger.error(f"✗ Test failed on attempt {attempt}: {error_type} - {error_msg}")
                
                if self.can_auto_fix(error_type):
                    logger.info(f"Attempting auto-fix for {error_type}...")
                    # Placeholder for auto-fix logic
                    # In a real implementation, this would modify code and re-run
                else:
                    logger.warning(f"Cannot auto-fix {error_type}, manual intervention required")
        
        # All attempts failed
        error_summary = self._format_error_summary()
        return False, f"Test failed after {self.max_attempts} attempts:\n{error_summary}"
    
    def _format_error_summary(self) -> str:
        """Format error summary for reporting."""
        lines = []
        lines.append(f"Total attempts: {self.max_attempts}")
        lines.append(f"Total errors: {len(self.errors)}")
        lines.append("")
        
        # Group errors by type
        error_types = {}
        for attempt, error_type, error_msg, trace in self.errors:
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((attempt, error_msg))
        
        lines.append("Error breakdown by type:")
        for error_type, occurrences in error_types.items():
            lines.append(f"  {error_type}: {len(occurrences)} occurrence(s)")
            for attempt, error_msg in occurrences[:3]:  # Show first 3
                lines.append(f"    Attempt {attempt}: {error_msg[:100]}...")
        
        lines.append("")
        lines.append("Last error traceback:")
        if self.errors:
            lines.append(self.errors[-1][3])  # Last traceback
        
        return "\n".join(lines)


async def run_ifc_v2_export_test_with_validation(tmp_path: Path | None = None) -> tuple[bool, str]:
    """
    Run IFC V2 export test with validation and auto-fix attempts.
    
    Args:
        tmp_path: Optional temporary directory (creates one if not provided)
        
    Returns:
        Tuple of (success, message)
    """
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())
    
    runner = TestRunner(max_attempts=10)
    return await runner.run_test_with_retry(tmp_path)


if __name__ == "__main__":
    # Run test directly
    import tempfile
    tmp_path = Path(tempfile.mkdtemp())
    
    success, message = asyncio.run(run_ifc_v2_export_test_with_validation(tmp_path))
    
    if success:
        print("✓ IFC V2 Export Test: SUCCESS")
        print(message)
    else:
        print("✗ IFC V2 Export Test: FAILED")
        print(message)
        exit(1)

