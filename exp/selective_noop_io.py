"""
Selective No-op I/O Module

This module provides functions for selectively silencing I/O operations during training,
specifically targeting J1/J2 visualization files while preserving original training
visualizations.

Functions:
- install_selective_noop_sinks: Replace I/O functions with selective no-op versions

This module is designed to suppress only J1/J2 specific visualizations while
retaining original training renders in the log/ directory.
"""

import os
import builtins
import types
import cv2
from PIL import Image


# Save original functions for selective replacement
original_cv2_imwrite = cv2.imwrite
original_pil_save = Image.save


def selective_cv2_imwrite(filename, img, *args, **kwargs):
    """
    Selective cv2.imwrite: Only suppress reports/ directory file saves.
    
    Args:
        filename: File path to save
        img: Image data
        *args, **kwargs: Additional arguments
        
    Returns:
        False if suppressed, original return value otherwise
    """
    # Check if file path contains reports/ (J1/J2 new visualizations)
    if 'reports/' in str(filename):
        print(f"[Selective No-op] Suppressed cv2.imwrite: {filename}")
        return False  # Return False but don't actually save
    else:
        # For other paths (like log/ directory), save normally
        return original_cv2_imwrite(filename, img, *args, **kwargs)


def selective_pil_save(self, filename, *args, **kwargs):
    """
    Selective PIL.Image.save: Only suppress reports/ directory file saves.
    
    Args:
        self: PIL Image instance
        filename: File path to save
        *args, **kwargs: Additional arguments
        
    Returns:
        None if suppressed, original return value otherwise
    """
    # Check if file path contains reports/ (J1/J2 new visualizations)
    if 'reports/' in str(filename):
        print(f"[Selective No-op] Suppressed PIL.Image.save: {filename}")
        return None  # Return None but don't actually save
    else:
        # For other paths (like log/ directory), save normally
        return original_pil_save(self, filename, *args, **kwargs)


def install_selective_noop_sinks(mute_print=False):
    """
    Selective silent mode: Replace common image saving functions with selective no-op.
    Only suppress reports/ directory visualization output, preserve original training visualizations.
    
    Args:
        mute_print: Whether to also mute console print statements (default: False)
        
    Returns:
        None
        
    Side effects:
        - Replaces cv2.imwrite with selective version
        - Replaces PIL.Image.save with selective version
        - Suppresses matplotlib saves (usually for J1/J2 visualizations)
        - Optionally replaces builtins.print with quiet version
    """
    print("[Selective No-op] Installing selective no-op sinks...")
    
    # Replace cv2.imwrite with selective version
    cv2.imwrite = selective_cv2_imwrite
    print("[Selective No-op] cv2.imwrite replaced with selective version")
    
    # Replace PIL.Image.save with selective version
    try:
        Image.save = selective_pil_save
        print("[Selective No-op] PIL.Image.save replaced with selective version")
    except Exception as e:
        print(f"[Selective No-op] Failed to replace PIL.Image.save: {e}")
    
    # Suppress matplotlib saves (usually for J1/J2 visualizations)
    try:
        import matplotlib.pyplot as plt
        def _nofig(*args, **kwargs): 
            return None
        plt.savefig = _nofig
        plt.clf = _nofig
        plt.close = _nofig
        print("[Selective No-op] matplotlib.pyplot functions replaced with no-op")
    except Exception as e:
        print(f"[Selective No-op] Failed to replace matplotlib functions: {e}")
    
    # Optional: Mute console output
    if mute_print:
        try:
            builtins_print = builtins.print
            
            def _quiet_print(*args, **kwargs):
                # Allow critical warnings to pass through
                txt = " ".join(map(str, args))
                if "[WARN]" in txt or "[ERROR]" in txt or "[CRITICAL]" in txt:
                    return builtins_print(*args, **kwargs)
                # Mute everything else
                return None
            
            builtins.print = _quiet_print
            print("[Selective No-op] Console output muted (warnings/errors still shown)")
        except Exception as e:
            print(f"[Selective No-op] Failed to mute console output: {e}")
    
    print("[Selective No-op] Selective no-op sinks installation complete")


def restore_selective_io():
    """
    Restore original I/O functions from selective no-op mode.
    
    Returns:
        None
    """
    print("[Selective No-op] Restoring original I/O functions...")
    
    try:
        # Restore cv2.imwrite
        cv2.imwrite = original_cv2_imwrite
        print("[Selective No-op] cv2.imwrite restored")
    except Exception as e:
        print(f"[Selective No-op] Failed to restore cv2.imwrite: {e}")
    
    try:
        # Restore PIL.Image.save
        Image.save = original_pil_save
        print("[Selective No-op] PIL.Image.save restored")
    except Exception as e:
        print(f"[Selective No-op] Failed to restore PIL.Image.save: {e}")
    
    try:
        # Restore print function
        import builtins
        if hasattr(builtins, '_original_print'):
            builtins.print = builtins._original_print
            print("[Selective No-op] Print function restored")
    except Exception as e:
        print(f"[Selective No-op] Failed to restore print function: {e}")
    
    print("[Selective No-op] I/O restoration complete")


def check_selective_status():
    """
    Check the status of selective no-op replacements.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "cv2_selective": False,
        "pil_selective": False,
        "matplotlib": False,
        "print": False
    }
    
    try:
        status["cv2_selective"] = cv2.imwrite.__name__ == "selective_cv2_imwrite"
    except:
        pass
    
    try:
        status["pil_selective"] = Image.save.__name__ == "selective_pil_save"
    except:
        pass
    
    try:
        import matplotlib.pyplot as plt
        status["matplotlib"] = plt.savefig.__name__ == "_nofig"
    except:
        pass
    
    try:
        import builtins
        status["print"] = builtins.print.__name__ == "_quiet_print"
    except:
        pass
    
    return status


def test_selective_mode():
    """
    Test selective no-op mode by attempting to save files to different directories.
    
    Returns:
        Dictionary with test results
    """
    import tempfile
    import numpy as np
    
    results = {
        "reports_suppressed": False,
        "log_preserved": False,
        "other_preserved": False
    }
    
    try:
        # Test reports/ directory (should be suppressed)
        with tempfile.NamedTemporaryFile(suffix=".png", dir="reports", delete=False) as f:
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = cv2.imwrite(f.name, test_img)
            results["reports_suppressed"] = (result == False)
            os.unlink(f.name)
    except Exception as e:
        print(f"[Selective No-op] Test failed for reports/: {e}")
    
    try:
        # Test log/ directory (should be preserved)
        with tempfile.NamedTemporaryFile(suffix=".png", dir="log", delete=False) as f:
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = cv2.imwrite(f.name, test_img)
            results["log_preserved"] = (result == True)
            os.unlink(f.name)
    except Exception as e:
        print(f"[Selective No-op] Test failed for log/: {e}")
    
    try:
        # Test other directory (should be preserved)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = cv2.imwrite(f.name, test_img)
            results["other_preserved"] = (result == True)
            os.unlink(f.name)
    except Exception as e:
        print(f"[Selective No-op] Test failed for other directory: {e}")
    
    return results